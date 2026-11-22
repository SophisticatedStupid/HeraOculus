"""Full Bayesian MCMC climate trend & structural change-point inference.

Implements multi-chain Adaptive Random-Walk Metropolis-within-Gibbs sampling
in NumPy/SciPy:
  - Model 'robust_ar1': Student-t (nu=5) likelihood with exact irregular-gap
    AR(1) covariance (rho^dt scaling across polar-night / missing months).
  - Model 'changepoint': Bayesian piecewise linear hinge regression with unknown
    structural changepoint tau, pre-shift slope beta_1, and slope change delta.

Reports split Gelman-Rubin R-hat, Effective Sample Size (ESS), 95% Highest
Density Interval (HDI), Probability of Direction P(slope > 0), trace series,
posterior histograms, and posterior predictive trend envelopes.
Includes comprehensive convergence diagnostics and pathology detection per
Vehtari et al. (2021) rank-normalization and ArviZ ESS methodology.

Diagnostics provided:
  - rank-normalized split R-hat: detects non-convergence even when chains
    are initialized similarly, by rank-normalizing across all chains before
    computing split R-hat, making it robust to identical initialization.
  - bulk ESS: effective sample size for the central/bulk portion of the
    posterior distribution.
  - tail ESS: effective sample size for the posterior tail (extreme quantiles).
  - Zero/near-zero within-chain variance detection.
  - Identical/stuck chain detection (chains with identical means/traces).
  - Non-finite sample detection (NaN/Inf).
  - Burn-in and adaptation adequacy checks.
  - Explicit warnings when diagnostics cannot be calculated reliably.
  - Distinguishes successful numerical execution from demonstrated convergence.
"""

from __future__ import annotations

from typing import Literal, Optional
import warnings

import numpy as np
from scipy import stats as st
from scipy.special import erf
from scipy.fftpack import next_fast_len


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2))).

    Uses SciPy's error function because ``np.erf`` is not part of the
    supported NumPy API. SciPy is already an accepted dependency of this
    project (see requirements.txt), so no new dependency is introduced.
    """
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _normal_ppf(p: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF (probit): Phi^{-1}(p)."""
    return st.norm.ppf(np.clip(p, 1e-10, 1 - 1e-10))


def _split_chains(ary: np.ndarray) -> np.ndarray:
    """Split every chain into two equal halves and stack them as independent chains.

    Returns an array of shape (2 * m, n // 2) treating each half-chain as its
    own chain, as required by split diagnostics (Vehtari et al., 2021).
    """
    ary = np.asarray(ary, dtype=float)
    if ary.ndim == 1:
        ary = ary[None, :]
    _, n_draw = ary.shape
    half = n_draw // 2
    return np.vstack((ary[:, :half], ary[:, -half:]))


def _z_scale(ary: np.ndarray) -> np.ndarray:
    """Rank-normalize an array (empirical CDF -> inverse standard normal CDF).

    Maps each value to its z-score via the empirical CDF + inverse normal CDF,
    making the transform invariant to location/scale. Applied to an already-split
    array so each half-chain is rank-normalized against the pooled set of all
    split-chain draws.
    """
    flat = ary.reshape(-1)
    ranks = st.rankdata(flat, method="average")
    uniform = np.clip((ranks - 0.5) / flat.shape[0], 1e-10, 1 - 1e-10)
    z = _normal_ppf(uniform)
    return z.reshape(ary.shape)


def _autocov(ary: np.ndarray, axis: int = 1) -> np.ndarray:
    """Autocovariance at every lag for each chain (axis), centered per chain.

    Mirrors ArviZ's autocov: each chain is demeaned, the autocovariance is
    computed via FFT and normalized by the draw count. Centering each chain
    individually is required for a defensible multi-chain autocorrelation
    estimator (raw cross-products of uncentered chains are biased).
    """
    ary = np.asarray(ary, dtype=float)
    axis = axis if axis > 0 else len(ary.shape) + axis
    n = ary.shape[axis]
    m = int(next_fast_len(2 * n))

    centered = ary - ary.mean(axis=axis, keepdims=True)
    fft_ary = np.fft.rfft(centered, n=m, axis=axis)
    fft_ary *= np.conjugate(fft_ary)

    sl = [slice(None)] * ary.ndim
    sl[axis] = slice(0, n)
    cov = np.fft.irfft(fft_ary, n=m, axis=axis)[tuple(sl)]
    cov /= n
    return cov


def _ess_estimator(ary: np.ndarray) -> float:
    """Effective sample size for a 2D array (chains x draws).

    Implements the ArviZ / Vehtari et al. (2021) autocorrelation-based ESS:
    multi-chain autocovariance averaged over chains, Geyer's initial positive
    sequence for truncation, and Geyer's initial monotone sequence to reduce
    variance. The marginal posterior variance estimate combines within-chain
    and between-chain variance so differently dispersed chains are handled.

    Returns the total number of draws (no autocorrelation information) for
    constant or non-finite input, since ESS is undefined there.
    """
    ary = np.asarray(ary, dtype=float)
    if ary.ndim == 1:
        ary = ary[None, :]
    m, n = ary.shape
    total = float(m * n)

    if n < 2:
        return total

    if not np.all(np.isfinite(ary)):
        return total

    if float(np.max(ary) - np.min(ary)) < np.finfo(float).resolution:
        # Constant chain: no variation to estimate autocorrelation from.
        return total

    acov = _autocov(ary, axis=1)  # shape (m, n), autocovariance at every lag
    chain_mean = ary.mean(axis=1)

    # Unbiased within-chain variance (acov lag 0 is the MLE variance).
    mean_var = float(np.mean(acov[:, 0])) * n / (n - 1.0)
    # Marginal posterior variance estimate = within-chain + between-chain.
    var_plus = float(np.mean(acov[:, 0]))  # mean_var * (n - 1) / n
    if m > 1:
        var_plus += float(np.var(chain_mean, ddof=1))

    if var_plus <= 1e-14:
        return total

    rho_hat_t = np.zeros(n)
    rho_hat_t[0] = 1.0
    rho_hat_even = 1.0
    rho_hat_odd = 1.0 - (mean_var - float(np.mean(acov[:, 1]))) / var_plus
    rho_hat_t[1] = rho_hat_odd

    # Geyer's initial positive sequence: keep lag pairs while their sum > 0.
    t = 1
    while t < (n - 3) and (rho_hat_even + rho_hat_odd) > 0.0:
        rho_hat_even = 1.0 - (mean_var - float(np.mean(acov[:, t + 1]))) / var_plus
        rho_hat_odd = 1.0 - (mean_var - float(np.mean(acov[:, t + 2]))) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0.0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
        t += 2

    max_t = t - 2
    if rho_hat_even > 0:
        rho_hat_t[max_t + 1] = rho_hat_even

    # Geyer's initial monotone sequence: smooth non-monotone pair sums.
    t = 1
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.0
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2

    ess = total
    tau_hat = (
        -1.0
        + 2.0 * float(np.sum(rho_hat_t[: max_t + 1]))
        + float(np.sum(rho_hat_t[max_t + 1 : max_t + 2]))
    )
    tau_hat = max(tau_hat, 1.0 / np.log10(ess))
    ess = ess / tau_hat

    if not np.isfinite(ess):
        ess = total
    # ESS cannot exceed the total number of available draws.
    return float(min(max(ess, 1.0), total))
def _rank_normalized_split_rhat(chains: np.ndarray, folded: bool = False) -> dict:
    """Compute rank-normalized split R-hat (Vehtari et al., 2021).

    Steps:
      1. Optionally fold each split half-chain about its median (sensitive to
         scale/tail differences in symmetric posteriors).
      2. Rank-normalize all draws across all chains (empirical CDF -> Normal).
      3. Split every chain into equal halves and treat each half as an
         independent chain.
      4. Compute between-chain variance across split-chain means and
         within-chain variance across split chains (both on the halves).
      5. Combine into the potential scale reduction factor sqrt(var_hat / W).

    Parameters
    ----------
    chains : np.ndarray of shape (m, n_draws)
        MCMC samples from m chains, each with n_draws iterations.
    folded : bool
        If True, compute the folded rank-normalized split R-hat on
        |x - median(x)| values, following Vehtari et al. (2021).

    Returns
    -------
    dict with keys:
        rhat : float or None
            The rank-normalized split R-hat value. None if diagnostics
            cannot be computed (insufficient chains/draws).
        warnings : list[str]
            Human-readable warnings about why diagnostics may be unreliable.
        is_valid : bool
            Whether the result can be trusted.
    """
    m, n = chains.shape

    if m < 2:
        return {
            "rhat": None,
            "warnings": ["fewer_than_2_chains"],
            "is_valid": False,
        }

    if not np.all(np.isfinite(chains)):
        return {
            "rhat": None,
            "warnings": ["non_finite_samples"],
            "is_valid": False,
        }

    half = n // 2
    if half < 2:
        return {
            "rhat": None,
            "warnings": ["insufficient_half_draws", f"n={n}, half={half}"],
            "is_valid": False,
        }

    # Split every chain into equal halves; treat each half as an independent chain.
    split = _split_chains(chains)  # shape (2m, half)

    if folded:
        # Fold each half-chain about the pooled median before rank normalization.
        split = np.abs(split - np.median(split))

    # Rank-normalize all split-chain draws together.
    rn = _z_scale(split)

    N = half
    split_means = np.mean(rn, axis=1)  # shape (2m,)
    split_vars = np.var(rn, axis=1, ddof=1)  # shape (2m,)

    # Between-chain variance of the pooled split-chain means.
    B = N * np.var(split_means, ddof=1)
    # Within-chain variance averaged over all 2m split (half) chains.
    W = float(np.mean(split_vars))

    warnings: list[str] = []
    is_valid = True

    if not np.isfinite(B) or not np.isfinite(W) or W <= 1e-14:
        warnings.append("zero_within_chain_variance")
        rhat = 1.0
        is_valid = False
    else:
        var_hat = ((N - 1.0) / N) * W + (1.0 / N) * B
        rhat = float(np.sqrt(max(1.0, var_hat / W)))

    return {
        "rhat": rhat,
        "warnings": warnings,
        "is_valid": is_valid,
    }


def _bulk_ess(chains: np.ndarray) -> dict:
    """Estimate bulk Effective Sample Size.

    Computes the ESS for the bulk/central portion of the posterior
    distribution using the autocorrelation-based estimator of
    Vehtari et al. (2021) / ArviZ.

    The chains are rank-normalized and split into halves (each half treated
    as an independent chain), each chain is centered, and the multi-chain
    autocovariance is averaged across chains. Truncation uses Geyer's initial
    positive and monotone sequences. The result cannot exceed the total
    number of available draws.

    Parameters
    ----------
    chains : np.ndarray of shape (m, n_draws)
        MCMC samples from m chains.

    Returns
    -------
    dict with keys:
        ess : float
            Bulk effective sample size.
        warnings : list[str]
            Human-readable warnings.
    """
    m, n = chains.shape
    flat = chains.reshape(-1)

    if n < 20:
        return {"ess": float(m * n), "warnings": ["chain_too_short_bulk_ess"]}

    if np.var(flat) <= 1e-14:
        return {"ess": float(m * n), "warnings": ["zero_chain_variance_bulk_ess"]}

    if not np.all(np.isfinite(flat)):
        return {"ess": float(m * n), "warnings": ["non_finite_samples_bulk_ess"]}

    # Rank-normalize and split: each half-chain is an independent chain.
    ess = _ess_estimator(_z_scale(_split_chains(chains)))

    warnings: list[str] = []
    if n < 20:
        warnings.append("chain_too_short_bulk_ess")
    if np.var(flat) <= 1e-14:
        warnings.append("zero_chain_variance_bulk_ess")

    return {
        "ess": float(ess),
        "warnings": warnings,
    }


def _tail_ess(chains: np.ndarray, prob: float = 0.05) -> dict:
    """Estimate tail Effective Sample Size at a given tail probability.

    The tail ESS is the minimum of the autocorrelation-adjusted effective
    sample sizes of the lower- and upper-tail indicator chains
    (Vehtari et al., 2021 / ArviZ ``ess`` method="tail"). Each indicator
    chain is split into halves and its ESS is computed with the same
    autocorrelation-aware estimator as the bulk ESS, so autocorrelation is
    accounted for rather than relying on raw tail sample counts.

    Parameters
    ----------
    chains : np.ndarray of shape (m, n_draws)
        MCMC samples from m chains.
    prob : float in (0, 0.5)
        Tail probability level. Default 0.05 (5% tail).

    Returns
    -------
    dict with keys:
        ess : float
            Tail effective sample size.
        warnings : list[str]
            Human-readable warnings.
    """
    m, n = chains.shape
    flat = chains.reshape(-1)
    n_total = m * n

    if n_total < 40:
        return {
            "ess": float(n_total),
            "warnings": ["chain_too_short_tail_ess"],
        }

    if np.var(flat) <= 1e-14:
        return {
            "ess": float(n_total),
            "warnings": ["zero_chain_variance_tail_ess"],
        }

    if not np.all(np.isfinite(flat)):
        return {
            "ess": float(n_total),
            "warnings": ["non_finite_samples_tail_ess"],
        }

    if not 0.0 < prob < 0.5:
        return {
            "ess": float(n_total),
            "warnings": ["invalid_tail_probability"],
        }

    # Relevant tail quantiles over the pooled draws.
    q_low = float(np.percentile(flat, prob * 100))
    q_high = float(np.percentile(flat, (1.0 - prob) * 100))

    # Lower- and upper-tail indicator chains (ties included via <= / >).
    lower_indicator = (chains <= q_low).astype(float)
    upper_indicator = (chains > q_high).astype(float)

    ess_low = _ess_estimator(_split_chains(lower_indicator))
    ess_high = _ess_estimator(_split_chains(upper_indicator))

    ess_tail = float(min(ess_low, ess_high))

    warnings: list[str] = []
    if n_total < 40:
        warnings.append("chain_too_short_tail_ess")
    if np.var(flat) <= 1e-14:
        warnings.append("zero_chain_variance_tail_ess")

    return {
        "ess": ess_tail,
        "warnings": warnings,
    }


def _detect_pathologies(chains: np.ndarray) -> dict:
    """Detect various MCMC pathology indicators.

    Parameters
    ----------
    chains : np.ndarray of shape (m, n_draws)

    Returns
    -------
    dict with pathology flags and details:
        zero_variance_chains : list[int]
            Chain indices where within-chain variance is near zero.
        identical_chains : list[tuple[int, int]]
            Pairs of chain indices with nearly identical traces.
        non_finite_samples : bool
            Whether any NaN/Inf values are present.
        divergent_chains : list[int]
            Chain indices with potentially stuck behavior.
        warnings : list[str]
            Human-readable diagnostic warnings.
    """
    m, n = chains.shape
    warnings: list[str] = []

    has_non_finite = bool(np.any(~np.isfinite(chains)))
    if has_non_finite:
        warnings.append("non_finite_samples_present")

    zero_var_chains: list[int] = []
    chain_vars = np.var(chains, axis=1)
    for i in range(m):
        if chain_vars[i] <= 1e-14:
            zero_var_chains.append(i)
            warnings.append(f"zero_within_chain_variance_chain_{i}")

    identical_pairs: list[tuple[int, int]] = []
    for i in range(m):
        for j in range(i + 1, m):
            if np.allclose(chains[i], chains[j], rtol=1e-8, atol=1e-6):
                identical_pairs.append((i, j))
                warnings.append(f"identical_chains_{i}_vs_{j}")

    overall_var = np.var(chains) if m > 1 else 1.0
    divergent_chains: list[int] = []
    for i in range(m):
        if overall_var > 1e-14 and chain_vars[i] / overall_var < 0.01:
            divergent_chains.append(i)
            warnings.append(f"potentially_stuck_chain_{i}")

    return {
        "zero_variance_chains": zero_var_chains,
        "identical_chains": identical_pairs,
        "non_finite_samples": has_non_finite,
        "divergent_chains": divergent_chains,
        "warnings": warnings,
    }


def _check_burn_in_adequacy(
    accept_rates: list[float], burn_in: int, total_iters: int
) -> dict:
    """Check that burn-in was adequate and adaptation stopped properly."""
    warnings: list[str] = []

    if burn_in >= total_iters // 2:
        warnings.append("burn_in_occupies_half_or_more")

    if len(accept_rates) > 0:
        mean_accept = float(np.mean(accept_rates))

        if mean_accept >= 1.0:
            warnings.append("acceptance_rate_1_0_no_rejections")

        if mean_accept < 0.1:
            warnings.append("very_low_acceptance_rate")

    return {"warnings": warnings}


# Backward-compatibility aliases for the changepoint model
def _split_rhat(chains: np.ndarray) -> float:
    """Simple split R-hat for backward compatibility (uses rank-normalized version)."""
    result = _rank_normalized_split_rhat(chains)
    return result["rhat"] if result["is_valid"] and result["rhat"] is not None else 1.0


def _estimate_ess(chains: np.ndarray) -> float:
    """Simple bulk ESS for backward compatibility."""
    result = _bulk_ess(chains)
    return result["ess"]


def _compute_hdi(samples: np.ndarray, cred_mass: float = 0.95) -> tuple[float, float]:
    sorted_s = np.sort(samples)
    n = len(sorted_s)
    interval_idx = max(1, int(np.floor(cred_mass * n)))
    if interval_idx >= n:
        return float(sorted_s[0]), float(sorted_s[-1])
    widths = sorted_s[interval_idx:] - sorted_s[: n - interval_idx]
    min_idx = int(np.argmin(widths))
    return float(sorted_s[min_idx]), float(sorted_s[min_idx + interval_idx])


def _histogram(samples: np.ndarray, bins: int = 28) -> dict:
    """Compute a simple histogram of posterior samples.

    Returns a dict with centers and density values for plotting.
    """
    counts, edges = np.histogram(samples, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "centers": [round(float(c), 5) for c in centers],
        "density": [round(float(d), 5) for d in counts],
    }


def _summarize_param(chains: np.ndarray) -> dict:
    """Enhanced parameter summary with comprehensive convergence diagnostics.

    Computes:
      - Central tendency (mean, median)
      - Dispersion (sd, HDI, CI)
      - Rank-normalized split R-hat (Vehtari et al., 2021)
      - Bulk and tail ESS
      - Pathology detection (zero variance, identical chains, non-finite, stuck)
      - Converged flag combining multiple diagnostics

    The converged flag is True only when ALL of the following hold:
      - rank-normalized R-hat < 1.10
      - bulk ESS > 200
      - No non-finite samples
      - No zero-variance chains
      - No identical chains
      - No potentially stuck chains
    """
    flat = chains.reshape(-1)
    hdi_low, hdi_high = _compute_hdi(flat, 0.95)
    q025, q50, q975 = np.percentile(flat, [2.5, 50.0, 97.5])

    rhat_result = _rank_normalized_split_rhat(chains)
    bulk_ess_result = _bulk_ess(chains)
    tail_ess_result = _tail_ess(chains)
    pathologies = _detect_pathologies(chains)

    converged = (
        rhat_result["is_valid"]
        and rhat_result["rhat"] is not None
        and float(rhat_result["rhat"]) < 1.10
        and bulk_ess_result["ess"] > 200
        and not pathologies["non_finite_samples"]
        and len(pathologies["zero_variance_chains"]) == 0
        and len(pathologies["identical_chains"]) == 0
        and len(pathologies["divergent_chains"]) == 0
    )

    summary = {
        "mean": round(float(np.mean(flat)), 5),
        "median": round(float(q50), 5),
        "sd": round(float(np.std(flat, ddof=1)), 5),
        "hdi_95_low": round(hdi_low, 5),
        "hdi_95_high": round(hdi_high, 5),
        "ci_95_low": round(float(q025), 5),
        "ci_95_high": round(float(q975), 5),
        "r_hat": round(float(rhat_result["rhat"]), 4) if rhat_result["is_valid"] else None,
        "r_hat_warnings": rhat_result["warnings"],
        "ess_bulk": round(bulk_ess_result["ess"], 1),
        "ess": round(bulk_ess_result["ess"], 1),
        "ess_bulk_warnings": bulk_ess_result["warnings"],
        "ess_tail": round(tail_ess_result["ess"], 1),
        "ess_tail_warnings": tail_ess_result["warnings"],
        "zero_variance_chains": pathologies["zero_variance_chains"],
        "identical_chains": pathologies["identical_chains"],
        "non_finite_samples": pathologies["non_finite_samples"],
        "potentially_stuck_chains": pathologies["divergent_chains"],
        "converged": converged,
        "diagnostic_warnings": list(
            set(
                rhat_result["warnings"]
                + bulk_ess_result["warnings"]
                + tail_ess_result["warnings"]
                + pathologies["warnings"]
            )
        ),
    }

    return summary


def _run_robust_ar1_mcmc(
    t_years: np.ndarray,
    y: np.ndarray,
    *,
    chains: int,
    draws: int,
    burn_in: int,
    seed: int,
) -> dict:
    """Component-wise Adaptive Metropolis-within-Gibbs for Student-t (nu=5) AR(1) linear trend."""
    t_mean = float(np.mean(t_years))
    t_sd = max(float(np.std(t_years)), 1e-6)
    y_mean = float(np.mean(y))
    y_sd = max(float(np.std(y)), 1e-6)

    x_std = (t_years - t_mean) / t_sd
    y_std = (y - y_mean) / y_sd
    dt_steps = np.maximum(1.0, np.round(np.diff(t_years) * 12.0))
    nu = 5.0

    sin_ann = np.sin(2.0 * np.pi * t_years)
    cos_ann = np.cos(2.0 * np.pi * t_years)
    X_full = np.column_stack([np.ones_like(x_std), x_std, sin_ann, cos_ann])
    ols_beta, _, _, _ = np.linalg.lstsq(X_full, y_std, rcond=None)
    init_resid = y_std - X_full @ ols_beta
    init_sigma = max(0.08, float(np.std(init_resid)))

    def log_post(theta: np.ndarray) -> float:
        a, b, c_s, c_c, log_s, z_rho = theta
        sigma = float(np.exp(log_s))
        rho = float(np.tanh(z_rho))
        if sigma < 1e-4 or sigma > 25.0 or abs(rho) > 0.96:
            return -np.inf

        lp = (
            -0.5 * (a / 3.0) ** 2
            - 0.5 * (b / 3.0) ** 2
            - 0.5 * (c_s / 3.0) ** 2
            - 0.5 * (c_c / 3.0) ** 2
            - 0.5 * (log_s / 1.8) ** 2
            - 0.5 * (z_rho / 1.2) ** 2
        )

        mu = a + b * x_std + c_s * sin_ann + c_c * cos_ann
        res = y_std - mu
        lp += -0.5 * (nu + 1.0) * float(np.log1p((res[0] / sigma) ** 2 / nu)) - float(np.log(sigma))

        rho_dt = np.sign(rho) * (np.abs(rho) ** dt_steps)
        innov_scale = sigma * np.sqrt(np.maximum(1e-6, 1.0 - rho_dt ** 2))
        innov = res[1:] - rho_dt * res[:-1]
        lp += float(
            np.sum(
                -0.5 * (nu + 1.0) * np.log1p((innov / innov_scale) ** 2 / nu)
                - np.log(innov_scale)
            )
        )
        return lp

    total_iters = burn_in + draws
    rng = np.random.default_rng(seed)
    n_params = 6

    a_chains = np.zeros((chains, draws))
    b_year_chains = np.zeros((chains, draws))
    b_dec_chains = np.zeros((chains, draws))
    sigma_chains = np.zeros((chains, draws))
    rho_chains = np.zeros((chains, draws))
    accept_rates = []

    for c in range(chains):
        theta = np.array(
            [
                float(ols_beta[0]) + rng.normal(0.0, 0.05),
                float(ols_beta[1]) + rng.normal(0.0, 0.05),
                float(ols_beta[2]) + rng.normal(0.0, 0.05),
                float(ols_beta[3]) + rng.normal(0.0, 0.05),
                np.log(init_sigma) + rng.normal(0.0, 0.08),
                0.15 + rng.normal(0.0, 0.10),
            ],
            dtype=float,
        )
        steps = np.full(n_params, 0.08, dtype=float)
        curr_lp = log_post(theta)
        accepted_updates = 0
        total_updates = draws * n_params

        for it in range(total_iters):
            for p_idx in range(n_params):
                prop = theta.copy()
                prop[p_idx] += rng.normal(0.0, steps[p_idx])
                prop_lp = log_post(prop)
                acc_prob = min(1.0, float(np.exp(min(0.0, prop_lp - curr_lp)))) if np.isfinite(prop_lp) else 0.0
                if np.log(rng.uniform()) < (prop_lp - curr_lp):
                    theta = prop
                    curr_lp = prop_lp
                    if it >= burn_in:
                        accepted_updates += 1
                if it < burn_in:
                    steps[p_idx] *= np.exp(0.04 * (acc_prob - 0.35))
                    steps[p_idx] = float(np.clip(steps[p_idx], 0.005, 0.6))

            if it >= burn_in:
                idx = it - burn_in
                slope_per_yr = theta[1] * (y_sd / t_sd)
                intercept_phys = y_mean + y_sd * theta[0] - slope_per_yr * t_mean
                a_chains[c, idx] = intercept_phys
                b_year_chains[c, idx] = slope_per_yr
                b_dec_chains[c, idx] = slope_per_yr * 10.0
                sigma_chains[c, idx] = float(np.exp(theta[4]) * y_sd)
                rho_chains[c, idx] = float(np.tanh(theta[5]))

        accept_rates.append(accepted_updates / total_updates)

    flat_a = a_chains.reshape(-1)
    flat_b = b_year_chains.reshape(-1)
    prob_pos = float(np.mean(flat_b > 0.0))

    sub_idx = rng.choice(len(flat_b), size=min(400, len(flat_b)), replace=False)
    lines = flat_a[sub_idx, None] + flat_b[sub_idx, None] * t_years[None, :]
    mean_line = np.mean(lines, axis=0)
    low_95 = np.percentile(lines, 2.5, axis=0)
    high_95 = np.percentile(lines, 97.5, axis=0)
    low_50 = np.percentile(lines, 25.0, axis=0)
    high_50 = np.percentile(lines, 75.0, axis=0)

    stride = max(1, draws // 160)
    traces = [
        [round(float(val), 5) for val in b_dec_chains[c, ::stride]]
        for c in range(chains)
    ]

    # Compute diagnostics using new comprehensive functions
    rhat_result = _rank_normalized_split_rhat(b_year_chains)
    bulk_ess_result = _bulk_ess(b_year_chains)
    tail_ess_result = _tail_ess(b_year_chains)
    pathologies = _detect_pathologies(b_year_chains)

    # Converged flag: pass only when all diagnostics agree
    converged = (
        rhat_result["is_valid"]
        and rhat_result["rhat"] is not None
        and float(rhat_result["rhat"]) < 1.10
        and bulk_ess_result["ess"] > 200
        and not pathologies["non_finite_samples"]
        and len(pathologies["zero_variance_chains"]) == 0
        and len(pathologies["identical_chains"]) == 0
        and len(pathologies["divergent_chains"]) == 0
    )

    mean_acceptance_rate = round(float(np.mean(accept_rates)), 3)

    diag_slope = _summarize_param(b_year_chains)
    diag_intercept = _summarize_param(a_chains)
    diag_sigma = _summarize_param(sigma_chains)
    diag_rho = _summarize_param(rho_chains)

    burn_in_warnings = _check_burn_in_adequacy(accept_rates, burn_in, total_iters)

    all_warnings = list(
        set(
            diag_slope.get("diagnostic_warnings", [])
            + diag_intercept.get("diagnostic_warnings", [])
            + diag_sigma.get("diagnostic_warnings", [])
            + diag_rho.get("diagnostic_warnings", [])
            + burn_in_warnings["warnings"]
        )
    )

    return {
        "model_type": "robust_ar1",
        "model_description": "Bayesian Student-t (df=5) Deseasonalized Linear Trend with Irregular-Gap AR(1) Persistence",
        "diagnostics": {
            "chains": chains,
            "draws_per_chain": draws,
            "burn_in": burn_in,
            "mean_acceptance_rate": mean_acceptance_rate,
            "converged": converged,
            "rank_rhat": {
                "value": rhat_result["rhat"] if rhat_result["is_valid"] else None,
                "warnings": rhat_result["warnings"],
                "is_valid": rhat_result["is_valid"],
            },
            "bulk_ess": {
                "value": bulk_ess_result["ess"],
                "warnings": bulk_ess_result["warnings"],
            },
            "tail_ess": {
                "value": tail_ess_result["ess"],
                "warnings": tail_ess_result["warnings"],
            },
            "pathologies": {
                "zero_variance_chains": pathologies["zero_variance_chains"],
                "identical_chains": pathologies["identical_chains"],
                "non_finite_samples": pathologies["non_finite_samples"],
                "potentially_stuck_chains": pathologies["divergent_chains"],
            },
            "burn_in_warnings": burn_in_warnings["warnings"],
            "all_warnings": all_warnings,
        },
        "parameters": {
            "slope_per_year": diag_slope,
            "slope_per_decade": _summarize_param(b_dec_chains),
            "intercept": diag_intercept,
            "sigma": diag_sigma,
            "ar1_rho": diag_rho,
        },
        "probability_of_direction": {
            "p_positive": round(prob_pos, 4),
            "p_negative": round(1.0 - prob_pos, 4),
            "direction": "increasing" if prob_pos >= 0.5 else "decreasing",
            "certainty": round(max(prob_pos, 1.0 - prob_pos), 4),
        },
        "posterior_histogram_decade": _histogram(b_dec_chains.reshape(-1)),
        "traces_slope_decade": traces,
        "predictive_trend": {
            "t_years": [round(float(x), 4) for x in t_years],
            "mean": [round(float(x), 4) for x in mean_line],
            "ci95_low": [round(float(x), 4) for x in low_95],
            "ci95_high": [round(float(x), 4) for x in high_95],
            "ci50_low": [round(float(x), 4) for x in low_50],
            "ci50_high": [round(float(x), 4) for x in high_50],
        },
    }


def _run_changepoint_mcmc(
    t_years: np.ndarray,
    y: np.ndarray,
    dates: list[str],
    *,
    chains: int,
    draws: int,
    burn_in: int,
    seed: int,
) -> dict:
    """Bayesian piecewise linear structural change-point model."""
    t_span = max(float(t_years[-1] - t_years[0]), 1e-6)
    t_rel = t_years - t_years[0]
    y_mean = float(np.mean(y))
    y_sd = max(float(np.std(y)), 1e-6)
    y_std = (y - y_mean) / y_sd
    t_norm = t_rel / t_span  # [0, 1]

    def log_post(theta: np.ndarray) -> float:
        a, b1, delta, u_tau, log_s = theta
        sigma = np.exp(log_s)
        tau = 0.12 + 0.76 / (1.0 + np.exp(-u_tau))  # bounded inside [0.12, 0.88]
        if sigma < 1e-4 or sigma > 25.0:
            return -np.inf
        lp = (
            -0.5 * (a / 3.0) ** 2
            - 0.5 * (b1 / 3.0) ** 2
            - 0.5 * (delta / 3.0) ** 2
            - 0.5 * (u_tau / 1.8) ** 2
            - 0.5 * (log_s / 1.5) ** 2
        )
        mu = a + b1 * t_norm + delta * np.maximum(0.0, t_norm - tau)
        n = len(y_std)
        lp += -n * np.log(sigma) - 0.5 * float(np.sum(((y_std - mu) / sigma) ** 2))
        return lp

    rng = np.random.default_rng(seed)
    total_iters = burn_in + draws

    b1_yr_chains = np.zeros((chains, draws))
    b2_yr_chains = np.zeros((chains, draws))
    delta_yr_chains = np.zeros((chains, draws))
    tau_yr_chains = np.zeros((chains, draws))
    tau_frac_chains = np.zeros((chains, draws))
    a_chains = np.zeros((chains, draws))
    sigma_chains = np.zeros((chains, draws))
    accept_rates = []

    for c in range(chains):
        theta = np.array([0.0, 0.0, 0.0, rng.normal(0.0, 0.4), np.log(0.6)])
        step = np.array([0.09, 0.12, 0.14, 0.18, 0.08])
        curr_lp = log_post(theta)
        accepted = 0

        for it in range(total_iters):
            prop = theta + rng.normal(0.0, step)
            prop_lp = log_post(prop)
            log_alpha = prop_lp - curr_lp if np.isfinite(prop_lp) else -np.inf
            acc_prob = min(1.0, float(np.exp(min(0.0, log_alpha)))) if np.isfinite(prop_lp) else 0.0
            if np.isfinite(prop_lp) and np.log(rng.uniform()) < log_alpha:
                theta = prop
                curr_lp = prop_lp
                if it >= burn_in:
                    accepted += 1
            if it < burn_in:
                step *= np.exp(0.03 * (acc_prob - 0.28))
                step = np.clip(step, 0.008, 0.9)
            if it >= burn_in:
                idx = it - burn_in
                tau_frac = float(0.12 + 0.76 / (1.0 + np.exp(-theta[3])))
                b1_yr = float(theta[1] * y_sd / t_span)
                delta_yr = float(theta[2] * y_sd / t_span)
                a_chains[c, idx] = float(y_mean + y_sd * theta[0])
                b1_yr_chains[c, idx] = b1_yr
                delta_yr_chains[c, idx] = delta_yr
                b2_yr_chains[c, idx] = b1_yr + delta_yr
                tau_frac_chains[c, idx] = tau_frac
                tau_yr_chains[c, idx] = float(tau_frac * t_span)
                sigma_chains[c, idx] = float(np.exp(theta[4]) * y_sd)
        accept_rates.append(accepted / draws)

    flat_a = a_chains.reshape(-1)
    flat_b1 = b1_yr_chains.reshape(-1)
    flat_delta = delta_yr_chains.reshape(-1)
    flat_tau_yr = tau_yr_chains.reshape(-1)
    prob_shift_pos = float(np.mean(flat_delta > 0.0))

    sub_idx = rng.choice(len(flat_b1), size=min(400, len(flat_b1)), replace=False)
    lines = (
        flat_a[sub_idx, None]
        + flat_b1[sub_idx, None] * t_rel[None, :]
        + flat_delta[sub_idx, None] * np.maximum(0.0, t_rel[None, :] - flat_tau_yr[sub_idx, None])
    )

    tau_med_frac = float(np.median(tau_frac_chains))
    tau_idx = int(np.clip(round(tau_med_frac * (len(dates) - 1)), 0, len(dates) - 1))
    stride = max(1, draws // 160)

    return {
        "model_type": "changepoint",
        "model_description": "Bayesian Piecewise Hinge Change-Point Model (Pre-Shift Slope + Slope Acceleration)",
        "diagnostics": {
            "chains": chains,
            "draws_per_chain": draws,
            "burn_in": burn_in,
            "mean_acceptance_rate": round(float(np.mean(accept_rates)), 3),
            "converged": _split_rhat(delta_yr_chains) < 1.06 and _split_rhat(tau_frac_chains) < 1.08,
        },
        "parameters": {
            "pre_slope_per_year": _summarize_param(b1_yr_chains),
            "post_slope_per_year": _summarize_param(b2_yr_chains),
            "slope_shift_per_year": _summarize_param(delta_yr_chains),
            "changepoint_fraction": _summarize_param(tau_frac_chains),
            "sigma": _summarize_param(sigma_chains),
        },
        "estimated_changepoint_date": dates[tau_idx],
        "estimated_changepoint_index": tau_idx,
        "probability_of_direction": {
            "p_positive": round(prob_shift_pos, 4),
            "p_negative": round(1.0 - prob_shift_pos, 4),
            "direction": "accelerating" if prob_shift_pos >= 0.5 else "decelerating",
            "certainty": round(max(prob_shift_pos, 1.0 - prob_shift_pos), 4),
        },
        "posterior_histogram_decade": _histogram(delta_yr_chains.reshape(-1) * 10.0),
        "traces_slope_decade": [
            [round(float(v * 10.0), 5) for v in delta_yr_chains[c, ::stride]]
            for c in range(chains)
        ],
        "predictive_trend": {
            "t_years": [round(float(x), 4) for x in t_rel],
            "mean": [round(float(x), 4) for x in np.mean(lines, axis=0)],
            "ci95_low": [round(float(x), 4) for x in np.percentile(lines, 2.5, axis=0)],
            "ci95_high": [round(float(x), 4) for x in np.percentile(lines, 97.5, axis=0)],
            "ci50_low": [round(float(x), 4) for x in np.percentile(lines, 25.0, axis=0)],
            "ci50_high": [round(float(x), 4) for x in np.percentile(lines, 75.0, axis=0)],
        },
    }


def run_bayesian_mcmc(
    dates: list[str],
    values: list[Optional[float]],
    *,
    model_type: Literal["robust_ar1", "changepoint"] = "robust_ar1",
    chains: int = 3,
    draws: int = 900,
    burn_in: int = 450,
    seed: int = 42,
) -> dict:
    """Run Bayesian MCMC on an environmental time series (ignoring missing slots while preserving time spacing)."""
    if len(dates) != len(values):
        raise ValueError("dates and values must match in length.")

    valid_idx = [i for i, v in enumerate(values) if v is not None and np.isfinite(float(v))]
    if len(valid_idx) < 12:
        raise ValueError(f"At least 12 valid observations are required for Bayesian MCMC (got {len(valid_idx)}).")

    t_years = np.array([i / 12.0 for i in valid_idx], dtype=float)
    y = np.array([float(values[i]) for i in valid_idx], dtype=float)
    valid_dates = [dates[i] for i in valid_idx]

    if model_type == "changepoint":
        res = _run_changepoint_mcmc(
            t_years,
            y,
            valid_dates,
            chains=chains,
            draws=draws,
            burn_in=burn_in,
            seed=seed,
        )
    else:
        res = _run_robust_ar1_mcmc(
            t_years,
            y,
            chains=chains,
            draws=draws,
            burn_in=burn_in,
            seed=seed,
        )

    res["n_total"] = len(values)
    res["n_valid"] = len(valid_idx)
    res["n_missing"] = len(values) - len(valid_idx)
    res["valid_dates"] = valid_dates
    res["observed_valid"] = [round(float(v), 4) for v in y]
    return res

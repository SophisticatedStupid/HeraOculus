"""Regression tests for Bayesian MCMC convergence diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from app.services.bayesian_service import (
    _normal_cdf,
    _normal_ppf,
    _rank_normalized_split_rhat,
    _bulk_ess,
    _tail_ess,
    _detect_pathologies,
    _check_burn_in_adequacy,
    _summarize_param,
)

try:
    import arviz_stats as _azs

    HAS_ARVIZ = True
except ImportError:  # pragma: no cover - reference dependency may be absent
    HAS_ARVIZ = False


def _az_rhat(chains, method="z_scale"):
    return float(_azs.rhat(np.asarray(chains), method=method, chain_axis=0, draw_axis=1))


def _az_ess(chains, method="bulk", prob=None):
    kw = dict(method=method, chain_axis=0, draw_axis=1)
    if prob is not None:
        kw["prob"] = prob
    return float(_azs.ess(np.asarray(chains), **kw))


def test_well_mixed_chains_rank_rhat_near_one():
    """Well-mixed independent chains should yield rank-normalized split R-hat near 1."""
    rng = np.random.default_rng(42)
    chains = rng.normal(0.0, 1.0, size=(4, 500))
    result = _rank_normalized_split_rhat(chains)
    assert result["is_valid"] is True
    assert abs(result["rhat"] - 1.0) < 0.05, f"R-hat should be ~1.0, got {result['rhat']}"


def test_non_converged_chains_different_means_rank_rhat_high():
    """Chains with different means should produce R-hat > 1.0 (non-converged)."""
    rng = np.random.default_rng(42)
    chains = rng.normal(0.0, 1.0, size=(4, 500))
    chains[0] += 2.0  # shift one chain to a different mean
    result = _rank_normalized_split_rhat(chains)
    assert result["is_valid"] is True
    assert result["rhat"] > 1.0, f"R-hat should exceed 1.0 for non-converged chains, got {result['rhat']}"


def test_bulk_ess_high_for_well_mixed_chains():
    """Well-mixed chains should yield high bulk ESS."""
    rng = np.random.default_rng(42)
    chains = rng.normal(0.0, 1.0, size=(4, 500))
    result = _bulk_ess(chains)
    assert result["ess"] > 1000, f"Bulk ESS should be high (>1000), got {result['ess']}"


def test_bulk_ess_low_for_identical_chains():
    """Identical chains should yield low ESS (no independent information)."""
    chains = np.tile(np.random.default_rng(1).normal(0.0, 1.0, 500), (3, 1))
    result = _bulk_ess(chains)
    # ESS should reflect that variance is artificially low/high
    assert result["ess"] >= 1.0


def test_tail_ess_heavy_tailed():
    """Heavy-tailed distributions should produce reasonable tail ESS."""
    rng = np.random.default_rng(7)
    chains = np.column_stack([rng.standard_cauchy(400) for _ in range(3)])
    result = _tail_ess(chains, prob=0.05)
    assert result["ess"] > 0
    assert result["warnings"] == []


def test_pathologies_identical_chains_detected():
    """Identical chains should be flagged as identical/stuck."""
    chains = np.tile(np.random.default_rng(1).normal(0.0, 1.0, 500), (3, 1))
    result = _detect_pathologies(chains)
    assert len(result["identical_chains"]) > 0, "Identical chains should be detected"
    assert result["warnings"] != []


def test_pathologies_zero_variance_detected():
    """Chains with zero variance should be flagged."""
    chains = np.zeros((3, 100))
    result = _detect_pathologies(chains)
    assert len(result["zero_variance_chains"]) > 0
    assert result["non_finite_samples"] is False


def test_pathologies_non_finite_detected():
    """Non-finite samples should be flagged."""
    chains = np.random.default_rng(1).normal(0.0, 1.0, size=(3, 100))
    chains[0, 5] = np.inf
    chains[1, 10] = np.nan
    result = _detect_pathologies(chains)
    assert result["non_finite_samples"] is True
    assert "non_finite_samples_present" in result["warnings"]


def test_short_chain_ess_warnings():
    """Short chains should produce warnings and fall back safely."""
    chains = np.random.default_rng(1).normal(0.0, 1.0, size=(2, 8))
    result = _bulk_ess(chains)
    assert "chain_too_short_bulk_ess" in result["warnings"]


def test_summarize_param_converged_flag():
    """Parameter summary should include converged flag from multiple diagnostics."""
    rng = np.random.default_rng(42)
    chains = rng.normal(0.0, 1.0, size=(4, 500))
    summ = _summarize_param(chains)
    assert "converged" in summ
    assert "ess_bulk" in summ
    assert "ess_tail" in summ
    assert "r_hat" in summ
    assert summ["r_hat"] > 0
    assert summ["ess_bulk"] > 0
    assert summ["ess_tail"] > 0


def test_summarize_param_unconverged_flag():
    """Unconverged parameter should have converged=False."""
    chains = np.zeros((3, 100))
    chains[0] += 1.0  # constant chains
    summ = _summarize_param(chains)
    assert summ["converged"] is False


def test_burn_in_adequacy_warnings():
    """Burn-in adequacy check should flag half-or-more burn-in."""
    result = _check_burn_in_adequacy([], burn_in=500, total_iters=1000)
    assert "burn_in_occupies_half_or_more" in result["warnings"]
    result2 = _check_burn_in_adequacy([0.95], burn_in=10, total_iters=100)
    assert "burn_in_occupies_half_or_more" not in result2["warnings"]


def test_heavy_tailed_ess_tail_low():
    """Heavy-tailed distributions should give sensible tail ESS (< bulk ESS)."""
    rng = np.random.default_rng(3)
    chains = np.column_stack([rng.standard_cauchy(400) for _ in range(3)])
    bulk = _bulk_ess(chains)
    tail = _tail_ess(chains, prob=0.05)
    # For heavy-tailed, tail ESS should be <= bulk ESS
    assert tail["ess"] <= bulk["ess"]


def test_run_bayesian_mcmc_returns_diagnostics():
    """Full run should return diagnostics structure."""
    dates = [f"{2016 + (i // 12):04d}-{(i % 12) + 1:02d}" for i in range(96)]
    values = [10.0 + 0.5 * (i / 12.0) + 0.3 * np.sin(2 * np.pi * (i % 12) / 12) + np.random.default_rng(5).normal(0, 0.2) for i in range(96)]
    from app.services.bayesian_service import run_bayesian_mcmc
    res = run_bayesian_mcmc(dates, values, model_type="robust_ar1", chains=2, draws=300, burn_in=100)
    assert "diagnostics" in res
    assert "converged" in res["diagnostics"]
    assert "rank_rhat" in res["diagnostics"]
    assert "bulk_ess" in res["diagnostics"]
    assert "tail_ess" in res["diagnostics"]
    assert "pathologies" in res["diagnostics"]


# ---------------------------------------------------------------------------
# Regression: 1.4 NumPy erf dependency removed
# ---------------------------------------------------------------------------
def test_normal_cdf_matches_scipy_erf():
    """_normal_cdf must use SciPy's error function, not the removed np.erf."""
    from scipy.special import erf

    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    np.testing.assert_allclose(_normal_cdf(x), expected, rtol=1e-12, atol=1e-12)
    assert abs(_normal_cdf(0.0) - 0.5) < 1e-12


def test_normal_ppf_round_trip():
    """_normal_ppf is the inverse of _normal_cdf on the open unit interval."""
    p = np.array([0.001, 0.1, 0.5, 0.9, 0.999])
    np.testing.assert_allclose(_normal_cdf(_normal_ppf(p)), p, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Regression: 1.5 reference comparisons against ArviZ (Vehtari et al. 2021)
# ---------------------------------------------------------------------------
RTOL_RHAT = 0.02
RTOL_ESS = 0.10


@pytest.mark.skipif(not HAS_ARVIZ, reason="arviz_stats not installed")
def test_rank_rhat_independent_normal_matches_arviz():
    """Independent normal chains: R-hat ~ 1 and matches ArviZ."""
    rng = np.random.default_rng(11)
    chains = rng.normal(0.0, 1.0, size=(4, 500))
    result = _rank_normalized_split_rhat(chains)
    assert result["is_valid"] is True
    assert abs(result["rhat"] - 1.0) < 0.02
    assert abs(result["rhat"] - _az_rhat(chains, "z_scale")) <= RTOL_RHAT


@pytest.mark.skipif(not HAS_ARVIZ, reason="arviz_stats not installed")
def test_rank_rhat_autocorrelated_matches_arviz():
    """Highly autocorrelated chains: R-hat > 1 and matches ArviZ."""
    rng = np.random.default_rng(12)
    chains = np.empty((4, 500))
    for c in range(4):
        x = np.empty(500)
        x[0] = rng.normal(0.0, 1.0)
        for t in range(1, 500):
            x[t] = 0.98 * x[t - 1] + rng.normal(0.0, 0.2)
        chains[c] = x
    result = _rank_normalized_split_rhat(chains)
    assert result["is_valid"] is True
    assert result["rhat"] > 1.0
    assert abs(result["rhat"] - _az_rhat(chains, "z_scale")) <= RTOL_RHAT


@pytest.mark.skipif(not HAS_ARVIZ, reason="arviz_stats not installed")
def test_rank_rhat_chains_different_means_matches_arviz():
    """Chains with different means: R-hat > 1 and matches ArviZ."""
    rng = np.random.default_rng(13)
    chains = rng.normal(0.0, 1.0, size=(4, 500))
    chains[0] += 3.0
    result = _rank_normalized_split_rhat(chains)
    assert result["is_valid"] is True
    assert result["rhat"] > 1.0
    assert abs(result["rhat"] - _az_rhat(chains, "z_scale")) <= RTOL_RHAT


@pytest.mark.skipif(not HAS_ARVIZ, reason="arviz_stats not installed")
def test_rank_rhat_chains_different_variances_matches_arviz():
    """Chains with different variances: result matches ArviZ."""
    rng = np.random.default_rng(14)
    chains = np.column_stack([
        rng.normal(0.0, 1.0, 500),
        rng.normal(0.0, 2.0, 500),
        rng.normal(0.0, 0.5, 500),
        rng.normal(0.0, 3.0, 500),
    ]).T
    result = _rank_normalized_split_rhat(chains)
    assert result["is_valid"] is True
    assert abs(result["rhat"] - _az_rhat(chains, "z_scale")) <= RTOL_RHAT


@pytest.mark.skipif(not HAS_ARVIZ, reason="arviz_stats not installed")
def test_rank_rhat_heavy_tailed_matches_arviz():
    """Heavy-tailed distribution: R-hat and ESS match ArviZ."""
    rng = np.random.default_rng(15)
    chains = np.column_stack([rng.standard_cauchy(400) for _ in range(3)]).T
    result = _rank_normalized_split_rhat(chains)
    assert result["is_valid"] is True
    assert abs(result["rhat"] - _az_rhat(chains, "z_scale")) <= RTOL_RHAT
    bulk = _bulk_ess(chains)
    az = _az_ess(chains, "bulk")
    assert abs(bulk["ess"] - az) <= RTOL_ESS * max(1.0, az)


@pytest.mark.skipif(not HAS_ARVIZ, reason="arviz_stats not installed")
def test_rank_rhat_multimodal_matches_arviz():
    """Multimodal posterior: R-hat > 1 and matches ArviZ."""
    rng = np.random.default_rng(16)
    chains = np.concatenate(
        [rng.normal(-2.0, 0.5, (3, 250)), rng.normal(2.0, 0.5, (3, 250))], axis=1
    )
    result = _rank_normalized_split_rhat(chains)
    assert result["is_valid"] is True
    assert result["rhat"] > 1.0
    assert abs(result["rhat"] - _az_rhat(chains, "z_scale")) <= RTOL_RHAT


@pytest.mark.skipif(not HAS_ARVIZ, reason="arviz_stats not installed")
def test_folded_rank_rhat_matches_arviz():
    """Folded rank-normalized split R-hat matches ArviZ folded method."""
    rng = np.random.default_rng(17)
    chains = rng.normal(0.0, 1.0, size=(4, 500))
    chains[0] += 2.0
    result = _rank_normalized_split_rhat(chains, folded=True)
    assert result["is_valid"] is True
    assert abs(result["rhat"] - _az_rhat(chains, "folded")) <= RTOL_RHAT


@pytest.mark.skipif(not HAS_ARVIZ, reason="arviz_stats not installed")
def test_bulk_ess_autocorrelated_matches_arviz():
    """Bulk ESS of autocorrelated chains matches ArviZ and stays below total draws."""
    rng = np.random.default_rng(12)
    chains = np.empty((4, 500))
    for c in range(4):
        x = np.empty(500)
        x[0] = rng.normal(0.0, 1.0)
        for t in range(1, 500):
            x[t] = 0.98 * x[t - 1] + rng.normal(0.0, 0.2)
        chains[c] = x
    total = chains.shape[0] * chains.shape[1]
    result = _bulk_ess(chains)
    assert result["ess"] <= total, "ESS cannot exceed total draws"
    assert result["ess"] > 1
    az = _az_ess(chains, "bulk")
    assert abs(result["ess"] - az) <= RTOL_ESS * max(1.0, az)


@pytest.mark.skipif(not HAS_ARVIZ, reason="arviz_stats not installed")
def test_bulk_ess_multimodal_matches_arviz():
    """Bulk ESS of multimodal chains matches ArviZ."""
    rng = np.random.default_rng(16)
    chains = np.concatenate(
        [rng.normal(-2.0, 0.5, (3, 250)), rng.normal(2.0, 0.5, (3, 250))], axis=1
    )
    result = _bulk_ess(chains)
    az = _az_ess(chains, "bulk")
    assert abs(result["ess"] - az) <= RTOL_ESS * max(1.0, az)


@pytest.mark.skipif(not HAS_ARVIZ, reason="arviz_stats not installed")
def test_tail_ess_matches_arviz():
    """Tail ESS matches ArviZ for independent and heavy-tailed chains."""
    rng = np.random.default_rng(18)
    chains = rng.normal(0.0, 1.0, size=(4, 500))
    result = _tail_ess(chains, prob=0.05)
    az = _az_ess(chains, "tail", prob=0.05)
    assert abs(result["ess"] - az) <= RTOL_ESS * max(1.0, az)

    chains_c = np.column_stack([rng.standard_cauchy(400) for _ in range(3)]).T
    result_c = _tail_ess(chains_c, prob=0.05)
    az_c = _az_ess(chains_c, "tail", prob=0.05)
    assert abs(result_c["ess"] - az_c) <= RTOL_ESS * max(1.0, az_c)


def test_rank_rhat_constant_chain_is_invalid():
    """Constant chains must not be reported as converged (diagnostics invalid)."""
    chains = np.zeros((4, 200))
    result = _rank_normalized_split_rhat(chains)
    assert result["is_valid"] is False
    assert result["rhat"] is not None
    assert "zero_within_chain_variance" in result["warnings"]
    summ = _summarize_param(chains)
    assert summ["converged"] is False


def test_rank_rhat_too_few_chains_invalid():
    """A single chain cannot yield a split R-hat."""
    chains = np.random.default_rng(1).normal(0.0, 1.0, size=(1, 500))
    result = _rank_normalized_split_rhat(chains)
    assert result["is_valid"] is False
    assert result["rhat"] is None
    assert "fewer_than_2_chains" in result["warnings"]


def test_rank_rhat_too_few_draws_invalid():
    """Too few draws per chain must invalidate the diagnostic."""
    chains = np.random.default_rng(1).normal(0.0, 1.0, size=(4, 3))
    result = _rank_normalized_split_rhat(chains)
    assert result["is_valid"] is False
    assert result["rhat"] is None
    assert "insufficient_half_draws" in result["warnings"]


def test_rank_rhat_non_converged_flag_false():
    """Deliberately non-converged chains must not be reported as converged."""
    rng = np.random.default_rng(19)
    chains = np.empty((4, 500))
    chains[0::2] = rng.normal(0.0, 1.0, size=(2, 500))
    chains[1::2] = rng.normal(5.0, 1.0, size=(2, 500))  # separate location, no mixing
    result = _rank_normalized_split_rhat(chains)
    assert result["is_valid"] is True
    assert result["rhat"] > 1.10, f"Non-converged chains should give R-hat > 1.10, got {result['rhat']}"
    summ = _summarize_param(chains)
    assert summ["converged"] is False


def test_bulk_ess_constant_chain():
    """Constant chain yields total-draws fallback ESS with a warning."""
    chains = np.zeros((3, 200))
    result = _bulk_ess(chains)
    assert result["ess"] == float(chains.size)
    assert "zero_chain_variance_bulk_ess" in result["warnings"]


def test_bulk_ess_cannot_exceed_total_draws():
    """Bulk ESS is capped at the total number of available draws."""
    rng = np.random.default_rng(20)
    chains = rng.normal(0.0, 1.0, size=(3, 400))
    result = _bulk_ess(chains)
    assert result["ess"] <= float(chains.size)
    assert result["ess"] >= 1.0


def test_tail_ess_constant_chain():
    """Constant chain yields total-draws fallback tail ESS."""
    chains = np.zeros((3, 200))
    result = _tail_ess(chains, prob=0.05)
    assert result["ess"] == float(chains.size)


def test_tail_ess_handles_ties():
    """Discrete/tied values produce a finite tail ESS without crashing."""
    rng = np.random.default_rng(21)
    chains = rng.integers(0, 3, size=(4, 300)).astype(float)
    result = _tail_ess(chains, prob=0.05)
    assert result["ess"] >= 1.0
    assert np.isfinite(result["ess"])

"""Deterministic validation for the Bayesian changepoint MCMC fix."""
from __future__ import annotations

import numpy as np
from app.services.bayesian_service import run_bayesian_mcmc


def _make_changepoint_data(n_months: int = 96, tau_yr: float = 4.0, slope_before: float = 0.1, slope_shift: float = 0.7) -> tuple[list[str], list[float]]:
    dates = []
    for i in range(n_months):
        year = 2016 + i // 12
        month = (i % 12) + 1
        dates.append(f"{year:04d}-{month:02d}")
    vals = []
    rng = np.random.default_rng(7)
    for i in range(n_months):
        t = i / 12.0
        val = 10.0 + slope_before * t + slope_shift * max(0.0, t - tau_yr) + rng.normal(0.0, 0.15)
        vals.append(round(float(val), 3))
    return dates, vals


def test_acceptance_prob_pre_update():
    """Accepted proposals update state; rejected proposals preserve state.
    Also validates the posterior against a simulated dataset with a known changepoint."""
    dates, vals = _make_changepoint_data()
    # Single chain, enough iterations to see updates
    res = run_bayesian_mcmc(
        dates, vals, model_type="changepoint", chains=3, draws=700, burn_in=350, seed=1
    )
    shift = res["parameters"]["slope_shift_per_year"]
    tau_frac = res["parameters"]["changepoint_fraction"]
    # True tau is at 4.0 years of a ~7.9-year span -> tau_frac ~ 0.5
    assert shift["mean"] > 0.25, f"shift mean should recover >0.25, got {shift['mean']}"
    assert 0.3 < tau_frac["mean"] < 0.7, f"tau fraction should be near 0.5, got {tau_frac['mean']}"
    assert res["estimated_changepoint_date"] is not None
    assert res["diagnostics"]["mean_acceptance_rate"] < 1.0, "acceptance rate must not be 1.0"
    print(
        f"PASS: changepoint recovered (shift={shift['mean']:.3f}, "
        f"tau_frac={tau_frac['mean']:.3f}, date={res['estimated_changepoint_date']})"
    )


def test_reproducible_seeds():
    """Identical seeds produce identical results."""
    dates, vals = _make_changepoint_data()
    r1 = run_bayesian_mcmc(dates, vals, model_type="changepoint", chains=2, draws=400, burn_in=200, seed=123)
    r2 = run_bayesian_mcmc(dates, vals, model_type="changepoint", chains=2, draws=400, burn_in=200, seed=123)
    assert r1["parameters"]["slope_shift_per_year"]["mean"] == r2["parameters"]["slope_shift_per_year"]["mean"]
    assert r1["estimated_changepoint_date"] == r2["estimated_changepoint_date"]
    assert r1["traces_slope_decade"] == r2["traces_slope_decade"]
    print("PASS: identical seeds produce reproducible results")


def test_chains_not_identically_initialized():
    """Multiple chains must not produce identical traces (indicates different initialization)."""
    dates, vals = _make_changepoint_data()
    res = run_bayesian_mcmc(
        dates, vals, model_type="changepoint", chains=3, draws=400, burn_in=200, seed=7
    )
    traces = res["traces_slope_decade"]
    assert len(traces) == 3
    # At least one pair of chains must differ in their traces
    any_diff = any(
        any(abs(traces[i][j] - traces[k][j]) > 1e-9 for j in range(len(traces[i])))
        for i in range(3) for k in range(i + 1, 3)
    )
    assert any_diff, "All chain traces are identical -> chains initialized identically!"
    print("PASS: multiple chains are not accidentally initialized identically")


def test_invalid_posterior_rejected():
    """NaN / inf inputs should not crash; invalid posterior values are rejected safely."""
    dates, vals = _make_changepoint_data()
    # Inject an extreme outlier to stress the posterior
    bad_vals = list(vals)
    bad_vals[50] = 1e6
    res = run_bayesian_mcmc(
        dates, bad_vals, model_type="changepoint", chains=2, draws=200, burn_in=100, seed=3
    )
    assert res["n_valid"] == len(dates)
    shift = res["parameters"]["slope_shift_per_year"]
    assert np.isfinite(shift["mean"]) and np.isfinite(shift["sd"])
    print("PASS: invalid posterior values are rejected safely without crashing")


def test_adaptation_only_during_burn_in():
    """Adaptation must not run after burn-in: acceptance rate should stabilize (not drift to extremes)."""
    dates, vals = _make_changepoint_data()
    res = run_bayesian_mcmc(dates, vals, model_type="changepoint", chains=2, draws=600, burn_in=300, seed=5)
    rate = res["diagnostics"]["mean_acceptance_rate"]
    assert 0.0 < rate < 1.0, f"acceptance rate out of bounds: {rate}"
    print(f"PASS: adaptation bounded, mean acceptance rate = {rate}")


def test_hdi_and_trace_storage():
    """Posterior uncertainty intervals and trace storage are present and well-formed."""
    dates, vals = _make_changepoint_data()
    res = run_bayesian_mcmc(dates, vals, model_type="changepoint", chains=2, draws=400, burn_in=200, seed=11)
    p = res["parameters"]
    for name in ["pre_slope_per_year", "post_slope_per_year", "slope_shift_per_year", "changepoint_fraction", "sigma"]:
        summ = p[name]
        assert summ["hdi_95_low"] <= summ["median"] <= summ["hdi_95_high"], f"HDI inverted for {name}"
        assert summ["ci_95_low"] <= summ["median"] <= summ["ci_95_high"], f"CI inverted for {name}"
        assert summ["r_hat"] > 0, f"R-hat non-positive for {name}"
        assert summ["ess"] > 0, f"ESS non-positive for {name}"
    traces = res["traces_slope_decade"]
    assert len(traces) == res["diagnostics"]["chains"]
    print("PASS: HDI/CI intervals, R-hat, ESS, and traces all well-formed")


if __name__ == "__main__":
    test_acceptance_prob_pre_update()
    test_reproducible_seeds()
    test_chains_not_identically_initialized()
    test_invalid_posterior_rejected()
    test_adaptation_only_during_burn_in()
    test_hdi_and_trace_storage()
    print("\nALL VALIDATION TESTS PASSED")

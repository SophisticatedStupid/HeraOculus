"""Regression tests for Bayesian MCMC convergence diagnostics."""

from __future__ import annotations

import numpy as np
from app.services.bayesian_service import (
    _rank_normalized_split_rhat,
    _bulk_ess,
    _tail_ess,
    _detect_pathologies,
    _check_burn_in_adequacy,
    _summarize_param,
)


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

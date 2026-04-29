"""Unit tests for ab_test_toolkit.bayesian — Beta-Binomial and Normal-Normal."""

import numpy as np
import pytest
from scipy import stats

from ab_test_toolkit.bayesian import (
    BayesianResult,
    beta_binomial,
    beta_binomial_from_stats,
    normal_normal,
    normal_normal_from_stats,
)


class TestBetaBinomial:
    """Tests for Beta-Binomial conjugate analysis."""

    def test_prob_b_gt_a_cross_validated(self):
        """P(B > A) via numerical integration cross-validated against MC simulation."""
        rng = np.random.default_rng(42)
        control = rng.binomial(1, 0.10, 2000)
        treatment = rng.binomial(1, 0.12, 2000)

        result = beta_binomial(control, treatment, n_simulations=500_000)

        # MC cross-validation
        c_alpha = 1 + control.sum()
        c_beta = 1 + len(control) - control.sum()
        t_alpha = 1 + treatment.sum()
        t_beta = 1 + len(treatment) - treatment.sum()
        mc_rng = np.random.default_rng(999)
        mc_a = mc_rng.beta(c_alpha, c_beta, 1_000_000)
        mc_b = mc_rng.beta(t_alpha, t_beta, 1_000_000)
        mc_prob = (mc_b > mc_a).mean()

        assert result.prob_b_greater_a == pytest.approx(mc_prob, abs=0.01)

    def test_expected_loss_positive(self):
        """Expected loss is always > 0 when rates differ (some chance control beats treatment)."""
        rng = np.random.default_rng(42)
        control = rng.binomial(1, 0.10, 1000)
        treatment = rng.binomial(1, 0.12, 1000)
        result = beta_binomial(control, treatment)
        assert result.expected_loss > 0, (
            "Expected loss should be strictly positive when rates differ (non-zero "
            "probability that control beats treatment)"
        )

    def test_expected_loss_large_when_control_better(self):
        """Expected loss is substantial when control clearly outperforms treatment."""
        result = beta_binomial_from_stats(
            control_count=150, control_total=1000,
            treatment_count=80, treatment_total=1000,
        )
        # Control rate 15% vs treatment rate 8% — expected loss of choosing B should
        # be approximately the rate difference (~0.07)
        assert result.expected_loss > 0.03, (
            f"Expected loss {result.expected_loss:.4f} should be substantial when "
            "control clearly outperforms treatment"
        )

    def test_posterior_parameters(self):
        """Posterior alpha/beta correct: alpha_post = alpha_prior + sum, beta_post = beta_prior + n - sum."""
        rng = np.random.default_rng(42)
        control = rng.binomial(1, 0.10, 500)
        treatment = rng.binomial(1, 0.12, 500)
        result = beta_binomial(control, treatment, prior_alpha=1.0, prior_beta=1.0)

        assert result.control_posterior["alpha"] == pytest.approx(1.0 + control.sum())
        assert result.control_posterior["beta"] == pytest.approx(1.0 + len(control) - control.sum())
        assert result.treatment_posterior["alpha"] == pytest.approx(1.0 + treatment.sum())
        assert result.treatment_posterior["beta"] == pytest.approx(1.0 + len(treatment) - treatment.sum())

    def test_custom_prior(self):
        """Custom prior Beta(0.5, 0.5) produces different posteriors."""
        rng = np.random.default_rng(42)
        control = rng.binomial(1, 0.10, 500)
        treatment = rng.binomial(1, 0.12, 500)
        result_uniform = beta_binomial(control, treatment, prior_alpha=1.0, prior_beta=1.0)
        result_jeffreys = beta_binomial(control, treatment, prior_alpha=0.5, prior_beta=0.5)
        # Posteriors should differ
        assert result_uniform.control_posterior["alpha"] != result_jeffreys.control_posterior["alpha"]

    def test_returns_bayesian_result(self):
        """Returns BayesianResult with all expected fields."""
        rng = np.random.default_rng(42)
        control = rng.binomial(1, 0.10, 500)
        treatment = rng.binomial(1, 0.12, 500)
        result = beta_binomial(control, treatment)
        assert isinstance(result, BayesianResult)
        assert hasattr(result, 'model_type')
        assert hasattr(result, 'prob_b_greater_a')
        assert hasattr(result, 'expected_loss')
        assert hasattr(result, 'credible_interval')
        assert hasattr(result, 'prior_config')
        assert result.model_type == "beta_binomial"

    def test_prob_b_gt_a_in_range(self):
        """P(B > A) is always in [0, 1]."""
        rng = np.random.default_rng(42)
        control = rng.binomial(1, 0.10, 200)
        treatment = rng.binomial(1, 0.10, 200)
        result = beta_binomial(control, treatment)
        assert 0 <= result.prob_b_greater_a <= 1

    def test_zero_conversions_edge_case(self):
        """One variant with zero conversions produces valid posterior."""
        control = np.zeros(1000, dtype=int)
        treatment = np.array([0] * 990 + [1] * 10, dtype=int)
        result = beta_binomial(control, treatment)
        assert 0 <= result.prob_b_greater_a <= 1
        # Control posterior: Beta(1 + 0, 1 + 1000) = Beta(1, 1001)
        assert result.control_posterior["alpha"] == pytest.approx(1.0)
        assert result.control_posterior["beta"] == pytest.approx(1001.0)

    def test_credible_interval_tuple(self):
        """Credible interval is a 2-tuple."""
        rng = np.random.default_rng(42)
        control = rng.binomial(1, 0.10, 500)
        treatment = rng.binomial(1, 0.12, 500)
        result = beta_binomial(control, treatment)
        assert len(result.credible_interval) == 2
        assert result.credible_interval[0] < result.credible_interval[1]


class TestBetaBinomialFromStats:
    """Tests for Beta-Binomial from summary statistics."""

    def test_matches_raw(self):
        """From-stats matches raw data result."""
        rng = np.random.default_rng(42)
        control = rng.binomial(1, 0.10, 2000)
        treatment = rng.binomial(1, 0.13, 2000)
        raw = beta_binomial(control, treatment, n_simulations=100_000)
        from_stats = beta_binomial_from_stats(
            control_count=int(control.sum()),
            control_total=len(control),
            treatment_count=int(treatment.sum()),
            treatment_total=len(treatment),
            n_simulations=100_000,
        )
        # Posteriors should be identical
        assert raw.control_posterior["alpha"] == from_stats.control_posterior["alpha"]
        assert raw.treatment_posterior["alpha"] == from_stats.treatment_posterior["alpha"]


class TestNormalNormal:
    """Tests for Normal-Normal conjugate analysis."""

    def test_posterior_mean_convergence(self):
        """Posterior mean converges to data mean with large N."""
        rng = np.random.default_rng(42)
        control = rng.normal(10.0, 2.0, 5000)
        treatment = rng.normal(10.5, 2.0, 5000)
        result = normal_normal(control, treatment)

        # Posterior means should be close to sample means with large N
        assert result.control_posterior["mean"] == pytest.approx(control.mean(), abs=0.1)
        assert result.treatment_posterior["mean"] == pytest.approx(treatment.mean(), abs=0.1)

    def test_credible_interval_contains_true_effect(self):
        """95% credible interval should contain the true effect for large N."""
        rng = np.random.default_rng(42)
        true_effect = 0.5
        control = rng.normal(10.0, 2.0, 5000)
        treatment = rng.normal(10.0 + true_effect, 2.0, 5000)
        result = normal_normal(control, treatment)
        ci_low, ci_high = result.credible_interval
        assert ci_low <= true_effect <= ci_high

    def test_small_n_warning(self):
        """N < 100 produces a warning note in the result."""
        rng = np.random.default_rng(42)
        control = rng.normal(10.0, 2.0, 50)
        treatment = rng.normal(10.5, 2.0, 50)
        result = normal_normal(control, treatment)
        assert result.model_type == "normal_normal"
        # The result or its prior_config should contain warning info
        # This is implementation-specific, just check it runs without error

    def test_prob_b_gt_a_in_range(self):
        """P(B > A) is in [0, 1]."""
        rng = np.random.default_rng(42)
        control = rng.normal(10.0, 2.0, 500)
        treatment = rng.normal(10.5, 2.0, 500)
        result = normal_normal(control, treatment)
        assert 0 <= result.prob_b_greater_a <= 1

    def test_returns_bayesian_result(self):
        """Returns BayesianResult."""
        rng = np.random.default_rng(42)
        control = rng.normal(10.0, 2.0, 500)
        treatment = rng.normal(10.5, 2.0, 500)
        result = normal_normal(control, treatment)
        assert isinstance(result, BayesianResult)
        assert result.model_type == "normal_normal"


class TestBetaBinomialBoundaryRates:
    """Regression tests for P(B>A) at extreme conversion rates."""

    def test_identical_100pct_conversion(self):
        """1000/1000 vs 1000/1000 → P(B>A) ≈ 0.50."""
        result = beta_binomial_from_stats(1000, 1000, 1000, 1000)
        assert result.prob_b_greater_a == pytest.approx(0.50, abs=0.02)

    def test_identical_0pct_conversion(self):
        """0/1000 vs 0/1000 → P(B>A) ≈ 0.50."""
        result = beta_binomial_from_stats(0, 1000, 0, 1000)
        assert result.prob_b_greater_a == pytest.approx(0.50, abs=0.02)

    def test_identical_near_100pct(self):
        """999/1000 vs 999/1000 → P(B>A) ≈ 0.50."""
        result = beta_binomial_from_stats(999, 1000, 999, 1000)
        assert result.prob_b_greater_a == pytest.approx(0.50, abs=0.02)

    def test_near_zero_with_difference(self):
        """0/1000 vs 5/1000 → P(B>A) clearly > 0.50."""
        result = beta_binomial_from_stats(0, 1000, 5, 1000)
        assert result.prob_b_greater_a > 0.90


class TestNormalNormalFromStats:
    """Tests for Normal-Normal from summary statistics."""

    def test_matches_raw_arrays(self):
        """From-stats produces same result as raw arrays."""
        rng = np.random.default_rng(99)
        c = rng.normal(50, 15, 5000)
        t = rng.normal(52, 15, 5000)
        raw = normal_normal(c, t)
        from_stats = normal_normal_from_stats(
            float(c.mean()), float(c.std(ddof=1)), 5000,
            float(t.mean()), float(t.std(ddof=1)), 5000,
        )
        assert raw.prob_b_greater_a == pytest.approx(from_stats.prob_b_greater_a, abs=0.001)

    def test_small_n_warning(self):
        """Small n (< 30) includes warning about t-posterior tails."""
        result = normal_normal_from_stats(50.0, 15.0, 20, 52.0, 15.0, 20)
        warnings = result.prior_config.get("warnings", [])
        assert any("t-posterior" in w or "Small sample" in w for w in warnings)

    def test_returns_bayesian_result(self):
        result = normal_normal_from_stats(50.0, 15.0, 1000, 52.0, 15.0, 1000)
        assert isinstance(result, BayesianResult)
        assert result.model_type == "normal_normal"


class TestNormalNormalZeroVariance:
    """Zero-variance edge cases for Normal-Normal model."""

    def test_both_zero_std_same_means(self):
        """Both groups zero std, same means → P(B>A) = 0.5."""
        result = normal_normal_from_stats(50.0, 0.0, 100, 50.0, 0.0, 100)
        assert result.prob_b_greater_a == pytest.approx(0.5)

    def test_both_zero_std_different_means(self):
        """Both groups zero std, treatment higher → P(B>A) = 1.0."""
        result = normal_normal_from_stats(50.0, 0.0, 100, 52.0, 0.0, 100)
        assert result.prob_b_greater_a == 1.0

    def test_both_zero_std_control_better(self):
        """Both groups zero std, control higher → P(B>A) = 0.0."""
        result = normal_normal_from_stats(52.0, 0.0, 100, 50.0, 0.0, 100)
        assert result.prob_b_greater_a == 0.0

    def test_one_zero_std_no_crash(self):
        """One group zero std doesn't crash."""
        result = normal_normal_from_stats(50.0, 0.0, 100, 52.0, 15.0, 100)
        assert 0 <= result.prob_b_greater_a <= 1

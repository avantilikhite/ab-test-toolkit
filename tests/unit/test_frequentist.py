"""Unit tests for ab_test_toolkit.frequentist — Z-test and Welch's t-test."""

import numpy as np
import pytest
from scipy import stats

from ab_test_toolkit.frequentist import (
    FrequentistResult,
    z_test,
    z_test_from_stats,
    welch_t_test,
    welch_t_test_from_stats,
)


class TestZTest:
    """Tests for two-proportion Z-test."""

    def test_p_value_matches_scipy(self):
        """Z-test p-value matches scipy.stats reference within tolerance."""
        rng = np.random.default_rng(42)
        control = rng.binomial(1, 0.10, 5000)
        treatment = rng.binomial(1, 0.12, 5000)

        result = z_test(control, treatment, alpha=0.05)

        # scipy reference (unpooled)
        p0 = control.mean()
        p1 = treatment.mean()
        n0, n1 = len(control), len(treatment)
        se = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
        z_stat = (p1 - p0) / se
        p_expected = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        assert result.p_value == pytest.approx(p_expected, abs=1e-6)
        assert result.test_type == "z_test"
        assert isinstance(result.is_significant, bool)

    def test_cohens_h(self):
        """Cohen's h = 2*arcsin(sqrt(p2)) - 2*arcsin(sqrt(p1))."""
        rng = np.random.default_rng(42)
        control = rng.binomial(1, 0.10, 1000)
        treatment = rng.binomial(1, 0.15, 1000)
        result = z_test(control, treatment)

        p0 = control.mean()
        p1 = treatment.mean()
        expected_h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p0))
        assert result.effect_size == pytest.approx(expected_h, abs=1e-6)

    def test_confidence_interval(self):
        """CI contains the point estimate."""
        rng = np.random.default_rng(42)
        control = rng.binomial(1, 0.10, 2000)
        treatment = rng.binomial(1, 0.12, 2000)
        result = z_test(control, treatment)
        assert result.ci_lower <= result.point_estimate <= result.ci_upper

    def test_returns_frequentist_result(self):
        """Returns a FrequentistResult with all expected fields."""
        rng = np.random.default_rng(42)
        control = rng.binomial(1, 0.10, 500)
        treatment = rng.binomial(1, 0.12, 500)
        result = z_test(control, treatment)
        assert isinstance(result, FrequentistResult)
        assert hasattr(result, 'statistic')
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'ci_lower')
        assert hasattr(result, 'ci_upper')
        assert hasattr(result, 'point_estimate')
        assert hasattr(result, 'effect_size')
        assert hasattr(result, 'alpha')
        assert hasattr(result, 'is_significant')

    def test_zero_conversions_edge_case(self):
        """One variant with zero conversions produces a valid result."""
        control = np.zeros(1000, dtype=int)
        treatment = np.array([0] * 990 + [1] * 10, dtype=int)
        result = z_test(control, treatment)
        assert 0 <= result.p_value <= 1


class TestZTestFromStats:
    """Tests for Z-test from summary statistics."""

    def test_matches_raw_data(self):
        """Summary stats input gives identical results to raw data."""
        rng = np.random.default_rng(42)
        control = rng.binomial(1, 0.10, 2000)
        treatment = rng.binomial(1, 0.13, 2000)

        raw_result = z_test(control, treatment)
        stats_result = z_test_from_stats(
            control_count=int(control.sum()),
            control_total=len(control),
            treatment_count=int(treatment.sum()),
            treatment_total=len(treatment),
        )
        assert raw_result.p_value == pytest.approx(stats_result.p_value, abs=1e-6)
        assert raw_result.point_estimate == pytest.approx(stats_result.point_estimate, abs=1e-6)


class TestWelchTTest:
    """Tests for Welch's t-test."""

    def test_p_value_matches_scipy(self):
        """Welch's t-test matches scipy.stats.ttest_ind(equal_var=False)."""
        rng = np.random.default_rng(42)
        control = rng.normal(10.0, 2.0, 500)
        treatment = rng.normal(10.5, 2.1, 500)

        result = welch_t_test(control, treatment)
        scipy_stat, scipy_p = stats.ttest_ind(treatment, control, equal_var=False)

        assert result.p_value == pytest.approx(scipy_p, abs=1e-6)
        assert result.test_type == "welch_t_test"

    def test_cohens_d(self):
        """Cohen's d = (mean2 - mean1) / s_pooled."""
        rng = np.random.default_rng(42)
        control = rng.normal(10.0, 2.0, 1000)
        treatment = rng.normal(11.0, 2.0, 1000)
        result = welch_t_test(control, treatment)

        s_pooled = np.sqrt(
            ((len(control) - 1) * control.std(ddof=1) ** 2 + (len(treatment) - 1) * treatment.std(ddof=1) ** 2)
            / (len(control) + len(treatment) - 2)
        )
        expected_d = (treatment.mean() - control.mean()) / s_pooled
        assert result.effect_size == pytest.approx(expected_d, abs=1e-6)

    def test_normality_check_present(self):
        """Normality check is present for continuous data."""
        rng = np.random.default_rng(42)
        control = rng.normal(10.0, 2.0, 50)
        treatment = rng.normal(10.5, 2.0, 50)
        result = welch_t_test(control, treatment)
        assert result.normality_check is not None
        assert "control" in result.normality_check
        assert "treatment" in result.normality_check
        assert result.normality_check["method"] == "shapiro_wilk"

    def test_normality_check_large_n_uses_skewness_kurtosis(self):
        """Large samples use skewness/kurtosis instead of Shapiro-Wilk."""
        rng = np.random.default_rng(42)
        control = rng.normal(10.0, 2.0, 6000)
        treatment = rng.normal(10.5, 2.0, 6000)
        result = welch_t_test(control, treatment)
        assert result.normality_check is not None
        assert result.normality_check["method"] == "skewness_kurtosis"
        assert "skewness" in result.normality_check["control"]
        assert "kurtosis" in result.normality_check["control"]
        assert "note" in result.normality_check

    def test_ci_contains_point_estimate(self):
        """Confidence interval contains point estimate."""
        rng = np.random.default_rng(42)
        control = rng.normal(10.0, 2.0, 300)
        treatment = rng.normal(10.5, 2.1, 300)
        result = welch_t_test(control, treatment)
        assert result.ci_lower <= result.point_estimate <= result.ci_upper


class TestWelchTTestFromStats:
    """Tests for Welch's t-test from summary stats."""

    def test_matches_raw_data(self):
        """Summary stats match raw data results."""
        rng = np.random.default_rng(42)
        control = rng.normal(10.0, 2.0, 500)
        treatment = rng.normal(10.5, 2.1, 500)

        raw_result = welch_t_test(control, treatment)
        stats_result = welch_t_test_from_stats(
            control_mean=float(control.mean()),
            control_std=float(control.std(ddof=1)),
            control_n=len(control),
            treatment_mean=float(treatment.mean()),
            treatment_std=float(treatment.std(ddof=1)),
            treatment_n=len(treatment),
        )
        assert raw_result.p_value == pytest.approx(stats_result.p_value, abs=1e-6)


class TestWelchEdgeCases:
    """Edge case tests for Welch's t-test."""

    def test_zero_variance_both_groups_different_means(self):
        """Zero std in both groups with different means → p=0."""
        result = welch_t_test_from_stats(50.0, 0.0, 100, 52.0, 0.0, 100)
        assert result.p_value == 0.0
        assert result.is_significant is True

    def test_zero_variance_both_groups_same_means(self):
        """Zero std in both groups with same means → p=1."""
        result = welch_t_test_from_stats(50.0, 0.0, 100, 50.0, 0.0, 100)
        assert result.p_value == 1.0
        assert result.is_significant is False

    def test_n_less_than_2_raises(self):
        """n < 2 raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            welch_t_test_from_stats(50.0, 15.0, 1, 52.0, 15.0, 100)

    def test_negative_std_raises(self):
        """Negative std raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            welch_t_test_from_stats(50.0, -1.0, 100, 52.0, 15.0, 100)


class TestZTestFromStatsValidation:
    """Input validation for z_test_from_stats."""

    def test_count_exceeds_total_raises(self):
        with pytest.raises(ValueError, match="exceed"):
            z_test_from_stats(600, 500, 50, 100)

    def test_negative_count_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            z_test_from_stats(-1, 100, 50, 100)

    def test_zero_total_raises(self):
        with pytest.raises(ValueError, match="positive"):
            z_test_from_stats(0, 0, 50, 100)

    def test_valid_inputs_work(self):
        result = z_test_from_stats(50, 500, 60, 500)
        assert 0 <= result.p_value <= 1


def test_newcombe_ci_clipped_to_unit_interval():
    """Newcombe CI must respect the natural [-1, 1] support of a risk difference."""
    from ab_test_toolkit.frequentist import _newcombe_diff_ci
    # 0/100 vs 100/100 → diff = 1.0; without clipping the upper bound exceeds 1
    lo, hi = _newcombe_diff_ci(0, 100, 100, 100, alpha=0.05)
    assert -1.0 <= lo <= 1.0
    assert -1.0 <= hi <= 1.0
    # 100/100 vs 0/100 → diff = -1.0
    lo, hi = _newcombe_diff_ci(100, 100, 0, 100, alpha=0.05)
    assert -1.0 <= lo <= 1.0
    assert -1.0 <= hi <= 1.0

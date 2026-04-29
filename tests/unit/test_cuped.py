"""Unit tests for ab_test_toolkit.cuped — CUPED variance reduction."""

import numpy as np
import pytest

from ab_test_toolkit.cuped import CUPEDResult, cuped_adjust


class TestCUPEDAdjust:
    """Tests for CUPED variance reduction."""

    def test_variance_reduction_with_correlated_covariate(self):
        """High correlation covariate produces substantial variance reduction."""
        rng = np.random.default_rng(42)
        n = 2000
        cov_ctrl = rng.normal(5.0, 1.0, n)
        cov_treat = rng.normal(5.0, 1.0, n)
        noise_ctrl = rng.normal(0, 0.7, n)
        noise_treat = rng.normal(0, 0.7, n)
        ctrl = 0.8 * cov_ctrl + noise_ctrl
        treat = 0.8 * cov_treat + noise_treat + 0.5
        result = cuped_adjust(ctrl, treat, cov_ctrl, cov_treat)
        assert isinstance(result, CUPEDResult)
        # rho ≈ 0.7, so variance reduction should be around 49%
        assert result.variance_reduction_pct > 30

    def test_pooled_theta(self):
        """Theta matches manual Cov(Y,X)/Var(X) calculation."""
        rng = np.random.default_rng(42)
        n = 1000
        cov = rng.normal(5.0, 1.0, 2 * n)
        outcome = 0.8 * cov + rng.normal(0, 1, 2 * n)
        ctrl_out, treat_out = outcome[:n], outcome[n:]
        ctrl_cov, treat_cov = cov[:n], cov[n:]
        result = cuped_adjust(ctrl_out, treat_out, ctrl_cov, treat_cov)
        # Manual theta: Cov(Y,X) / Var(X) pooled
        all_y = np.concatenate([ctrl_out, treat_out])
        all_x = np.concatenate([ctrl_cov, treat_cov])
        expected_theta = np.cov(all_y, all_x)[0, 1] / np.var(all_x, ddof=1)
        assert result.theta == pytest.approx(expected_theta, rel=0.01)

    def test_zero_correlation_degrades_gracefully(self):
        """Zero-correlation covariate: adjusted ≈ unadjusted."""
        rng = np.random.default_rng(42)
        n = 1000
        ctrl = rng.normal(10.0, 2.0, n)
        treat = rng.normal(10.5, 2.0, n)
        cov_ctrl = rng.normal(0, 1, n)  # Uncorrelated
        cov_treat = rng.normal(0, 1, n)
        result = cuped_adjust(ctrl, treat, cov_ctrl, cov_treat)
        assert abs(result.adjusted_estimate - result.unadjusted_estimate) < 0.5
        assert result.variance_reduction_pct < 10

    def test_adjusted_ci_narrower(self):
        """Adjusted CI strictly narrower than unadjusted when rho > 0."""
        rng = np.random.default_rng(42)
        n = 2000
        cov_ctrl = rng.normal(5.0, 1.0, n)
        cov_treat = rng.normal(5.0, 1.0, n)
        ctrl = 0.8 * cov_ctrl + rng.normal(0, 0.7, n)
        treat = 0.8 * cov_treat + rng.normal(0, 0.7, n) + 0.5
        result = cuped_adjust(ctrl, treat, cov_ctrl, cov_treat)
        unadj_width = result.unadjusted_ci[1] - result.unadjusted_ci[0]
        adj_width = result.adjusted_ci[1] - result.adjusted_ci[0]
        assert adj_width < unadj_width

    def test_unequal_group_sizes(self):
        """Works with unequal group sizes."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(10.0, 2.0, 800)
        treat = rng.normal(10.5, 2.0, 1200)
        cov_ctrl = rng.normal(5, 1, 800)
        cov_treat = rng.normal(5, 1, 1200)
        result = cuped_adjust(ctrl, treat, cov_ctrl, cov_treat)
        assert isinstance(result, CUPEDResult)

    def test_correlation_field(self):
        """Correlation field is accurate."""
        rng = np.random.default_rng(42)
        n = 2000
        cov = rng.normal(5.0, 1.0, 2 * n)
        outcome = 0.8 * cov + rng.normal(0, 0.7, 2 * n)
        result = cuped_adjust(outcome[:n], outcome[n:], cov[:n], cov[n:])
        all_y = np.concatenate([outcome[:n], outcome[n:]])
        all_x = np.concatenate([cov[:n], cov[n:]])
        expected_corr = np.corrcoef(all_y, all_x)[0, 1]
        assert result.correlation == pytest.approx(expected_corr, abs=0.01)


def test_winsorize_clips_extreme_values():
    import numpy as np
    from ab_test_toolkit.cuped import winsorize
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 10_000)
    x[0] = 1e6  # outlier
    x[1] = -1e6
    w = winsorize(x, p=0.99)
    assert w.max() < 1e6
    assert w.min() > -1e6
    # mid-quantile values are unchanged
    assert np.allclose(np.median(w), np.median(x), atol=0.05)


def test_winsorize_validation():
    import numpy as np
    import pytest
    from ab_test_toolkit.cuped import winsorize
    with pytest.raises(ValueError):
        winsorize(np.array([np.nan, 1.0]))
    with pytest.raises(ValueError):
        winsorize(np.array([1.0, 2.0]), p=1.5)

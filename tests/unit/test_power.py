"""Unit tests for ab_test_toolkit.power — sample size and power analysis."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ab_test_toolkit.power import PowerResult, required_sample_size, required_sample_size_continuous, power_curve


class TestRequiredSampleSizeBalanced:
    """Tests for balanced (50/50) sample size calculation."""

    def test_balanced_matches_scipy_reference(self):
        """Balanced design matches manual scipy.stats.norm.ppf calculation."""
        # Known parameters
        baseline = 0.10
        mde = 0.02
        alpha = 0.05
        power_val = 0.80

        # Manual calculation using Z-test power formula
        p0, p1 = baseline, baseline + mde
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power_val)
        n_manual = ((z_alpha + z_beta) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1))) / (p1 - p0) ** 2
        n_manual = int(np.ceil(n_manual))

        result = required_sample_size(baseline_rate=baseline, mde=mde, alpha=alpha, power=power_val)
        assert isinstance(result, PowerResult)
        assert result.n_control == n_manual
        assert result.n_treatment == n_manual
        assert result.n_total == 2 * n_manual
        assert result.power_loss_pct == pytest.approx(0.0, abs=0.01)

    def test_returns_power_result(self):
        """Returns a PowerResult with all expected fields."""
        result = required_sample_size(baseline_rate=0.10, mde=0.02)
        assert hasattr(result, 'n_control')
        assert hasattr(result, 'n_treatment')
        assert hasattr(result, 'n_total')
        assert hasattr(result, 'n_effective')
        assert hasattr(result, 'power_loss_pct')
        assert hasattr(result, 'estimated_days')

    def test_known_answer_10pct_baseline(self):
        """Hardcoded known answer: 10% baseline, 2pp MDE → 3,839 per arm.

        This guards against formula regressions that would pass the formula-
        recomputation test above (which uses the same math).
        """
        result = required_sample_size(
            baseline_rate=0.10, mde=0.02, alpha=0.05, power=0.80,
        )
        # n = ceil((Z_{0.975} + Z_{0.80})^2 * (0.10*0.90 + 0.12*0.88) / 0.02^2)
        # = ceil(7.8489 * 0.1956 / 0.0004) = ceil(3838.10) = 3839
        assert result.n_control == 3839
        assert result.n_treatment == 3839
        assert result.n_total == 7678


class TestRequiredSampleSizeUnequal:
    """Tests for unequal allocation."""

    def test_unequal_allocation_power_loss(self):
        """Unequal allocation (80/20) has positive power loss."""
        result = required_sample_size(
            baseline_rate=0.10, mde=0.02, allocation_ratio=4.0,
        )
        assert result.power_loss_pct > 0
        assert result.n_treatment > result.n_control

    def test_n_effective_formula(self):
        """n_effective = 4*n1*n2 / (n1+n2) for allocation_ratio != 1."""
        result = required_sample_size(
            baseline_rate=0.10, mde=0.02, allocation_ratio=2.0,
        )
        expected_neff = 4 * result.n_control * result.n_treatment / (result.n_control + result.n_treatment)
        assert result.n_effective == pytest.approx(expected_neff, rel=1e-6)


class TestRequiredSampleSizeMDEMode:
    """Tests for relative MDE mode."""

    def test_relative_mde_conversion(self):
        """Relative MDE converted to absolute: mde_abs = baseline * mde_relative."""
        baseline = 0.10
        relative_mde = 0.20  # 20% relative lift → 0.02 absolute
        result_relative = required_sample_size(
            baseline_rate=baseline, mde=relative_mde, mde_mode="relative",
        )
        result_absolute = required_sample_size(
            baseline_rate=baseline, mde=baseline * relative_mde, mde_mode="absolute",
        )
        assert result_relative.n_control == result_absolute.n_control


class TestRequiredSampleSizeDuration:
    """Tests for duration estimation."""

    def test_duration_with_daily_traffic(self):
        """estimated_days computed when daily_traffic provided."""
        result = required_sample_size(
            baseline_rate=0.10, mde=0.02, daily_traffic=1000,
        )
        assert result.estimated_days is not None
        assert result.estimated_days > 0
        # Duration = ceil(n_total / daily_traffic)
        expected_days = int(np.ceil(result.n_total / 1000))
        assert result.estimated_days == expected_days

    def test_no_duration_without_traffic(self):
        """estimated_days is None when daily_traffic not provided."""
        result = required_sample_size(baseline_rate=0.10, mde=0.02)
        assert result.estimated_days is None


class TestRequiredSampleSizeValidation:
    """Tests for input validation."""

    def test_invalid_baseline_rate(self):
        with pytest.raises(ValueError):
            required_sample_size(baseline_rate=0.0, mde=0.02)
        with pytest.raises(ValueError):
            required_sample_size(baseline_rate=1.0, mde=0.02)

    def test_invalid_mde(self):
        with pytest.raises(ValueError):
            required_sample_size(baseline_rate=0.10, mde=0.0)
        with pytest.raises(ValueError):
            required_sample_size(baseline_rate=0.10, mde=-0.01)

    def test_resulting_rate_out_of_bounds(self):
        with pytest.raises(ValueError):
            required_sample_size(baseline_rate=0.95, mde=0.10)


class TestPowerCurve:
    """Tests for power_curve function."""

    def test_returns_dataframe(self):
        """power_curve returns a DataFrame with expected columns."""
        df = power_curve(baseline_rate=0.10, mde_range=(0.01, 0.05))
        assert isinstance(df, pd.DataFrame)
        assert "mde" in df.columns
        assert "n_control" in df.columns
        assert "n_treatment" in df.columns
        assert "n_total" in df.columns

    def test_n_decreases_as_mde_increases(self):
        """Required N monotonically decreases as MDE increases."""
        df = power_curve(baseline_rate=0.10, mde_range=(0.01, 0.05), n_points=20)
        # n_total should decrease as mde increases
        assert all(df["n_total"].iloc[i] >= df["n_total"].iloc[i + 1] for i in range(len(df) - 1))

    def test_n_points(self):
        """Output has n_points rows."""
        df = power_curve(baseline_rate=0.10, mde_range=(0.01, 0.05), n_points=25)
        assert len(df) == 25


class TestRequiredSampleSizeContinuous:
    """Tests for continuous metric sample size calculation."""

    def test_balanced_matches_formula(self):
        """Balanced design matches manual two-sample t-test formula."""
        sigma = 2.0
        mde = 0.5
        alpha = 0.05
        power_val = 0.80

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power_val)
        n_manual = int(np.ceil((z_alpha + z_beta) ** 2 * 2 * sigma**2 / mde**2))

        result = required_sample_size_continuous(
            baseline_mean=10.0, baseline_std=sigma, mde=mde,
            alpha=alpha, power=power_val,
        )
        assert isinstance(result, PowerResult)
        assert result.n_control == n_manual
        assert result.n_treatment == n_manual
        assert result.n_total == 2 * n_manual
        assert result.power_loss_pct == pytest.approx(0.0, abs=0.01)

    def test_unequal_allocation_power_loss(self):
        """Unequal allocation has positive power loss."""
        result = required_sample_size_continuous(
            baseline_mean=10.0, baseline_std=2.0, mde=0.5,
            allocation_ratio=3.0,
        )
        assert result.power_loss_pct > 0
        assert result.n_treatment > result.n_control

    def test_unequal_variance(self):
        """Supports different treatment std."""
        result_equal = required_sample_size_continuous(
            baseline_mean=10.0, baseline_std=2.0, mde=0.5,
        )
        result_higher = required_sample_size_continuous(
            baseline_mean=10.0, baseline_std=2.0, mde=0.5,
            treatment_std=3.0,
        )
        assert result_higher.n_total > result_equal.n_total

    def test_duration_estimate(self):
        """Duration is computed when daily_traffic provided."""
        result = required_sample_size_continuous(
            baseline_mean=10.0, baseline_std=2.0, mde=0.5,
            daily_traffic=500,
        )
        assert result.estimated_days is not None
        assert result.estimated_days == int(np.ceil(result.n_total / 500))

    def test_validation_errors(self):
        """Invalid inputs raise ValueError."""
        with pytest.raises(ValueError):
            required_sample_size_continuous(
                baseline_mean=10.0, baseline_std=0.0, mde=0.5,
            )
        with pytest.raises(ValueError):
            required_sample_size_continuous(
                baseline_mean=10.0, baseline_std=2.0, mde=-1.0,
            )

    def test_smaller_mde_needs_more_samples(self):
        """Smaller effect size requires more samples."""
        result_large = required_sample_size_continuous(
            baseline_mean=10.0, baseline_std=2.0, mde=1.0,
        )
        result_small = required_sample_size_continuous(
            baseline_mean=10.0, baseline_std=2.0, mde=0.5,
        )
        assert result_small.n_total > result_large.n_total

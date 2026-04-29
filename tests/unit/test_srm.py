"""Unit tests for ab_test_toolkit.srm — Sample Ratio Mismatch detection."""

import pytest
from scipy import stats as sp_stats

from ab_test_toolkit.srm import SRMResult, check_srm


class TestCheckSRM:
    """Tests for SRM chi-square test."""

    def test_balanced_no_mismatch(self):
        """50/50 split with balanced data shows no mismatch."""
        result = check_srm(observed=(5000, 5000))
        assert isinstance(result, SRMResult)
        assert result.has_mismatch is False
        assert result.p_value > 0.01

    def test_large_imbalance_triggers_mismatch(self):
        """Large allocation bias detected as mismatch."""
        result = check_srm(observed=(4000, 6000))
        assert result.has_mismatch is True
        assert result.p_value < 0.01

    def test_custom_expected_ratio(self):
        """Custom 80/20 ratio matching observed shows no mismatch."""
        result = check_srm(observed=(8000, 2000), expected_ratio=(0.8, 0.2))
        assert result.has_mismatch is False
        assert result.p_value > 0.01

    def test_configurable_threshold(self):
        """Custom threshold changes mismatch detection."""
        # Moderate imbalance: p-value between strict and lenient thresholds
        # (4800, 5200) on 50/50 has chi2=16, p~6e-5, so both strict and lenient detect
        # Use a milder imbalance that falls between thresholds
        result_strict = check_srm(observed=(4900, 5100), threshold=0.10)
        result_lenient = check_srm(observed=(4900, 5100), threshold=0.001)
        # Same statistic regardless of threshold
        assert result_strict.chi2_statistic == result_lenient.chi2_statistic
        assert result_strict.p_value == result_lenient.p_value
        # Strict threshold flags it, lenient does not (or vice versa) — they must differ
        # (4900 vs 5100 on 50/50: chi2=4, p≈0.046)
        assert result_strict.has_mismatch is True, "Strict threshold (0.10) should flag p≈0.046"
        assert result_lenient.has_mismatch is False, "Lenient threshold (0.001) should not flag p≈0.046"

    def test_observed_ratio_accuracy(self):
        """Observed ratio tuple is accurate."""
        result = check_srm(observed=(3000, 7000))
        assert result.observed_ratio[0] == pytest.approx(0.3, abs=0.001)
        assert result.observed_ratio[1] == pytest.approx(0.7, abs=0.001)

    def test_expected_ratio_stored(self):
        """Expected ratio is stored in result."""
        result = check_srm(observed=(5000, 5000), expected_ratio=(0.6, 0.4))
        assert result.expected_ratio == (0.6, 0.4)

    def test_equal_counts(self):
        """Exactly equal counts should pass."""
        result = check_srm(observed=(1000, 1000))
        assert result.has_mismatch is False
        assert result.chi2_statistic == pytest.approx(0.0)

    def test_chi2_matches_scipy(self):
        """Chi-square statistic matches scipy.stats.chisquare."""
        observed = (4500, 5500)
        total = sum(observed)
        expected = (total * 0.5, total * 0.5)
        scipy_result = sp_stats.chisquare(observed, f_exp=expected)
        result = check_srm(observed=observed)
        assert result.chi2_statistic == pytest.approx(scipy_result.statistic, abs=1e-6)
        assert result.p_value == pytest.approx(scipy_result.pvalue, abs=1e-6)


class TestSRMEdgeCases:
    """Edge cases for SRM check."""

    def test_zero_total_returns_no_mismatch(self):
        """observed=(0,0) returns no mismatch instead of crashing."""
        result = check_srm(observed=(0, 0))
        assert result.has_mismatch is False
        assert result.p_value == 1.0

    def test_one_zero_group(self):
        """One group with zero users triggers mismatch."""
        result = check_srm(observed=(1000, 0))
        assert result.has_mismatch is True


def test_srm_small_cell_warning():
    from ab_test_toolkit.srm import check_srm
    res = check_srm((0, 5))  # min expected ~2.5 < 5
    assert res.warning is not None
    assert "small" in res.warning.lower() or "expected" in res.warning.lower()


def test_check_srm_by_stratum_detects_per_day_mismatch():
    import pandas as pd
    from ab_test_toolkit.srm import check_srm_by_stratum
    rows = []
    # Days 1-4 balanced, day 5 grossly imbalanced
    for d in range(1, 5):
        rows += [{"day": d, "group": "control"}] * 50
        rows += [{"day": d, "group": "treatment"}] * 50
    rows += [{"day": 5, "group": "control"}] * 80
    rows += [{"day": 5, "group": "treatment"}] * 20
    df = pd.DataFrame(rows)
    out = check_srm_by_stratum(df, group_col="group", stratum_col="day")
    assert out.n_strata == 5
    assert out.any_mismatch is True
    assert out.n_mismatches >= 1

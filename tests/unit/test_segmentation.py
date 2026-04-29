"""Unit tests for ab_test_toolkit.segmentation — HTE and Simpson's Paradox."""

import numpy as np
import pandas as pd
import pytest

from ab_test_toolkit.segmentation import SegmentResult, segment_analysis


class TestSegmentAnalysis:
    """Tests for per-segment analysis."""

    def test_per_segment_results_returned(self, segmented_df):
        """Each segment has estimate, CI, n, and p_value."""
        result = segment_analysis(segmented_df)
        assert isinstance(result, SegmentResult)
        assert len(result.segment_results) == 2  # mobile, desktop
        for seg in result.segment_results:
            assert "segment" in seg
            assert "estimate" in seg
            assert "ci" in seg
            assert "n" in seg
            assert "p_value" in seg

    def test_simpsons_paradox_detection(self, simpsons_paradox_df):
        """Simpson's Paradox detected when aggregate and segment signs differ."""
        result = segment_analysis(simpsons_paradox_df)
        assert result.simpsons_paradox is True
        assert result.simpsons_details is not None

    def test_no_paradox(self, segmented_df):
        """No paradox when all segments agree with aggregate."""
        result = segment_analysis(segmented_df)
        assert result.simpsons_paradox is False

    def test_n_segments_count(self, segmented_df):
        """n_segments matches actual number of segments."""
        result = segment_analysis(segmented_df)
        assert result.n_segments == 2

    def test_multiple_comparisons_note(self, segmented_df):
        """Disclaimer about unadjusted p-values is present."""
        result = segment_analysis(segmented_df)
        assert result.multiple_comparisons_note is not None
        assert "2" in result.multiple_comparisons_note  # should mention segment count

    def test_aggregate_estimate(self, segmented_df):
        """Aggregate estimate is computed."""
        result = segment_analysis(segmented_df)
        assert result.aggregate_estimate is not None
        assert result.aggregate_ci is not None
        assert len(result.aggregate_ci) == 2

    def test_single_segment(self):
        """Single segment degenerate case works."""
        df = pd.DataFrame({
            "group": ["control"] * 200 + ["treatment"] * 200,
            "value": np.random.default_rng(42).binomial(1, 0.10, 400),
            "segment": ["only_seg"] * 400,
        })
        result = segment_analysis(df)
        assert result.n_segments == 1
        assert result.simpsons_paradox is False


class TestHolmBonferroni:
    """Tests for Holm-Bonferroni adjusted p-values in segment analysis."""

    def test_adjusted_p_values_present(self, segmented_df):
        """Each segment result contains an adjusted p-value."""
        result = segment_analysis(segmented_df)
        for seg in result.segment_results:
            assert "p_value_adjusted" in seg
            assert 0 <= seg["p_value_adjusted"] <= 1.0

    def test_adjusted_ge_raw(self, segmented_df):
        """Adjusted p-values are always >= raw p-values."""
        result = segment_analysis(segmented_df)
        for seg in result.segment_results:
            assert seg["p_value_adjusted"] >= seg["p_value"] - 1e-10

    def test_single_segment_unchanged(self):
        """With one segment, adjusted = raw."""
        df = pd.DataFrame({
            "group": ["control"] * 500 + ["treatment"] * 500,
            "value": np.random.default_rng(42).binomial(1, 0.10, 1000),
            "segment": ["only"] * 1000,
        })
        result = segment_analysis(df)
        seg = result.segment_results[0]
        assert seg["p_value_adjusted"] == pytest.approx(seg["p_value"], abs=1e-10)

    def test_note_mentions_holm(self, segmented_df):
        """Multiple comparisons note references Holm-Bonferroni."""
        result = segment_analysis(segmented_df)
        assert "Holm-Bonferroni" in result.multiple_comparisons_note

    def test_three_segments(self):
        """Holm-Bonferroni works with 3 segments."""
        rng = np.random.default_rng(42)
        dfs = []
        for seg_name, c_rate, t_rate in [
            ("A", 0.10, 0.12), ("B", 0.10, 0.11), ("C", 0.10, 0.15),
        ]:
            n = 300
            ctrl = rng.binomial(1, c_rate, n)
            treat = rng.binomial(1, t_rate, n)
            seg_df = pd.DataFrame({
                "group": ["control"] * n + ["treatment"] * n,
                "value": np.concatenate([ctrl, treat]),
                "segment": [seg_name] * (2 * n),
            })
            dfs.append(seg_df)
        df = pd.concat(dfs, ignore_index=True)
        result = segment_analysis(df)
        assert result.n_segments == 3
        # Smallest raw p-value should have smallest adjusted p-value
        raw_ps = [s["p_value"] for s in result.segment_results]
        adj_ps = [s["p_value_adjusted"] for s in result.segment_results]
        min_raw_idx = raw_ps.index(min(raw_ps))
        min_adj_idx = adj_ps.index(min(adj_ps))
        assert min_raw_idx == min_adj_idx


class TestHolmNaN:
    """Tests for NaN handling in Holm-Bonferroni correction."""

    def test_nan_p_value_passed_through(self):
        """NaN p-values are preserved, not turned into 0."""
        from ab_test_toolkit.segmentation import _holm_bonferroni
        import math
        result = _holm_bonferroni([0.01, float("nan"), 0.05])
        assert result[0] == pytest.approx(0.02, abs=1e-10)
        assert math.isnan(result[1])
        assert result[2] == pytest.approx(0.05, abs=1e-10)

    def test_all_nan(self):
        """All NaN p-values returned as NaN."""
        from ab_test_toolkit.segmentation import _holm_bonferroni
        import math
        result = _holm_bonferroni([float("nan"), float("nan")])
        assert all(math.isnan(r) for r in result)

    def test_degenerate_segment_no_crash(self):
        """Segment with all-zero values (zero variance) produces NaN p but no crash."""
        df = pd.DataFrame({
            "group": ["control"] * 100 + ["treatment"] * 100 + ["control"] * 5 + ["treatment"] * 5,
            "value": list(np.random.default_rng(42).binomial(1, 0.1, 100)) + list(np.random.default_rng(43).binomial(1, 0.12, 100)) + [0] * 5 + [0] * 5,
            "segment": ["A"] * 200 + ["B"] * 10,
        })
        result = segment_analysis(df)
        assert result.n_segments == 2
        seg_b = [s for s in result.segment_results if s["segment"] == "B"][0]
        import math
        assert math.isnan(seg_b["p_value"]) or seg_b["p_value"] >= 0


class TestSimpsonsParadoxMajority:
    """Simpson's Paradox requires majority of segments to disagree."""

    def test_single_segment_disagree_not_flagged(self):
        """One segment disagreeing out of 3 should NOT trigger Simpson's."""
        rng = np.random.default_rng(42)
        dfs = []
        # Segments A and B: positive effect (agree with aggregate)
        for seg in ["A", "B"]:
            n = 500
            ctrl = rng.normal(10, 2, n)
            treat = rng.normal(11, 2, n)  # positive
            dfs.append(pd.DataFrame({
                "group": ["control"] * n + ["treatment"] * n,
                "value": np.concatenate([ctrl, treat]),
                "segment": [seg] * (2 * n),
            }))
        # Segment C: negative effect (disagrees)
        n = 50
        ctrl = rng.normal(10, 2, n)
        treat = rng.normal(9, 2, n)  # negative
        dfs.append(pd.DataFrame({
            "group": ["control"] * n + ["treatment"] * n,
            "value": np.concatenate([ctrl, treat]),
            "segment": ["C"] * (2 * n),
        }))
        df = pd.concat(dfs, ignore_index=True)
        result = segment_analysis(df)
        # Aggregate should be positive (dominated by A, B)
        # Only 1/3 segments disagrees — should NOT be flagged
        assert result.simpsons_paradox is False

    def test_tie_segments_not_flagged(self):
        """When exactly half the segments disagree (tie), Simpson's should NOT trigger."""
        rng = np.random.default_rng(99)
        dfs = []
        # Segment A: strong positive effect (large n, dominates aggregate)
        n = 1000
        ctrl = rng.normal(10, 2, n)
        treat = rng.normal(11, 2, n)
        dfs.append(pd.DataFrame({
            "group": ["control"] * n + ["treatment"] * n,
            "value": np.concatenate([ctrl, treat]),
            "segment": ["A"] * (2 * n),
        }))
        # Segment B: small negative effect (small n)
        n = 50
        ctrl = rng.normal(10, 1, n)
        treat = rng.normal(9, 1, n)
        dfs.append(pd.DataFrame({
            "group": ["control"] * n + ["treatment"] * n,
            "value": np.concatenate([ctrl, treat]),
            "segment": ["B"] * (2 * n),
        }))
        df = pd.concat(dfs, ignore_index=True)
        result = segment_analysis(df)
        # 1/2 disagrees — tie should NOT be flagged as Simpson's
        assert result.simpsons_paradox is False

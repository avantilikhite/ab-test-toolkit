"""Unit tests for round-3 P1 review fixes.

Covers:
- CUPED zero-variance fast path (DS-3)
- io.py covariate finite check (DS-4)
- io.py day numeric coercion (DS-5)
- Stratified SRM Holm correction & per-stratum expected_ratio (DS-2, EL-8)
- Manifest schema: full SHA-256 + drift detection (EL-1, EL-2, EL-3)
- Simpson SwM consistency uses _has_significant_conflicting_segment (EL-4)
- Power Loss → Sample Inflation chart relabeling (DS-6)
"""
import numpy as np
import pandas as pd
import pytest

from ab_test_toolkit.cuped import cuped_adjust
from ab_test_toolkit.io import load_experiment_data
from ab_test_toolkit.srm import check_srm_by_stratum
from ab_test_toolkit.recommendation import generate_recommendation
from ab_test_toolkit.visualization import power_loss_curve

# Reuse the recommendation-test helpers
from tests.unit.test_recommendation import _make_freq, _make_bayes, _make_srm, _make_segment


# ─────────────────────────────────────────────────────────────────────────────
# CUPED zero-variance
# ─────────────────────────────────────────────────────────────────────────────
class TestCUPEDZeroVariance:
    def test_zero_outcome_variance_returns_finite_ci(self):
        """When both arms are constant, CI should collapse to the point estimate, not NaN."""
        rng = np.random.default_rng(0)
        control = np.zeros(200)
        treatment = np.ones(200) * 0.05
        cov_c = rng.normal(size=200)
        cov_t = rng.normal(size=200)
        result = cuped_adjust(control, treatment, cov_c, cov_t)
        assert np.isfinite(result.adjusted_ci[0])
        assert np.isfinite(result.adjusted_ci[1])
        assert np.isfinite(result.adjusted_estimate)


# ─────────────────────────────────────────────────────────────────────────────
# io.py validation
# ─────────────────────────────────────────────────────────────────────────────
class TestIOValidation:
    def test_covariate_inf_rejected(self):
        df = pd.DataFrame({
            "group": ["control", "treatment"] * 5,
            "value": [0, 1] * 5,
            "covariate": [1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })
        with pytest.raises(ValueError, match=r"finite|inf"):
            load_experiment_data(df)

    def test_day_strings_coerced(self):
        df_ok = pd.DataFrame({
            "group": ["control", "treatment"] * 3,
            "value": [0, 1] * 3,
            "day": ["1", "2", "10", "1", "2", "10"],
        })
        out, _ = load_experiment_data(df_ok)
        assert pd.api.types.is_numeric_dtype(out["day"])
        # Sort works numerically: 1,2,10 — not lexicographic 1,10,2
        assert sorted(out["day"].unique().tolist()) == [1, 2, 10]

    def test_day_non_numeric_rejected(self):
        df_bad = pd.DataFrame({
            "group": ["control", "treatment"] * 3,
            "value": [0, 1] * 3,
            "day": ["mon", "tue", "wed", "mon", "tue", "wed"],
        })
        with pytest.raises(ValueError, match=r"day"):
            load_experiment_data(df_bad)


# ─────────────────────────────────────────────────────────────────────────────
# Stratified SRM Holm correction + per-stratum ratio
# ─────────────────────────────────────────────────────────────────────────────
class TestStratifiedSRM:
    def test_holm_dampens_false_positives_across_strata(self):
        """20 perfectly-balanced strata should not produce spurious mismatches under Holm."""
        rng = np.random.default_rng(42)
        rows = []
        for d in range(20):
            for _ in range(500):
                rows.append({"group": rng.choice(["control", "treatment"]), "day": d})
        df = pd.DataFrame(rows)
        out = check_srm_by_stratum(df, threshold=0.01)
        assert out.n_mismatches == 0
        assert out.n_strata == 20
        for r in out.stratum_results:
            assert r["p_value_adjusted"] >= r["p_value"] - 1e-12

    def test_per_stratum_expected_ratio_dict(self):
        """A ramp design (10/90 → 50/50) should not flag SRM when the planned ratio matches."""
        rng = np.random.default_rng(1)
        rows = []
        for d, ratio in [(1, (0.9, 0.1)), (2, (0.5, 0.5))]:
            for _ in range(2000):
                grp = rng.choice(["control", "treatment"], p=ratio)
                rows.append({"group": grp, "day": d})
        df = pd.DataFrame(rows)
        out = check_srm_by_stratum(
            df,
            expected_ratio=(0.5, 0.5),
            expected_ratio_by_stratum={1: (0.9, 0.1), 2: (0.5, 0.5)},
            threshold=0.01,
        )
        assert out.n_mismatches == 0


# ─────────────────────────────────────────────────────────────────────────────
# Manifest: full SHA-256 + drift
# ─────────────────────────────────────────────────────────────────────────────
class TestManifestDrift:
    def test_full_sha256_stored(self):
        rec = generate_recommendation(
            frequentist=_make_freq(),
            bayesian=_make_bayes(),
            srm=_make_srm(),
            manifest={"experiment_id": "x", "alpha": 0.05},
        )
        assert rec.manifest_hash_full is not None
        assert len(rec.manifest_hash_full) == 64
        assert rec.manifest_hash == rec.manifest_hash_full[:16]

    def test_alpha_drift_flagged(self):
        rec = generate_recommendation(
            frequentist=_make_freq(),  # alpha=0.05
            bayesian=_make_bayes(),
            srm=_make_srm(),
            manifest={"experiment_id": "x", "alpha": 0.01},
        )
        assert any("alpha" in d for d in rec.manifest_drift)
        assert any("Manifest drift" in f for f in rec.flags)

    def test_matching_policy_no_drift(self):
        rec = generate_recommendation(
            frequentist=_make_freq(),
            bayesian=_make_bayes(),
            srm=_make_srm(),
            manifest={
                "experiment_id": "x",
                "alpha": 0.05,
                "loss_tolerance": None,
                "allow_ship_with_monitoring": False,
                "monitoring_prob_threshold": 0.85,
                "twyman_min_baseline": 0.01,
                "lift_warning_threshold": 0.50,
            },
        )
        assert rec.manifest_drift == []
        assert not any("Manifest drift" in f for f in rec.flags)

    def test_no_manifest_no_drift_field(self):
        rec = generate_recommendation(
            frequentist=_make_freq(), bayesian=_make_bayes(), srm=_make_srm(),
        )
        assert rec.manifest_hash is None
        assert rec.manifest_hash_full is None
        assert rec.manifest_drift == []


# ─────────────────────────────────────────────────────────────────────────────
# Simpson SwM consistency
# ─────────────────────────────────────────────────────────────────────────────
class TestSimpsonSwMConsistency:
    def test_significant_simpson_blocks_swm(self):
        """SwM should NOT trigger when a significant conflicting segment exists."""
        rec = generate_recommendation(
            frequentist=_make_freq(is_significant=False, p_value=0.12, point_estimate=0.01),
            bayesian=_make_bayes(prob_b_gt_a=0.88, expected_loss=0.0001),
            srm=_make_srm(),
            segmentation=_make_segment(simpsons_paradox=True),
            allow_ship_with_monitoring=True,
            monitoring_prob_threshold=0.85,
        )
        assert rec.recommendation != "Ship with Monitoring"


# ─────────────────────────────────────────────────────────────────────────────
# Power Loss → Sample Inflation relabeling
# ─────────────────────────────────────────────────────────────────────────────
class TestSampleInflationRelabel:
    def test_chart_uses_sample_inflation_label(self):
        fig = power_loss_curve(baseline_rate=0.10, mde=0.02)
        title = (fig.layout.title.text or "").lower()
        assert "sample" in title and "inflation" in title
        yaxis_title = (fig.layout.yaxis.title.text or "").lower()
        assert "vs balanced" in yaxis_title or "extra sample" in yaxis_title
        assert any("Sample Inflation" in (tr.name or "") for tr in fig.data)





# ─────────────────────────────────────────────────────────────────────────────
# Planned-N drift detection (final review fix)
# ─────────────────────────────────────────────────────────────────────────────
class TestPlannedNDrift:
    def test_planned_n_shortfall_flagged(self):
        """Drift fires when as-run total < 50% of registered planned_n_total."""
        freq = _make_freq()
        freq.n_control, freq.n_treatment = 200, 200
        rec = generate_recommendation(
            frequentist=freq,
            bayesian=_make_bayes(),
            srm=_make_srm(),
            manifest={"experiment_id": "x", "alpha": 0.05, "planned_n_total": 2000},
        )
        assert any("planned_n_total" in d for d in rec.manifest_drift), rec.manifest_drift
        assert any("under-powered" in d for d in rec.manifest_drift)

    def test_planned_n_close_to_plan_no_drift(self):
        """No drift when as-run is within the threshold."""
        freq = _make_freq()
        freq.n_control, freq.n_treatment = 1000, 1000
        rec = generate_recommendation(
            frequentist=freq,
            bayesian=_make_bayes(),
            srm=_make_srm(),
            manifest={"experiment_id": "x", "alpha": 0.05, "planned_n_total": 2000},
        )
        assert not any("planned_n_total" in d for d in rec.manifest_drift)


class TestFrequentistResultPopulatesN:
    """Ensure n_control / n_treatment land on FrequentistResult so downstream
    manifest drift detection (planned-N) can actually fire."""

    def test_z_test_populates_n(self):
        from ab_test_toolkit.frequentist import z_test
        control = np.array([0] * 90 + [1] * 10)
        treatment = np.array([0] * 80 + [1] * 20)
        result = z_test(control, treatment)
        assert result.n_control == 100
        assert result.n_treatment == 100

    def test_welch_t_test_populates_n(self):
        from ab_test_toolkit.frequentist import welch_t_test
        rng = np.random.default_rng(0)
        control = rng.normal(0, 1, 150)
        treatment = rng.normal(0.1, 1, 175)
        result = welch_t_test(control, treatment)
        assert result.n_control == 150
        assert result.n_treatment == 175

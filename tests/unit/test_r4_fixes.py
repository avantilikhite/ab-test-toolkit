"""Unit tests for round-4 review fixes.

Covers:
- Survival-function p-values (no underflow to exactly 0 at moderate z)
- Planned-N gate (interim analyses cannot certify Ship)
- higher_is_better metric-direction support
- Configurable Bayesian credible_level
- user_id duplicate guard in io.load_experiment_data
- Traffic-mix (group × segment) imbalance diagnostic
- Scale-aware sign-reversal noise floor in segmentation
"""
import numpy as np
import pandas as pd
import pytest

from ab_test_toolkit.bayesian import beta_binomial_from_stats, normal_normal_from_stats
from ab_test_toolkit.frequentist import welch_t_test_from_stats, z_test_from_stats
from ab_test_toolkit.io import load_experiment_data
from ab_test_toolkit.recommendation import generate_recommendation
from ab_test_toolkit.segmentation import segment_analysis
from tests.unit.test_recommendation import _make_bayes, _make_freq, _make_srm


# ─────────────────────────────────────────────────────────────────────────────
# Survival-function p-values
# ─────────────────────────────────────────────────────────────────────────────
class TestPValueNoUnderflow:
    def test_z_test_extreme_effect_p_positive(self):
        """A huge but finite z should give a tiny positive p, not exactly 0."""
        result = z_test_from_stats(1000, 100_000, 2000, 100_000)
        assert result.statistic > 8
        assert 0 < result.p_value < 1e-10

    def test_welch_extreme_effect_p_positive(self):
        # t ≈ 10: the old 2*(1 - cdf) path returns exactly 0.0 here
        # (cdf rounds to 1.0 beyond |t| ≈ 8); sf keeps precision.
        result = welch_t_test_from_stats(10.0, 1.0, 20_000, 10.1, 1.0, 20_000)
        assert result.statistic > 8
        assert 0 < result.p_value < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Planned-N gate
# ─────────────────────────────────────────────────────────────────────────────
class TestPlannedNGate:
    def test_interim_shortfall_withholds_ship(self):
        freq = _make_freq()  # significant, positive → would be Ship
        freq.n_control, freq.n_treatment = 800, 800
        rec = generate_recommendation(
            frequentist=freq,
            bayesian=_make_bayes(),
            srm=_make_srm(),
            manifest={"experiment_id": "x", "alpha": 0.05, "planned_n_total": 2000},
        )
        assert rec.recommendation == "Inconclusive"
        assert "interim" in rec.reason.lower()
        assert any("Interim analysis" in f for f in rec.flags)
        assert any("Continue the experiment" in s for s in rec.next_steps)

    def test_interim_does_not_withhold_no_ship(self):
        """Harm signals on interim data are safety information — not withheld."""
        freq = _make_freq(point_estimate=-0.02)
        freq.ci_lower, freq.ci_upper = -0.04, -0.001
        freq.n_control, freq.n_treatment = 800, 800
        rec = generate_recommendation(
            frequentist=freq,
            bayesian=_make_bayes(prob_b_gt_a=0.02),
            srm=_make_srm(),
            manifest={"experiment_id": "x", "alpha": 0.05, "planned_n_total": 2000},
        )
        assert rec.recommendation == "No-Ship"
        assert any("Interim analysis" in f for f in rec.flags)

    def test_planned_n_reached_no_gate(self):
        freq = _make_freq()
        freq.n_control, freq.n_treatment = 1000, 1000
        rec = generate_recommendation(
            frequentist=freq,
            bayesian=_make_bayes(),
            srm=_make_srm(),
            manifest={"experiment_id": "x", "alpha": 0.05, "planned_n_total": 2000},
        )
        assert rec.recommendation == "Ship"
        assert not any("Interim analysis" in f for f in rec.flags)

    def test_no_manifest_no_gate(self):
        freq = _make_freq()
        freq.n_control, freq.n_treatment = 10, 10
        rec = generate_recommendation(
            frequentist=freq, bayesian=_make_bayes(), srm=_make_srm(),
        )
        assert rec.recommendation == "Ship"


# ─────────────────────────────────────────────────────────────────────────────
# higher_is_better
# ─────────────────────────────────────────────────────────────────────────────
class TestMetricDirection:
    def test_lower_is_better_win_ships(self):
        """A significant metric *decrease* should Ship when lower is better."""
        freq = _make_freq(point_estimate=-0.02, effect_size=-0.1)
        freq.ci_lower, freq.ci_upper = -0.04, -0.001
        rec = generate_recommendation(
            frequentist=freq,
            bayesian=_make_bayes(prob_b_gt_a=0.03),  # raw: treatment likely lower
            srm=_make_srm(),
            higher_is_better=False,
        )
        assert rec.recommendation == "Ship"
        assert any("lower-is-better" in f for f in rec.flags)
        assert rec.supporting_metrics["policy"]["higher_is_better"] is False
        # Reported in improvement space: positive effect, flipped probability
        assert rec.supporting_metrics["prob_b_gt_a"] == pytest.approx(0.97)

    def test_lower_is_better_harm_no_ships(self):
        """A significant metric *increase* is harm when lower is better."""
        freq = _make_freq(point_estimate=0.02, effect_size=0.1)
        rec = generate_recommendation(
            frequentist=freq,
            bayesian=_make_bayes(prob_b_gt_a=0.97),
            srm=_make_srm(),
            higher_is_better=False,
        )
        assert rec.recommendation == "No-Ship"

    def test_default_direction_unchanged(self):
        rec = generate_recommendation(
            frequentist=_make_freq(), bayesian=_make_bayes(), srm=_make_srm(),
        )
        assert rec.recommendation == "Ship"
        assert not any("lower-is-better" in f for f in rec.flags)


# ─────────────────────────────────────────────────────────────────────────────
# Configurable credible level
# ─────────────────────────────────────────────────────────────────────────────
class TestCredibleLevel:
    def test_narrower_level_narrower_interval(self):
        wide = beta_binomial_from_stats(500, 5000, 550, 5000, credible_level=0.95)
        narrow = beta_binomial_from_stats(500, 5000, 550, 5000, credible_level=0.50)
        w_width = wide.credible_interval[1] - wide.credible_interval[0]
        n_width = narrow.credible_interval[1] - narrow.credible_interval[0]
        assert n_width < w_width
        assert narrow.prior_config["credible_level"] == 0.50

    def test_normal_normal_credible_level(self):
        wide = normal_normal_from_stats(10, 2, 500, 10.5, 2, 500, credible_level=0.99)
        base = normal_normal_from_stats(10, 2, 500, 10.5, 2, 500, credible_level=0.95)
        assert (wide.credible_interval[1] - wide.credible_interval[0]) > (
            base.credible_interval[1] - base.credible_interval[0]
        )

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="credible_level"):
            beta_binomial_from_stats(50, 100, 60, 100, credible_level=1.5)


# ─────────────────────────────────────────────────────────────────────────────
# user_id duplicate guard
# ─────────────────────────────────────────────────────────────────────────────
class TestUserIdGuard:
    def test_duplicate_user_ids_rejected(self):
        df = pd.DataFrame({
            "group": ["control", "control", "treatment", "treatment"],
            "value": [0, 1, 1, 0],
            "user_id": ["u1", "u1", "u2", "u3"],
        })
        with pytest.raises(ValueError, match="duplicate"):
            load_experiment_data(df)

    def test_unique_user_ids_accepted(self):
        df = pd.DataFrame({
            "group": ["control", "control", "treatment", "treatment"],
            "value": [0, 1, 1, 0],
            "user_id": ["u1", "u2", "u3", "u4"],
        })
        out, _ = load_experiment_data(df)
        assert len(out) == 4

    def test_no_user_id_column_still_accepted(self):
        df = pd.DataFrame({
            "group": ["control", "treatment"] * 3,
            "value": [0, 1] * 3,
        })
        out, _ = load_experiment_data(df)
        assert len(out) == 6


# ─────────────────────────────────────────────────────────────────────────────
# Traffic-mix diagnostic + scale-aware reversal floor
# ─────────────────────────────────────────────────────────────────────────────
class TestTrafficMix:
    def test_mix_imbalance_detected(self):
        rng = np.random.default_rng(7)
        rows = []
        # Control: 800 seg A / 200 seg B.  Treatment: 200 seg A / 800 seg B.
        for grp, n_a, n_b in [("control", 800, 200), ("treatment", 200, 800)]:
            for seg, n in [("A", n_a), ("B", n_b)]:
                for _ in range(n):
                    rows.append({"group": grp, "value": float(rng.normal(10, 2)), "segment": seg})
        result = segment_analysis(pd.DataFrame(rows))
        assert result.mix_imbalance is True
        assert result.mix_p_value is not None and result.mix_p_value < 0.001
        assert result.mix_details is not None and "mix differs" in result.mix_details.lower()

    def test_balanced_mix_not_flagged(self):
        rng = np.random.default_rng(8)
        rows = []
        for grp in ["control", "treatment"]:
            for seg in ["A", "B"]:
                for _ in range(500):
                    rows.append({"group": grp, "value": float(rng.normal(10, 2)), "segment": seg})
        result = segment_analysis(pd.DataFrame(rows))
        assert result.mix_imbalance is False


class TestScaleAwareReversalFloor:
    def test_tiny_reversal_relative_to_aggregate_not_flagged(self):
        """A reversal < 1% of the aggregate magnitude is noise, regardless of share."""
        rng = np.random.default_rng(9)
        dfs = []
        # Segment A: huge positive effect, dominates the aggregate.
        n = 600
        dfs.append(pd.DataFrame({
            "group": ["control"] * n + ["treatment"] * n,
            "value": np.concatenate([rng.normal(10, 1, n), rng.normal(20, 1, n)]),
            "segment": ["A"] * (2 * n),
        }))
        # Segment B: 40% of the sample, trivially small reversal (~0.3% of aggregate).
        n = 400
        dfs.append(pd.DataFrame({
            "group": ["control"] * n + ["treatment"] * n,
            "value": np.concatenate([rng.normal(10, 0.1, n), rng.normal(9.98, 0.1, n)]),
            "segment": ["B"] * (2 * n),
        }))
        result = segment_analysis(pd.concat(dfs, ignore_index=True))
        # Under the old fixed 0.001 floor this would have flagged; the
        # scale-aware floor treats it as noise.
        assert result.simpsons_paradox is False

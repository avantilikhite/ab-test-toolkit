"""Unit tests for ab_test_toolkit.data_generator — generate_experiment_data."""

import numpy as np
import pandas as pd
import pytest

from ab_test_toolkit.data_generator import generate_experiment_data


class TestGenerateExperimentDataBasic:
    """Tests for basic data generation."""

    def test_output_schema_basic(self):
        """Output DataFrame has required columns."""
        df = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02, n_control=500, n_treatment=500,
        )
        assert "group" in df.columns
        assert "value" in df.columns
        assert "covariate" in df.columns  # Always included for CUPED

    def test_group_labels(self):
        """Output contains exactly control and treatment groups."""
        df = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02, n_control=500, n_treatment=500,
        )
        assert set(df["group"].unique()) == {"control", "treatment"}

    def test_sample_sizes(self):
        """Output has correct number of rows per group."""
        df = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02, n_control=300, n_treatment=700,
        )
        assert len(df[df["group"] == "control"]) == 300
        assert len(df[df["group"] == "treatment"]) == 700

    def test_baseline_rate_accuracy(self):
        """Control group mean is close to baseline_rate for large N."""
        df = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02, n_control=10000, n_treatment=10000,
            random_seed=42,
        )
        control_mean = df[df["group"] == "control"]["value"].mean()
        assert abs(control_mean - 0.10) < 0.02  # Within 2% tolerance for large N

    def test_treatment_rate_accuracy(self):
        """Treatment group mean reflects baseline + effect for large N."""
        df = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.03, n_control=10000, n_treatment=10000,
            random_seed=42,
        )
        treatment_mean = df[df["group"] == "treatment"]["value"].mean()
        assert abs(treatment_mean - 0.13) < 0.02

    def test_binary_values(self):
        """Values are binary 0/1 for proportion data."""
        df = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02, n_control=1000, n_treatment=1000,
        )
        assert set(df["value"].unique()).issubset({0, 1})


class TestGenerateExperimentDataSeed:
    """Tests for seed reproducibility."""

    def test_seed_reproducibility(self):
        """Same seed produces identical output."""
        df1 = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02, n_control=500, n_treatment=500,
            random_seed=42,
        )
        df2 = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02, n_control=500, n_treatment=500,
            random_seed=42,
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_produces_different_output(self):
        """Different seeds produce different output."""
        df1 = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02, n_control=500, n_treatment=500,
            random_seed=42,
        )
        df2 = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02, n_control=500, n_treatment=500,
            random_seed=99,
        )
        assert not df1["value"].equals(df2["value"])


class TestGenerateExperimentDataNovelty:
    """Tests for novelty effect injection."""

    def test_novelty_adds_day_column(self):
        """inject_novelty=True adds 'day' column."""
        df = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02, n_control=1000, n_treatment=1000,
            inject_novelty=True, novelty_days=3,
        )
        assert "day" in df.columns

    def test_novelty_inflates_early_days(self):
        """Treatment effect in early days is inflated compared to later days."""
        df = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02, n_control=5000, n_treatment=5000,
            inject_novelty=True, novelty_days=3, novelty_multiplier=3.0,
            random_seed=42,
        )
        treat = df[df["group"] == "treatment"]
        early = treat[treat["day"] <= 3]["value"].mean()
        late = treat[treat["day"] > 3]["value"].mean()
        # Early should be higher due to inflated effect
        assert early > late


class TestGenerateExperimentDataSimpsons:
    """Tests for Simpson's Paradox injection."""

    def test_simpsons_adds_segment_column(self):
        """inject_simpsons=True adds 'segment' column."""
        df = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02, n_control=1000, n_treatment=1000,
            inject_simpsons=True,
        )
        assert "segment" in df.columns

    def test_simpsons_creates_sign_flip(self):
        """Simpson's injection produces a sign flip between aggregate and at least one segment."""
        df = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02, n_control=2000, n_treatment=2000,
            inject_simpsons=True, random_seed=42,
        )
        # Aggregate effect
        ctrl_mean = df[df["group"] == "control"]["value"].mean()
        treat_mean = df[df["group"] == "treatment"]["value"].mean()
        agg_effect = treat_mean - ctrl_mean

        # Per-segment effects
        sign_flip = False
        for seg in df["segment"].unique():
            seg_data = df[df["segment"] == seg]
            seg_ctrl = seg_data[seg_data["group"] == "control"]["value"].mean()
            seg_treat = seg_data[seg_data["group"] == "treatment"]["value"].mean()
            seg_effect = seg_treat - seg_ctrl
            if (agg_effect > 0 and seg_effect < 0) or (agg_effect < 0 and seg_effect > 0):
                sign_flip = True
                break
        assert sign_flip, "Simpson's Paradox injection should create a sign flip"

    def test_demo_defaults_trip_simpsons_detector(self):
        """The shipped case-study seed must actually trip the Simpson detector.

        Regression guard: a previous strict-majority rule meant the canonical
        demo (seed=42) silently failed to fire — undermining the toolkit's
        primary teaching example.
        """
        from ab_test_toolkit.segmentation import segment_analysis

        df = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02,
            n_control=5000, n_treatment=5000,
            inject_simpsons=True, random_seed=42,
        )
        seg = segment_analysis(df)
        assert seg.simpsons_paradox, (
            "Demo seed must trip Simpson's detector — otherwise the case study "
            "tells a story the engine cannot demonstrate."
        )

    def test_demo_defaults_trip_novelty_detector(self):
        """The shipped case-study novelty parameters must trip check_novelty."""
        from ab_test_toolkit.recommendation import check_novelty

        df = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02,
            n_control=5000, n_treatment=5000,
            inject_novelty=True, novelty_multiplier=3.0,
            random_seed=42,
        )
        nov = check_novelty(df)
        assert nov.has_novelty, (
            "Demo with novelty_multiplier=3.0 must trip the novelty detector."
        )


class TestGenerateExperimentDataSRM:
    """Tests for SRM injection."""

    def test_srm_skews_allocation(self):
        """inject_srm=True skews group allocation."""
        df = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02, n_control=1000, n_treatment=1000,
            inject_srm=True, srm_actual_ratio=0.60,
        )
        n_ctrl = len(df[df["group"] == "control"])
        n_treat = len(df[df["group"] == "treatment"])
        actual_ratio = n_treat / (n_ctrl + n_treat)
        # Should be close to 0.60, not 0.50
        assert abs(actual_ratio - 0.60) < 0.05


class TestGenerateExperimentDataComposition:
    """Tests for composable flag behavior."""

    def test_all_flags_together(self):
        """All injection flags can be used simultaneously without error."""
        df = generate_experiment_data(
            baseline_rate=0.10, effect_size=0.02, n_control=1000, n_treatment=1000,
            inject_novelty=True, inject_simpsons=True, inject_srm=True,
            random_seed=42,
        )
        assert "day" in df.columns
        assert "segment" in df.columns
        assert "covariate" in df.columns
        assert len(df) > 0


class TestEdgeCaseRates:
    """Tests for extreme baseline + effect combinations."""

    def test_treatment_rate_clamped_above_one(self):
        """baseline_rate + effect_size > 1 should not crash."""
        df = generate_experiment_data(
            baseline_rate=0.8, effect_size=0.3,
            n_control=100, n_treatment=100,
        )
        assert len(df) == 200
        assert df["value"].isin([0, 1]).all()

    def test_novelty_treatment_rate_clamped(self):
        """Novelty path also clamps normal treatment rate."""
        df = generate_experiment_data(
            baseline_rate=0.8, effect_size=0.3,
            n_control=100, n_treatment=100,
            inject_novelty=True,
        )
        assert len(df) == 200
        assert "day" in df.columns

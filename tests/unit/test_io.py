"""Unit tests for ab_test_toolkit.io — load_experiment_data."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ab_test_toolkit import MetricType
from ab_test_toolkit.io import load_experiment_data


class TestLoadExperimentDataFromDataFrame:
    """Tests for DataFrame pass-through path."""

    def test_valid_proportion_df(self, proportion_df):
        """Valid binary DataFrame returns PROPORTION metric type."""
        df, metric = load_experiment_data(proportion_df)
        assert metric == MetricType.PROPORTION
        assert set(df["group"].unique()) == {"control", "treatment"}

    def test_valid_continuous_df(self, continuous_df):
        """Valid continuous DataFrame returns CONTINUOUS metric type."""
        df, metric = load_experiment_data(continuous_df)
        assert metric == MetricType.CONTINUOUS

    def test_case_insensitive_groups(self, case_insensitive_df):
        """Group labels are normalized to lowercase."""
        df, metric = load_experiment_data(case_insensitive_df)
        assert set(df["group"].unique()) == {"control", "treatment"}

    def test_segment_nan_filled(self):
        """NaN in segment column becomes 'unknown'."""
        df_input = pd.DataFrame({
            "group": ["control", "treatment", "control", "treatment"],
            "value": [1, 0, 1, 0],
            "segment": ["mobile", np.nan, "desktop", np.nan],
        })
        df, _ = load_experiment_data(df_input)
        assert "unknown" in df["segment"].values
        assert df["segment"].isna().sum() == 0

    def test_optional_covariate_present(self, covariate_df):
        """Covariate column passes through when present."""
        df, _ = load_experiment_data(covariate_df)
        assert "covariate" in df.columns


class TestLoadExperimentDataValidation:
    """Tests for validation and error messages."""

    def test_missing_group_column(self, missing_group_col_df):
        """Missing group column raises ValueError."""
        with pytest.raises(ValueError, match="(?i)group"):
            load_experiment_data(missing_group_col_df)

    def test_missing_value_column(self, missing_value_col_df):
        """Missing value column raises ValueError."""
        with pytest.raises(ValueError, match="(?i)value"):
            load_experiment_data(missing_value_col_df)

    def test_wrong_group_labels(self, wrong_group_labels_df):
        """Non-control/treatment labels raise ValueError."""
        with pytest.raises(ValueError, match="(?i)control.*treatment|treatment.*control"):
            load_experiment_data(wrong_group_labels_df)

    def test_nan_in_value_column(self, nan_values_df):
        """NaN in value column raises ValueError."""
        with pytest.raises(ValueError, match="(?i)nan|missing|null"):
            load_experiment_data(nan_values_df)

    def test_nan_in_covariate_column(self):
        """NaN in covariate column raises ValueError."""
        df_input = pd.DataFrame({
            "group": ["control", "treatment"],
            "value": [1.0, 0.0],
            "covariate": [1.0, np.nan],
        })
        with pytest.raises(ValueError, match="(?i)covariate.*nan|nan.*covariate|missing"):
            load_experiment_data(df_input)

    def test_empty_dataframe(self, empty_df):
        """Empty DataFrame raises ValueError."""
        with pytest.raises(ValueError):
            load_experiment_data(empty_df)

    def test_non_numeric_values_rejected(self):
        """Non-numeric strings in value column raise ValueError."""
        df_input = pd.DataFrame({
            "group": ["control", "treatment", "control", "treatment"],
            "value": ["foo", "bar", "1.0", "2.0"],
        })
        with pytest.raises(ValueError, match="(?i)non-numeric"):
            load_experiment_data(df_input)

    def test_non_numeric_covariate_rejected(self):
        """Non-numeric strings in covariate column raise ValueError."""
        df_input = pd.DataFrame({
            "group": ["control", "treatment"],
            "value": [1.0, 2.0],
            "covariate": ["abc", "def"],
        })
        with pytest.raises(ValueError, match="(?i)non-numeric"):
            load_experiment_data(df_input)


class TestLoadExperimentDataFromCSV:
    """Tests for CSV file loading path."""

    def test_load_from_csv_path(self, proportion_df, tmp_path):
        """Load from CSV file path works."""
        csv_path = tmp_path / "test.csv"
        proportion_df.to_csv(csv_path, index=False)
        df, metric = load_experiment_data(str(csv_path))
        assert metric == MetricType.PROPORTION
        assert len(df) == len(proportion_df)

    def test_load_from_pathlib_path(self, proportion_df, tmp_path):
        """Load from pathlib.Path works."""
        csv_path = tmp_path / "test.csv"
        proportion_df.to_csv(csv_path, index=False)
        df, metric = load_experiment_data(csv_path)
        assert metric == MetricType.PROPORTION


class TestMetricTypeAutoDetection:
    """Tests for automatic metric type detection."""

    def test_all_binary_is_proportion(self):
        """All 0/1 values detected as PROPORTION."""
        df = pd.DataFrame({
            "group": ["control"] * 50 + ["treatment"] * 50,
            "value": [0, 1] * 50,
        })
        _, metric = load_experiment_data(df)
        assert metric == MetricType.PROPORTION

    def test_continuous_values_is_continuous(self):
        """Non-binary values detected as CONTINUOUS."""
        df = pd.DataFrame({
            "group": ["control"] * 50 + ["treatment"] * 50,
            "value": np.random.default_rng(42).normal(10, 2, 100),
        })
        _, metric = load_experiment_data(df)
        assert metric == MetricType.CONTINUOUS

    def test_integer_binary_still_proportion(self):
        """Integer 0 and 1 values detected as PROPORTION."""
        df = pd.DataFrame({
            "group": ["control"] * 50 + ["treatment"] * 50,
            "value": [0] * 25 + [1] * 25 + [0] * 25 + [1] * 25,
        })
        _, metric = load_experiment_data(df)
        assert metric == MetricType.PROPORTION

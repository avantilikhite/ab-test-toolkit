"""Shared test fixtures for ab_test_toolkit tests."""

import numpy as np
import pandas as pd
import pytest

from ab_test_toolkit import MetricType


# ---------------------------------------------------------------------------
# Proportion (binary 0/1) DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def proportion_df():
    """Balanced proportion experiment data with known effect."""
    rng = np.random.default_rng(42)
    n = 1000
    control_vals = rng.binomial(1, 0.10, n)
    treatment_vals = rng.binomial(1, 0.12, n)
    return pd.DataFrame({
        "group": ["control"] * n + ["treatment"] * n,
        "value": np.concatenate([control_vals, treatment_vals]),
    })


@pytest.fixture
def proportion_df_large():
    """Large proportion dataset for power testing."""
    rng = np.random.default_rng(99)
    n = 5000
    control_vals = rng.binomial(1, 0.10, n)
    treatment_vals = rng.binomial(1, 0.13, n)
    return pd.DataFrame({
        "group": ["control"] * n + ["treatment"] * n,
        "value": np.concatenate([control_vals, treatment_vals]),
    })


# ---------------------------------------------------------------------------
# Continuous DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def continuous_df():
    """Balanced continuous experiment data with known effect."""
    rng = np.random.default_rng(42)
    n = 500
    control_vals = rng.normal(10.0, 2.0, n)
    treatment_vals = rng.normal(10.5, 2.1, n)
    return pd.DataFrame({
        "group": ["control"] * n + ["treatment"] * n,
        "value": np.concatenate([control_vals, treatment_vals]),
    })


# ---------------------------------------------------------------------------
# Segmented DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def segmented_df():
    """Experiment data with segment column for HTE analysis."""
    rng = np.random.default_rng(42)
    n_per_seg = 500
    segments = []
    groups = []
    values = []
    for seg_name, ctrl_rate, treat_rate in [("mobile", 0.08, 0.11), ("desktop", 0.12, 0.14)]:
        for group_label, rate in [("control", ctrl_rate), ("treatment", treat_rate)]:
            vals = rng.binomial(1, rate, n_per_seg)
            segments.extend([seg_name] * n_per_seg)
            groups.extend([group_label] * n_per_seg)
            values.extend(vals.tolist())
    return pd.DataFrame({"group": groups, "value": values, "segment": segments})


@pytest.fixture
def simpsons_paradox_df():
    """Data engineered to exhibit Simpson's Paradox.

    Aggregate: treatment appears better (driven by imbalanced segment D).
    Segments A, B, C: treatment is worse in each.
    Segment D (heavily over-allocated to treatment): pulls aggregate positive.
    Strict majority (3/4) of segments disagree with aggregate.
    """
    rng = np.random.default_rng(123)
    # Segments A, B, C — treatment worse (clear negative effect)
    dfs = []
    for seg, n in [("A", 500), ("B", 500), ("C", 500)]:
        ctrl = rng.binomial(1, 0.15, n)
        treat = rng.binomial(1, 0.08, n)
        dfs.append(pd.DataFrame({
            "group": ["control"] * n + ["treatment"] * n,
            "value": np.concatenate([ctrl, treat]).tolist(),
            "segment": [seg] * (2 * n),
        }))
    # Segment D — small control, huge treatment, much higher rate
    n_d_ctrl = 100
    n_d_treat = 3000
    ctrl_d = rng.binomial(1, 0.05, n_d_ctrl)
    treat_d = rng.binomial(1, 0.30, n_d_treat)
    dfs.append(pd.DataFrame({
        "group": ["control"] * n_d_ctrl + ["treatment"] * n_d_treat,
        "value": np.concatenate([ctrl_d, treat_d]).tolist(),
        "segment": ["D"] * (n_d_ctrl + n_d_treat),
    }))
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Covariate DataFrames (for CUPED)
# ---------------------------------------------------------------------------

@pytest.fixture
def covariate_df():
    """Experiment data with correlated pre-experiment covariate for CUPED."""
    rng = np.random.default_rng(42)
    n = 1000
    # Covariate correlated with outcome (ρ ≈ 0.7)
    cov_control = rng.normal(5.0, 1.0, n)
    cov_treatment = rng.normal(5.0, 1.0, n)
    noise_control = rng.normal(0, 0.7, n)
    noise_treatment = rng.normal(0, 0.7, n)
    val_control = 0.8 * cov_control + noise_control
    val_treatment = 0.8 * cov_treatment + noise_treatment + 0.5  # True effect = 0.5
    return pd.DataFrame({
        "group": ["control"] * n + ["treatment"] * n,
        "value": np.concatenate([val_control, val_treatment]),
        "covariate": np.concatenate([cov_control, cov_treatment]),
    })


# ---------------------------------------------------------------------------
# Edge-case DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_df():
    """Empty DataFrame with correct columns."""
    return pd.DataFrame({"group": pd.Series(dtype=str), "value": pd.Series(dtype=float)})


@pytest.fixture
def missing_group_col_df():
    """DataFrame missing the 'group' column."""
    return pd.DataFrame({"value": [1, 0, 1, 0]})


@pytest.fixture
def missing_value_col_df():
    """DataFrame missing the 'value' column."""
    return pd.DataFrame({"group": ["control", "treatment"]})


@pytest.fixture
def wrong_group_labels_df():
    """DataFrame with non-standard group labels."""
    return pd.DataFrame({
        "group": ["a", "a", "b", "b"],
        "value": [1, 0, 1, 0],
    })


@pytest.fixture
def nan_values_df():
    """DataFrame with NaN in value column."""
    return pd.DataFrame({
        "group": ["control", "treatment", "control", "treatment"],
        "value": [1.0, np.nan, 0.0, 1.0],
    })


@pytest.fixture
def case_insensitive_df():
    """DataFrame with mixed-case group labels."""
    return pd.DataFrame({
        "group": ["Control", "CONTROL", "Treatment", "TREATMENT"],
        "value": [1, 0, 1, 0],
    })

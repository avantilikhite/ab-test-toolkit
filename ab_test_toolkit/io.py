"""CSV and DataFrame loading, validation, and metric type auto-detection.

Provides the primary data ingestion pipeline for the AB Test Toolkit.
Accepts CSV file paths, pathlib Paths, or pandas DataFrames and returns
a validated DataFrame with a detected MetricType.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from ab_test_toolkit import MetricType


def load_experiment_data(
    source: Union[str, Path, pd.DataFrame],
) -> tuple[pd.DataFrame, MetricType]:
    """Load and validate experiment data from CSV path or DataFrame.

    Parameters
    ----------
    source : str | Path | DataFrame
        CSV file path or an already-loaded DataFrame.

    Returns
    -------
    tuple[DataFrame, MetricType]
        Validated DataFrame and auto-detected metric type.

    Raises
    ------
    ValueError
        If required columns are missing, group labels are invalid,
        or value column contains NaN.
    """
    # --- Load from path if necessary ---
    if isinstance(source, (str, Path)):
        source = pd.read_csv(source)

    df = source.copy()

    # --- Validate required columns ---
    if "group" not in df.columns:
        raise ValueError(
            "Missing required column: 'group'. DataFrame must contain a 'group' column."
        )

    if "value" not in df.columns:
        raise ValueError(
            "Missing required column: 'value'. DataFrame must contain a 'value' column."
        )

    # --- Validate non-empty ---
    if len(df) == 0:
        raise ValueError("DataFrame is empty. Must contain at least one row of data.")

    # --- Normalize group labels ---
    df["group"] = df["group"].astype(str).str.lower().str.strip()

    unique_groups = set(df["group"].unique())
    if unique_groups != {"control", "treatment"}:
        raise ValueError(
            f"Group column must contain exactly 'control' and 'treatment' "
            f"(case-insensitive). Found: {sorted(unique_groups)}"
        )

    # --- Validate value column ---
    try:
        df["value"] = pd.to_numeric(df["value"], errors="raise")
    except (ValueError, TypeError):
        raise ValueError(
            "Value column contains non-numeric data. "
            "All values must be numbers (integers or decimals)."
        )
    if df["value"].isna().any():
        raise ValueError(
            "Value column contains NaN/missing values. "
            "All observations must have valid numeric values."
        )
    import numpy as _np
    if not _np.all(_np.isfinite(df["value"].to_numpy(dtype=float))):
        raise ValueError(
            "Value column contains Inf/-Inf values. "
            "All observations must be finite numbers."
        )

    # --- Validate covariate column (if present) ---
    if "covariate" in df.columns:
        try:
            df["covariate"] = pd.to_numeric(df["covariate"], errors="raise")
        except (ValueError, TypeError):
            raise ValueError(
                "Covariate column contains non-numeric data. "
                "All covariate values must be numbers."
            )
        if df["covariate"].isna().any():
            raise ValueError(
                "Covariate column contains NaN/missing values. "
                "All covariate values must be numeric."
            )
        if not _np.all(_np.isfinite(df["covariate"].to_numpy(dtype=float))):
            raise ValueError(
                "Covariate column contains Inf/-Inf values. "
                "All covariate values must be finite numbers."
            )

    # --- Validate day column (if present) ---
    # Coerce to numeric to avoid lexicographic ordering bugs in time-series
    # diagnostics (e.g. "1","10","2" sorted as strings would mis-bucket the
    # early/late windows of the novelty heuristic).
    if "day" in df.columns:
        try:
            df["day"] = pd.to_numeric(df["day"], errors="raise")
        except (ValueError, TypeError):
            raise ValueError(
                "Day column contains non-numeric data. "
                "Use integer day-from-start (0, 1, 2, ...) or numeric timestamps; "
                "string labels like 'Mon'/'Tue' are not supported."
            )
        if df["day"].isna().any():
            raise ValueError("Day column contains NaN/missing values.")

    # --- Handle segment NaN ---
    if "segment" in df.columns:
        df["segment"] = df["segment"].fillna("unknown").astype(str)

    # --- Auto-detect metric type ---
    metric_type = _detect_metric_type(df["value"])

    return df, metric_type


def _detect_metric_type(values: pd.Series) -> MetricType:
    """Detect whether values are proportion (binary 0/1) or continuous.

    Parameters
    ----------
    values : pd.Series
        The value column from the experiment data.

    Returns
    -------
    MetricType
        PROPORTION if all values are 0 or 1, CONTINUOUS otherwise.
    """
    unique_vals = set(values.unique())
    # Check if all values are exactly 0 or 1 (works for int and float)
    if unique_vals.issubset({0, 1, 0.0, 1.0}):
        return MetricType.PROPORTION
    return MetricType.CONTINUOUS

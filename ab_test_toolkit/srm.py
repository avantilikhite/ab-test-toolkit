"""SRM (Sample Ratio Mismatch) detection using chi-square goodness-of-fit."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import pandas as pd
from scipy import stats as sp_stats


@dataclass
class SRMResult:
    """Result of an SRM check."""

    expected_ratio: Tuple[float, float]
    observed_ratio: Tuple[float, float]
    chi2_statistic: float
    p_value: float
    has_mismatch: bool
    warning: Optional[str] = None


@dataclass
class StratifiedSRMResult:
    """Per-stratum SRM result and aggregated diagnostics."""

    stratum_results: list = field(default_factory=list)  # list[dict]: {stratum, n, p_value, has_mismatch}
    n_strata: int = 0
    n_mismatches: int = 0
    any_mismatch: bool = False
    warning: Optional[str] = None


def check_srm(
    observed: Tuple[int, int],
    expected_ratio: Tuple[float, float] = (0.5, 0.5),
    threshold: float = 0.01,
) -> SRMResult:
    """Run a chi-square test to detect sample ratio mismatch.

    Parameters
    ----------
    observed : tuple of two ints
        Observed counts for (control, treatment).
    expected_ratio : tuple of two floats
        Expected allocation ratio, default 50/50.
    threshold : float
        P-value threshold below which mismatch is flagged.
    """
    if len(expected_ratio) != 2:
        raise ValueError("expected_ratio must be a 2-tuple (control_share, treatment_share).")
    if any(r < 0 for r in expected_ratio):
        raise ValueError("expected_ratio entries must be non-negative.")
    if sum(expected_ratio) <= 0:
        raise ValueError("expected_ratio entries must sum to a positive value.")
    if len(observed) != 2:
        raise ValueError("observed must be a 2-tuple (control_count, treatment_count).")
    if any(o < 0 for o in observed):
        raise ValueError("observed counts must be non-negative.")
    if not (0 < threshold < 1):
        raise ValueError("threshold must be in (0, 1).")

    total = sum(observed)
    if total <= 0:
        return SRMResult(
            expected_ratio=expected_ratio,
            observed_ratio=(0.0, 0.0),
            chi2_statistic=0.0,
            p_value=1.0,
            has_mismatch=False,
        )
    # Normalize expected_ratio if it doesn't sum to 1
    ratio_sum = sum(expected_ratio)
    if abs(ratio_sum - 1.0) > 1e-6:
        expected_ratio = (expected_ratio[0] / ratio_sum, expected_ratio[1] / ratio_sum)
    expected_counts = (total * expected_ratio[0], total * expected_ratio[1])

    result = sp_stats.chisquare(observed, f_exp=expected_counts)

    observed_ratio = (observed[0] / total, observed[1] / total)

    warning = None
    min_expected = min(expected_counts)
    if min_expected < 5:
        warning = (
            f"Small expected cell count (min={min_expected:.1f}). The chi-square "
            f"approximation is unreliable below 5; treat the p-value as indicative."
        )

    return SRMResult(
        expected_ratio=expected_ratio,
        observed_ratio=observed_ratio,
        chi2_statistic=result.statistic,
        p_value=result.pvalue,
        has_mismatch=bool(result.pvalue < threshold),
        warning=warning,
    )


def check_srm_by_stratum(
    df: pd.DataFrame,
    group_col: str = "group",
    stratum_col: str = "day",
    expected_ratio: Tuple[float, float] = (0.5, 0.5),
    threshold: float = 0.01,
    expected_ratio_by_stratum: Optional[dict] = None,
) -> StratifiedSRMResult:
    """Run an SRM check independently within each stratum (e.g. day, segment).

    A test can be balanced overall while still showing systematic imbalance on
    specific days or in specific segments — usually a sign of a bucketing bug,
    targeted exposure, or a dropped event source.  This helper surfaces those
    cases so a per-stratum mismatch is not hidden by the aggregate.

    Multiple-testing
    ----------------
    Per-stratum p-values are independently distributed under the null, so with
    K strata you'd expect roughly ``α·K`` spurious "mismatches" by chance.
    This helper applies a **Holm–Bonferroni** correction across strata; the
    ``has_mismatch`` flag on each stratum result reflects the *adjusted*
    p-value vs ``threshold``.  The raw p-value is still returned for
    transparency.

    Parameters
    ----------
    expected_ratio : tuple
        Default (control, treatment) share applied when no per-stratum
        override is supplied.
    expected_ratio_by_stratum : dict, optional
        Map ``{stratum_key: (control_share, treatment_share)}`` for designs
        where the planned allocation varies (e.g. a ramp).  Strata not in
        the map fall back to ``expected_ratio``.
    """
    if df is None or len(df) == 0:
        return StratifiedSRMResult(warning="empty input")
    if group_col not in df.columns or stratum_col not in df.columns:
        raise ValueError(f"DataFrame must have columns '{group_col}' and '{stratum_col}'.")

    results = []
    raw_p_values = []
    for stratum, sub in df.groupby(stratum_col, sort=True):
        cnt_c = int((sub[group_col] == "control").sum())
        cnt_t = int((sub[group_col] == "treatment").sum())
        if cnt_c + cnt_t == 0:
            continue
        ratio = expected_ratio
        if expected_ratio_by_stratum and stratum in expected_ratio_by_stratum:
            ratio = expected_ratio_by_stratum[stratum]
        sr = check_srm((cnt_c, cnt_t), expected_ratio=ratio, threshold=threshold)
        results.append({
            "stratum": stratum,
            "n_control": cnt_c,
            "n_treatment": cnt_t,
            "expected_ratio": ratio,
            "p_value": sr.p_value,
            "p_value_adjusted": sr.p_value,  # placeholder; filled below
            "has_mismatch": False,
        })
        raw_p_values.append(sr.p_value)

    # Holm–Bonferroni across strata
    if raw_p_values:
        m = len(raw_p_values)
        order = sorted(range(m), key=lambda i: raw_p_values[i])
        adjusted = [1.0] * m
        running_max = 0.0
        for rank, i in enumerate(order):
            adj = min((m - rank) * raw_p_values[i], 1.0)
            running_max = max(running_max, adj)
            adjusted[i] = running_max
        for r, adj in zip(results, adjusted):
            r["p_value_adjusted"] = adj
            r["has_mismatch"] = adj < threshold

    n_mm = sum(1 for r in results if r["has_mismatch"])

    warn = None
    if n_mm > 0:
        warn = (
            f"{n_mm}/{len(results)} strata show SRM after Holm-correction at "
            f"threshold {threshold}. Investigate bucketing or exposure logic "
            f"for these strata."
        )

    return StratifiedSRMResult(
        stratum_results=results,
        n_strata=len(results),
        n_mismatches=n_mm,
        any_mismatch=n_mm > 0,
        warning=warn,
    )

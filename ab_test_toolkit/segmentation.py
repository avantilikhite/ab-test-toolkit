"""Segmentation analysis — per-segment treatment effects and Simpson's Paradox."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


@dataclass
class SegmentResult:
    """Result of segmented experiment analysis.

    Note on naming: ``simpsons_paradox`` flags a *segment sign reversal* —
    at least one non-trivial segment whose effect opposes the aggregate
    sign.  A textbook Simpson's paradox additionally requires a segment-mix
    imbalance between arms to drive the reversal; that upstream signature
    is reported separately via ``mix_imbalance`` / ``mix_p_value``.  The
    field name is retained for API stability.
    """

    segment_results: List[Dict[str, Any]]
    n_segments: int
    aggregate_estimate: float
    aggregate_ci: Tuple[float, float]
    simpsons_paradox: bool
    simpsons_details: Optional[str] = None
    multiple_comparisons_note: Optional[str] = None
    mix_imbalance: bool = False
    mix_p_value: Optional[float] = None
    mix_details: Optional[str] = None


def _treatment_effect(ctrl: np.ndarray, treat: np.ndarray) -> Dict[str, Any]:
    """Compute treatment effect, CI, and p-value using Welch's t-test."""
    estimate = float(np.mean(treat) - np.mean(ctrl))
    n = len(ctrl) + len(treat)

    t_stat, p_value = sp_stats.ttest_ind(treat, ctrl, equal_var=False)

    n_c, n_t = len(ctrl), len(treat)
    var_c = np.var(ctrl, ddof=1)
    var_t = np.var(treat, ddof=1)
    se = np.sqrt(var_c / n_c + var_t / n_t)

    # Welch-Satterthwaite df. Fall back to a wide CI when both groups have zero
    # variance (e.g., all-identical observations) to avoid 0/0 RuntimeWarning.
    num = (var_c / n_c + var_t / n_t) ** 2
    denom = (var_c / n_c) ** 2 / (n_c - 1) + (var_t / n_t) ** 2 / (n_t - 1)
    if denom <= 0 or not np.isfinite(denom):
        ci = (float("-inf"), float("inf")) if se == 0 else (estimate, estimate)
    else:
        df = num / denom
        t_crit = sp_stats.t.ppf(0.975, df)
        ci = (estimate - t_crit * se, estimate + t_crit * se)

    return {"estimate": estimate, "ci": ci, "n": n, "p_value": float(p_value)}


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to a list of p-values.

    Returns adjusted p-values (capped at 1.0), preserving input order.
    NaN p-values (from degenerate segments) are passed through as NaN.
    """
    n = len(p_values)
    if n <= 1:
        return list(p_values)

    # Separate NaN and valid p-values, correct only the valid ones
    indexed = [(i, p) for i, p in enumerate(p_values)]
    valid = [(i, p) for i, p in indexed if not (p != p)]  # NaN != NaN
    nan_indices = {i for i, p in indexed if p != p}

    adjusted = [float("nan")] * n

    if not valid:
        return adjusted

    # Use n_valid (not total n) for Holm correction: segments with
    # too-small samples are excluded before testing, so the family
    # size equals only the segments that produced valid p-values.
    n_valid = len(valid)
    sorted_valid = sorted(valid, key=lambda x: x[1])
    cumulative_max = 0.0
    for rank, (orig_idx, p) in enumerate(sorted_valid):
        corrected = p * (n_valid - rank)
        cumulative_max = max(cumulative_max, corrected)
        adjusted[orig_idx] = min(cumulative_max, 1.0)

    return adjusted


def segment_analysis(
    df: pd.DataFrame,
    group_col: str = "group",
    value_col: str = "value",
    segment_col: str = "segment",
    max_segments: int = 20,
) -> SegmentResult:
    """Run per-segment treatment effect analysis with Simpson's Paradox detection.

    Parameters
    ----------
    df : DataFrame
        Must contain group_col ('control'/'treatment'), value_col, segment_col.
    max_segments : int
        Soft cap on segment cardinality.  Above this, a warning is appended
        to the result's ``multiple_comparisons_note`` because per-segment
        p-values become brittle and Holm correction becomes very conservative.
    """
    segments = df[segment_col].unique()
    n_segments = len(segments)
    _high_card_warning = (
        f" High segment cardinality ({n_segments} > {max_segments}) inflates "
        "multiple-comparisons risk and weakens per-segment power; consider "
        "grouping rare segments into 'other' or pre-registering a smaller "
        "list of decision-grade segments."
        if n_segments > max_segments
        else ""
    )

    # Aggregate effect
    ctrl_all = df.loc[df[group_col] == "control", value_col].to_numpy()
    treat_all = df.loc[df[group_col] == "treatment", value_col].to_numpy()
    agg = _treatment_effect(ctrl_all, treat_all)
    aggregate_estimate = agg["estimate"]
    aggregate_ci = agg["ci"]
    agg_sign = np.sign(aggregate_estimate)

    # Per-segment
    segment_results: List[Dict[str, Any]] = []
    paradox_segments: List[str] = []
    paradox_share: float = 0.0
    n_valid_segments = 0
    valid_total_n = 0

    for seg in sorted(segments):
        seg_df = df[df[segment_col] == seg]
        ctrl = seg_df.loc[seg_df[group_col] == "control", value_col].to_numpy()
        treat = seg_df.loc[seg_df[group_col] == "treatment", value_col].to_numpy()

        if len(ctrl) < 2 or len(treat) < 2:
            result = {"estimate": 0.0, "ci": (0.0, 0.0), "n": len(ctrl) + len(treat), "p_value": float("nan")}
            result["segment"] = seg
            segment_results.append(result)
            continue

        result = _treatment_effect(ctrl, treat)
        result["segment"] = seg
        segment_results.append(result)
        n_valid_segments += 1
        valid_total_n += result["n"]

        seg_sign = np.sign(result["estimate"])
        # Scale-aware noise floor: ignore reversals smaller than 1% of the
        # aggregate effect's magnitude (a fixed absolute epsilon would be
        # meaningless for dollar-scale or rare-rate metrics).
        reversal_floor = 0.01 * abs(aggregate_estimate)
        if agg_sign != 0 and seg_sign != 0 and seg_sign != agg_sign and abs(result["estimate"]) > reversal_floor:
            paradox_segments.append(str(seg))
            paradox_share += result["n"]

    # Segment sign reversal: flag when any non-trivial segment (>=20% of valid
    # sample) reverses the aggregate effect's sign. The textbook Simpson case
    # is a 2-segment example where one large segment contradicts the aggregate,
    # so a strict-majority rule is too conservative — a meaningful contradiction
    # is enough to require human review.
    paradox_fraction = paradox_share / valid_total_n if valid_total_n > 0 else 0.0
    simpsons_paradox = (
        len(paradox_segments) > 0
        and n_valid_segments >= 2
        and paradox_fraction >= 0.20
    )

    # Traffic-mix diagnostic: the upstream signature of a true Simpson's
    # paradox is that the two arms have different segment mixes.  Chi-square
    # test of independence on the group × segment contingency table.  A
    # strict threshold (0.001) keeps false alarms rare, mirroring SRM logic.
    mix_imbalance = False
    mix_p_value: Optional[float] = None
    mix_details: Optional[str] = None
    try:
        crosstab = pd.crosstab(df[group_col], df[segment_col])
        if crosstab.shape[0] == 2 and crosstab.shape[1] >= 2 and (crosstab.values.sum(axis=1) > 0).all():
            _, mix_p, _, _ = sp_stats.chi2_contingency(crosstab.values)
            mix_p_value = float(mix_p)
            if mix_p_value < 0.001:
                mix_imbalance = True
                shares = crosstab.div(crosstab.sum(axis=1), axis=0)
                gap = (shares.iloc[0] - shares.iloc[1]).abs()
                worst_seg = gap.idxmax()
                mix_details = (
                    f"Segment mix differs between arms (chi-square p={mix_p_value:.2e}); "
                    f"largest share gap in segment '{worst_seg}' "
                    f"({gap.max():.1%} between arms). This is the structural setup for "
                    f"Simpson's paradox — investigate the assignment mechanism."
                )
    except (ValueError, ZeroDivisionError):
        mix_p_value = None

    simpsons_details = None
    if simpsons_paradox:
        simpsons_details = (
            f"Segment sign reversal: aggregate effect sign differs from segment(s) "
            f"{', '.join(paradox_segments)} representing "
            f"{paradox_fraction:.0%} of valid sample."
        )
        if mix_imbalance:
            simpsons_details += (
                " A segment-mix imbalance between arms was also detected — "
                "consistent with a true Simpson's paradox (composition effect)."
            )
        else:
            simpsons_details += (
                " Segment mixes are balanced across arms, so this is heterogeneous "
                "treatment effects (HTE), not a composition artifact."
            )

    # Holm-Bonferroni adjusted p-values
    raw_p_values = [seg["p_value"] for seg in segment_results]
    adjusted_p_values = _holm_bonferroni(raw_p_values)
    for seg, adj_p in zip(segment_results, adjusted_p_values):
        seg["p_value_adjusted"] = adj_p

    multiple_comparisons_note = (
        f"Holm-Bonferroni adjusted p-values provided alongside raw values "
        f"across {n_segments} segment(s)." + _high_card_warning
    )

    return SegmentResult(
        segment_results=segment_results,
        n_segments=n_segments,
        aggregate_estimate=aggregate_estimate,
        aggregate_ci=aggregate_ci,
        simpsons_paradox=simpsons_paradox,
        simpsons_details=simpsons_details,
        multiple_comparisons_note=multiple_comparisons_note,
        mix_imbalance=mix_imbalance,
        mix_p_value=mix_p_value,
        mix_details=mix_details,
    )

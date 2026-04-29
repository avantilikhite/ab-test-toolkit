"""Frequentist hypothesis tests for A/B testing: Z-test and Welch's t-test."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class FrequentistResult:
    """Result container for frequentist hypothesis tests."""

    test_type: str
    statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    point_estimate: float
    effect_size: float
    alpha: float
    is_significant: bool
    n_control: Optional[int] = None
    n_treatment: Optional[int] = None
    normality_check: Optional[dict] = field(default=None)


# ---------------------------------------------------------------------------
# Two-proportion Z-test (unpooled SE)
# ---------------------------------------------------------------------------

def _wilson_bounds(count: int, total: int, z: float) -> tuple[float, float]:
    """Wilson score interval bounds for a single proportion."""
    if total == 0:
        return 0.0, 1.0
    p = count / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _newcombe_diff_ci(
    count_c: int, total_c: int, count_t: int, total_t: int, alpha: float
) -> tuple[float, float]:
    """Newcombe hybrid score CI for the difference of two proportions.

    Uses Wilson score bounds for each proportion and combines them. Has well-
    behaved coverage at the boundaries (e.g. 0/n vs 0/n) where the Wald
    interval collapses to a misleading [0, 0].
    """
    z = stats.norm.ppf(1 - alpha / 2)
    l1, u1 = _wilson_bounds(count_c, total_c, z)
    l2, u2 = _wilson_bounds(count_t, total_t, z)
    p0 = count_c / total_c
    p1 = count_t / total_t
    diff = p1 - p0
    delta_lo = np.sqrt((p0 - l1) ** 2 + (u2 - p1) ** 2)
    delta_hi = np.sqrt((u1 - p0) ** 2 + (p1 - l2) ** 2)
    # The natural support of a risk difference is [-1, 1]; clip to avoid
    # impossible bounds at extreme corner cases (e.g. 0/n vs n/n).
    lo = max(-1.0, float(diff - delta_lo))
    hi = min(1.0, float(diff + delta_hi))
    return lo, hi


def z_test(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> FrequentistResult:
    """Two-proportion Z-test from raw binary arrays (unpooled SE)."""
    control = np.asarray(control)
    treatment = np.asarray(treatment)
    if control.size == 0 or treatment.size == 0:
        raise ValueError("z_test requires non-empty control and treatment arrays.")
    if not np.all(np.isfinite(control)) or not np.all(np.isfinite(treatment)):
        raise ValueError("z_test inputs must be finite (no NaN/Inf).")
    unique_vals = np.unique(np.concatenate([control, treatment]))
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(
            "z_test expects binary 0/1 outcomes. For continuous metrics use welch_t_test()."
        )
    return z_test_from_stats(
        control_count=int(control.sum()),
        control_total=len(control),
        treatment_count=int(treatment.sum()),
        treatment_total=len(treatment),
        alpha=alpha,
    )


def z_test_from_stats(
    control_count: int,
    control_total: int,
    treatment_count: int,
    treatment_total: int,
    alpha: float = 0.05,
) -> FrequentistResult:
    """Two-proportion test using Wilson-derived Newcombe CI for the risk difference."""
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")
    if control_total <= 0 or treatment_total <= 0:
        raise ValueError("Totals must be positive integers.")
    if control_count < 0 or treatment_count < 0:
        raise ValueError("Counts must be non-negative.")
    if control_count > control_total or treatment_count > treatment_total:
        raise ValueError("Count cannot exceed total.")

    p0 = control_count / control_total
    p1 = treatment_count / treatment_total
    n0, n1 = control_total, treatment_total

    se = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    point_estimate = p1 - p0

    # Z statistic and p-value still use the unpooled SE — Newcombe is for the CI only.
    z_stat = point_estimate / se if se > 0 else (float("inf") if point_estimate != 0 else 0.0)
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat)))) if se > 0 else (0.0 if point_estimate != 0 else 1.0)

    # Newcombe hybrid score CI — well-behaved at boundaries where Wald collapses to 0.
    ci_lower, ci_upper = _newcombe_diff_ci(control_count, control_total, treatment_count, treatment_total, alpha)

    # Cohen's h
    effect_size = float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p0)))

    return FrequentistResult(
        test_type="z_test",
        statistic=float(z_stat),
        p_value=p_value,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        point_estimate=float(point_estimate),
        effect_size=effect_size,
        alpha=alpha,
        is_significant=p_value < alpha,
        n_control=int(control_total),
        n_treatment=int(treatment_total),
    )


# ---------------------------------------------------------------------------
# Welch's t-test
# ---------------------------------------------------------------------------

def welch_t_test(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> FrequentistResult:
    """Welch's t-test from raw continuous arrays."""
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    if control.size == 0 or treatment.size == 0:
        raise ValueError("welch_t_test requires non-empty control and treatment arrays.")
    if not (np.all(np.isfinite(control)) and np.all(np.isfinite(treatment))):
        raise ValueError("welch_t_test inputs must be finite (no NaN/Inf).")
    if control.size < 2 or treatment.size < 2:
        raise ValueError("Each group must have at least 2 observations for Welch's t-test.")

    result = welch_t_test_from_stats(
        control_mean=float(control.mean()),
        control_std=float(control.std(ddof=1)),
        control_n=len(control),
        treatment_mean=float(treatment.mean()),
        treatment_std=float(treatment.std(ddof=1)),
        treatment_n=len(treatment),
        alpha=alpha,
    )

    # Normality diagnostics: Shapiro-Wilk for small N, skewness/kurtosis for large N
    normality_threshold = 5000
    if len(control) <= normality_threshold and len(treatment) <= normality_threshold:
        sw_control = stats.shapiro(control)
        sw_treatment = stats.shapiro(treatment)
        result.normality_check = {
            "method": "shapiro_wilk",
            "control": {
                "statistic": float(sw_control.statistic),
                "p_value": float(sw_control.pvalue),
            },
            "treatment": {
                "statistic": float(sw_treatment.statistic),
                "p_value": float(sw_treatment.pvalue),
            },
        }
    else:
        result.normality_check = {
            "method": "skewness_kurtosis",
            "note": (
                f"N > {normality_threshold}: Shapiro-Wilk is unreliable at this scale. "
                "Reporting skewness/kurtosis instead. Welch's t-test is robust via CLT."
            ),
            "control": {
                "skewness": float(stats.skew(control)),
                "kurtosis": float(stats.kurtosis(control)),
            },
            "treatment": {
                "skewness": float(stats.skew(treatment)),
                "kurtosis": float(stats.kurtosis(treatment)),
            },
        }
    return result


def welch_t_test_from_stats(
    control_mean: float,
    control_std: float,
    control_n: int,
    treatment_mean: float,
    treatment_std: float,
    treatment_n: int,
    alpha: float = 0.05,
) -> FrequentistResult:
    """Welch's t-test from summary statistics."""
    if control_n < 2 or treatment_n < 2:
        raise ValueError("Each group must have at least 2 observations for Welch's t-test.")
    if control_std < 0 or treatment_std < 0:
        raise ValueError("Standard deviations must be non-negative.")

    point_estimate = treatment_mean - control_mean

    # Handle zero-variance edge case
    if control_std == 0 and treatment_std == 0:
        return FrequentistResult(
            test_type="welch_t_test",
            statistic=float("inf") if point_estimate != 0 else 0.0,
            p_value=0.0 if point_estimate != 0 else 1.0,
            ci_lower=point_estimate,
            ci_upper=point_estimate,
            point_estimate=point_estimate,
            effect_size=0.0,
            alpha=alpha,
            is_significant=point_estimate != 0,
            n_control=int(control_n),
            n_treatment=int(treatment_n),
        )

    se = np.sqrt(control_std**2 / control_n + treatment_std**2 / treatment_n)
    t_stat = point_estimate / se if se > 0 else (float("inf") if point_estimate != 0 else 0.0)

    # Welch-Satterthwaite degrees of freedom
    num = (control_std**2 / control_n + treatment_std**2 / treatment_n) ** 2
    denom = (
        (control_std**2 / control_n) ** 2 / (control_n - 1)
        + (treatment_std**2 / treatment_n) ** 2 / (treatment_n - 1)
    )
    df = num / denom if denom > 0 else 1.0

    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = point_estimate - t_crit * se
    ci_upper = point_estimate + t_crit * se

    # Cohen's d with pooled std
    s_pooled = np.sqrt(
        ((control_n - 1) * control_std**2 + (treatment_n - 1) * treatment_std**2)
        / (control_n + treatment_n - 2)
    )
    effect_size = float(point_estimate / s_pooled) if s_pooled > 0 else 0.0

    return FrequentistResult(
        test_type="welch_t_test",
        statistic=float(t_stat),
        p_value=p_value,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        point_estimate=float(point_estimate),
        effect_size=effect_size,
        alpha=alpha,
        is_significant=p_value < alpha,
        n_control=int(control_n),
        n_treatment=int(treatment_n),
    )

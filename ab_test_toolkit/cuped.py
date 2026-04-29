"""CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import stats as sp_stats


def winsorize(arr: np.ndarray, p: float = 0.99) -> np.ndarray:
    """Symmetrically winsorize an array at the (1-p)/2 and (1+p)/2 quantiles.

    Replaces extreme values with the cutoff to limit the influence of outliers
    in heavy-tailed metrics (revenue, sessions/day) before t-tests or CUPED.
    Returns a new array; the original is not modified.

    Parameters
    ----------
    arr : np.ndarray
        Numeric array (NaN/Inf raise).
    p : float
        Coverage in (0, 1).  Default 0.99 winsorizes the top/bottom 0.5%.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr.copy()
    if not np.all(np.isfinite(arr)):
        raise ValueError("winsorize inputs must be finite (no NaN/Inf).")
    if not (0 < p < 1):
        raise ValueError("p must be in (0, 1).")
    tail = (1.0 - p) / 2.0
    lo = float(np.quantile(arr, tail))
    hi = float(np.quantile(arr, 1.0 - tail))
    return np.clip(arr, lo, hi)


@dataclass
class CUPEDResult:
    """Result of CUPED variance adjustment."""

    theta: float
    correlation: float
    variance_reduction_pct: float
    unadjusted_estimate: float
    adjusted_estimate: float
    unadjusted_ci: Tuple[float, float]
    adjusted_ci: Tuple[float, float]
    realized_variance_reduction_pct: float = 0.0


def _welch_ci(group_a: np.ndarray, group_b: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """Compute Welch's t-interval for mean(b) - mean(a)."""
    n_a, n_b = len(group_a), len(group_b)
    mean_diff = np.mean(group_b) - np.mean(group_a)
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)
    # Zero-variance fast path: if both arms are constant the Welch CI
    # is undefined (0/0 df).  Return a degenerate interval at the
    # observed mean difference.
    if var_a == 0 and var_b == 0:
        return (mean_diff, mean_diff)
    se = np.sqrt(var_a / n_a + var_b / n_b)
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df)
    return (mean_diff - t_crit * se, mean_diff + t_crit * se)


def _ancova_treatment_ci(
    ctrl_outcome: np.ndarray,
    treat_outcome: np.ndarray,
    ctrl_covariate: np.ndarray,
    treat_covariate: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, Tuple[float, float]]:
    """OLS ANCOVA: Y = b0 + b1*T + b2*X + e.

    Returns the treatment-effect point estimate (b1) and its (1-alpha)
    confidence interval, computed from the OLS sampling distribution.  The
    SE properly accounts for theta-from-data uncertainty (unlike the Welch
    CI on residuals), which is the right fix for the canonical CUPED
    "ignored degrees of freedom" critique.
    """
    n_c, n_t = len(ctrl_outcome), len(treat_outcome)
    n = n_c + n_t
    y = np.concatenate([ctrl_outcome, treat_outcome])
    t = np.concatenate([np.zeros(n_c), np.ones(n_t)])
    x = np.concatenate([ctrl_covariate, treat_covariate])
    X = np.column_stack([np.ones(n), t, x])
    # Solve (X'X) β = X'y
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        beta = np.linalg.solve(XtX, Xty)
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # Singular design matrix (e.g., constant covariate) — fall back to Welch.
        return float(np.mean(treat_outcome) - np.mean(ctrl_outcome)), _welch_ci(
            ctrl_outcome, treat_outcome, alpha
        )
    resid = y - X @ beta
    df = n - 3
    if df <= 0:
        return float(beta[1]), (float(beta[1]), float(beta[1]))
    sigma2 = float(resid @ resid / df)
    se_b1 = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df)
    est = float(beta[1])
    return est, (est - t_crit * se_b1, est + t_crit * se_b1)


def cuped_adjust(
    ctrl_outcome: np.ndarray,
    treat_outcome: np.ndarray,
    ctrl_covariate: np.ndarray,
    treat_covariate: np.ndarray,
    alpha: float = 0.05,
) -> CUPEDResult:
    """Apply CUPED variance reduction.

    Implementation notes
    --------------------
    * ``theta`` is computed via the canonical pooled formula
      ``Cov(Y, X) / Var(X)`` (Deng et al., 2013).
    * ``adjusted_estimate`` and ``adjusted_ci`` are taken from an OLS
      ANCOVA fit ``Y ~ T + X``, so the CI accounts for uncertainty in
      ``theta`` itself.  For randomized experiments with balanced
      covariates the ANCOVA point estimate equals the residual-based
      CUPED estimate to first order, but the SE is correctly larger
      than the naive Welch-on-residuals SE in finite samples.
    * ``variance_reduction_pct`` (= ρ² × 100) is the *theoretical*
      asymptotic reduction.  ``realized_variance_reduction_pct`` is
      computed from the observed CI widths and reflects what actually
      happened on this dataset.  Note: the unadjusted Welch CI uses
      ``df`` based on Welch–Satterthwaite while the adjusted ANCOVA CI
      uses ``df = n − 3``; for this reason the realized number is an
      approximation that loses ~1 df of conservativeness in small
      samples.  Use it as a sanity-check, not a reportable statistic.
    """
    ctrl_outcome = np.asarray(ctrl_outcome, dtype=float)
    treat_outcome = np.asarray(treat_outcome, dtype=float)
    ctrl_covariate = np.asarray(ctrl_covariate, dtype=float)
    treat_covariate = np.asarray(treat_covariate, dtype=float)

    if len(ctrl_outcome) != len(ctrl_covariate):
        raise ValueError(
            f"Control outcome ({len(ctrl_outcome)}) and covariate ({len(ctrl_covariate)}) "
            f"arrays must have the same length."
        )
    if len(treat_outcome) != len(treat_covariate):
        raise ValueError(
            f"Treatment outcome ({len(treat_outcome)}) and covariate ({len(treat_covariate)}) "
            f"arrays must have the same length."
        )
    if len(ctrl_outcome) < 2 or len(treat_outcome) < 2:
        raise ValueError("CUPED requires at least 2 observations per group.")
    for arr, name in [
        (ctrl_outcome, "ctrl_outcome"),
        (treat_outcome, "treat_outcome"),
        (ctrl_covariate, "ctrl_covariate"),
        (treat_covariate, "treat_covariate"),
    ]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite (no NaN/Inf).")

    all_y = np.concatenate([ctrl_outcome, treat_outcome])
    all_x = np.concatenate([ctrl_covariate, treat_covariate])

    cov_yx = np.cov(all_y, all_x)[0, 1]
    var_x = np.var(all_x, ddof=1)

    unadjusted_estimate = float(np.mean(treat_outcome) - np.mean(ctrl_outcome))
    unadjusted_ci = _welch_ci(ctrl_outcome, treat_outcome, alpha)
    unadj_width = unadjusted_ci[1] - unadjusted_ci[0]

    if var_x == 0 or np.isnan(var_x):
        return CUPEDResult(
            theta=0.0,
            correlation=0.0,
            variance_reduction_pct=0.0,
            unadjusted_estimate=unadjusted_estimate,
            adjusted_estimate=unadjusted_estimate,
            unadjusted_ci=unadjusted_ci,
            adjusted_ci=unadjusted_ci,
            realized_variance_reduction_pct=0.0,
        )

    theta = cov_yx / var_x
    correlation = np.corrcoef(all_y, all_x)[0, 1]

    adjusted_estimate, adjusted_ci = _ancova_treatment_ci(
        ctrl_outcome, treat_outcome, ctrl_covariate, treat_covariate, alpha
    )

    # Theoretical (asymptotic) reduction
    variance_reduction_pct = float(correlation ** 2 * 100)

    # Realized reduction from observed CI widths.  CI half-width ∝ SE, so
    # var-reduction = 1 - (SE_adj / SE_unadj)^2 = 1 - (W_adj / W_unadj)^2.
    adj_width = adjusted_ci[1] - adjusted_ci[0]
    if unadj_width > 0:
        realized = 1.0 - (adj_width / unadj_width) ** 2
        realized_variance_reduction_pct = float(max(realized, 0.0) * 100)
    else:
        realized_variance_reduction_pct = 0.0

    return CUPEDResult(
        theta=float(theta),
        correlation=float(correlation),
        variance_reduction_pct=variance_reduction_pct,
        unadjusted_estimate=unadjusted_estimate,
        adjusted_estimate=float(adjusted_estimate),
        unadjusted_ci=unadjusted_ci,
        adjusted_ci=adjusted_ci,
        realized_variance_reduction_pct=realized_variance_reduction_pct,
    )

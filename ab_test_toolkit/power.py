"""Sample size and power analysis for proportion and continuous metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class PowerResult:
    """Result of a sample size / power calculation."""

    n_control: int
    n_treatment: int
    n_total: int
    n_effective: float
    sample_inflation_pct: float
    estimated_days: Optional[int]

    @property
    def power_loss_pct(self) -> float:
        """Deprecated alias for sample_inflation_pct.

        Retained for backward compatibility — the metric is sample-size
        inflation vs a balanced design, not power loss at fixed N.
        """
        return self.sample_inflation_pct


def required_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0,
    mde_mode: str = "absolute",
    daily_traffic: Optional[int] = None,
) -> PowerResult:
    """Compute per-arm sample sizes for a two-proportion Z-test.

    Parameters
    ----------
    baseline_rate : float
        Control group conversion rate, must be in (0, 1).
    mde : float
        Minimum detectable effect (absolute or relative).
    alpha : float
        Significance level (two-sided).
    power : float
        Desired statistical power (1 − β).
    allocation_ratio : float
        n_treatment / n_control ratio. 1.0 = balanced.
    mde_mode : str
        ``"absolute"`` or ``"relative"``.
    daily_traffic : int, optional
        Total daily visitors; used to estimate experiment duration.
    """
    # --- validation ---
    if baseline_rate <= 0 or baseline_rate >= 1:
        raise ValueError("baseline_rate must be in (0, 1)")
    if mde <= 0:
        raise ValueError("mde must be positive")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if not (0 < power < 1):
        raise ValueError("power must be in (0, 1)")
    if allocation_ratio <= 0:
        raise ValueError("allocation_ratio must be positive")
    if mde_mode not in ("absolute", "relative"):
        raise ValueError("mde_mode must be 'absolute' or 'relative'")
    if daily_traffic is not None and daily_traffic <= 0:
        raise ValueError("daily_traffic must be positive when provided")

    mde_abs = mde if mde_mode == "absolute" else baseline_rate * mde

    p0 = baseline_rate
    p1 = p0 + mde_abs
    if p1 <= 0 or p1 >= 1:
        raise ValueError("baseline_rate + mde must be in (0, 1)")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    r = allocation_ratio

    # Per-control-arm sample size for unequal allocation:
    # n_control = (z_α + z_β)² × [p0(1-p0) + p1(1-p1)/r] / (p1-p0)²
    n_control_raw = (
        (z_alpha + z_beta) ** 2
        * (p0 * (1 - p0) + p1 * (1 - p1) / r)
        / (p1 - p0) ** 2
    )
    n_control = int(math.ceil(n_control_raw))
    n_treatment = int(math.ceil(n_control * r))
    n_total = n_control + n_treatment

    # Effective sample size
    n_effective = 4 * n_control * n_treatment / (n_control + n_treatment)

    # Sample-size inflation vs balanced design (the cost of unequal allocation)
    if allocation_ratio == 1.0:
        sample_inflation_pct = 0.0
    else:
        balanced = required_sample_size(
            baseline_rate=baseline_rate,
            mde=mde,
            alpha=alpha,
            power=power,
            allocation_ratio=1.0,
            mde_mode=mde_mode,
        )
        balanced_n_total = balanced.n_total
        sample_inflation_pct = (n_total - balanced_n_total) / balanced_n_total * 100

    # Duration estimate
    estimated_days: Optional[int] = None
    if daily_traffic is not None:
        estimated_days = int(math.ceil(n_total / daily_traffic))

    return PowerResult(
        n_control=n_control,
        n_treatment=n_treatment,
        n_total=n_total,
        n_effective=n_effective,
        sample_inflation_pct=sample_inflation_pct,
        estimated_days=estimated_days,
    )


def power_curve(
    baseline_rate: float,
    mde_range: Tuple[float, float],
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0,
    n_points: int = 50,
) -> pd.DataFrame:
    """Generate a power curve DataFrame over a range of MDE values."""
    mde_values = np.linspace(mde_range[0], mde_range[1], n_points)
    rows = []
    for m in mde_values:
        res = required_sample_size(
            baseline_rate=baseline_rate,
            mde=float(m),
            alpha=alpha,
            power=power,
            allocation_ratio=allocation_ratio,
        )
        rows.append(
            {
                "mde": float(m),
                "n_control": res.n_control,
                "n_treatment": res.n_treatment,
                "n_total": res.n_total,
            }
        )
    return pd.DataFrame(rows)


def required_sample_size_continuous(
    baseline_mean: float,
    baseline_std: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0,
    treatment_std: Optional[float] = None,
    daily_traffic: Optional[int] = None,
) -> PowerResult:
    """Compute per-arm sample sizes for a two-sample t-test (continuous metrics).

    Method note
    -----------
    Uses the **Normal (z)** approximation rather than solving the exact
    non-central t-distribution.  At the sample sizes typical for product
    A/B tests (n ≥ a few hundred per arm) the difference is well under
    1%; for very small designs (n < ~30 per arm) you may want to inflate
    by a few percent or use a dedicated power package.

    Parameters
    ----------
    baseline_mean : float
        Control group mean (used for context; does not affect sample size).
    baseline_std : float
        Control group standard deviation, must be positive.
    mde : float
        Minimum detectable effect (absolute difference in means), must be positive.
    alpha : float
        Significance level (two-sided).
    power : float
        Desired statistical power (1 − β).
    allocation_ratio : float
        n_treatment / n_control ratio. 1.0 = balanced.
    treatment_std : float, optional
        Treatment group standard deviation. If None, assumed equal to baseline_std.
    daily_traffic : int, optional
        Total daily visitors; used to estimate experiment duration.
    """
    if baseline_std <= 0:
        raise ValueError("baseline_std must be positive")
    if mde <= 0:
        raise ValueError("mde must be positive")
    if treatment_std is not None and treatment_std <= 0:
        raise ValueError("treatment_std must be positive when provided")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if not (0 < power < 1):
        raise ValueError("power must be in (0, 1)")
    if allocation_ratio <= 0:
        raise ValueError("allocation_ratio must be positive")
    if daily_traffic is not None and daily_traffic <= 0:
        raise ValueError("daily_traffic must be positive when provided")

    sigma_c = baseline_std
    sigma_t = treatment_std if treatment_std is not None else baseline_std
    r = allocation_ratio

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Per-control-arm sample size for unequal allocation:
    # n_control = (z_α + z_β)² × (σ_c² + σ_t²/r) / mde²
    n_control_raw = (z_alpha + z_beta) ** 2 * (sigma_c**2 + sigma_t**2 / r) / mde**2
    n_control = int(math.ceil(n_control_raw))
    n_treatment = int(math.ceil(n_control * r))
    n_total = n_control + n_treatment

    n_effective = 4 * n_control * n_treatment / (n_control + n_treatment)

    if allocation_ratio == 1.0:
        sample_inflation_pct = 0.0
    else:
        balanced = required_sample_size_continuous(
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            mde=mde,
            alpha=alpha,
            power=power,
            allocation_ratio=1.0,
            treatment_std=treatment_std,
        )
        sample_inflation_pct = (n_total - balanced.n_total) / balanced.n_total * 100

    estimated_days: Optional[int] = None
    if daily_traffic is not None:
        estimated_days = int(math.ceil(n_total / daily_traffic))

    return PowerResult(
        n_control=n_control,
        n_treatment=n_treatment,
        n_total=n_total,
        n_effective=n_effective,
        sample_inflation_pct=sample_inflation_pct,
        estimated_days=estimated_days,
    )

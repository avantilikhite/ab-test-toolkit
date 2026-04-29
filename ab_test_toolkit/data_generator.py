"""Synthetic A/B experiment data generation with configurable anomalies.

Generates controlled datasets for testing, demonstration, and education.
Each injection flag (novelty, Simpson's Paradox, SRM) is independent and
composable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_experiment_data(
    baseline_rate: float,
    effect_size: float,
    n_control: int,
    n_treatment: int,
    *,
    inject_novelty: bool = False,
    novelty_days: int = 3,
    novelty_multiplier: float = 2.0,
    inject_simpsons: bool = False,
    inject_srm: bool = False,
    srm_actual_ratio: float = 0.55,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic A/B test data with configurable anomalies.

    Parameters
    ----------
    baseline_rate : float
        Control group expected conversion rate, in (0, 1).
    effect_size : float
        True treatment effect (absolute difference from baseline).
    n_control : int
        Number of observations in the control group.
    n_treatment : int
        Number of observations in the treatment group.
    inject_novelty : bool, default False
        If True, inflate treatment effect for the first ``novelty_days``.
    novelty_days : int, default 3
        Number of days with inflated effect.
    novelty_multiplier : float, default 2.0
        Effect multiplier during the novelty period.
    inject_simpsons : bool, default False
        If True, engineer Simpson's Paradox into segment-level data.
    inject_srm : bool, default False
        If True, skew group allocation to create Sample Ratio Mismatch.
    srm_actual_ratio : float, default 0.55
        Treatment fraction of total when SRM is injected.
    random_seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: group, value, covariate (always), segment (if inject_simpsons),
        day (if inject_novelty).
    """
    if not (0 < baseline_rate < 1):
        raise ValueError("baseline_rate must be in (0, 1).")
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("n_control and n_treatment must be positive integers.")
    if novelty_days < 0:
        raise ValueError("novelty_days must be non-negative.")
    if novelty_multiplier <= 0:
        raise ValueError("novelty_multiplier must be positive.")
    if not (0 < srm_actual_ratio < 1):
        raise ValueError("srm_actual_ratio must be in (0, 1).")

    rng = np.random.default_rng(random_seed)

    # --- SRM: adjust actual sample sizes ---
    if inject_srm:
        total = n_control + n_treatment
        n_treatment = int(total * srm_actual_ratio)
        n_control = total - n_treatment

    # --- Base data generation ---
    treatment_rate = np.clip(baseline_rate + effect_size, 0.0, 1.0)

    if inject_simpsons and inject_novelty:
        # Compose both: use Simpson's base data, then apply novelty inflation
        df = _generate_simpsons_data(
            rng, baseline_rate, effect_size, n_control, n_treatment,
        )
        # Add day column and inflate early treatment effect
        total_days = 14
        df["day"] = rng.integers(1, total_days + 1, size=len(df))
        early_mask = (df["group"] == "treatment") & (df["day"] <= novelty_days)
        # Boost early treatment values toward higher conversion
        inflated_rate = min(baseline_rate + effect_size * novelty_multiplier, 0.99)
        n_early = early_mask.sum()
        if n_early > 0:
            df.loc[early_mask, "value"] = rng.binomial(1, inflated_rate, n_early)
    elif inject_simpsons:
        # Engineer Simpson's Paradox via segment imbalance
        df = _generate_simpsons_data(
            rng, baseline_rate, effect_size, n_control, n_treatment,
        )
    elif inject_novelty:
        df = _generate_novelty_data(
            rng, baseline_rate, effect_size, n_control, n_treatment,
            novelty_days, novelty_multiplier,
        )
    else:
        control_vals = rng.binomial(1, baseline_rate, n_control)
        treatment_vals = rng.binomial(1, treatment_rate, n_treatment)
        df = pd.DataFrame({
            "group": ["control"] * n_control + ["treatment"] * n_treatment,
            "value": np.concatenate([control_vals, treatment_vals]),
        })

    # --- Add novelty day column if not already present ---
    if inject_novelty and "day" not in df.columns:
        total_days = 14
        df["day"] = rng.integers(1, total_days + 1, size=len(df))

    # --- Always add covariate for CUPED testing ---
    if "covariate" not in df.columns:
        n_total = len(df)
        covariate_base = rng.normal(5.0, 1.0, n_total)
        # Make covariate correlated with outcome
        values = df["value"].values.astype(float)
        df["covariate"] = 0.5 * covariate_base + 0.3 * values + rng.normal(0, 0.5, n_total)

    return df.reset_index(drop=True)


def _generate_novelty_data(
    rng: np.random.Generator,
    baseline_rate: float,
    effect_size: float,
    n_control: int,
    n_treatment: int,
    novelty_days: int,
    novelty_multiplier: float,
) -> pd.DataFrame:
    """Generate data with novelty effect — inflated early treatment effect."""
    total_days = 14
    # Assign days uniformly
    ctrl_days = rng.integers(1, total_days + 1, size=n_control)
    treat_days = rng.integers(1, total_days + 1, size=n_treatment)

    # Control: same rate every day
    control_vals = rng.binomial(1, baseline_rate, n_control)

    # Treatment: inflated effect during novelty period
    treatment_vals = np.zeros(n_treatment, dtype=int)
    early_mask = treat_days <= novelty_days
    late_mask = ~early_mask

    inflated_rate = min(baseline_rate + effect_size * novelty_multiplier, 0.99)
    normal_rate = np.clip(baseline_rate + effect_size, 0.0, 1.0)

    treatment_vals[early_mask] = rng.binomial(1, inflated_rate, early_mask.sum())
    treatment_vals[late_mask] = rng.binomial(1, normal_rate, late_mask.sum())

    df = pd.DataFrame({
        "group": ["control"] * n_control + ["treatment"] * n_treatment,
        "value": np.concatenate([control_vals, treatment_vals]),
        "day": np.concatenate([ctrl_days, treat_days]),
    })
    return df


def _generate_simpsons_data(
    rng: np.random.Generator,
    baseline_rate: float,
    effect_size: float,
    n_control: int,
    n_treatment: int,
) -> pd.DataFrame:
    """Generate data with Simpson's Paradox.

    Creates two segments with engineered imbalance:
    - Segment A (large): treatment is WORSE than control
    - Segment B (small control, large treatment): treatment is MUCH BETTER
    The imbalance causes aggregate to show positive effect while
    at least one segment shows negative.
    """
    # Segment A: large groups, treatment worse
    n_a_ctrl = int(n_control * 0.8)
    n_a_treat = int(n_treatment * 0.4)  # Fewer treatments in segment A

    # Segment B: small control, large treatment, treatment much better
    n_b_ctrl = n_control - n_a_ctrl
    n_b_treat = n_treatment - n_a_treat

    # Rates engineered for sign flip
    rate_a_ctrl = baseline_rate + effect_size  # Higher baseline in A
    rate_a_treat = baseline_rate  # Treatment WORSE in A

    rate_b_ctrl = baseline_rate * 0.5  # Low baseline in B
    rate_b_treat = baseline_rate + effect_size * 3  # Much better treatment in B

    # Clamp rates
    rate_a_ctrl = np.clip(rate_a_ctrl, 0.01, 0.99)
    rate_a_treat = np.clip(rate_a_treat, 0.01, 0.99)
    rate_b_ctrl = np.clip(rate_b_ctrl, 0.01, 0.99)
    rate_b_treat = np.clip(rate_b_treat, 0.01, 0.99)

    groups = []
    values = []
    segments = []

    for seg, n_c, n_t, r_c, r_t in [
        ("A", n_a_ctrl, n_a_treat, rate_a_ctrl, rate_a_treat),
        ("B", n_b_ctrl, n_b_treat, rate_b_ctrl, rate_b_treat),
    ]:
        ctrl_v = rng.binomial(1, r_c, n_c)
        treat_v = rng.binomial(1, r_t, n_t)
        groups.extend(["control"] * n_c + ["treatment"] * n_t)
        values.extend(ctrl_v.tolist() + treat_v.tolist())
        segments.extend([seg] * (n_c + n_t))

    return pd.DataFrame({"group": groups, "value": values, "segment": segments})

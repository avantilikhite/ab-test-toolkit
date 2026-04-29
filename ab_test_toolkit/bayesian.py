"""Bayesian A/B testing analysis.

Two models are exposed:

* ``beta_binomial`` — Beta-Binomial conjugate update for proportions.
  Uses the standard Beta(α, β) prior and returns the analytic posterior.

* ``normal_normal`` / ``normal_normal_from_stats`` — analysis of two
  means with **unknown variances** under the standard non-informative
  reference prior ``p(μ, σ²) ∝ 1/σ²``.  The marginal posterior for each
  group's mean is **Student-t** with ``df = n − 1``.  This is *not* a
  Normal-Normal conjugate model with a known variance; the older
  Normal-Normal name is retained for API stability but the implemented
  model propagates variance uncertainty correctly and converges to a
  Normal as ``n`` grows.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
from scipy import stats


@dataclass
class BayesianResult:
    """Result container for Bayesian A/B test analysis."""

    model_type: str
    prob_b_greater_a: float
    expected_loss: float
    control_posterior: Dict[str, float]
    treatment_posterior: Dict[str, float]
    credible_interval: Tuple[float, float]
    prior_config: Dict[str, Any] = field(default_factory=dict)


def beta_binomial(
    control: np.ndarray,
    treatment: np.ndarray,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_simulations: int = 100_000,
    random_state: int | None = 0,
) -> BayesianResult:
    """Beta-Binomial conjugate analysis from raw binary arrays."""
    control = np.asarray(control)
    treatment = np.asarray(treatment)
    if control.size == 0 or treatment.size == 0:
        raise ValueError("beta_binomial requires non-empty arrays.")
    if not (np.all(np.isfinite(control)) and np.all(np.isfinite(treatment))):
        raise ValueError("beta_binomial inputs must be finite (no NaN/Inf).")
    unique_vals = np.unique(np.concatenate([control, treatment]))
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(
            "beta_binomial expects binary 0/1 outcomes. For continuous metrics use normal_normal()."
        )
    return beta_binomial_from_stats(
        control_count=int(control.sum()),
        control_total=len(control),
        treatment_count=int(treatment.sum()),
        treatment_total=len(treatment),
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        n_simulations=n_simulations,
        random_state=random_state,
    )


def beta_binomial_from_stats(
    control_count: int,
    control_total: int,
    treatment_count: int,
    treatment_total: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_simulations: int = 100_000,
    random_state: int | None = 0,
) -> BayesianResult:
    """Beta-Binomial conjugate analysis from summary statistics."""
    if control_total <= 0 or treatment_total <= 0:
        raise ValueError("Totals must be positive integers.")
    if control_count < 0 or treatment_count < 0:
        raise ValueError("Counts must be non-negative.")
    if control_count > control_total or treatment_count > treatment_total:
        raise ValueError("Count cannot exceed total.")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("prior_alpha and prior_beta must be positive.")
    if n_simulations < 1000:
        raise ValueError("n_simulations must be at least 1000 for stable estimates.")

    c_alpha = prior_alpha + control_count
    c_beta = prior_beta + control_total - control_count
    t_alpha = prior_alpha + treatment_count
    t_beta = prior_beta + treatment_total - treatment_count

    # MC sampling for P(B>A), expected loss, and credible interval
    rng = np.random.default_rng(random_state)
    samples_a = rng.beta(c_alpha, c_beta, n_simulations)
    samples_b = rng.beta(t_alpha, t_beta, n_simulations)

    prob_b_gt_a = float((samples_b > samples_a).mean())
    expected_loss = float(np.maximum(samples_a - samples_b, 0).mean())

    diff = samples_b - samples_a
    ci = (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))

    return BayesianResult(
        model_type="beta_binomial",
        prob_b_greater_a=prob_b_gt_a,
        expected_loss=expected_loss,
        control_posterior={"alpha": float(c_alpha), "beta": float(c_beta)},
        treatment_posterior={"alpha": float(t_alpha), "beta": float(t_beta)},
        credible_interval=ci,
        prior_config={"alpha": prior_alpha, "beta": prior_beta},
    )


def normal_normal(
    control: np.ndarray,
    treatment: np.ndarray,
    prior_variance_multiplier: float = 100.0,
    n_simulations: int = 100_000,
    random_state: int | None = 0,
) -> BayesianResult:
    """Bayesian analysis of two means with unknown variances (Student-t posterior)."""
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    if control.size < 2 or treatment.size < 2:
        raise ValueError("normal_normal requires at least 2 observations per group.")
    if not (np.all(np.isfinite(control)) and np.all(np.isfinite(treatment))):
        raise ValueError("normal_normal inputs must be finite (no NaN/Inf).")

    return normal_normal_from_stats(
        control_mean=float(control.mean()),
        control_std=float(control.std(ddof=1)),
        control_n=len(control),
        treatment_mean=float(treatment.mean()),
        treatment_std=float(treatment.std(ddof=1)),
        treatment_n=len(treatment),
        prior_variance_multiplier=prior_variance_multiplier,
        n_simulations=n_simulations,
        random_state=random_state,
    )


def normal_normal_from_stats(
    control_mean: float,
    control_std: float,
    control_n: int,
    treatment_mean: float,
    treatment_std: float,
    treatment_n: int,
    prior_variance_multiplier: float = 100.0,
    n_simulations: int = 100_000,
    random_state: int | None = 0,
) -> BayesianResult:
    """Bayesian analysis of two means with **unknown variances**.

    Posterior approach:
        Under the standard non-informative reference prior
        ``p(mu, sigma^2) ∝ 1/sigma^2``, the marginal posterior for each group's
        mean is Student-t with ``df = n - 1``, location = sample mean,
        and scale = ``s / sqrt(n)`` where ``s`` is the sample standard
        deviation. We sample independently from each group's posterior and
        combine them to get ``P(B > A)``, expected loss, and a credible
        interval for the mean difference.

    Why this differs from the previous implementation:
        The earlier model treated the standard deviations as known and
        centred a wide prior on the pooled sample mean.  Both choices
        understated uncertainty for small / moderate ``n``.  The Student-t
        posterior propagates variance uncertainty correctly and converges
        to the Normal as ``n`` grows, so large-N behaviour is unchanged.

    Parameters
    ----------
    prior_variance_multiplier
        Retained for backward compatibility.  Ignored under the reference
        prior; included so callers and `BayesianResult.prior_config` keep a
        stable schema.
    """
    n_c, n_t = control_n, treatment_n
    if n_c < 2 or n_t < 2:
        raise ValueError("normal_normal_from_stats requires n >= 2 per group.")
    if control_std < 0 or treatment_std < 0:
        raise ValueError("Standard deviations must be non-negative.")
    if n_simulations < 1000:
        raise ValueError("n_simulations must be at least 1000 for stable posterior estimates.")

    x_bar_c, x_bar_t = control_mean, treatment_mean
    s_c, s_t = max(control_std, 1e-12), max(treatment_std, 1e-12)

    # Zero-variance fast path
    if control_std == 0 and treatment_std == 0:
        diff = x_bar_t - x_bar_c
        return BayesianResult(
            model_type="normal_normal",
            prob_b_greater_a=1.0 if diff > 0 else (0.0 if diff < 0 else 0.5),
            expected_loss=max(-diff, 0.0),
            control_posterior={"mean": x_bar_c, "scale": 0.0, "df": float(n_c - 1)},
            treatment_posterior={"mean": x_bar_t, "scale": 0.0, "df": float(n_t - 1)},
            credible_interval=(diff, diff),
            prior_config={"reference_prior": "p(mu, sigma^2) ∝ 1/sigma^2", "warnings": ["Both groups zero variance."]},
        )

    df_c, df_t = n_c - 1, n_t - 1
    scale_c, scale_t = s_c / np.sqrt(n_c), s_t / np.sqrt(n_t)

    rng = np.random.default_rng(random_state)
    samples_c = stats.t.rvs(df_c, loc=x_bar_c, scale=scale_c, size=n_simulations, random_state=rng)
    samples_t = stats.t.rvs(df_t, loc=x_bar_t, scale=scale_t, size=n_simulations, random_state=rng)

    diff_samples = samples_t - samples_c
    prob_b_gt_a = float((diff_samples > 0).mean())
    expected_loss = float(np.maximum(samples_c - samples_t, 0).mean())
    ci = (float(np.percentile(diff_samples, 2.5)), float(np.percentile(diff_samples, 97.5)))

    warnings = []
    if n_c < 30 or n_t < 30:
        warnings.append(
            f"Small sample size (n_c={n_c}, n_t={n_t}): t-posterior used. "
            "Heavy tails widen the credible interval relative to a Normal approximation."
        )

    return BayesianResult(
        model_type="normal_normal",
        prob_b_greater_a=prob_b_gt_a,
        expected_loss=expected_loss,
        control_posterior={"mean": float(x_bar_c), "scale": float(scale_c), "df": float(df_c)},
        treatment_posterior={"mean": float(x_bar_t), "scale": float(scale_t), "df": float(df_t)},
        credible_interval=ci,
        prior_config={
            "reference_prior": "p(mu, sigma^2) ∝ 1/sigma^2",
            "posterior": "Student-t marginal for each group's mean",
            "prior_variance_multiplier_ignored": prior_variance_multiplier,
            "warnings": warnings,
        },
    )

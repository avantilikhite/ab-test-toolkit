# Data Model: AB Test Toolkit

**Feature**: 001-ab-test-toolkit | **Date**: 2026-04-10

## Overview

The AB Test Toolkit operates on in-memory data structures. There is no persistent storage — all data enters via CSV upload or manual entry and is processed in-memory. The data model defines the entities, their fields, validation rules, and relationships.

---

## Entity: ExperimentData

The primary input container representing raw experiment observations.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `group` | `pd.Series[str]` | Yes | Group label: `"control"` or `"treatment"` (case-insensitive) |
| `value` | `pd.Series[float]` | Yes | 0/1 for proportions, continuous for revenue/time metrics |
| `segment` | `pd.Series[str]` | No | Categorical segment label (e.g., "mobile", "desktop") |
| `covariate` | `pd.Series[float]` | No | Pre-experiment covariate value for CUPED |

**Representation**: `pandas.DataFrame` with columns matching the field names above.

**Validation rules**:
- `group` column must exist and contain exactly two unique values that map to "control" and "treatment" (case-insensitive).
- `value` column must exist and be numeric (int or float). NaN values raise an error.
- If `segment` is present, it must be a string/categorical column. NaN values in segment are grouped as "unknown".
- If `covariate` is present, it must be numeric. NaN values raise an error.

**Metric type auto-detection**:
- If all values in `value` are 0 or 1 → `MetricType.PROPORTION`
- Otherwise → `MetricType.CONTINUOUS`

---

## Entity: SummaryStats

An alternative input path: pre-aggregated summary statistics instead of raw data.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `control_count` | `int` | Yes (proportion) | Number of conversions in control |
| `control_total` | `int` | Yes (proportion) | Total observations in control |
| `treatment_count` | `int` | Yes (proportion) | Number of conversions in treatment |
| `treatment_total` | `int` | Yes (proportion) | Total observations in treatment |
| `control_mean` | `float` | Yes (continuous) | Mean of control group |
| `control_std` | `float` | Yes (continuous) | Standard deviation of control |
| `control_n` | `int` | Yes (continuous) | Sample size of control |
| `treatment_mean` | `float` | Yes (continuous) | Mean of treatment group |
| `treatment_std` | `float` | Yes (continuous) | Standard deviation of treatment |
| `treatment_n` | `int` | Yes (continuous) | Sample size of treatment |

**Validation rules**:
- Counts must be non-negative integers.
- `control_count ≤ control_total`, same for treatment.
- Standard deviations must be non-negative.
- Sample sizes must be ≥ 1.

---

## Entity: PowerConfig

Configuration for sample size / power analysis (FR-001).

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `baseline_rate` | `float` | Yes | — | Control group expected conversion rate (0, 1) |
| `mde` | `float` | Yes | — | Minimum detectable effect (absolute percentage points) |
| `mde_mode` | `str` | No | `"absolute"` | `"absolute"` or `"relative"`. If relative, converted internally: `mde_abs = baseline × mde_relative` |
| `alpha` | `float` | No | `0.05` | Significance level |
| `power` | `float` | No | `0.80` | Desired statistical power |
| `allocation_ratio` | `float` | No | `1.0` | Ratio n_treatment / n_control |
| `daily_traffic` | `int` | No | `None` | Daily user volume for duration estimation |

**Validation rules**:
- `baseline_rate` ∈ (0, 1) exclusive.
- `mde` > 0.
- `alpha` ∈ (0, 1) exclusive.
- `power` ∈ (0, 1) exclusive.
- `allocation_ratio` > 0.
- If `mde_mode == "relative"`, result rate `baseline + baseline × mde` must be ∈ (0, 1).

---

## Entity: PowerResult

Output of sample size / power calculation.

| Field | Type | Description |
|-------|------|-------------|
| `n_control` | `int` | Required sample size for control group |
| `n_treatment` | `int` | Required sample size for treatment group |
| `n_total` | `int` | Total required sample size |
| `n_effective` | `float` | Effective sample size accounting for imbalance |
| `sample_inflation_pct` | `float` | Percentage extra sample required vs a balanced (50/50) design. 0 when allocation is balanced. Available as `power_loss_pct` for backward compatibility. |
| `estimated_days` | `int \| None` | Estimated test duration (if daily_traffic provided) |

---

## Entity: FrequentistResult

Output of Z-test (proportions) or Welch's t-test (continuous) (FR-002, FR-003).

| Field | Type | Description |
|-------|------|-------------|
| `test_type` | `str` | `"z_test"` or `"welch_t_test"` |
| `statistic` | `float` | Z-statistic or t-statistic |
| `p_value` | `float` | Two-tailed p-value |
| `ci_lower` | `float` | Lower bound of confidence interval for the difference |
| `ci_upper` | `float` | Upper bound of confidence interval for the difference |
| `point_estimate` | `float` | Observed difference (treatment - control) |
| `effect_size` | `float` | Cohen's h (proportions) or Cohen's d (continuous) |
| `alpha` | `float` | Significance level used |
| `is_significant` | `bool` | `p_value < alpha` |
| `normality_check` | `dict \| None` | Shapiro-Wilk result for continuous data (FR-003a) |

---

## Entity: BayesianResult

Output of Bayesian analysis (FR-004, FR-005).

| Field | Type | Description |
|-------|------|-------------|
| `model_type` | `str` | `"beta_binomial"` or `"normal_normal"` |
| `prob_b_greater_a` | `float` | P(treatment > control) |
| `expected_loss` | `float` | Expected loss from choosing the wrong variant |
| `control_posterior` | `dict` | Posterior parameters (e.g., `{"alpha": 101, "beta": 901}`) |
| `treatment_posterior` | `dict` | Posterior parameters |
| `credible_interval` | `tuple[float, float]` | 95% credible interval for the difference |
| `prior_config` | `dict` | Prior hyperparameters used |

---

## Entity: SRMResult

Output of Sample Ratio Mismatch check (FR-006).

| Field | Type | Description |
|-------|------|-------------|
| `expected_ratio` | `tuple[float, float]` | Expected allocation (e.g., (0.5, 0.5)) |
| `observed_ratio` | `tuple[float, float]` | Observed allocation |
| `chi2_statistic` | `float` | Chi-square test statistic |
| `p_value` | `float` | P-value of the chi-square test |
| `has_mismatch` | `bool` | `p_value < 0.01` (strict threshold for SRM) |

---

## Entity: CUPEDResult

Output of CUPED variance reduction (FR-007).

| Field | Type | Description |
|-------|------|-------------|
| `theta` | `float` | Estimated regression coefficient |
| `correlation` | `float` | Correlation between covariate and outcome (ρ) |
| `variance_reduction_pct` | `float` | Percentage variance reduction achieved |
| `adjusted_estimate` | `float` | CUPED-adjusted treatment effect |
| `adjusted_ci` | `tuple[float, float]` | Adjusted confidence interval |
| `unadjusted_estimate` | `float` | Original treatment effect (for comparison) |
| `unadjusted_ci` | `tuple[float, float]` | Original confidence interval (for comparison) |

---

## Entity: SegmentResult

Output of per-segment HTE analysis (FR-008).

| Field | Type | Description |
|-------|------|-------------|
| `aggregate_estimate` | `float` | Overall treatment effect |
| `aggregate_ci` | `tuple[float, float]` | Overall confidence interval |
| `segment_results` | `list[dict]` | Per-segment: `{"segment": str, "estimate": float, "ci": tuple, "n": int, "p_value": float}` |
| `simpsons_paradox` | `bool` | True if sign flip detected between aggregate and any segment |
| `simpsons_details` | `str \| None` | Description of the sign flip if detected |
| `n_segments` | `int` | Number of segments tested |
| `multiple_comparisons_note` | `str` | Disclaimer about unadjusted p-values |

---

## Entity: Recommendation

Output of the executive summary engine (FR-009).

| Field | Type | Description |
|-------|------|-------------|
| `recommendation` | `str` | `"Ship"`, `"No-Ship"`, or `"Inconclusive"` |
| `flags` | `list[str]` | Diagnostic warnings (e.g., Twyman's Law for >30% relative lift) |
| `supporting_metrics` | `dict` | `{"significance": bool, "p_value": float, "effect_size": float, "srm_status": str, "prob_b_gt_a": float}` |

**State transitions** (decision logic):
```
IF srm.has_mismatch:
    → "Inconclusive" + flag "Sample Ratio Mismatch detected"
ELIF simpsons_paradox:
    → "Inconclusive" + flag "Simpson's Paradox detected"
ELIF frequentist.is_significant AND bayesian.prob_b_greater_a > 0.95 AND point_estimate > 0:
    → "Ship"
ELIF frequentist.is_significant AND point_estimate < 0:
    → "No-Ship" (significant negative effect)
ELSE:
    → "Inconclusive"

ALWAYS CHECK: if relative_lift > 30%:
    → add flag "Twyman's Law: effect suspiciously large (>30% relative lift)"
```

---

## Entity: DataGeneratorConfig

Configuration for synthetic data generation (FR-010).

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `baseline_rate` | `float` | Yes | — | Control conversion rate |
| `effect_size` | `float` | Yes | — | True treatment effect (absolute) |
| `n_control` | `int` | Yes | — | Control group sample size |
| `n_treatment` | `int` | Yes | — | Treatment group sample size |
| `inject_novelty` | `bool` | No | `False` | Add novelty effect (inflated early lift) |
| `novelty_days` | `int` | No | `3` | Number of days with novelty effect |
| `novelty_multiplier` | `float` | No | `2.0` | Effect multiplier during novelty period |
| `inject_simpsons` | `bool` | No | `False` | Engineer Simpson's Paradox into segments |
| `inject_srm` | `bool` | No | `False` | Inject sample ratio mismatch |
| `srm_actual_ratio` | `float` | No | `0.55` | Actual allocation ratio when SRM injected |
| `random_seed` | `int` | No | `42` | Seed for reproducibility |

---

## Enum: MetricType

```python
class MetricType(str, Enum):
    PROPORTION = "proportion"
    CONTINUOUS = "continuous"
```

---

## Relationships

```
ExperimentData ──1:1──▶ MetricType (auto-detected)
ExperimentData ──1:N──▶ FrequentistResult (one per metric type)
ExperimentData ──1:N──▶ BayesianResult (one per metric type)
ExperimentData ──1:1──▶ SRMResult
ExperimentData ──1:1──▶ CUPEDResult (if covariate present)
ExperimentData ──1:N──▶ SegmentResult (if segment present)
FrequentistResult ─┐
BayesianResult ────┤
SRMResult ─────────┼──▶ Recommendation
SegmentResult ─────┘
DataGeneratorConfig ──1:1──▶ ExperimentData (generates)
PowerConfig ──1:1──▶ PowerResult
```

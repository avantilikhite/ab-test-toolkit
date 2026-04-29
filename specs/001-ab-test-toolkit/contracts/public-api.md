# Public API Contract: ab_test_toolkit

**Feature**: 001-ab-test-toolkit | **Date**: 2026-04-10
**Contract Type**: Python library public API

This document defines the public interface of the `ab_test_toolkit` package. All functions listed here are importable from the top-level package or their respective modules. Signatures use Python type annotations (NumPy-style docstrings in implementation).

---

## Module: `ab_test_toolkit.power`

### `required_sample_size`

```python
def required_sample_size(
    baseline_rate: float,
    mde: float,
    *,
    mde_mode: Literal["absolute", "relative"] = "absolute",
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0,
) -> PowerResult:
    """
    Calculate required sample size per group for a two-proportion Z-test.

    Parameters
    ----------
    baseline_rate : float
        Expected conversion rate of the control group, in (0, 1).
    mde : float
        Minimum detectable effect. Interpretation depends on mde_mode.
    mde_mode : {"absolute", "relative"}, default "absolute"
        If "relative", mde is converted to absolute: baseline_rate × mde.
    alpha : float, default 0.05
        Significance level (two-tailed).
    power : float, default 0.80
        Desired statistical power.
    allocation_ratio : float, default 1.0
        Ratio of treatment to control group size (n_treatment / n_control).

    Returns
    -------
    PowerResult
        Named result with n_control, n_treatment, n_total, n_effective,
        sample_inflation_pct, estimated_days. (`power_loss_pct` is retained as a deprecated alias.)

    Raises
    ------
    ValueError
        If baseline_rate not in (0,1), mde <= 0, or resulting rate not in (0,1).
    """
```

### `power_curve`

```python
def power_curve(
    baseline_rate: float,
    mde_range: tuple[float, float],
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation_ratio: float = 1.0,
    n_points: int = 50,
) -> pd.DataFrame:
    """
    Generate MDE-vs-N data for plotting power curves.

    Returns DataFrame with columns: mde, n_control, n_treatment, n_total.
    """
```

---

## Module: `ab_test_toolkit.frequentist`

### `z_test`

```python
def z_test(
    control: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    *,
    alpha: float = 0.05,
) -> FrequentistResult:
    """
    Two-proportion Z-test with unpooled standard error.

    Reports p-value, confidence interval for the difference, and Cohen's h.

    Parameters
    ----------
    control : array-like
        Binary (0/1) outcomes for the control group.
    treatment : array-like
        Binary (0/1) outcomes for the treatment group.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    FrequentistResult
    """
```

### `z_test_from_stats`

```python
def z_test_from_stats(
    control_count: int,
    control_total: int,
    treatment_count: int,
    treatment_total: int,
    *,
    alpha: float = 0.05,
) -> FrequentistResult:
    """Z-test from pre-aggregated summary statistics."""
```

### `welch_t_test`

```python
def welch_t_test(
    control: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    *,
    alpha: float = 0.05,
) -> FrequentistResult:
    """
    Welch's t-test for continuous metrics (unequal variance).

    Includes Shapiro-Wilk normality check (FR-003a).
    Reports p-value, CI, and Cohen's d.
    """
```

### `welch_t_test_from_stats`

```python
def welch_t_test_from_stats(
    control_mean: float,
    control_std: float,
    control_n: int,
    treatment_mean: float,
    treatment_std: float,
    treatment_n: int,
    *,
    alpha: float = 0.05,
) -> FrequentistResult:
    """Welch's t-test from pre-aggregated summary statistics."""
```

---

## Module: `ab_test_toolkit.bayesian`

### `beta_binomial`

```python
def beta_binomial(
    control: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_simulations: int = 1_000_000,
) -> BayesianResult:
    """
    Bayesian analysis for proportions using Beta-Binomial conjugate model.

    Returns P(B > A) via numerical integration and Expected Loss via
    Monte Carlo simulation.

    Parameters
    ----------
    prior_alpha, prior_beta : float
        Beta prior hyperparameters. Default Beta(1,1) = uniform.
    n_simulations : int
        Number of Monte Carlo samples for Expected Loss calculation.
    """
```

### `beta_binomial_from_stats`

```python
def beta_binomial_from_stats(
    control_count: int,
    control_total: int,
    treatment_count: int,
    treatment_total: int,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_simulations: int = 1_000_000,
) -> BayesianResult:
    """Beta-Binomial analysis from summary statistics."""
```

### `normal_normal`

```python
def normal_normal(
    control: np.ndarray | pd.Series,
    treatment: np.ndarray | pd.Series,
    *,
    prior_mean: float | None = None,
    prior_variance_multiplier: float = 10.0,
    n_simulations: int = 1_000_000,
) -> BayesianResult:
    """
    Bayesian analysis for continuous metrics using Normal-Normal conjugate.

    Uses sample variance treated as known. Warns when N < 100 per group.
    Default prior: mean = pooled sample mean, variance = multiplier × pooled variance.
    """
```

---

## Module: `ab_test_toolkit.srm`

### `check_srm`

```python
def check_srm(
    observed: tuple[int, int],
    expected_ratio: tuple[float, float] = (0.5, 0.5),
    *,
    threshold: float = 0.01,
) -> SRMResult:
    """
    Chi-square test for Sample Ratio Mismatch.

    Parameters
    ----------
    observed : tuple of int
        Observed counts per group (n_control, n_treatment).
    expected_ratio : tuple of float
        Expected allocation ratio. Default 50/50.
    threshold : float
        P-value threshold for flagging mismatch. Default 0.01 (strict).
    """
```

---

## Module: `ab_test_toolkit.cuped`

### `cuped_adjust`

```python
def cuped_adjust(
    control_outcome: np.ndarray | pd.Series,
    treatment_outcome: np.ndarray | pd.Series,
    control_covariate: np.ndarray | pd.Series,
    treatment_covariate: np.ndarray | pd.Series,
    *,
    alpha: float = 0.05,
) -> CUPEDResult:
    """
    CUPED variance reduction with a single pre-experiment covariate.

    Theta estimated via pooled OLS. Handles unequal group sizes.
    Returns both adjusted and unadjusted estimates for comparison.
    """
```

---

## Module: `ab_test_toolkit.segmentation`

### `segment_analysis`

```python
def segment_analysis(
    data: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> SegmentResult:
    """
    Per-segment treatment effect analysis with Simpson's Paradox detection.

    Parameters
    ----------
    data : DataFrame
        Must contain 'group', 'value', and 'segment' columns.
    alpha : float
        Significance level for per-segment tests.

    Returns
    -------
    SegmentResult
        Includes aggregate + per-segment results and paradox flag.
        Per-segment p-values are unadjusted (disclaimer included).
    """
```

---

## Module: `ab_test_toolkit.recommendation`

### `generate_recommendation`

```python
def generate_recommendation(
    frequentist: FrequentistResult,
    bayesian: BayesianResult,
    srm: SRMResult,
    segmentation: SegmentResult | None = None,
    *,
    lift_warning_threshold: float = 0.30,
) -> Recommendation:
    """
    Generate Ship / No-Ship / Inconclusive recommendation.

    Applies Twyman's Law check for relative lift > threshold.
    """
```

---

## Module: `ab_test_toolkit.data_generator`

### `generate_experiment_data`

```python
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
    """
    Generate synthetic A/B test data with configurable anomalies.

    Each injection flag (novelty, Simpson's, SRM) is independent and composable.
    Returns DataFrame with columns: group, value, segment (if inject_simpsons),
    covariate (always included for CUPED testing), day (if inject_novelty).
    """
```

---

## Module: `ab_test_toolkit.visualization`

All functions return `plotly.graph_objects.Figure`.

### Chart Functions

```python
def ci_comparison_plot(results: FrequentistResult, ...) -> go.Figure
def posterior_plot(results: BayesianResult, ...) -> go.Figure
def mde_vs_n_curve(power_data: pd.DataFrame, ...) -> go.Figure
def cumulative_lift_chart(data: pd.DataFrame, ...) -> go.Figure
def daily_treatment_effect(data: pd.DataFrame, ...) -> go.Figure
def power_loss_curve(allocation_ratios: list[float], ...) -> go.Figure
def segment_comparison_chart(results: SegmentResult, ...) -> go.Figure
```

---

## Module: `ab_test_toolkit.io`

### `load_experiment_data`

```python
def load_experiment_data(
    source: str | Path | pd.DataFrame,
) -> tuple[pd.DataFrame, MetricType]:
    """
    Load and validate experiment data from CSV path or DataFrame.

    Auto-detects metric type from value column.
    Validates group labels are 'control' and 'treatment' (case-insensitive).
    Raises clear ValueError on invalid input (not stack traces).
    """
```

---

## CSV Input Schema

```
group,value[,segment][,covariate]
control,0
control,1
treatment,1
treatment,0,mobile,4.50
```

- `group`: Required. Values: "control", "treatment" (case-insensitive).
- `value`: Required. Numeric. Binary (0/1) for proportions, continuous otherwise.
- `segment`: Optional. String categorical.
- `covariate`: Optional. Numeric pre-experiment value.

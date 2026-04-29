# Research: AB Test Toolkit

**Feature**: 001-ab-test-toolkit | **Date**: 2026-04-10

## Research Summary

All NEEDS CLARIFICATION items resolved. No unknowns remain.

---

## R1: Power Analysis — Sample Size Formula with Unequal Allocation

**Decision**: Use the two-proportion Z-test power formula with explicit allocation ratio parameter.

**Rationale**: This is the standard formula used by scipy and all major experimentation platforms. It naturally handles unequal allocation via the `r` parameter without balanced-design shortcuts.

**Formula**:
```
n₁ = (z_α + z_β)² · [p₀(1-p₀)/1 + p₁(1-p₁)/r] / (p₁ - p₀)²

where:
  z_α = norm.ppf(1 - α/2)      # two-tailed critical value
  z_β = norm.ppf(power)         # power critical value
  r   = n₂/n₁ allocation ratio
  p₀  = baseline rate (control)
  p₁  = baseline + MDE (treatment)
```

**Effective sample size for imbalance visualization**:
```
n_eff = 4·n₁·n₂ / (n₁ + n₂)
```

| Allocation | Power Loss |
|------------|-----------|
| 1:1 (50/50) | 0% |
| 1:2 (67/33) | 11% |
| 1:3 (75/25) | 25% |
| 1:4 (80/20) | 36% |

**Alternatives considered**:
- `statsmodels.stats.power`: Rejected — adds a heavy dependency for one calculation we can implement in ~30 lines with scipy.stats.norm.
- Simulation-based power: Rejected — slow for interactive Streamlit use; analytical formula computes in <1ms.

**Implementation**: `ab_test_toolkit/power.py` using `scipy.stats.norm.ppf()`.

---

## R2: Frequentist Inference — Z-Test and Welch's T-Test

**Decision**: Implement Z-test for proportions with unpooled SE; Welch's t-test for continuous metrics. Validate against scipy.stats reference implementations.

**Rationale**: Unpooled SE is more conservative and correct when treatment/control rates differ. Welch's t-test handles unequal variance without assumption violations.

**Z-test (proportions)**:
```
SE = sqrt(p̂₁(1-p̂₁)/n₁ + p̂₂(1-p̂₂)/n₂)    # unpooled
z  = (p̂₂ - p̂₁) / SE
p  = 2 · (1 - norm.cdf(|z|))
Cohen's h = 2·arcsin(√p₂) - 2·arcsin(√p₁)
```

**Welch's t-test (continuous)**:
```
SE = sqrt(s₁²/n₁ + s₂²/n₂)
t  = (x̄₂ - x̄₁) / SE
df = Welch-Satterthwaite approximation
Cohen's d = (x̄₂ - x̄₁) / s_pooled
```

**Normality check**: Shapiro-Wilk via `scipy.stats.shapiro()`. Warn if N < 30 and p < 0.05; note CLT applicability if N ≥ 30.

**Alternatives considered**:
- Pooled SE for Z-test: Rejected — assumes equal proportions under H₀, which is a stronger assumption.
- Mann-Whitney U as default: Rejected — less interpretable for product teams; include as a note in documentation only.

**Validation**: Compare against `scipy.stats.proportions_ztest` and `scipy.stats.ttest_ind(equal_var=False)` with tolerance ≤ 1e-6.

---

## R3: Bayesian Inference — Conjugate Priors Only

**Decision**: Beta-Binomial for proportions, Normal-Normal for continuous metrics. No MCMC.

**Rationale**: Conjugate priors give closed-form posteriors. Fast enough for interactive Streamlit use. Satisfies constitution principle I (Statistical Rigor) and IV (Minimal Dependencies).

### Beta-Binomial (proportions)

```
Prior:     p ~ Beta(α₀, β₀)         # default: Beta(1,1) uniform
Posterior: p | data ~ Beta(α₀ + x, β₀ + n - x)
```

**P(B > A)**: Numerical integration using `scipy.special.betainc`:
```python
p_vals = np.linspace(0.001, 0.999, 10000)
prob_A_less_p = betainc(α_A, β_A, p_vals)
pdf_B = beta.pdf(p_vals, α_B, β_B)
P_B_gt_A = np.trapz(prob_A_less_p * pdf_B, p_vals)
```

**Expected Loss**: Monte Carlo (1M samples) — simple, accurate, and validates against numerical integration.

### Normal-Normal (continuous)

```
Prior:     μ ~ N(μ₀, σ₀²)
Posterior: μ | data ~ N(μ_post, σ²_post)

σ²_post = 1 / (1/σ₀² + n/σ²)
μ_post  = σ²_post · (μ₀/σ₀² + n·x̄/σ²)
```

**Default prior**: Weakly informative — μ₀ = pooled sample mean, σ₀² = 10 × pooled sample variance.
**Warn when N < 100** per group (sample variance as known approximation less reliable).

**Alternatives considered**:
- PyMC / Stan: Rejected per constitution (Principle IV). Unnecessary for conjugate models.
- Jeffreys prior Beta(0.5, 0.5): Considered but Beta(1,1) is more intuitive for non-statisticians and specified in FR-004.
- Grid approximation for P(B > A): Rejected — numerical integration is faster and more accurate for 1D.

---

## R4: CUPED Variance Reduction

**Decision**: Implement CUPED with pooled OLS theta estimation, handling unequal group sizes natively.

**Rationale**: CUPED is a production differentiator. The pooled OLS approach is the standard method used at Microsoft (Deng et al., 2013) and handles unequal groups correctly without per-group estimation.

**Formula**:
```
Y_adj = Y - θ̂ · (X - X̄)
θ̂ = Cov(Y, X) / Var(X)           # pooled across all observations
Var(Y_adj) = Var(Y) · (1 - ρ²)   # where ρ = Corr(Y, X)
```

**Critical**: θ must be estimated from pooled data (all observations), NOT per-group. Per-group estimation introduces bias with unequal N.

**Expected variance reduction**:
| Covariate Correlation (ρ) | Variance Reduction | CI Width Reduction |
|--------------------------|-------------------|--------------------|
| 0.3 | 9% | ~5% |
| 0.5 | 25% | ~13% |
| 0.7 | 51% | ~29% |
| 0.8 | 64% | ~40% |

**Alternatives considered**:
- Per-group theta: Rejected — introduces bias with unequal N.
- Multiple covariates: Rejected per FR-007 (single covariate). Extensible later.
- Regression adjustment (Lin, 2013): More powerful but adds complexity beyond spec scope.

---

## R5: Visualization Strategy — Plotly Everywhere

**Decision**: Use Plotly for all 7 visualization types. No matplotlib.

**Rationale**: The spec originally mentioned matplotlib for notebooks, but Plotly works in both Streamlit (`st.plotly_chart`) and Jupyter (`fig.show()`) with identical API. Unifying on Plotly eliminates dual rendering code and provides interactive charts everywhere.

**Pattern**: Every visualization function returns a `plotly.graph_objects.Figure` object with sensible defaults (height=400px, template="plotly_white"). Consumers control rendering context.

**7 required chart types** (FR-013):
1. Cumulative lift chart
2. CI comparison plot
3. Bayesian posterior distributions
4. MDE-vs-N curve (power)
5. Time-series daily treatment effect
6. Power loss vs. allocation ratio curve
7. Per-segment comparison chart

**Alternatives considered**:
- matplotlib for notebooks + Plotly for Streamlit: Rejected — doubles maintenance, no interactivity in notebooks.
- Altair: Rejected — less control over statistical chart customization; Plotly has better support for error bars and distributions.

---

## R6: Package Structure and Build System

**Decision**: Flat package at repo root (`ab_test_toolkit/`), Streamlit app in `app/`, pyproject.toml with optional dependency groups, uv for environment management.

**Rationale**: Flat structure is the standard for scientific Python packages with ≤15 modules. Optional dependency groups keep the core library lightweight (3 deps) while allowing Streamlit consumers to install app dependencies.

**Dependency groups**:
- Core: `numpy>=1.24`, `scipy>=1.10`, `pandas>=2.0`
- `[viz]`: `plotly>=5.10`
- `[app]`: `streamlit>=1.28`, `plotly>=5.10`
- `[notebook]`: `jupyter`, `nbconvert`
- `[dev]`: `pytest>=7.4`, `pytest-cov>=4.1`, `hypothesis>=6.80`, `ruff>=0.10`

**Alternatives considered**:
- src/ layout: Rejected — adds import complexity without benefit for a single-package project.
- Monorepo with separate packages: Rejected — one package with optional deps is simpler and sufficient.

---

## R7: Streamlit Multi-Page App Architecture

**Decision**: Native Streamlit multi-page app with `pages/` auto-discovery, global settings in sidebar via `st.session_state`, initialization in `app.py`.

**Rationale**: Streamlit 1.12+ natively supports multi-page apps. Auto-discovery via numbered prefixes (`01_`, `02_`) gives deterministic page ordering.

**Global settings pattern**: Alpha/confidence level selector in sidebar of `app.py`, propagated via `session_state` to all pages. Single initialization function prevents flickering.

**4 pages** per FR-011:
1. Experiment Design (power calculator)
2. Analyze Results (CSV upload → full pipeline)
3. Sensitivity Analysis (post-experiment MDE detection)
4. Case Study Demo (pre-loaded walkthrough)

**Alternatives considered**:
- Manual page routing: Rejected — reinvents built-in Streamlit navigation.
- Dash / Panel: Rejected — Streamlit is specified in constitution and has lowest friction for portfolio demos.

---

## R8: Testing Strategy

**Decision**: pytest for unit/integration tests, hypothesis for property-based testing, nbconvert for notebook validation. Target ≥90% coverage.

**Rationale**: Constitution Principle III mandates TDD. pytest is the standard. hypothesis catches edge cases that hand-written tests miss (e.g., extreme allocation ratios, boundary conversion rates).

**Testing layers**:
1. **Unit tests** (`tests/unit/`): Each module tested independently against scipy reference values or hand-calculated results. Tolerance ≤ 1e-6.
2. **Property-based tests** (hypothesis): Invariants like p-values ∈ [0,1], CI contains point estimate, posterior narrows with more data.
3. **Integration tests** (`tests/integration/`): Full pipeline (data → analysis → recommendation). Notebook execution via nbconvert.

**Fixture strategy**: Shared `conftest.py` with parametrized fixtures for varying sample sizes, allocation ratios, and effect sizes.

**Alternatives considered**:
- unittest: Rejected — pytest is more expressive and is the constitution standard.
- Manual edge case enumeration: Supplemented (not replaced) by hypothesis.

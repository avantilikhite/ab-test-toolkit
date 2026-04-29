# Feature Specification: AB Test Toolkit

**Feature Branch**: `001-ab-test-toolkit`
**Created**: 2026-04-10
**Status**: Draft
**Input**: Production-grade Python toolkit for designing, auditing, and analyzing A/B experiments

## Delivery Artifacts

| Artifact | Purpose |
|---|---|
| **Python package** (`ab_test_toolkit/`) | Clean, importable statistical engine |
| **Streamlit app** | Interactive demo — power calculator, upload results, get recommendation |
| **Jupyter notebook** | End-to-end case study (e-commerce or real estate domain) as a narrative walkthrough |

## User Scenarios & Testing

### User Story 1 — Power Analysis & Experiment Design (Priority: P1)

A data scientist planning a new A/B test needs to determine the required sample size and expected test duration before launch. They input baseline conversion rate, minimum detectable effect (expressed as either absolute percentage points or relative percentage of baseline), desired power, significance level, allocation ratio, and daily traffic volume.

**Why this priority**: Experiment design happens before any test runs. If sample size is wrong, the entire experiment is wasted. This is the entry point for every user.

**Independent Test**: Can be tested by providing known statistical inputs and verifying the output matches scipy.stats reference calculations.

**Acceptance Scenarios**:

1. **Given** a baseline conversion of 10%, MDE of 2%, power of 0.8, alpha of 0.05, and 1:1 allocation, **When** the user runs the power calculator, **Then** the system returns the correct required N (validated against scipy).
2. **Given** an 80/20 allocation ratio, **When** the power calculator computes required N per group, **Then** it uses the standard power formula with the allocation ratio plugged into the variance term (not a balanced-design shortcut). The power loss visualization uses $n_{eff} = \frac{4 \cdot n_1 \cdot n_2}{n_1 + n_2}$ to illustrate the cost of imbalance, and the power loss curve is displayed.
3. **Given** a daily traffic volume of 5,000 users, **When** the user adjusts MDE from 2% to 5%, **Then** the estimated test duration updates in real time in the Streamlit app.
4. **Given** a baseline of 20% and MDE mode set to "relative 5%", **When** the calculator runs, **Then** it converts to an absolute MDE of 1 percentage point (20% × 0.05) and returns the same N as entering 1% absolute.

---

### User Story 2 — Frequentist Analysis of Experiment Results (Priority: P1)

An analyst has collected A/B test results (two groups, with counts or continuous metric values) and needs to determine statistical significance via frequentist methods.

**Why this priority**: Frequentist inference is the default language of most product teams. This is the core analysis path.

**Independent Test**: Run Z-test and Welch's t-test against known datasets with pre-calculated p-values and CIs.

**Acceptance Scenarios**:

1. **Given** two groups with conversion counts, **When** the Z-test is run with unpooled SE (Standard Error), **Then** the p-value and 95% CI match scipy.stats.proportions_ztest output.
2. **Given** two groups with continuous revenue data of unequal size, **When** Welch's t-test is run, **Then** the result correctly handles unequal N and unequal variance.
3. **Given** results, **When** CI overlap plot is generated, **Then** the plot shows both group CIs with clear labeling.
4. **Given** proportion data, **When** the Z-test completes, **Then** it also reports Cohen's h effect size. **Given** continuous data, **When** Welch's t-test completes, **Then** it also reports Cohen's d effect size.
5. **Given** continuous metric data, **When** the analysis pipeline runs, **Then** a Shapiro-Wilk normality check is performed first. If N < 30 and p < 0.05, a warning is issued; if N ≥ 30, a note confirms Central Limit Theorem applicability.
6. **Given** summary statistics (mean, std dev, count per group) entered manually instead of CSV, **When** the user runs analysis, **Then** the results are identical to uploading raw data with those statistics.

---

### User Story 3 — Bayesian Analysis of Experiment Results (Priority: P1)

An analyst wants a probabilistic interpretation: "What is the probability that B is better than A?" and "What is the expected loss if we choose the wrong variant?"

**Why this priority**: Bayesian results are increasingly expected at experiment-mature companies. Showing both paradigms side-by-side is a core differentiator.

**Independent Test**: Beta-Binomial posterior with known priors and observed data; verify P(B > A) via numerical integration against Monte Carlo simulation.

**Acceptance Scenarios**:

1. **Given** conversion data for two variants, **When** the Bayesian engine runs, **Then** it returns P(B > A) and Expected Loss, validated against Monte Carlo simulation with tolerance ≤ 0.01.
2. **Given** unequal sample sizes (e.g., 80/20 split), **When** posteriors are plotted, **Then** the larger group has a visibly tighter distribution.
3. **Given** continuous metric data, **When** the Normal-Normal conjugate model runs, **Then** posterior summaries (mean, CI) are returned.

---

### User Story 4 — Sample Ratio Mismatch Detection (Priority: P1)

Before analyzing results, the analyst needs to verify that the assignment engine didn't introduce bias. They check whether observed allocation matches expected allocation.

**Why this priority**: If Sample Ratio Mismatch exists, all downstream results are unreliable. This is a prerequisite check.

**Independent Test**: Generate data with known allocation bias and verify the chi-square test detects it at the appropriate significance level.

**Acceptance Scenarios**:

1. **Given** 50/50 expected allocation and 51/49 observed with large N, **When** Sample Ratio Mismatch check runs, **Then** it flags a statistically significant mismatch.
2. **Given** 80/20 expected allocation and 80/20 observed, **When** Sample Ratio Mismatch check runs, **Then** it returns no mismatch.

---

### User Story 5 — CUPED Variance Reduction (Priority: P1)

An analyst has pre-experiment covariate data and wants to reduce metric variance to shorten the test or detect smaller effects.

**Why this priority**: CUPED is a production differentiator used at major tech companies. Most portfolio toolkits skip this entirely.

**Independent Test**: Compare CI width with and without CUPED on synthetic data with known covariate correlation.

**Acceptance Scenarios**:

1. **Given** pre-experiment covariate data correlated with the outcome, **When** CUPED adjustment is applied, **Then** the adjusted CI is narrower than the unadjusted CI.
2. **Given** unequal group sizes, **When** CUPED covariance estimation runs, **Then** it correctly handles per-group covariance without assuming equal N.

---

### User Story 6 — Segmentation & HTE Analysis (Priority: P1)

An analyst wants to understand how the treatment effect varies across user segments (e.g., platform, cohort, geography).

**Why this priority**: Heterogeneous treatment effects are critical for "Ship to whom?" decisions. Also the vehicle for demonstrating Simpson's Paradox awareness.

**Independent Test**: Generate synthetic data where the aggregate effect and per-segment effects have opposite signs (Simpson's Paradox). Verify the detector flags it.

**Acceptance Scenarios**:

1. **Given** experiment data with segment labels, **When** HTE analysis runs, **Then** per-segment point estimates + CIs are returned alongside the aggregate.
2. **Given** data where treatment wins aggregate but loses in every segment, **When** Simpson's Paradox detector runs, **Then** it flags the sign flip.

---

### User Story 7 — Automated Executive Summary (Priority: P2)

After running all checks, the analyst (or their stakeholder) wants a plain-English recommendation: Ship, No-Ship, or Inconclusive, with supporting evidence.

**Why this priority**: Communication is half the job. This demonstrates product sense, not just statistical skill.

**Independent Test**: Feed known result combinations (significant + no Sample Ratio Mismatch + no paradox → "Ship"; Sample Ratio Mismatch detected → "Inconclusive with warning") and verify output text.

**Acceptance Scenarios**:

1. **Given** a statistically significant positive result with no Sample Ratio Mismatch and no segment issues, **When** the summary generates, **Then** it returns "Ship" with supporting metrics.
2. **Given** a relative lift > 50% (the Twyman threshold), **When** the Twyman's Law check runs, **Then** it flags the result as suspiciously large.

---

### User Story 8 — Interactive Streamlit App (Priority: P2)

A hiring manager or interviewer opens the Streamlit app and explores the toolkit interactively without reading code.

**Why this priority**: This is the demo artifact. It must be polished and self-explanatory.

**Independent Test**: Launch the app, interact with every input control, verify no crashes and charts update.

**App Page Structure**:

| Page | Content |
|---|---|
| Page 1: Experiment Design | Power calculator with all interactive inputs, MDE-vs-N chart, duration estimator |
| Page 2: Analyze Results | CSV upload or manual entry → full pipeline (frequentist, Bayesian, Sample Ratio Mismatch, segmentation, recommendation) |
| Page 3: Sensitivity Analysis | Post-experiment MDE detection view — "what effect could this test have reliably detected?" |
| Page 4: Case Study Demo | Pre-loaded synthetic data walkthrough with all diagnostic steps |

**Acceptance Scenarios**:

1. **Given** the app is running, **When** the user adjusts MDE slider from 2% to 5%, **Then** required N, duration estimate, and MDE-vs-N chart update in real time.
2. **Given** a CSV upload with two-group experiment data, **When** the user clicks "Analyze," **Then** the full pipeline runs and displays frequentist, Bayesian, Sample Ratio Mismatch, and recommendation outputs.
3. **Given** the segment selector, **When** the user picks "platform," **Then** per-segment results and Simpson's Paradox check are displayed.
4. **Given** the app analysis page, **When** the user selects "Manual Entry" mode, **Then** they can input summary statistics (conversions/total or mean/std/n) without uploading a file, and receive the same analysis outputs.
5. **Given** a global alpha selector set to 90%, **When** any analysis runs, **Then** all CIs, p-value thresholds, and recommendations use alpha = 0.10.
6. **Given** completed experiment results, **When** the user opens the "Sensitivity Analysis" tab, **Then** the app displays the minimum detectable effect (MDE) at 80% power given the observed sample size, answering "what effect could this test have reliably detected?"

---

### User Story 9 — End-to-End Case Study (Priority: P2)

A reviewer reads the Jupyter notebook and follows a complete experiment lifecycle from hypothesis through recommendation, using a realistic e-commerce or real estate domain.

**Why this priority**: The narrative walkthrough ties every module together and demonstrates domain awareness.

**Independent Test**: Run the notebook top-to-bottom with `nbconvert --execute` and verify zero errors.

**Case Study Narrative Structure**:

1. **Hypothesis**: New search filter layout increases search-to-save rate.
2. **Design**: Power analysis with 80/20 allocation split (risk mitigation scenario).
3. **Data generation**: Synthetic data with embedded novelty effect and segment imbalance.
4. **Sample Ratio Mismatch check**: Confirm assignment integrity.
5. **Analysis**:
   - Frequentist + Bayesian results side by side.
   - CUPED-adjusted vs. unadjusted CIs.
6. **Diagnostics**:
   - Time-series daily treatment effect plot → identify novelty effect → apply burn-in exclusion (drop first 3 days) and re-run.
   - Segmentation analysis → surface Simpson's Paradox (treatment wins aggregate, loses per-segment or vice versa).
   - Peeking illustration: "Had we checked at Day 3, p = 0.03 — a false positive."
7. **Recommendation**: Automated summary with Twyman's Law check.

**Acceptance Scenarios**:

1. **Given** the notebook, **When** executed end-to-end, **Then** it completes without error and produces all expected plots (minimum 6: power curve, CI comparison, Bayesian posterior distributions, time-series daily treatment effect, per-segment comparison, cumulative lift chart) and summaries.
2. **Given** the synthetic data, **When** the novelty effect section runs, **Then** the time-series plot shows inflated early lift, and the burn-in exclusion + re-analysis is demonstrated.
3. **Given** the synthetic data (generated with a fixed random seed and engineered parameters), **When** the peeking illustration runs, **Then** it shows a p-value < 0.05 at Day 3 that would have been a false positive.
4. **Given** the 80/20 allocation, **When** power analysis is shown, **Then** it demonstrates the cost of unequal allocation vs. 50/50.

---

### Edge Cases

**Acceptance Scenarios (promoted from design notes)**:

1. **Given** one variant has zero conversions, **When** the Bayesian engine runs, **Then** the Beta posterior is still valid (informed by the prior) and the Z-test issues a warning about zero conversion rate.
2. **Given** an uploaded CSV with missing values or wrong column names, **When** the user clicks "Analyze" in Streamlit, **Then** a clear, actionable error message is displayed (not a stack trace).

**Design Notes (guide implementation but not formally tested)**:

- What happens when sample sizes are extremely unequal (e.g., 99/1)? (Power calculator should warn about near-zero effective sample size)
- What happens when the treatment effect is exactly zero? (Bayesian P(B > A) ≈ 0.5; frequentist p-value should be large)
- What happens when pre-experiment covariate has zero correlation with the outcome? (CUPED should degrade gracefully to unadjusted estimate)

## Requirements

### Functional Requirements

- **FR-001**: System MUST calculate required sample size given baseline rate, MDE (absolute or relative), power, alpha, and allocation ratio. When MDE mode is "relative", the system converts to absolute internally via `absolute_mde = baseline_rate × relative_mde`.
- **FR-002**: System MUST perform Z-test for proportions with unpooled standard error and report Cohen's h effect size.
- **FR-003**: System MUST perform Welch's t-test for continuous metrics and report Cohen's d effect size.
- **FR-003a**: System MUST run Shapiro-Wilk normality check on continuous metric data before parametric tests. If N < 30 (heuristic threshold) and non-normal, issue a warning; if N ≥ 30, note Central Limit Theorem applicability.
- **FR-004**: System MUST calculate Bayesian posterior P(B > A) and Expected Loss for proportions (Beta-Binomial). Default prior is Beta(1,1) (uniform/uninformative). The prior MUST be user-configurable via function parameter.
- **FR-005**: System MUST calculate Bayesian posterior for continuous metrics (Normal-Normal conjugate). Uses sample variance treated as known. Default prior uses data-driven hyperparameters: prior mean = pooled sample mean, prior variance = large multiple of sample variance (weakly informative). Prior hyperparameters MUST be user-configurable. This approximation is valid for N > 100 per group; the system SHOULD warn when N < 100.
- **FR-006**: System MUST detect Sample Ratio Mismatch via chi-square test for arbitrary expected ratios.
- **FR-007**: System MUST implement CUPED variance reduction with a single pre-experiment covariate, handling unequal group sizes. Theta is estimated via OLS regression of outcome on covariate.
- **FR-008**: System MUST compute per-segment treatment effects and detect Simpson's Paradox (sign flip between aggregate and segment-level effects). Per-segment results MUST include a disclaimer noting that p-values are unadjusted for multiple comparisons, and state the number of segments tested.
- **FR-009**: System MUST generate a "Ship / No-Ship / No Effect / Inconclusive" recommendation as a Python dict with keys: `recommendation` (str), `flags` (list of diagnostic warnings including Twyman's Law when relative lift exceeds the configured threshold — default 50%, see `lift_warning_threshold` in `recommendation.py`), and `supporting_metrics` (dict of significance, effect size, Sample Ratio Mismatch status). Enhanced narrative output is deferred to FR-017 (Tier 2).
- **FR-010**: System MUST generate synthetic experiment data with configurable: baseline rate, effect size, sample size, allocation ratio, novelty effect, Simpson's Paradox, and Sample Ratio Mismatch injection. Each data generation feature (novelty, Simpson's, Sample Ratio Mismatch) is an independent boolean flag — implementation should be composable, with each feature as a separate function that transforms a base dataset.
- **FR-011**: System MUST provide a Streamlit app with interactive power calculator (real-time chart updates on parameter change), CSV upload → full analysis pipeline, and manual summary-statistics entry mode (conversions/total or mean/std/n).
- **FR-011a**: System MUST provide a global alpha/confidence level selector (90%, 95%, 99%) that propagates to all analysis outputs.
- **FR-011b**: System MUST provide a sensitivity analysis view showing the minimum detectable effect at 80% power given the observed sample size (reframed from post-hoc power to avoid the known critique that observed power is a 1-to-1 function of the p-value).
- **FR-012**: System MUST provide a Jupyter notebook case study demonstrating a realistic experiment end-to-end (hypothesis → design → analysis → diagnostics → recommendation) using an e-commerce or real estate domain.
- **FR-013**: System MUST produce: cumulative lift charts, CI comparison plots, Bayesian posterior plots, MDE-vs-N curves, time-series daily treatment effect plot, power loss vs. allocation ratio curve, and per-segment comparison chart (7 visualization types total). All visualizations use **Plotly** for interactive charts in both Streamlit and Jupyter.

### Tier 2 Functional Requirements (Good to Have)

- **FR-014**: System SHOULD implement Sequential Probability Ratio Test (SPRT) for valid early stopping.
- **FR-015**: System SHOULD apply Bonferroni and Benjamini-Hochberg FDR corrections when multiple metrics are evaluated.
- **FR-016**: System SHOULD provide winsorization and trimmed mean utilities for outlier management.
- **FR-017**: System SHOULD generate enhanced narrative summaries with effect size context ("equivalent to X additional conversions per week").

### Tier 3 Functional Requirements (If Time Permits)

- **FR-018**: System MAY implement formal temporal effects detection via time-series changepoint analysis.
- **FR-019**: System MAY implement A/A testing validation for false positive rate calibration.

### Key Entities

- **Experiment**: A single A/B test instance with control group, treatment group, primary metric, and optional guardrail metrics.
- **Variant**: A group in the experiment (control or treatment) with observed data (conversions/totals for proportions, or continuous values).
- **Metric**: A measurable outcome — either a proportion (conversion rate) or continuous (revenue, time). Has a baseline value and optional pre-experiment covariate.
- **Segment**: A user-defined categorical split (platform, cohort, geography) for HTE analysis.
- **Recommendation**: The output of the executive summary engine — Ship / No-Ship / Inconclusive — with supporting evidence and diagnostic flags.

## Success Criteria

### Measurable Outcomes

- **SC-001**: All Tier 1 functional requirements (FR-001 through FR-013) are implemented and pass tests.
- **SC-002**: Test coverage ≥ 90% on all modules under `ab_test_toolkit/` (the library). Streamlit pages under `app/pages/` are out of scope for unit-test coverage and are exercised via manual demo and notebook integration tests.
- **SC-003**: Statistical functions validated against scipy reference or hand-calculated values with tolerance ≤ 1e-6.
- **SC-004**: Streamlit app launches without error and all interactive controls produce expected output.
- **SC-005**: Jupyter notebook executes end-to-end via `nbconvert --execute` with zero errors.
- **SC-006**: A reviewer unfamiliar with the codebase can understand the case study narrative and executive summary without reading source code.

## Data Input Schema

The toolkit expects CSV data in **long format** with the following columns:

| Column | Required | Description |
|---|---|---|
| `group` | Yes | Group label (e.g., "control", "treatment") |
| `value` | Yes | 0/1 for proportion metrics, continuous for revenue/time metrics |
| `segment` | No | Categorical segment label for HTE analysis (e.g., "mobile", "desktop") |
| `covariate` | No | Pre-experiment covariate value for CUPED variance reduction |

The metric type (proportion vs. continuous) is **auto-detected** from the `value` column: if all values are 0 or 1, it is treated as proportion data; otherwise, continuous.

Group labels in the `group` column **must** be `"control"` and `"treatment"` (case-insensitive). The toolkit raises a clear error if other labels are found.

## Assumptions

- Target audience is hiring managers and interviewers at experiment-mature product companies.
- All data is synthetic — no real user data is used or required.
- The toolkit analyzes experiment results; it does not assign traffic or run experiments (traffic assignment and feature flagging platforms are out of scope).
- Python 3.11+ is the minimum supported version.
- No MCMC or heavy probabilistic programming frameworks (PyMC, Stan) — conjugate priors only.
- Mobile/responsive design for Streamlit is not a priority.
- The project is scoped for ~1.5 hours/day over 6 weeks alongside other portfolio projects (~63 total hours; ~50 effective coding hours with ~13 hours buffer for debugging and scope surprises).

## Out of Scope

| Item | Reason |
|---|---|
| Multivariate / Split URL / Multipage testing | Implementation-layer concerns. This toolkit analyzes results, not assigns traffic. |
| MCMC / PyMC-based Bayesian inference | Heavy dependency weight. Conjugate priors are sufficient for the portfolio use case. |
| Experimentation Decision Matrix (as code) | Better as README documentation, not a runnable module. |
| Go/No-Go Framework (as code) | Process doc, not code. |
| Real-time traffic assignment | Out of scope — this is an analysis toolkit, not an experimentation platform. |

## Documentation (README)

The README should include:

- **Motivation**: "I built this because I saw X gap between theory and practice."
- One paragraph on unequal allocation: why it exists, why 50/50 is optimal but not always practical.
- Experimentation Decision Matrix as a markdown table (A/B vs. Multivariate — when to use which).
- Go/No-Go framework as a short checklist.
- Installation, usage, and links to the Streamlit app and case study notebook.

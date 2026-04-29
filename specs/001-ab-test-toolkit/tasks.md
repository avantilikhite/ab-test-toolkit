# Tasks: AB Test Toolkit

**Input**: Design documents from `/specs/001-ab-test-toolkit/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/public-api.md ✅

**Tests**: TDD is mandatory per constitution (Principle III). Test tasks are included before implementation in every phase. Tests validate against scipy reference values with tolerance ≤ 1e-6. Property-based testing via hypothesis for statistical invariants.

**Organization**: Tasks are grouped by user story (US1–US9) to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story (US1–US9) this task serves
- All paths are relative to repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization — directory tree, packaging, package skeleton

- [X] T001 Create project directory structure with ab_test_toolkit/, app/pages/, notebooks/, tests/unit/, tests/integration/ per plan.md
- [X] T002 Create pyproject.toml with dependency groups (core: numpy>=1.24, scipy>=1.10, pandas>=2.0; viz: plotly>=5.10; app: streamlit>=1.28, plotly>=5.10; notebook: jupyter, nbconvert; dev: pytest>=7.4, pytest-cov>=4.1, hypothesis>=6.80, ruff>=0.10) and project metadata in pyproject.toml
- [X] T003 [P] Create package init with MetricType enum (PROPORTION, CONTINUOUS), package docstring, and version string in ab_test_toolkit/__init__.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data ingestion and synthetic data generation — required by ALL downstream stories for testing and data input

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Tests for Foundational Phase

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T004 Create shared test fixtures with sample proportion DataFrames (binary 0/1), continuous DataFrames, segmented DataFrames, covariate DataFrames, and edge-case DataFrames (empty, missing cols, wrong types) in tests/conftest.py
- [X] T005 [P] Write unit tests for load_experiment_data covering CSV path load, DataFrame pass-through, group column validation (case-insensitive "control"/"treatment"), MetricType auto-detection (all 0/1 → PROPORTION, else CONTINUOUS), optional segment/covariate handling, and clear ValueError messages for invalid input in tests/unit/test_io.py
- [X] T006 [P] Write unit tests for generate_experiment_data covering baseline rate accuracy, each anomaly flag independently (inject_novelty inflates early days, inject_simpsons creates segment sign-flip, inject_srm skews allocation), flag composition, seed reproducibility, and output DataFrame schema (group, value, segment, covariate, day columns) in tests/unit/test_data_generator.py

### Implementation for Foundational Phase

- [X] T007 Implement load_experiment_data accepting str/Path/DataFrame, CSV parsing with pandas, group column validation (exactly "control" and "treatment" case-insensitive), value column NaN rejection, MetricType auto-detection, optional segment (NaN → "unknown") and covariate validation, and clear ValueError messages per data-model.md ExperimentData rules in ab_test_toolkit/io.py
- [X] T008 Implement generate_experiment_data with composable injection functions: base binary data from np.random.binomial, inject_novelty (inflated effect for first N days with multiplier), inject_simpsons (engineered segment imbalance for sign flip), inject_srm (skewed allocation via srm_actual_ratio), always include covariate column for CUPED testing, per data-model.md DataGeneratorConfig in ab_test_toolkit/data_generator.py

**Checkpoint**: Foundation ready — `pytest tests/unit/test_io.py tests/unit/test_data_generator.py -v` all green. Data loading pipeline and synthetic data generator are functional. User story implementation can now begin.

---

## Phase 3: User Story 1 — Power Analysis & Experiment Design (Priority: P1) 🎯 MVP

**Goal**: Calculate required sample size per group and estimated test duration given baseline rate, MDE (absolute or relative), power, alpha, allocation ratio, and daily traffic. Visualize MDE-vs-N trade-off and power loss from unequal allocation.

**Independent Test**: Provide known statistical parameters and verify output matches scipy.stats.norm.ppf-based hand calculations within tolerance ≤ 1e-6.

**FRs**: FR-001, FR-013 (MDE-vs-N curve, power loss curve)

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T009 [P] [US1] Write unit tests for required_sample_size (balanced 50/50 validated against scipy, unequal 80/20 with power_loss_pct > 0, relative MDE conversion matching absolute equivalent, boundary baseline rates, daily_traffic duration estimation) and power_curve (returns DataFrame with mde/n_control/n_treatment/n_total columns, N monotonically decreases as MDE increases) in tests/unit/test_power.py
- [X] T010 [P] [US1] Write unit tests for mde_vs_n_curve and power_loss_curve verifying each returns plotly.graph_objects.Figure, correct number of traces, axis labels present, and data ranges match input in tests/unit/test_visualization.py

### Implementation for User Story 1

- [X] T011 [US1] Implement PowerResult dataclass (n_control, n_treatment, n_total, n_effective, power_loss_pct, estimated_days), required_sample_size with two-proportion Z-test power formula using scipy.stats.norm.ppf and explicit allocation ratio parameter per research.md R1, relative-to-absolute MDE conversion, n_effective = 4·n₁·n₂/(n₁+n₂), and power_curve returning DataFrame over MDE range in ab_test_toolkit/power.py
- [X] T012 [US1] Create visualization module with mde_vs_n_curve (line chart of MDE vs required N from power_curve DataFrame) and power_loss_curve (allocation ratio vs power loss percentage) using Plotly with plotly_white template and height=400px defaults in ab_test_toolkit/visualization.py

**Checkpoint**: `pytest tests/unit/test_power.py tests/unit/test_visualization.py -v` all green. Power calculator returns correct sample sizes for balanced and unbalanced designs. Interactive charts render correctly.

---

## Phase 4: User Story 2 — Frequentist Analysis (Priority: P1)

**Goal**: Perform two-proportion Z-test (unpooled SE, Cohen's h) and Welch's t-test (unequal variance, Cohen's d) on experiment data. Include Shapiro-Wilk normality check for continuous metrics. Support both raw data arrays and pre-aggregated summary statistics.

**Independent Test**: Run Z-test and Welch's t-test against known datasets and verify p-values and CIs match scipy.stats.proportions_ztest and scipy.stats.ttest_ind(equal_var=False) within tolerance ≤ 1e-6.

**FRs**: FR-002, FR-003, FR-003a, FR-013 (CI comparison plot)

### Tests for User Story 2

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T013 [P] [US2] Write unit tests for z_test and z_test_from_stats (p-value/CI validated against scipy.stats.proportions_ztest, Cohen's h = 2·arcsin(√p₂)−2·arcsin(√p₁)), welch_t_test and welch_t_test_from_stats (validated against scipy.stats.ttest_ind equal_var=False, Cohen's d), Shapiro-Wilk normality gating (warn if N<30 + non-normal, CLT note if N≥30), summary stats equivalence (from_stats matches raw data results), and **edge case: one variant with zero conversions triggers a warning about zero conversion rate** in tests/unit/test_frequentist.py
- [X] T014 [P] [US2] Add unit tests for ci_comparison_plot verifying Figure has traces for both groups with error bars, effect size annotation, and labeled axes to tests/unit/test_visualization.py

### Implementation for User Story 2

- [X] T015 [US2] Implement FrequentistResult dataclass (test_type, statistic, p_value, ci_lower, ci_upper, point_estimate, effect_size, alpha, is_significant, normality_check), z_test with unpooled SE per research.md R2, z_test_from_stats, welch_t_test with Welch-Satterthwaite df, welch_t_test_from_stats, Shapiro-Wilk check via scipy.stats.shapiro (warn N<30 + non-normal, note CLT N≥30), Cohen's h and Cohen's d in ab_test_toolkit/frequentist.py
- [X] T016 [US2] Add ci_comparison_plot function producing error bar chart with control/treatment CIs, point estimates, labeled axes, and effect size annotation to ab_test_toolkit/visualization.py

**Checkpoint**: `pytest tests/unit/test_frequentist.py -v` all green. Z-test and Welch's t-test match scipy reference. Summary stats input gives identical results to raw data.

---

## Phase 5: User Story 3 — Bayesian Analysis (Priority: P1)

**Goal**: Compute Bayesian posteriors for proportions (Beta-Binomial conjugate) and continuous metrics (Normal-Normal conjugate). Return P(B > A) via numerical integration, Expected Loss via Monte Carlo, posterior parameters, and 95% credible intervals. Support configurable priors.

**Independent Test**: Beta-Binomial with known priors and observed data; verify P(B > A) via numerical integration matches Monte Carlo simulation within tolerance ≤ 0.01.

**FRs**: FR-004, FR-005, FR-013 (Bayesian posterior plot)

### Tests for User Story 3

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T017 [P] [US3] Write unit tests for beta_binomial and beta_binomial_from_stats (P(B>A) cross-validated against MC simulation ≤0.01 tolerance, expected loss > 0, posterior alpha/beta correctness, custom prior Beta(0.5,0.5)), normal_normal (posterior mean convergence to data, N<100 warning, credible interval contains true effect), hypothesis property-based tests (P(B>A) ∈ [0,1], posterior narrows as N increases, expected_loss ≥ 0), and **edge case: one variant with zero conversions produces a valid Beta posterior informed by the prior (Beta(1, N+1))** in tests/unit/test_bayesian.py
- [X] T018 [P] [US3] Add unit tests for posterior_plot verifying Figure has two distribution traces (control/treatment), shaded 95% credible intervals, and labeled axes to tests/unit/test_visualization.py

### Implementation for User Story 3

- [X] T019 [US3] Implement BayesianResult dataclass (model_type, prob_b_greater_a, expected_loss, control_posterior, treatment_posterior, credible_interval, prior_config), beta_binomial with numerical integration P(B>A) using scipy.special.betainc + np.trapz per research.md R3, MC expected loss (1M samples default), beta_binomial_from_stats, and normal_normal with weakly informative data-driven prior (μ₀=pooled mean, σ₀²=10×pooled variance) and N<100 warning in ab_test_toolkit/bayesian.py
- [X] T020 [US3] Add posterior_plot function producing overlaid posterior PDF curves for control/treatment with shaded 95% credible interval bands and probability annotation to ab_test_toolkit/visualization.py

**Checkpoint**: `pytest tests/unit/test_bayesian.py -v` all green. Bayesian engine returns correct posteriors for both metric types. Posterior plot shows visibly tighter distribution for larger samples.

---

## Phase 6: User Story 4 — Sample Ratio Mismatch Detection (Priority: P1)

**Goal**: Detect allocation bias via chi-square test comparing observed vs expected group sizes. Flag mismatch at strict p < 0.01 threshold. Support arbitrary expected ratios (not just 50/50).

**Independent Test**: Generate data with known allocation bias and verify chi-square test detects it. Verify balanced allocation passes.

**FRs**: FR-006

### Tests for User Story 4

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T021 [P] [US4] Write unit tests for check_srm covering: balanced 50/50 no-mismatch (p_value > 0.01), large-N slight imbalance triggers mismatch, custom expected ratio 80/20 matching observed returns no mismatch, configurable threshold sensitivity, observed/expected ratio tuple accuracy, and edge case of exactly equal counts in tests/unit/test_srm.py

### Implementation for User Story 4

- [X] T022 [US4] Implement SRMResult dataclass (expected_ratio, observed_ratio, chi2_statistic, p_value, has_mismatch) and check_srm using scipy.stats.chisquare with expected counts derived from expected_ratio, configurable threshold (default 0.01), and observed/expected ratio reporting in ab_test_toolkit/srm.py

**Checkpoint**: `pytest tests/unit/test_srm.py -v` all green. SRM detector correctly identifies allocation mismatches and passes for balanced data.

---

## Phase 7: User Story 5 — CUPED Variance Reduction (Priority: P1)

**Goal**: Reduce metric variance using a pre-experiment covariate via CUPED adjustment. Estimate theta via pooled OLS (NOT per-group). Handle unequal group sizes. Return both adjusted and unadjusted estimates for side-by-side comparison.

**Independent Test**: Compare CI width with and without CUPED on synthetic data with known covariate correlation ρ. Verify variance reduction ≈ ρ².

**FRs**: FR-007

### Tests for User Story 5

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T023 [P] [US5] Write unit tests for cuped_adjust covering: variance reduction with known ρ=0.7 (expect ~49% reduction), pooled theta matches manual Cov(Y,X)/Var(X) calculation, unequal group sizes handled correctly, zero-correlation covariate degrades gracefully (adjusted ≈ unadjusted), adjusted CI strictly narrower than unadjusted CI when ρ > 0, and correlation field accuracy in tests/unit/test_cuped.py

### Implementation for User Story 5

- [X] T024 [US5] Implement CUPEDResult dataclass (theta, correlation, variance_reduction_pct, adjusted_estimate, adjusted_ci, unadjusted_estimate, unadjusted_ci) and cuped_adjust with pooled OLS theta estimation θ̂=Cov(Y,X)/Var(X) across all observations per research.md R4, ρ calculation, variance_reduction_pct = ρ²×100, adjusted treatment effect, and adjusted CI in ab_test_toolkit/cuped.py

**Checkpoint**: `pytest tests/unit/test_cuped.py -v` all green. CUPED produces narrower CIs when covariate is correlated and degrades gracefully when ρ ≈ 0.

---

## Phase 8: User Story 6 — Segmentation & HTE Analysis (Priority: P1)

**Goal**: Compute per-segment treatment effects and detect Simpson's Paradox (sign flip between aggregate and any segment-level effect). Include multiple comparisons disclaimer with segment count.

**Independent Test**: Generate synthetic data where aggregate effect and per-segment effects have opposite signs. Verify detector flags the sign flip.

**FRs**: FR-008, FR-013 (per-segment comparison chart)

### Tests for User Story 6

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T025 [P] [US6] Write unit tests for segment_analysis covering: per-segment point estimates and CIs returned, Simpson's Paradox detection (positive aggregate + negative segment triggers simpsons_paradox=True with details string), no-paradox case (simpsons_paradox=False), single-segment degenerate case, n_segments count accuracy, and multiple_comparisons_note present with segment count in tests/unit/test_segmentation.py
- [X] T026 [P] [US6] Add unit tests for segment_comparison_chart verifying per-segment grouped bars with error bars, aggregate overlay line, Simpson's Paradox annotation when flagged, and labeled axes to tests/unit/test_visualization.py

### Implementation for User Story 6

- [X] T027 [US6] Implement SegmentResult dataclass (aggregate_estimate, aggregate_ci, segment_results list, simpsons_paradox, simpsons_details, n_segments, multiple_comparisons_note) and segment_analysis performing per-segment frequentist tests, aggregate treatment effect, sign-flip detection between aggregate and each segment, and disclaimer noting unadjusted p-values with segment count in ab_test_toolkit/segmentation.py
- [X] T028 [US6] Add segment_comparison_chart function producing grouped bar chart with per-segment treatment effect CIs, horizontal aggregate effect line, and Simpson's Paradox annotation callout when detected to ab_test_toolkit/visualization.py

**Checkpoint**: `pytest tests/unit/test_segmentation.py -v` all green. Segmentation analysis surfaces per-segment heterogeneity and correctly detects Simpson's Paradox on engineered data.

---

## Phase 9: User Story 7 — Automated Executive Summary (Priority: P2)

**Goal**: Generate a Ship / No-Ship / Inconclusive recommendation with diagnostic flags (SRM mismatch, Simpson's Paradox, Twyman's Law for >30% relative lift) and supporting metrics dict. Decision logic follows the state machine defined in data-model.md Recommendation entity.

**Independent Test**: Feed known result object combinations and verify output matches expected decision and flags.

**FRs**: FR-009

### Tests for User Story 7

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T029 [P] [US7] Write unit tests for generate_recommendation covering all state transitions: Ship (is_significant + prob_b_gt_a > 0.95 + positive effect + no SRM + no paradox), No-Ship (significant negative effect), Inconclusive-SRM (has_mismatch=True), Inconclusive-Simpson's (simpsons_paradox=True), Inconclusive-default (not significant), Twyman's Law flag (relative lift > 30%), and None segmentation optional path in tests/unit/test_recommendation.py

### Implementation for User Story 7

- [X] T030 [US7] Implement Recommendation dataclass (recommendation, flags, supporting_metrics) and generate_recommendation with decision state machine: check SRM → check Simpson's → check significance + Bayesian + direction → Ship/No-Ship/Inconclusive, Twyman's Law relative lift check against configurable threshold (default 0.30), and supporting_metrics dict (significance, p_value, effect_size, srm_status, prob_b_gt_a) in ab_test_toolkit/recommendation.py

**Checkpoint**: `pytest tests/unit/test_recommendation.py -v` all green. Recommendation engine produces correct decisions for all documented state transitions.

---

## Phase 10: User Story 8 — Interactive Streamlit App (Priority: P2)

**Goal**: Deliver a polished 4-page Streamlit multi-page app: Experiment Design (interactive power calculator), Analyze Results (CSV upload or manual entry → full analysis pipeline), Sensitivity Analysis (post-experiment MDE at 80% power), and Case Study Demo (pre-loaded walkthrough). Global alpha/confidence selector in sidebar propagates to all analyses.

**Independent Test**: Launch app with `streamlit run app/app.py`, interact with every input control, verify no crashes and charts update in real time.

**FRs**: FR-011, FR-011a, FR-011b, FR-013

### Implementation for User Story 8

- [X] T031 [US8] Create Streamlit app entry point with page config (title, icon, layout), global sidebar alpha/confidence selector (90%/95%/99% mapped to alpha 0.10/0.05/0.01 via st.session_state), and app initialization guard to prevent flickering in app/app.py
- [X] T032 [US8] Create Streamlit helper utilities for session_state management (safe get/set), consistent page layout wrapper, error display formatting (user-friendly messages not stack traces), and chart rendering helper in app/utils.py
- [X] T033 [P] [US8] Implement Experiment Design page with sliders/inputs for baseline rate, MDE (absolute/relative toggle), power, alpha (from global), allocation ratio, daily traffic; real-time MDE-vs-N chart (power.power_curve → visualization.mde_vs_n_curve), power loss curve, duration estimate, and sample size summary in app/pages/01_experiment_design.py
- [X] T034 [P] [US8] Implement Analyze Results page with CSV file_uploader and manual summary stats entry toggle (conversions/total for proportions, mean/std/n for continuous), full analysis pipeline execution (io.load_experiment_data → frequentist → bayesian → srm → optional cuped/segmentation → recommendation), all relevant charts (CI comparison, posterior, segment), and recommendation display in app/pages/02_analyze_results.py
- [X] T035 [P] [US8] Implement Sensitivity Analysis page showing minimum detectable effect at 80% power given observed sample size (inverse power calculation), with explanation text clarifying this avoids observed-power critique in app/pages/03_sensitivity_analysis.py
- [X] T036 [US8] Implement Case Study Demo page with pre-loaded synthetic data from data_generator (novelty + Simpson's + covariate), step-by-step walkthrough of all diagnostic checks, all 7 visualization types displayed, and narrative text explaining each step in app/pages/04_case_study_demo.py

**Checkpoint**: `streamlit run app/app.py` launches without error. All 4 pages interactive. Global alpha propagates correctly. CSV upload triggers full pipeline.

---

## Phase 11: User Story 9 — End-to-End Case Study Notebook (Priority: P2)

**Goal**: Create a Jupyter notebook demonstrating a complete experiment lifecycle — hypothesis → power analysis (80/20 split) → synthetic data generation → SRM check → frequentist + Bayesian → CUPED → time-series novelty detection → segmentation + Simpson's → peeking illustration → automated recommendation. Minimum 6 Plotly interactive charts.

**Independent Test**: Execute notebook top-to-bottom with `jupyter nbconvert --execute` and verify zero errors.

**FRs**: FR-012, FR-013 (cumulative lift chart, daily treatment effect chart)

### Tests for User Story 9

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T037 [P] [US9] Write notebook execution test using nbconvert --execute to verify case_study.ipynb runs top-to-bottom with zero errors and all cells produce output in tests/integration/test_notebook.py
- [X] T038 [P] [US9] Add unit tests for cumulative_lift_chart (verify Figure traces, x-axis is observation index, y-axis is cumulative lift) and daily_treatment_effect (verify time-series traces, date axis, effect line with CI band) to tests/unit/test_visualization.py

### Implementation for User Story 9

- [X] T039 [US9] Implement cumulative_lift_chart (running cumulative treatment effect line with CI band) and daily_treatment_effect (daily point estimates with CI error bars, horizontal zero line) in ab_test_toolkit/visualization.py
- [X] T040 [US9] Create case study notebook with narrative sections: (1) Hypothesis and domain context, (2) Power analysis with 80/20 split showing allocation cost, (3) Synthetic data generation with novelty + Simpson's + covariate, (4) SRM integrity check, (5) Frequentist + Bayesian side-by-side, (6) CUPED adjusted vs unadjusted CIs, (7) Time-series daily effect → identify novelty → burn-in exclusion (drop 3 days) → re-analyze, (8) Segmentation → Simpson's Paradox, (9) Peeking illustration (Day 3 false positive), (10) Automated recommendation with Twyman's Law check; minimum 6 Plotly charts (power curve, CI comparison, posterior, daily effect, segment comparison, cumulative lift) in notebooks/case_study.ipynb

**Checkpoint**: `jupyter nbconvert --execute notebooks/case_study.ipynb` succeeds with zero errors. All 6+ charts render. Narrative reads as a self-contained portfolio piece.

---

## Phase 12: Polish & Cross-Cutting Concerns

**Purpose**: Integration testing, documentation, package exports, coverage validation, and final cleanup

- [X] T041 [P] Write integration test for full analysis pipeline (generate_experiment_data → load_experiment_data → z_test → beta_binomial → check_srm → cuped_adjust → segment_analysis → generate_recommendation) validating end-to-end data flow and recommendation output in tests/integration/test_pipeline.py
- [X] T042 [P] Create README.md with motivation paragraph, one-paragraph unequal allocation discussion, experimentation decision matrix (A/B vs Multivariate markdown table), Go/No-Go checklist, installation instructions (uv + pip), library usage quickstart, Streamlit launch command, notebook link, and project structure overview in README.md
- [X] T043 [P] Update ab_test_toolkit/__init__.py with complete public API exports: all result dataclasses (PowerResult, FrequentistResult, BayesianResult, SRMResult, CUPEDResult, SegmentResult, Recommendation), all public functions from each module, and MetricType enum in ab_test_toolkit/__init__.py
- [X] T044 Run full test suite with `pytest --cov=ab_test_toolkit --cov-report=term-missing`, validate ≥90% line coverage on all modules, and execute quickstart.md code examples as smoke test for documentation accuracy
- [X] T045 [P] Run `ruff check ab_test_toolkit/` with zero warnings, verify all public function signatures have type annotations, and verify all public functions have NumPy-style docstrings per constitution quality standards
- [X] T046 [P] Write Streamlit integration test using `streamlit.testing.AppTest` or subprocess health check — validate app launches without error (`streamlit run app/app.py --server.headless true`), all 4 pages load, and interactive controls render without exceptions in tests/integration/test_streamlit.py

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup (Phase 1) — **BLOCKS all user stories**
- **US1–US6 (Phases 3–8)**: All depend on Foundational completion. **US1–US6 are independent of each other** and can proceed in parallel or in priority order
- **US7 (Phase 9)**: Depends on US2, US3, US4 completion (requires FrequentistResult, BayesianResult, SRMResult). Optionally uses US6 (SegmentResult)
- **US8 (Phase 10)**: Depends on US1–US7 completion (all analysis modules must exist for full pipeline)
- **US9 (Phase 11)**: Depends on US1–US7 completion (notebook exercises all modules)
- **Polish (Phase 12)**: Depends on all desired user stories being complete

### User Story Dependencies

| Story | Depends On | Can Start After |
|-------|-----------|-----------------|
| US1 (Power Analysis) | Foundational only | Phase 2 complete |
| US2 (Frequentist) | Foundational only | Phase 2 complete |
| US3 (Bayesian) | Foundational only | Phase 2 complete |
| US4 (SRM Detection) | Foundational only | Phase 2 complete |
| US5 (CUPED) | Foundational only | Phase 2 complete |
| US6 (Segmentation) | Foundational only | Phase 2 complete |
| US7 (Recommendation) | US2, US3, US4, US6 | All P1 stories complete |
| US8 (Streamlit App) | US1–US7 | All analysis modules complete |
| US9 (Case Study) | US1–US7 | All analysis modules complete |

### Within Each User Story (TDD Cycle)

1. Tests MUST be written FIRST and FAIL before implementation begins
2. Result dataclasses defined alongside their module logic
3. Core statistical computation before visualization functions
4. Verify all tests PASS after implementation
5. Commit after each completed story

### Parallel Opportunities

- **Phase 1**: T002 and T003 can run in parallel
- **Phase 2**: T005 and T006 can run in parallel (different test files)
- **US1–US6**: All six P1 stories can start in parallel after Phase 2 (independent modules, different files)
- **Within each story**: Test tasks targeting different files can run in parallel (marked [P])
- **US8 Streamlit**: Pages 01, 02, 03 can run in parallel after app.py and utils.py are created
- **US9**: T037 and T038 can run in parallel (different test files)
- **Phase 12**: T041, T042, T043 can run in parallel (different files)

---

## Parallel Example: P1 User Stories (After Phase 2)

```
# All six P1 stories target independent modules — zero cross-dependencies:

Stream A:
  US1: tests/unit/test_power.py → ab_test_toolkit/power.py → visualization.py (MDE + power loss charts)

Stream B:
  US2: tests/unit/test_frequentist.py → ab_test_toolkit/frequentist.py → visualization.py (CI plot)

Stream C:
  US3: tests/unit/test_bayesian.py → ab_test_toolkit/bayesian.py → visualization.py (posterior)

Stream D:
  US4: tests/unit/test_srm.py → ab_test_toolkit/srm.py
  US5: tests/unit/test_cuped.py → ab_test_toolkit/cuped.py
  US6: tests/unit/test_segmentation.py → ab_test_toolkit/segmentation.py → visualization.py (segment chart)
```

## Parallel Example: Streamlit Pages (After US1–US7)

```
# Three pages can be built simultaneously (different files):

Task T033: app/pages/01_experiment_design.py
Task T034: app/pages/02_analyze_results.py
Task T035: app/pages/03_sensitivity_analysis.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (io + data_generator)
3. Complete Phase 3: User Story 1 (power analysis + charts)
4. **STOP and VALIDATE**: `pytest tests/unit/test_power.py tests/unit/test_visualization.py -v`
5. Power calculator is independently usable as a library and demoable

### Full Statistical Engine (US1–US7)

1. Setup + Foundational → Foundation ready
2. US1 → US2 → US3 → US4 → US5 → US6 (each independently testable after completion)
3. US7 (Recommendation) → ties all analysis modules into Ship/No-Ship/Inconclusive
4. **VALIDATE**: `pytest --cov=ab_test_toolkit` — should approach ≥90% coverage

### Complete Delivery (All 3 Artifacts)

1. Full Statistical Engine (above) → Python package artifact complete
2. US8 (Streamlit App) → interactive demo artifact complete
3. US9 (Case Study Notebook) → narrative portfolio artifact complete
4. Phase 12 (Polish) → README, integration tests, coverage gate, quickstart validation

---

## Notes

- **[P]** tasks = different files, no dependencies on incomplete tasks within the same phase
- **[Story]** label maps task to specific user story for traceability
- Each user story is independently completable and testable
- TDD: Verify tests FAIL before implementing, then PASS after
- Commit after each task or logical group
- Stop at any checkpoint to validate independently
- All visualization functions accumulate in a single `ab_test_toolkit/visualization.py` — phases are sequential so no file conflicts
- Avoid: vague tasks, same-file conflicts within parallel tasks, cross-story dependencies that break independence

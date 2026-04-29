# Implementation Plan: AB Test Toolkit

**Branch**: `001-ab-test-toolkit` | **Date**: 2026-04-10 | **Spec**: `specs/001-ab-test-toolkit/spec.md`
**Input**: Feature specification from `/specs/001-ab-test-toolkit/spec.md`

## Summary

Build a production-grade Python toolkit for designing, auditing, and analyzing A/B experiments. The toolkit delivers three artifacts: an importable Python package (`ab_test_toolkit/`) with statistical modules for power analysis, frequentist/Bayesian inference, variance reduction (CUPED), segmentation, and automated recommendations; a multi-page Streamlit interactive demo app; and a Jupyter notebook case study. Architecture follows a module-first approach where every statistical capability is a self-contained, independently testable module, and UI layers (Streamlit, Jupyter) strictly consume the package without containing core logic.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: numpy, scipy, pandas (core engine); plotly (all visualizations); streamlit (interactive app); jupyter/nbconvert (case study)
**Storage**: N/A (in-memory data processing; CSV file upload only)
**Testing**: pytest (unit/integration), hypothesis (property-based edge-case generation)
**Target Platform**: Cross-platform (macOS/Linux/Windows), local Python environment
**Project Type**: Library + interactive demo app + narrative notebook
**Performance Goals**: All statistical computations complete in <1s for datasets up to 1M rows; Streamlit UI updates feel real-time (<500ms on parameter change)
**Constraints**: No MCMC/heavy frameworks (conjugate priors only); minimal dependency footprint; ~50 effective coding hours budget
**Scale/Scope**: 13 Tier 1 functional requirements, 4 Tier 2, 2 Tier 3; 7 visualization types; 4-page Streamlit app; 1 case study notebook

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Statistical Rigor First | ✅ PASS | Conjugate priors only (Beta-Binomial, Normal-Normal). Welch's t-test for unequal variance. All results include point estimates + CIs/credible intervals. No MCMC. |
| II. Module-First Architecture | ✅ PASS | Every capability lives in its own module under `ab_test_toolkit/`. Streamlit/Jupyter consume the package only. No circular dependencies by design. |
| III. Test-Driven Development | ✅ PASS | TDD mandatory per constitution. Tests validate against scipy reference or hand-calculated values. Red-Green-Refactor enforced. |
| IV. Minimal Dependencies | ✅ PASS | Core: numpy, scipy, pandas. Viz: plotly. Demo: streamlit. Testing: pytest, hypothesis. All justified in spec. |
| V. Portfolio-Grade Communication | ✅ PASS | Automated executive summary produces Ship/No-Ship/Inconclusive. All plots labeled. Case study reads as narrative. |
| VI. Simplicity & YAGNI | ✅ PASS | Tiered scope (Tier 1 mandatory, Tier 2/3 optional). No abstractions beyond module boundaries. Data generation is composable functions, not a framework. |

**Gate Result: PASS** — No violations detected. Proceeding to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/001-ab-test-toolkit/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
ab_test_toolkit/             # Importable Python package
├── __init__.py              # Package exports
├── power.py                 # FR-001: Sample size & power analysis
├── frequentist.py           # FR-002, FR-003, FR-003a: Z-test, Welch's t, normality
├── bayesian.py              # FR-004, FR-005: Beta-Binomial & Normal-Normal conjugate
├── srm.py                   # FR-006: Sample Ratio Mismatch detection
├── cuped.py                 # FR-007: CUPED variance reduction
├── segmentation.py          # FR-008: HTE & Simpson's Paradox detection
├── recommendation.py        # FR-009: Executive summary engine
├── data_generator.py        # FR-010: Synthetic data generation
├── visualization.py         # FR-013: All 7 chart types (Plotly)
└── io.py                    # CSV parsing, validation, auto-detection

app/                         # Streamlit multi-page app
├── app.py                   # Main entry point
├── pages/
│   ├── 01_experiment_design.py
│   ├── 02_analyze_results.py
│   ├── 03_sensitivity_analysis.py
│   └── 04_case_study_demo.py
└── utils.py                 # Streamlit-specific helpers (state, layout)

notebooks/
└── case_study.ipynb         # FR-012: End-to-end narrative walkthrough

tests/
├── conftest.py              # Shared fixtures, synthetic data factories
├── unit/
│   ├── test_power.py
│   ├── test_frequentist.py
│   ├── test_bayesian.py
│   ├── test_srm.py
│   ├── test_cuped.py
│   ├── test_segmentation.py
│   ├── test_recommendation.py
│   ├── test_data_generator.py
│   ├── test_visualization.py
│   └── test_io.py
└── integration/
    ├── test_pipeline.py     # Full analysis pipeline end-to-end
    └── test_notebook.py     # nbconvert execution test

pyproject.toml               # Modern Python packaging (uv-compatible)
README.md                    # Motivation, decision matrix, quickstart
```

**Structure Decision**: Single-project Python package layout. The `ab_test_toolkit/` directory is the importable package at repository root. Streamlit app lives in `app/` as a consumer. Tests mirror source structure under `tests/unit/` with an `integration/` directory for pipeline and notebook execution tests. This follows the constitution's Module-First Architecture principle — clean separation between statistical engine, UI layer, and test layer.

## Post-Design Constitution Re-Check

*Re-evaluated after Phase 1 design artifacts (data-model.md, contracts/, quickstart.md) are complete.*

| Principle | Status | Post-Design Evidence |
|-----------|--------|---------------------|
| I. Statistical Rigor First | ✅ PASS | All formulas documented in research.md with scipy references. Contracts specify tolerance ≤ 1e-6 for validation. P(B > A) computed via numerical integration (not simulation alone). |
| II. Module-First Architecture | ✅ PASS | 11 modules in `ab_test_toolkit/`, each with single responsibility. Contracts define clean interfaces. No module imports another module's internals — they communicate via result types defined in data-model.md. |
| III. Test-Driven Development | ✅ PASS | Test file structure mirrors source. Contracts define testable signatures. Property-based testing strategy documented in research.md (hypothesis invariants). |
| IV. Minimal Dependencies | ✅ PASS | Unified on Plotly (dropped matplotlib). Core remains numpy+scipy+pandas only. Optional dependency groups in pyproject.toml keep library install lightweight. |
| V. Portfolio-Grade Communication | ✅ PASS | Recommendation entity returns structured dict per FR-009. Visualization contract specifies all 7 chart types. Quickstart demonstrates end-to-end usage in <20 lines. |
| VI. Simplicity & YAGNI | ✅ PASS | No abstract base classes, no plugin system, no ORM. Data model uses plain dicts/dataclasses. Data generator uses composable boolean flags, not a pipeline framework. |

**Post-Design Gate Result: PASS** — Design artifacts are consistent with constitution. No violations introduced.

## Complexity Tracking

> No violations detected — table intentionally left empty.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| *(none)* | — | — |

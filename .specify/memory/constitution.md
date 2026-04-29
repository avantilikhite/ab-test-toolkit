<!--
SYNC IMPACT REPORT
===================
Version change: 1.0.0 → 1.1.0 (MINOR)
Bump rationale: Material change to approved dependency list —
  removed matplotlib, narrowing the allowed visualization stack.

Modified principles:
  - IV. Minimal Dependencies: Updated visualization dependency
    from "matplotlib (notebook) + plotly (Streamlit)" →
    "plotly for all contexts (Streamlit and Jupyter)".
    Also strengthened "must" → "MUST" for dependency justification.

Added sections: None
Removed sections: None

Technology Stack changes:
  - Visualization row: "matplotlib, plotly" → "plotly"
  - Rationale updated to "Unified interactive charts for both
    Streamlit and Jupyter"

Templates requiring updates:
  - .specify/templates/plan-template.md    ✅ No update needed
    (generic template; no library-specific references)
  - .specify/templates/spec-template.md    ✅ No update needed
    (generic template; no library-specific references)
  - .specify/templates/tasks-template.md   ✅ No update needed
    (generic template; no library-specific references)

Downstream artifacts (already consistent):
  - specs/001-ab-test-toolkit/spec.md      ✅ FR-013 says Plotly
  - specs/001-ab-test-toolkit/plan.md      ✅ "Unified on Plotly"
  - specs/001-ab-test-toolkit/tasks.md     ✅ All viz tasks use Plotly
  - specs/001-ab-test-toolkit/research.md  ✅ R5 confirms Plotly-only

Follow-up TODOs: None — all placeholders resolved.
-->
# AB Test Toolkit Constitution

## Core Principles

### I. Statistical Rigor First

Every statistical method must be mathematically correct and well-referenced. No shortcuts on inference — conjugate priors only (no MCMC), proper handling of unequal sample sizes via Welch's t-test and unpooled standard errors, and explicit assumptions stated for every test. Results must always include both point estimates and uncertainty quantification (confidence intervals or credible intervals).

### II. Module-First Architecture

Every analytical capability lives in its own importable module under `ab_test_toolkit/`. Modules must be self-contained, independently testable, and composable. No circular dependencies. The Streamlit app and Jupyter notebook consume the package — they never contain core statistical logic.

### III. Test-Driven Development (NON-NEGOTIABLE)

TDD mandatory for all statistical modules: write tests with known analytical results first (e.g., Z-test on pre-computed data must return expected p-value), verify tests fail, then implement. Every public function must have at least one test validating correctness against a hand-calculated or reference result. Red-Green-Refactor cycle strictly enforced.

### IV. Minimal Dependencies

Core engine depends only on: `numpy`, `scipy`, `pandas`. Visualization uses `plotly` for all contexts (Streamlit and Jupyter). No heavy frameworks (PyMC, statsmodels) unless a specific function is unavailable in scipy. Every dependency MUST be justified.

### V. Portfolio-Grade Communication

Every output — plot, summary, recommendation — must be interpretable by a non-technical product manager. The automated executive summary must produce plain-English "Ship / No-Ship / Inconclusive" recommendations. Plots must have clear titles, axis labels, and legends. The case study must read as a narrative, not a code dump.

### VI. Simplicity & YAGNI

Do not over-engineer. No abstractions for one-time operations. No feature creep beyond the tiered scope defined in the specification. If a concept can be demonstrated via the case study narrative, do not build a separate module for it.

## Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Core engine | Python 3.11+, numpy, scipy, pandas | Standard scientific Python stack |
| Visualization | plotly | Unified interactive charts for both Streamlit and Jupyter |
| Interactive demo | Streamlit | Low-friction interactive app framework |
| Case study | Jupyter Notebook | Narrative-first format for walkthrough |
| Testing | pytest, hypothesis (property-based) | pytest for unit tests, hypothesis for edge-case generation |
| Packaging | pyproject.toml, uv | Modern Python packaging |

## Quality Standards

- **Test coverage**: ≥90% on all modules under `ab_test_toolkit/`.
- **Type hints**: All public function signatures must have type annotations.
- **Docstrings**: All public functions must have a one-line summary + parameter descriptions. Use NumPy-style docstrings.
- **Linting**: `ruff` for linting and formatting. Zero warnings on CI.
- **Numerical validation**: Statistical functions must be validated against scipy reference implementations or hand-calculated values with tolerance ≤ 1e-6.

## Development Workflow

1. Branch per feature (spec-kit convention: `NNN-feature-name`).
2. Write failing tests first.
3. Implement until tests pass.
4. Add visualization / demo integration.
5. Update case study notebook if the module is referenced.
6. PR review against spec compliance.

## Governance

This constitution supersedes all ad-hoc decisions. Scope changes require updating the specification (`spec.md`) first, then amending this document. The tiered scope (Tier 1 / Tier 2 / Tier 3) in the specification is the single source of truth for what gets built.

**Version**: 1.1.0 | **Ratified**: 2026-04-10 | **Last Amended**: 2026-04-10

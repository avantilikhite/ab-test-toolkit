# 🧪 AB Test Toolkit

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ab-test-toolkit.streamlit.app/)
[![Tests](https://img.shields.io/badge/tests-234%20passing-brightgreen)](./tests)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](./pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](./pyproject.toml)

A production-grade Python toolkit and interactive app for designing, analyzing, and making decisions on A/B experiments.

**🚀 Try it live → [ab-test-toolkit.streamlit.app](https://ab-test-toolkit.streamlit.app/)**

**Why this exists:** Most experimentation guides stop at "run a t-test." This toolkit covers the full lifecycle: power analysis with unequal allocation, integrity checks (SRM), frequentist testing with an optional Bayesian layer for richer decision metrics (P(B>A), expected loss), variance reduction (CUPED), segmentation with Simpson's Paradox detection, and an automated decision engine — all in a single, well-tested package.

> **Note:** Statistical primitives (distributions, critical values, hypothesis tests, χ²) are built on [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html). The toolkit's contribution is the experiment design framework, integrity checks, dual-evidence decision engine, and end-to-end workflow layered on top of those primitives.

---

## What the App Does

The Streamlit app walks through the full experimentation lifecycle — from designing an experiment to making a Ship / No-Ship / Inconclusive decision. It works in two modes: **manual entry** (paste summary stats) and **CSV upload** (raw user-level data).

### 🎯 Page 1: Experiment Design

Interactive power calculator that answers "how many users do I need?"

| Parameter | Options | What It Means |
|-----------|---------|---------------|
| **Metric Type** | `Proportion` · `Continuous` | **Proportion** = binary outcomes like converted yes/no (use Z-test). **Continuous** = numeric outcomes like revenue per user (use Welch's t-test). If your metric is an aggregate (total revenue), reframe it as per-user (revenue/user). |
| **MDE Mode** | `Relative` · `Absolute` | **Relative** (default): % change from baseline — e.g., 0.05 means "detect a 5% lift" (10% → 10.5%). Industry standard for product experiments. **Absolute**: raw difference — e.g., 0.02 means "detect a 2pp change" (10% → 12%). Prefer absolute when baseline is near 0% or 100%, or for very low-rate metrics like error rates. |
| **Baseline** | Slider/input | Your current metric value before the experiment. For proportions: conversion rate (e.g., 0.10 = 10%). For continuous: mean and standard deviation (e.g., $50 ± $15). |
| **Allocation Ratio** | 0.1 – 4.0 | Treatment/control ratio. `1.0` = 50/50 split (optimal power). `0.25` = 80% control / 20% treatment (limits risk exposure but costs ~64% more samples vs balanced). `2.0` = 67% treatment / 33% control. |
| **Daily Traffic** | 100 – 10M | Expected visitors per day — used to estimate experiment duration in calendar days. |
| **Confidence Level** | 90% · 95% · 99% | Set in sidebar. Controls α (false positive rate): 95% → α=0.05 is the industry default. |

**Outputs:** Control n · Treatment n · Total n · Estimated days · Sample inflation % (extra sample required vs balanced design at the configured allocation ratio)

**Warnings:** Flags when estimated duration < 7 days (day-of-week seasonality risk) and when unequal allocation inflates required sample size by >5% vs a balanced design.

**Charts:** MDE vs Sample Size curve · Sample Size Inflation vs Allocation Ratio

**Pre-registration manifest export:** the page also lets you snapshot the full experiment plan — `experiment_id`, hypothesis, primary metric, MDE, planned N, allocation, alpha/power, the **fixed dual-evidence decision rule**, and the structured policy block (`loss_tolerance`, `allow_ship_with_monitoring`, `monitoring_prob_threshold`, `twyman_min_baseline`, `practical_significance_threshold`, `lift_warning_threshold`) — into a JSON manifest. The manifest is versioned, stamped with the toolkit version, and verified for drift on the Analyze page.

### 📊 Page 2: Analyze Results

Full analysis pipeline in 5 steps, with optional **pre-registration verification**:

| Step | What It Does |
|------|-------------|
| **0. Pre-reg manifest** *(optional)* | Upload the JSON manifest exported from Page 1. Engine **auto-loads** the registered policy fields (α, loss tolerance, monitoring threshold, etc.) into the controls and **detects drift** between the registered plan and the as-run analysis — surfaced as a red "Manifest drift detected" card on the recommendation. Stamps the result with the full SHA-256 of the manifest for audit. |
| **1. SRM Check** | Chi-squared test on traffic split — catches broken randomization before anything else. When a `day` column is present, also runs **per-day stratified SRM** with **Holm-Bonferroni correction** so day-level bucketing bugs don't hide inside an aggregate-balanced split. |
| **2. Frequentist** | Z-test (proportions) or Welch's t-test (continuous) → p-value, CI, effect size, relative lift |
| **3. Bayesian** | Beta-Binomial conjugate (proportions) or Student-t reference-prior posterior (continuous) → P(B>A), expected loss, credible interval |
| **4. CUPED** | Pre-experiment covariate adjustment → variance reduction %, tighter CIs (CSV mode with covariate column) |
| **5. Segments** | Per-segment Welch's t with Holm-Bonferroni correction → heterogeneous effects, Simpson's Paradox detection (CSV mode with segment column) |

**Charts:** Confidence Interval · Posterior Distributions · Segment Comparison. *Cumulative Lift and Daily Treatment Effect charts are demonstrated in the Case Study notebook (require a `day` column).*

### 🔍 Page 3: Sensitivity Analysis

Post-experiment check: "given my sample size, what's the smallest effect I could have detected at 80% power?" Avoids the [post-hoc power fallacy](https://en.wikipedia.org/wiki/Post_hoc_analysis) by solving for MDE instead of observed power.

**Why 80% power?** Power = 1 − β, the probability of correctly detecting a real effect when one exists. 80% (β = 0.20) is the long-standing industry default popularized by Cohen (1988) — a pragmatic balance between two costs: lower power means more false negatives (missing real wins); higher power (e.g., 90–95%) requires substantially more sample size (~30% more for 90%, ~67% more for 95%). For high-stakes decisions (medical, payments, irreversible launches) you may want 90%+; for cheap, reversible product changes 80% is the standard floor.

### 📚 Page 4: Case Study Demo

Pre-loaded walkthrough with synthetic data (5,000 users/group, 10% baseline, 2pp lift) and **injected anomalies** — novelty effect and Simpson's Paradox — so you can see every diagnostic fire in a single run.

### 📅 Duration Guidance

Even when the statistical sample size is reached quickly, the app recommends a minimum runtime to avoid temporal biases:

- **At least 7 days** to capture day-of-week seasonality (weekday vs weekend behavior differs for most consumer products).
- **At least 14 days** for experiments affecting **habitual behavior** (e.g., notifications, recommendations, retention loops) — users need time to adapt, and short runs conflate novelty with the true steady-state effect.

The Experiment Design page surfaces this warning automatically when the calculated duration is under 7 days.

---

## Issues & Diagnostics the App Surfaces

The recommendation engine synthesizes all evidence and flags problems automatically:

| Diagnostic | How It's Detected | Impact on Decision |
|-----------|-------------------|-------------------|
| **Sample Ratio Mismatch** | χ² test on observed vs expected traffic split (aggregate + per-day stratified with Holm correction) | Blocks decision → Inconclusive (randomization broken) |
| **Manifest Drift** | Compares registered pre-reg manifest against as-run α, loss tolerance, monitoring policy, and planned N | Flags drift on the recommendation card; recommendation reflects as-run settings (treat as exploratory unless drift is justified) |
| **Simpson's Paradox** | At least one segment that *opposes* the aggregate effect-sign and represents ≥20% of the valid sample is significant under Holm-adjusted p-values | Blocks decision → Inconclusive (composition effect is real) |
| **Twyman's Law** | \|effect size\| ≥ 1.0 or relative lift ≥ 50% | Downgrades Ship → Inconclusive (suspiciously large) |
| **Novelty Effect** | Early treatment effect > 2× late effect | Warning + suggests burn-in re-analysis |
| **Heavy-Tailed Data** | Kurtosis > 10 | Suggests P99 winsorization before analysis |
| **Framework Disagreement** | Frequentist significant but Bayesian P(B>A) < 95% | Inconclusive with moderate signal |
| **Underpowered Test** | Not significant + low Bayesian confidence | Inconclusive + guidance on extending the test |
| **Practical Significance** | CI lower bound below minimum threshold | Flags that effect may not be business-meaningful |

### Pre-Registration & Drift Detection

Pre-registration is the single highest-leverage habit for credible experimentation — and one of the easiest to neglect. This toolkit makes it concrete:

1. **Export a versioned manifest** from the Experiment Design page before the test runs. The JSON snapshot includes the toolkit version, hypothesis, primary metric, MDE/power/α, planned sample size, the fixed decision rule, every structured policy knob, and Bayesian seed parameters.
2. **Upload the manifest** on the Analyze page when reading results to verify the as-run analysis matches the registered plan.

> **Note on the decision rule:** the toolkit hard-codes a dual-evidence rule (frequentist p < α **and** Bayesian P(B>A) ≥ 95% **and** no SRM **and** no significant Simpson reversal **and** not Twyman-flagged). This is intentionally stricter than industry default — most teams pick **one** school per experiment (see the per-school decision matrices below). The dual-evidence default in this toolkit is a teaching device that forces you to inspect both lenses and surfaces conflicts.


### When to Use Which Framework

| Framework | Best For | Typical Adopters |
|-----------|----------|---------|
| **Frequentist** | Regulatory/executive buy-in, binary go/no-go decisions, teams accustomed to p-values | Most large-traffic platforms |
| **Bayesian** | Probability statements ("94% chance B is better"), expected loss / regret framing, strong priors from past experiments, smaller companies with less traffic | Streaming, recommendation, and smaller-traffic teams |
| **Sequential** | Peeking at results mid-experiment, formal early stopping (efficacy + futility), expensive traffic or high-risk features | Increasingly common as practical middle ground |

**The framework choice matters far less than:** (1) pre-registering your hypothesis and MDE, (2) not peeking without correction, and (3) running for at least one full business cycle. Pick one and commit before the experiment starts.

### Decision Matrices

Most teams commit to one framework pre-experiment. The toolkit reports both, but the two matrices below are the standalone decision rules you'd use depending on your team's choice.

#### Frequentist Decision Matrix

| Scenario | p-value | 95% CI | Decision |
|---|---|---|---|
| ✅ Clear Ship | < α | Lower bound > practical-significance threshold (positive) | **Ship** |
| 🚫 Clear No-Ship | < α | Entire CI on the negative side | **No-Ship** |
| ⬜ Confident Null | > α | CI entirely within ±practical-significance margin | **No Effect** |
| 🟡 Underpowered Null | > α | Wide CI crosses zero AND extends past the practical margin | **Inconclusive** |
| ⚠️ Marginal | p ≈ α (e.g., 0.04–0.05) | CI lower bound just above zero but below the practical threshold | **Inconclusive / borderline** |

#### Bayesian Decision Matrix

| Scenario | P(B>A) | Expected Loss of Shipping B | Decision |
|---|---|---|---|
| ✅ Clear Ship | > 95% | Below loss tolerance | **Ship** (gated by `loss_tolerance` if supplied) |
| 🚫 Clear No-Ship | < 5% | High | **No-Ship** |
| 🟡 Moderate Signal | 85–95% | Moderate | **Ship with Monitoring** when `allow_ship_with_monitoring=True`, otherwise **Inconclusive** |
| ⚪ Weak Signal | 60–85% | Non-trivial | **Inconclusive** |
| ⬜ Confident Null | ~50% (40–60%) | Very low in either direction | **No Effect** |
| 🔴 Likely Harmful | < 5% | High | **No-Ship** (investigate before any future launch) |

#### Cross-Framework Sanity Check (this toolkit's default)

For demonstration, this toolkit's recommendation engine runs **both** frameworks in parallel and treats agreement as a confidence boost and disagreement as a flag to investigate. This is a deliberate portfolio design choice — **not** industry standard. Most production systems pre-commit to one framework. The dual requirement here is more conservative (fewer false Ships) and surfaces borderline results sensitive to priors or noise.

| Frequentist | Bayesian P(B>A) | Effect | Decision | Signal Strength |
|------------|----------------|--------|----------|----------|
| Significant | > 95% | Positive | **Ship** (downgraded if `expected_loss > loss_tolerance`) | Strong |
| Significant | Any | Negative | **No-Ship** | Strong |
| Significant | ≤ 95% | — | Inconclusive (disagreement) | Moderate |
| Not significant | ≥ 95% | Positive | Inconclusive (Bayesian-only signal) | Moderate |
| Not significant | 85–95% | Positive | **Ship with Monitoring** (opt-in via `allow_ship_with_monitoring=True`) | Moderate |
| Not significant | 70–85% | — | Inconclusive | Weak |
| Not significant | < 70% | — | Inconclusive | None |

Every recommendation includes **context-aware next steps** and **supporting metrics** (p-value, effect size, SRM status, P(B>A), novelty ratio, plus an optional pre-registration manifest hash for reproducibility).

> **Advisory framing.** The decision states above are *advisories*, not autonomous launch gates. They consolidate the statistical evidence; the launch decision belongs to the team and should weigh strategic, qualitative, and risk factors not captured in the summary. The toolkit always exposes the underlying p-value, CI, P(B>A), and expected loss so reviewers can override the headline.

### Scale and operational caveats

- **Single-metric framing.** The recommendation engine evaluates one primary metric. Real launches typically track several guardrails; if you act on multiple metrics simultaneously, apply Bonferroni / Holm or use a hierarchical decision rule. The segmentation module does Holm-correct *within* a segment family — it does not correct across metrics.
- **Stationarity assumption.** Power and decision logic assume the experiment runs to its planned horizon. For sequential / always-valid analyses, plug into a sequential testing framework (group-sequential, AGRAPA, mSPRT) — not this toolkit.
- **No interference / spillover handling.** All tests assume the Stable Unit Treatment Value Assumption (SUTVA). Two-sided marketplaces, social-graph products, ad auctions, and inventory-shared experiences violate SUTVA and require cluster randomization or budget-split designs.
- **A/A pre-flight.** Run an A/A test before any new logging pipeline goes live. The toolkit does not auto-gate on A/A signal; it is the experimenter's responsibility.

---

## Quick Start

### Installation

```bash
git clone https://github.com/avantilikhite/ab-test-toolkit.git
cd ab-test-toolkit

# Using uv (recommended)
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run the App

```bash
streamlit run app/app.py
```

Configure the **confidence level** (90% / 95% / 99%) in the sidebar — it applies across all pages.

### Library Usage

```python
from ab_test_toolkit.power import required_sample_size
from ab_test_toolkit.frequentist import z_test
from ab_test_toolkit.bayesian import beta_binomial
from ab_test_toolkit.srm import check_srm
from ab_test_toolkit.recommendation import generate_recommendation

# 1. Design
power = required_sample_size(baseline_rate=0.10, mde=0.02, allocation_ratio=1.0)
print(f"Need {power.n_total:,} total users")

# 2. Analyze
import numpy as np
ctrl = np.random.binomial(1, 0.10, 5000)
treat = np.random.binomial(1, 0.12, 5000)

freq = z_test(ctrl, treat)
bayes = beta_binomial(ctrl, treat)
srm = check_srm(observed=(len(ctrl), len(treat)))

# 3. Decide
rec = generate_recommendation(freq, bayes, srm)
print(rec.recommendation)    # Ship / No-Ship / Inconclusive
print(rec.signal_strength)   # Strong / Moderate / Weak / None
print(rec.next_steps)        # Context-aware action items
```

### CSV Format

Upload a CSV with these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `group` | ✅ | `"control"` or `"treatment"` (case-insensitive) |
| `value` | ✅ | Metric value (0/1 for proportions, numeric for continuous) |
| `segment` | Optional | Segment label (e.g., `"mobile"`, `"US"`) — enables HTE analysis |
| `covariate` | Optional | Pre-experiment metric — enables CUPED variance reduction |
| `day` | Optional | Day index (1, 2, …) — enables novelty detection & temporal charts |

**Validation enforced at load time** (`io.load_experiment_data`):
- `value` must be numeric and finite (NaN / `±Inf` rows are rejected with a row-count error message).
- `covariate`, when present, must also be **finite** (no NaN, no `Inf`, no `-Inf`) — non-finite values silently propagate through CUPED's regression and produce broken CIs, so they fail fast instead.
- `day`, when present, is **coerced to numeric**; non-numeric values like `"Mon"` or `"week 1"` are rejected. This prevents the lexicographic-sort trap (`"1", "10", "2"` → wrong novelty windows).
- `group` values are normalized to lowercase; anything outside `{control, treatment}` is rejected.

### Pre-Registration Manifest Schema

When you click **Export pre-registration manifest** on the Experiment Design page, the JSON file contains:

| Field | Notes |
|---|---|
| `manifest_version` | Currently `2`. Bumped when fields change in a backwards-incompatible way. |
| `toolkit_version` | Pulled from `ab_test_toolkit.__version__` — surfaces drift across releases. |
| `experiment_id`, `primary_metric`, `hypothesis` | Free-text identifiers for human reviewers. |
| `decision_rule` | The **fixed** dual-evidence rule string (recorded for reviewers; not parsed). |
| `alpha`, `power`, `mde`, `mde_mode`, `metric_type`, `allocation_ratio` | Statistical-design inputs. |
| `planned_n_control`, `planned_n_treatment`, `planned_n_total`, `estimated_days` | Power-analysis outputs. |
| `loss_tolerance`, `allow_ship_with_monitoring`, `monitoring_prob_threshold`, `twyman_min_baseline`, `practical_significance_threshold`, `lift_warning_threshold` | The structured policy block — these are the fields the engine actually enforces and the drift-checker diffs at analysis time. |
| `bayesian_random_state`, `bayesian_n_simulations` | Bayesian sampling reproducibility. |

The manifest is hashed (SHA-256) when stamped on the recommendation; the full hash is stored in supporting metrics for audit, with a 16-character display prefix in the UI.

### Jupyter Notebook

```bash
jupyter notebook notebooks/case_study.ipynb
```

End-to-end case study covering power analysis, SRM, dual statistical analysis, CUPED, novelty detection, Simpson's Paradox, and automated recommendation.

---

## Running Tests

```bash
# All tests
pytest --quiet

# With coverage (library only — Streamlit pages are not unit-tested)
pytest --cov=ab_test_toolkit --cov-report=term-missing

# Specific module
pytest tests/unit/test_power.py -v

# Integration tests only
pytest tests/integration/ -v
```

> **Coverage scope:** The ≥90% coverage target applies to the `ab_test_toolkit/`
> library where the statistics live. The Streamlit pages (`app/pages/`) are
> validated through manual exercise of the demo flow and the executable
> notebook integration test, not through page-level unit tests.

---

## Project Structure

```
ab_test_toolkit/         # Importable Python package
├── __init__.py          # Public API exports + MetricType enum + version
├── power.py             # Sample size & power analysis (incl. unequal allocation)
├── frequentist.py       # Z-test, Welch's t-test, Newcombe interval
├── bayesian.py          # Beta-Binomial conjugate + Student-t reference-prior posterior
├── srm.py               # Aggregate + per-stratum (Holm-corrected) SRM detection
├── cuped.py             # CUPED variance reduction (ANCOVA-correct CIs, winsorization)
├── segmentation.py      # Heterogeneous treatment effects + Simpson's Paradox (Holm)
├── recommendation.py    # Decision engine + manifest hashing/drift detection
├── data_generator.py    # Synthetic data with injectable anomalies
├── visualization.py     # Plotly chart builders
└── io.py                # CSV loading & validation (covariate finiteness, day numeric)

app/                     # Streamlit interactive app
├── app.py               # Entry point + sidebar (α, practical significance)
├── utils.py             # Shared helpers + manifest accessors
└── pages/
    ├── 01_experiment_design.py    # Power calculator + manifest export (v2 schema)
    ├── 02_analyze_results.py      # Full pipeline + manifest verify + stratified SRM
    ├── 03_sensitivity_analysis.py # Post-experiment MDE calculator
    └── 04_case_study_demo.py      # Walkthrough with synthetic data

notebooks/
└── case_study.ipynb     # End-to-end Jupyter walkthrough

tests/                   # 234 tests (unit + integration)
├── conftest.py          # Shared fixtures
├── unit/                # Per-module tests, including round-3 P1 fix coverage
└── integration/         # Pipeline, notebook & app tests
```

## License

MIT

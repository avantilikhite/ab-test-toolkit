# Quickstart: AB Test Toolkit

**Feature**: 001-ab-test-toolkit | **Date**: 2026-04-10

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
# Clone and install in development mode
git clone <repo-url>
cd a_b_testing

# Using uv (recommended)
uv sync
uv pip install -e ".[dev,app]"

# Or using pip
pip install -e ".[dev,app]"
```

## Quick Usage — As a Library

### Power Analysis

```python
from ab_test_toolkit.power import required_sample_size

result = required_sample_size(
    baseline_rate=0.10,
    mde=0.02,
    alpha=0.05,
    power=0.80,
    allocation_ratio=1.0,
)
print(f"Need {result.n_total} total observations ({result.n_control} per group)")
```

### Frequentist Analysis

```python
import numpy as np
from ab_test_toolkit.frequentist import z_test

control = np.random.binomial(1, 0.10, 5000)
treatment = np.random.binomial(1, 0.12, 5000)

result = z_test(control, treatment, alpha=0.05)
print(f"p-value: {result.p_value:.4f}, significant: {result.is_significant}")
print(f"Effect size (Cohen's h): {result.effect_size:.4f}")
```

### Bayesian Analysis

```python
from ab_test_toolkit.bayesian import beta_binomial

result = beta_binomial(control, treatment)
print(f"P(B > A): {result.prob_b_greater_a:.4f}")
print(f"Expected Loss: {result.expected_loss:.6f}")
```

### Full Pipeline

```python
from ab_test_toolkit.io import load_experiment_data
from ab_test_toolkit.frequentist import z_test
from ab_test_toolkit.bayesian import beta_binomial
from ab_test_toolkit.srm import check_srm
from ab_test_toolkit.recommendation import generate_recommendation

# Load data
data, metric_type = load_experiment_data("experiment_results.csv")

control = data[data["group"] == "control"]["value"]
treatment = data[data["group"] == "treatment"]["value"]

# Run analyses
freq_result = z_test(control, treatment)
bayes_result = beta_binomial(control, treatment)
srm_result = check_srm(
    observed=(len(control), len(treatment)),
    expected_ratio=(0.5, 0.5),
)

# Get recommendation
rec = generate_recommendation(freq_result, bayes_result, srm_result)
print(f"Recommendation: {rec.recommendation}")
print(f"Flags: {rec.flags}")
```

## Quick Usage — Streamlit App

```bash
# Launch the interactive app
streamlit run app/app.py
```

The app provides 4 pages:
1. **Experiment Design** — Interactive power calculator with real-time MDE-vs-N chart
2. **Analyze Results** — Upload CSV or enter summary stats → full analysis pipeline
3. **Sensitivity Analysis** — Post-experiment MDE detection view
4. **Case Study Demo** — Pre-loaded walkthrough with synthetic data

## Quick Usage — Jupyter Notebook

```bash
# Open the case study
jupyter notebook notebooks/case_study.ipynb
```

## Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific module tests
pytest tests/unit/test_power.py -v

# Run with hypothesis verbose output
pytest tests/unit/test_frequentist.py -v --hypothesis-show-statistics
```

## Project Structure

```
ab_test_toolkit/         # Importable Python package
├── power.py             # Sample size & power analysis
├── frequentist.py       # Z-test, Welch's t-test, normality checks
├── bayesian.py          # Beta-Binomial & Normal-Normal conjugate
├── srm.py               # Sample Ratio Mismatch detection
├── cuped.py             # CUPED variance reduction
├── segmentation.py      # HTE & Simpson's Paradox detection
├── recommendation.py    # Executive summary engine
├── data_generator.py    # Synthetic data generation
├── visualization.py     # All 7 Plotly chart types
└── io.py                # CSV loading & validation

app/                     # Streamlit multi-page app
notebooks/               # Jupyter case study
tests/                   # pytest + hypothesis tests
```

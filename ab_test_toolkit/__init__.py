"""
AB Test Toolkit — Production-grade A/B experiment design, audit, and analysis.

A modular Python package for power analysis, frequentist/Bayesian inference,
CUPED variance reduction, segmentation, and automated recommendations.
"""

from enum import Enum

__version__ = "0.1.0"


class MetricType(str, Enum):
    """Type of metric being analyzed."""

    PROPORTION = "proportion"
    CONTINUOUS = "continuous"


# --- Public API Exports ---

from ab_test_toolkit.bayesian import (
    BayesianResult,
    beta_binomial,
    beta_binomial_from_stats,
    normal_normal,
)
from ab_test_toolkit.cuped import CUPEDResult, cuped_adjust
from ab_test_toolkit.data_generator import generate_experiment_data
from ab_test_toolkit.frequentist import (
    FrequentistResult,
    welch_t_test,
    welch_t_test_from_stats,
    z_test,
    z_test_from_stats,
)
from ab_test_toolkit.io import load_experiment_data
from ab_test_toolkit.power import (
    PowerResult,
    power_curve,
    required_sample_size,
    required_sample_size_continuous,
)
from ab_test_toolkit.recommendation import (
    NoveltyCheckResult,
    Recommendation,
    check_novelty,
    generate_recommendation,
)
from ab_test_toolkit.segmentation import SegmentResult, segment_analysis
from ab_test_toolkit.srm import SRMResult, check_srm
from ab_test_toolkit.visualization import (
    ci_comparison_plot,
    cumulative_lift_chart,
    daily_treatment_effect,
    mde_vs_n_curve,
    posterior_plot,
    power_loss_curve,
    segment_comparison_chart,
)

__all__ = [
    # Core types
    "MetricType",
    # Result dataclasses
    "PowerResult",
    "FrequentistResult",
    "BayesianResult",
    "SRMResult",
    "CUPEDResult",
    "SegmentResult",
    "Recommendation",
    # Power analysis
    "required_sample_size",
    "required_sample_size_continuous",
    "power_curve",
    # Frequentist
    "z_test",
    "z_test_from_stats",
    "welch_t_test",
    "welch_t_test_from_stats",
    # Bayesian
    "beta_binomial",
    "beta_binomial_from_stats",
    "normal_normal",
    # SRM
    "check_srm",
    # CUPED
    "cuped_adjust",
    # Segmentation
    "segment_analysis",
    # Recommendation
    "generate_recommendation",
    "check_novelty",
    "NoveltyCheckResult",
    # Data generation
    "generate_experiment_data",
    # I/O
    "load_experiment_data",
    # Visualization
    "mde_vs_n_curve",
    "power_loss_curve",
    "ci_comparison_plot",
    "posterior_plot",
    "segment_comparison_chart",
    "cumulative_lift_chart",
    "daily_treatment_effect",
]

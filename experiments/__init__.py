# experiments/__init__.py
from experiments.stats_utils import (
    paired_t_test,
    mann_whitney_test,
    ks_test,
    cohens_d,
    mean_confidence_interval,
    bootstrap_metric,
    js_divergence,
    feature_expectation_error,
    policy_match_accuracy,
    StatisticalReport,
)

from experiments.compare_baselines import (
    BaselineComparator,
    ComparisonResult,
    generate_synthetic_results,
    generate_latex_table,
    BASELINE_METHODS,
    METRIC_NAMES,
)

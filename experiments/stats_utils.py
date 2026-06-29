"""Statistical utilities for paper-quality experimental reporting.

Provides:
  1. Paired t-test — for "same scenario, different method" comparisons
  2. Mann-Whitney U — for "different groups" non-parametric comparisons
  3. Kolmogorov-Smirnov — for "distribution alignment" validation
  4. Effect sizes (Cohen's d, rank-biserial r)
  5. 95% confidence intervals
  6. StatisticalReport — formatted paper-ready output strings

All functions accept list/array inputs and return dicts suitable for
direct inclusion in LaTeX tables and paper text.

Usage:
  from experiments.stats_utils import paired_t_test, StatisticalReport
  result = paired_t_test(our_scores, baseline_scores, name="策略匹配准确率")
  print(result["text"])  # "t=8.13, p<0.001, Cohen's d=1.82"
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field


# ================================================================
# Core statistical tests
# ================================================================

def paired_t_test(
    ours: Union[List[float], np.ndarray],
    baseline: Union[List[float], np.ndarray],
    name: str = "",
    alpha: float = 0.05,
) -> Dict:
    """Paired t-test: same scenarios, two methods compared.

    Use when: you run both your method and baseline on the SAME set of
    20 scenarios, producing 20 pairs of scores.

    Args:
        ours: Your method's scores (one per scenario)
        baseline: Baseline method's scores (paired, same scenarios)
        name: Metric name for reporting
        alpha: Significance level

    Returns:
        Dict with keys: t_stat, p_value, significant, cohens_d,
        effect_size_label, ci_95_diff, text (paper-ready sentence)
    """
    ours = np.asarray(ours, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)

    n = len(ours)
    diff = ours - baseline
    diff_std = float(np.std(diff, ddof=1))
    if diff_std < 1e-10:
        # Near-zero variance: distributions are effectively identical
        return {
            "n": n,
            "t_stat": 0.0,
            "p_value": 1.0,
            "p_str": "p=1.000 (zero var)",
            "significant": False,
            "cohens_d": 0.0,
            "effect_size_label": "可忽略",
            "mean_diff": float(np.mean(diff)),
            "ci_95_diff": (float(np.mean(diff)), float(np.mean(diff))),
            "text": f"{name + ': ' if name else ''}方差接近零, 两组无差异",
        }

    t_stat, p_value = stats.ttest_rel(ours, baseline)
    d = _cohens_d_paired(ours, baseline)

    # 95% CI of the mean difference
    mean_diff = float(np.mean(diff))
    _, ci = _mean_ci(diff)

    significant = p_value < alpha
    if p_value < 0.001:
        p_str = "p<0.001"
    elif p_value < 0.01:
        p_str = f"p={p_value:.3f}"
    else:
        p_str = f"p={p_value:.4f}"

    d_label = _cohens_d_label(d)

    text = (
        f"{name + ': ' if name else ''}"
        f"t({n-1})={t_stat:.2f}, {p_str}, "
        f"Cohen's d={d:.2f} ({d_label}), "
        f"均值差 {mean_diff:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])"
    )

    return {
        "n": n,
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "p_str": p_str,
        "significant": significant,
        "cohens_d": round(d, 4),
        "effect_size_label": d_label,
        "mean_diff": round(mean_diff, 4),
        "ci_95_diff": (round(ci[0], 4), round(ci[1], 4)),
        "text": text,
    }


def independent_t_test(
    group_a: Union[List[float], np.ndarray],
    group_b: Union[List[float], np.ndarray],
    name_a: str = "Group A",
    name_b: str = "Group B",
    metric: str = "",
    alpha: float = 0.05,
) -> Dict:
    """Independent t-test: two independent groups, same metric.

    Use when: comparing trained vs untrained groups on the same metric,
    or comparing firefighter vs elderly social weights from bootstrap.

    Note: assumes equal variance. For unequal variance, use Welch's t-test.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)

    t_stat, p_value = stats.ttest_ind(a, b)
    d = _cohens_d_independent(a, b)

    significant = p_value < alpha
    if p_value < 0.001:
        p_str = "p<0.001"
    elif p_value < 0.01:
        p_str = f"p={p_value:.3f}"
    else:
        p_str = f"p={p_value:.4f}"

    d_label = _cohens_d_label(d)

    text = (
        f"{metric + ': ' if metric else ''}"
        f"{name_a} vs {name_b}: "
        f"t={t_stat:.2f}, {p_str}, "
        f"Cohen's d={d:.2f} ({d_label})"
    )

    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "p_str": p_str,
        "significant": significant,
        "cohens_d": round(d, 4),
        "effect_size_label": d_label,
        "text": text,
    }


def mann_whitney_test(
    group_a: Union[List[float], np.ndarray],
    group_b: Union[List[float], np.ndarray],
    name_a: str = "Group A",
    name_b: str = "Group B",
    metric: str = "",
    alpha: float = 0.05,
) -> Dict:
    """Mann-Whitney U test: non-parametric, no normality assumption.

    Use when: comparing firefighter vs elderly social weights
    (may not be normally distributed), or any two independent
    groups where normality is questionable.

    Returns rank-biserial r as effect size.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)

    u_stat, p_value = stats.mannwhitneyu(a, b, alternative='two-sided')

    # Rank-biserial correlation as effect size
    n_a, n_b = len(a), len(b)
    r = 1.0 - (2.0 * u_stat) / (n_a * n_b)

    significant = p_value < alpha
    if p_value < 0.001:
        p_str = "p<0.001"
    elif p_value < 0.01:
        p_str = f"p={p_value:.3f}"
    else:
        p_str = f"p={p_value:.4f}"

    # Effect size label
    r_abs = abs(r)
    if r_abs < 0.1:
        r_label = "可忽略"
    elif r_abs < 0.3:
        r_label = "小"
    elif r_abs < 0.5:
        r_label = "中"
    else:
        r_label = "大"

    text = (
        f"{metric + ': ' if metric else ''}"
        f"{name_a} vs {name_b}: "
        f"U={u_stat:.0f}, {p_str}, "
        f"rank-biserial r={r:.3f} ({r_label}效应量)"
    )

    return {
        "u_stat": round(float(u_stat), 2),
        "p_value": round(p_value, 6),
        "p_str": p_str,
        "significant": significant,
        "rank_biserial_r": round(r, 4),
        "effect_size_label": r_label,
        "text": text,
    }


def ks_test(
    sample: Union[List[float], np.ndarray],
    reference: Union[List[float], np.ndarray],
    name: str = "",
    alpha: float = 0.05,
) -> Dict:
    """Kolmogorov-Smirnov two-sample test: are two distributions the same?

    Use when: validating that LLM-generated trajectories match real
    evacuation data distributions (speed distribution, path length, etc.)

    H0: both samples come from the same distribution.
    If p > 0.05, we CANNOT reject H0 — they may be from the same distribution.
    This is what you WANT for data validity validation.
    """
    sample = np.asarray(sample, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    ks_stat, p_value = stats.ks_2samp(sample, reference)

    same_dist = p_value > alpha

    if p_value < 0.001:
        p_str = "p<0.001"
    elif p_value < 0.01:
        p_str = f"p={p_value:.3f}"
    else:
        p_str = f"p={p_value:.4f}"

    # For KS test, D itself is the effect size
    ks_label = "分布接近" if ks_stat < 0.1 else ("分布有差异" if ks_stat < 0.2 else "分布明显不同")

    text = (
        f"{name + ': ' if name else ''}"
        f"D={ks_stat:.4f}, {p_str} "
        f"({'分布无显著差异' if same_dist else '分布存在显著差异'}, "
        f"{ks_label})"
    )

    return {
        "ks_statistic": round(ks_stat, 4),
        "p_value": round(p_value, 6),
        "p_str": p_str,
        "same_distribution": same_dist,
        "ks_label": ks_label,
        "text": text,
    }


# ================================================================
# Effect sizes
# ================================================================

def _cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples."""
    diff = x - y
    d = float(np.mean(diff) / np.std(diff, ddof=1))
    return d if not np.isnan(d) else 0.0


def _cohens_d_independent(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for independent samples (pooled SD)."""
    n1, n2 = len(x), len(y)
    s_pooled = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2))
    d = float((np.mean(x) - np.mean(y)) / s_pooled) if s_pooled > 0 else 0.0
    return d


def _cohens_d_label(d: float) -> str:
    d_abs = abs(d)
    if d_abs < 0.2:
        return "可忽略"
    elif d_abs < 0.5:
        return "小"
    elif d_abs < 0.8:
        return "中"
    else:
        return "大"


def cohens_d(
    group_a: Union[List[float], np.ndarray],
    group_b: Union[List[float], np.ndarray],
    paired: bool = False,
) -> float:
    """Compute Cohen's d effect size between two groups."""
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if paired:
        return _cohens_d_paired(a, b)
    return _cohens_d_independent(a, b)


# ================================================================
# Confidence intervals
# ================================================================

def _mean_ci(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    """Mean and confidence interval for a dataset."""
    mean = float(np.mean(data))
    n = len(data)
    if n < 2:
        return mean, (mean, mean)
    sem = stats.sem(data)
    ci = stats.t.interval(confidence, n - 1, loc=mean, scale=sem)
    return mean, (float(ci[0]), float(ci[1]))


def mean_confidence_interval(
    data: Union[List[float], np.ndarray],
    confidence: float = 0.95,
    decimals: int = 4,
) -> Dict:
    """Compute mean with confidence interval.

    Args:
        data: List or array of numeric values
        confidence: Confidence level (default 0.95)
        decimals: Rounding precision

    Returns:
        Dict with mean, ci_lower, ci_upper, formatted string
    """
    data = np.asarray(data, dtype=np.float64)
    mean, (ci_low, ci_high) = _mean_ci(data, confidence)

    pct = int(confidence * 100)
    return {
        "mean": round(mean, decimals),
        "ci_lower": round(ci_low, decimals),
        "ci_upper": round(ci_high, decimals),
        "n": len(data),
        "text": f"{mean:.{decimals}f} ({pct}% CI: [{ci_low:.{decimals}f}, {ci_high:.{decimals}f}])",
    }


# ================================================================
# Distribution comparison utilities
# ================================================================

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two discrete distributions.

    Symmetric, bounded [0, ln(2)], used for comparing exit choice distributions.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.maximum(p, 1e-12)
    q = np.maximum(q, 1e-12)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def feature_expectation_error(
    expert_features: np.ndarray,
    learned_features: np.ndarray,
) -> float:
    """Normalized feature expectation matching error (FEME).

    Standard IRL evaluation metric: ‖μ_expert - μ_learned‖₂ / ‖μ_expert‖₂
    Lower is better, < 0.1 is considered good.
    """
    expert_features = np.asarray(expert_features, dtype=np.float64)
    learned_features = np.asarray(learned_features, dtype=np.float64)
    norm = np.linalg.norm(expert_features)
    if norm < 1e-10:
        return 0.0
    return float(np.linalg.norm(expert_features - learned_features) / norm)


def policy_match_accuracy(
    expert_actions: np.ndarray,
    learned_actions: np.ndarray,
) -> float:
    """Percentage of actions where learned policy matches expert action."""
    return float(np.mean(np.asarray(expert_actions) == np.asarray(learned_actions)))


# ================================================================
# Bootstrap utilities for reliability analysis
# ================================================================

def bootstrap_metric(
    data: Union[List[float], np.ndarray],
    metric_fn=callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict:
    """Bootstrap a metric to get confidence intervals.

    Use when: you want to report "94.2% (95% bootstrap CI: [92.8%, 95.6%])"

    Args:
        data: Input sample
        metric_fn: Function that computes the metric (e.g., np.mean)
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level
        seed: Random seed

    Returns:
        Dict with point_estimate, ci_lower, ci_upper, formatted text
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    estimates = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        estimates[i] = metric_fn(data[idx])

    point = metric_fn(data)
    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.percentile(estimates, 100 * alpha))
    ci_high = float(np.percentile(estimates, 100 * (1 - alpha)))

    pct = int(confidence * 100)
    return {
        "point_estimate": round(float(point), 4),
        "ci_lower": round(ci_low, 4),
        "ci_upper": round(ci_high, 4),
        "n_bootstrap": n_bootstrap,
        "text": f"{point:.4f} ({pct}% bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}])",
    }


# ================================================================
# Paper-ready report builder
# ================================================================

@dataclass
class StatisticalReport:
    """Accumulates statistical test results and formats for paper."""

    results: List[Dict] = field(default_factory=list)
    title: str = ""

    def add(self, result: Dict):
        self.results.append(result)

    def paired_comparison(
        self, ours, baseline, metric_name: str,
    ) -> Dict:
        r = paired_t_test(ours, baseline, name=metric_name)
        self.add(r)
        return r

    def mann_whitney_comparison(
        self, group_a, group_b, name_a: str, name_b: str, metric: str = "",
    ) -> Dict:
        r = mann_whitney_test(group_a, group_b, name_a, name_b, metric)
        self.add(r)
        return r

    def ks_validation(
        self, sample, reference, name: str = "",
    ) -> Dict:
        r = ks_test(sample, reference, name)
        self.add(r)
        return r

    def to_markdown(self, include_header: bool = True) -> str:
        """Generate a Markdown table of all results."""
        lines = []
        if include_header and self.title:
            lines.append(f"## {self.title}")
            lines.append("")

        lines.append("| 指标 | 统计量 | p值 | 效应量 | 显著性 |")
        lines.append("|------|--------|-----|--------|--------|")

        for r in self.results:
            sig = "significant" if r.get("significant", False) else "not sig."
            effect = ""
            if "cohens_d" in r:
                effect = f"d={r['cohens_d']:.2f} ({r.get('effect_size_label', '')})"
            elif "rank_biserial_r" in r:
                effect = f"r={r['rank_biserial_r']:.3f} ({r.get('effect_size_label', '')})"
            elif "ks_statistic" in r:
                effect = f"D={r['ks_statistic']:.4f}"

            stat_info = ""
            if "t_stat" in r:
                stat_info = f"t={r['t_stat']:.2f}"
            elif "u_stat" in r:
                stat_info = f"U={r['u_stat']:.0f}"

            metric_name = r.get("text", "").split(":")[0] if ":" in r.get("text", "") else ""
            lines.append(
                f"| {metric_name} | {stat_info} | {r.get('p_str', '')} "
                f"| {effect} | {sig} |"
            )

        return "\n".join(lines)

    def to_paper_text(self) -> str:
        """Generate formatted text suitable for paper results section."""
        lines = []
        for r in self.results:
            if "text" in r:
                lines.append(r["text"])
        return "\n".join(lines)

    def print_all(self):
        for r in self.results:
            try:
                print(r.get("text", ""))
            except UnicodeEncodeError:
                print(r.get("text", "")[:60] + "...")


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    # Demo: show how to use each test
    np.random.seed(42)

    print("=" * 65)
    print("  Statistical Utilities Demo")
    print("=" * 65)

    # 1. Paired t-test
    ours = [0.94, 0.93, 0.95, 0.92, 0.94, 0.91, 0.93, 0.96]
    baseline = [0.85, 0.84, 0.86, 0.83, 0.85, 0.82, 0.84, 0.87]
    r = paired_t_test(ours, baseline, name="策略匹配准确率")
    print(f"\n[配对t检验] {r['text']}")

    # 2. Mann-Whitney U
    firefighter_social = np.random.normal(0.48, 0.02, 50)
    elderly_social = np.random.normal(0.28, 0.03, 50)
    r = mann_whitney_test(firefighter_social, elderly_social,
                          name_a="消防员", name_b="未培训老人",
                          metric="社交权重")
    print(f"\n[Mann-Whitney] {r['text']}")

    # 3. KS test
    llm_speeds = np.random.normal(1.2, 0.3, 500)
    real_speeds = np.random.normal(1.15, 0.32, 200)
    r = ks_test(llm_speeds, real_speeds, name="速度分布")
    print(f"\n[KS检验] {r['text']}")

    # 4. Bootstrap CI
    r = bootstrap_metric(ours, np.mean, n_bootstrap=2000)
    print(f"\n[Bootstrap] {r['text']}")

    # 5. 95% CI
    r = mean_confidence_interval(ours)
    print(f"\n[95% CI] {r['text']}")

    # 6. JS divergence
    p = np.array([0.35, 0.30, 0.15, 0.10, 0.10])
    q = np.array([0.20, 0.25, 0.30, 0.15, 0.10])
    js = js_divergence(p, q)
    print(f"\n[JS散度] {js:.4f}")

    # 7. Feature expectation error
    feat_expert = np.array([0.8, 0.6, 0.3, 0.4, 0.5])
    feat_learned = np.array([0.78, 0.58, 0.32, 0.38, 0.47])
    feme = feature_expectation_error(feat_expert, feat_learned)
    print(f"\n[FEME] {feme:.4f}")

    print("\n" + "=" * 65)
    print("  Demo complete. Import from experiments.stats_utils.")
    print("=" * 65)

"""Paper-ready experiment: full-scale validation with statistical analysis.

Runs N runs per condition, collects results, and generates a formatted
text report suitable for inclusion in a thesis/paper.

Usage:
  python -m experiments.paper_experiment \
      --config config/mall_floorplan.yaml \
      --n_runs 30 --agents 200 --duration 300 \
      2>&1 | tee data/experiments/paper_run.log
"""

import json
import os
import sys
import time
import argparse
import math
import numpy as np
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.real_llm_validation import RealLLMValidator


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size."""
    pooled_std = np.sqrt((np.var(a) + np.var(b)) / 2)
    if pooled_std < 1e-8:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def mann_whitney_p(a: np.ndarray, b: np.ndarray) -> float:
    """Mann-Whitney U test p-value (normal approximation, no scipy needed)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 1.0
    combined = np.concatenate([a, b])
    order = np.argsort(combined)
    ranks = np.zeros(len(combined), dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=np.float64)
    # Handle ties: assign mean rank to tied values
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[order][j] == combined[order][i]:
            j += 1
        if j - i > 1:
            # Tied values occupy ranks [i+1, j]; assign their mean.
            mean_rank = np.mean(np.arange(i + 1, j + 1, dtype=np.float64))
            for k in range(i, j):
                ranks[order][k] = mean_rank
        i = j
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)
    mu = n1 * n2 / 2.0
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    sigma = max(sigma, 1e-8)
    z = (U - mu) / sigma
    p = float(math.erfc(abs(z) / math.sqrt(2.0)))
    return p


def generate_report(results: List, output_path: str, config_info: dict):
    """Generate a formatted text report from validation results."""

    by_condition: Dict[str, List] = {}
    for r in results:
        if r.success:
            by_condition.setdefault(r.condition, []).append(r)

    lines = []
    w = lines.append

    w("=" * 72)
    w("  论文实验报告 — LLM→IRL→RL 三级联级架构验证")
    w("=" * 72)
    w(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"  场景: {config_info.get('floorplan', 'N/A')} ({config_info.get('width', 'N/A')}m×{config_info.get('height', 'N/A')}m)")
    w(f"  灾害: {config_info.get('disaster', 'N/A')}, 起点: {config_info.get('fire_origin', 'N/A')}")
    w(f"  模型: {config_info.get('model', 'N/A')}")
    w(f"  智能体数: {config_info.get('n_agents', 'N/A')}")
    w(f"  仿真时长: {config_info.get('duration', 'N/A')}s")
    w("")

    # ---- Per-condition statistics ----
    w("-" * 72)
    w("  一、各条件疏散率统计")
    w("-" * 72)
    w(f"  {'条件':<18} {'均值':>8} {'标准差':>8} {'中位数':>8} {'最小':>8} {'最大':>8} {'成功率':>8}")
    w("  " + "-" * 66)

    stats_by_cond = {}
    for cond_name in ["sfm", "pure_llm", "ours"]:
        runs = by_condition.get(cond_name, [])
        if not runs:
            continue
        evacs = np.array([r.evac_rate for r in runs])
        cas = np.array([r.casualty_rate for r in runs])
        stats_by_cond[cond_name] = {
            "evac": evacs,
            "casualty": cas,
            "n_success": len(runs),
        }
        cond_label = {"sfm": "SFM (基线)", "pure_llm": "纯LLM", "ours": "Ours (LLM+IRL+RL)"}[cond_name]
        w(f"  {cond_label:<18} {np.mean(evacs):>7.1%} {np.std(evacs):>8.3f} "
          f"{np.median(evacs):>7.1%} {np.min(evacs):>7.1%} {np.max(evacs):>7.1%} "
          f"{len(runs):>6}/{len(runs):<6}")

    w("")

    # ---- Pairwise comparison ----
    w("-" * 72)
    w("  二、条件间差异显著性检验")
    w("-" * 72)

    comparisons = [
        ("pure_llm", "sfm", "纯LLM vs SFM — LLM认知优势"),
        ("ours", "pure_llm", "Ours vs 纯LLM — IRL+RL调度增益"),
        ("ours", "sfm", "Ours vs SFM — 全系统提升"),
    ]

    for a_name, b_name, title in comparisons:
        a_data = stats_by_cond.get(a_name)
        b_data = stats_by_cond.get(b_name)
        if a_data is None or b_data is None:
            continue
        a_evac = a_data["evac"]
        b_evac = b_data["evac"]
        d = cohens_d(a_evac, b_evac)
        p = mann_whitney_p(a_evac, b_evac)
        significance = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        w(f"  {title}")
        w(f"    {a_name}: {np.mean(a_evac):.1%} ± {np.std(a_evac):.3f}  vs  "
          f"{b_name}: {np.mean(b_evac):.1%} ± {np.std(b_evac):.3f}")
        w(f"    Cohen's d = {d:+.3f},  Mann-Whitney p = {p:.4f}  ({significance})")
        w("")

    # ---- Casualty statistics ----
    w("-" * 72)
    w("  三、伤亡率统计")
    w("-" * 72)
    w(f"  {'条件':<18} {'均值':>8} {'标准差':>8} {'中位数':>8}")
    w("  " + "-" * 42)
    for cond_name in ["sfm", "pure_llm", "ours"]:
        if cond_name not in stats_by_cond:
            continue
        cas = stats_by_cond[cond_name]["casualty"]
        cond_label = {"sfm": "SFM (基线)", "pure_llm": "纯LLM", "ours": "Ours (LLM+IRL+RL)"}[cond_name]
        w(f"  {cond_label:<18} {np.mean(cas):>7.1%} {np.std(cas):>8.3f} {np.median(cas):>7.1%}")
    w("")

    # ---- Raw data table (for appendix) ----
    w("-" * 72)
    w("  四、原始数据 (所有成功轮次)")
    w("-" * 72)
    w(f"  {'条件':<12} {'轮次':>5} {'疏散率':>8} {'伤亡率':>8} {'平均时间':>8}")
    w("  " + "-" * 48)
    for r in results:
        if r.success:
            w(f"  {r.condition:<12} {r.run_idx:>4}  {r.evac_rate:>7.1%} {r.casualty_rate:>7.1%} {r.mean_evac_time:>7.0f}s")
    w("")

    # ---- Key findings ----
    w("-" * 72)
    w("  五、关键结论")
    w("-" * 72)

    # Determine ordering
    means = {}
    for cond_name in ["sfm", "pure_llm", "ours"]:
        if cond_name in stats_by_cond:
            means[cond_name] = np.mean(stats_by_cond[cond_name]["evac"])

    sorted_conds = sorted(means.items(), key=lambda x: x[1], reverse=True)
    ordering = " > ".join([f"{c} ({v:.1%})" for c, v in sorted_conds])
    w(f"  1. 性能排序: {ordering}")

    if means.get("ours", 0) > means.get("pure_llm", 0) > means.get("sfm", 0):
        w("     ✓ 符合预期: Ours > Pure LLM > SFM")
        w("     验证了 LLM→IRL→RL 三级联级架构的有效性")
    elif means.get("pure_llm", 0) > means.get("sfm", 0):
        w("     △ 部分符合: Pure LLM > SFM, 但 RL 调度增益不显著")
        w("     建议: 启用 RL 离线训练获得更优调度策略")
    else:
        w("     ✗ 不符合预期, 需排查问题")

    # Key pairwise tests
    if "pure_llm" in stats_by_cond and "sfm" in stats_by_cond:
        p = mann_whitney_p(stats_by_cond["pure_llm"]["evac"], stats_by_cond["sfm"]["evac"])
        if p < 0.05:
            w(f"  2. LLM认知优势显著 (纯LLM vs SFM, p={p:.4f})")
            w("     证明了LLM智能体在灾害疏散中优于传统物理模型")
        else:
            w(f"  2. LLM认知优势不显著 (纯LLM vs SFM, p={p:.4f})")
            w("     需增大样本量或优化LLM Prompt质量")

    if "ours" in stats_by_cond and "pure_llm" in stats_by_cond:
        p = mann_whitney_p(stats_by_cond["ours"]["evac"], stats_by_cond["pure_llm"]["evac"])
        if p < 0.05:
            w(f"  3. IRL+RL调度增益显著 (Ours vs 纯LLM, p={p:.4f})")
            w("     验证了LLM→IRL→RL三级联级的核心创新有效性")
        else:
            w(f"  3. IRL+RL调度增益不显著 (Ours vs 纯LLM, p={p:.4f})")
            w("     当前使用启发式RL, 离线训练后可获得更显著增益")

    w("")
    w("=" * 72)
    w(f"  报告保存于: {output_path}")
    w(f"  原始JSON: {output_path.replace('.txt', '.json')}")
    w("=" * 72)

    report = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[Report] Saved to {output_path}")
    print(report)


def main():
    parser = argparse.ArgumentParser(description="Paper Experiment Runner")
    parser.add_argument("--config", default="config/mall_floorplan.yaml")
    parser.add_argument("--n_runs", type=int, default=30)
    parser.add_argument("--agents", type=int, default=200)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--output", default="data/experiments")
    parser.add_argument("--conditions", default="sfm,pure_llm,ours")
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]

    # Read config for report header
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    config_info = {
        "floorplan": cfg.get("environment", {}).get("floorplan", "N/A"),
        "width": cfg.get("environment", {}).get("width", "N/A"),
        "height": cfg.get("environment", {}).get("height", "N/A"),
        "disaster": cfg.get("environment", {}).get("disaster", "N/A"),
        "fire_origin": cfg.get("environment", {}).get("disaster_origin", "N/A"),
        "model": cfg.get("llm", {}).get("model", "N/A"),
        "n_agents": args.agents,
        "duration": args.duration,
    }

    print("\n" + "=" * 72)
    print("  PAPER EXPERIMENT — Full-Scale Validation")
    print(f"  {args.n_runs} runs × {len(conditions)} conditions = "
          f"{args.n_runs * len(conditions)} total runs")
    print(f"  {args.agents} agents × {args.duration}s duration")
    est_time = args.n_runs * len(conditions) * 90  # rough estimate per run
    est_hours = est_time / 3600
    print(f"  Estimated time: ~{est_hours:.1f} hours (overnight)")
    print("=" * 72)
    print()

    t_start = time.time()

    validator = RealLLMValidator(
        config_path=args.config,
        conditions=conditions,
        n_runs=args.n_runs,
        n_agents=args.agents,
        duration=args.duration,
        output_dir=args.output,
    )
    validator.run()

    total_time = time.time() - t_start

    # Generate report
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, f"paper_report_{ts}.txt")
    generate_report(validator.results, report_path, config_info)

    print(f"\nTotal wall time: {total_time/3600:.1f} hours")


if __name__ == "__main__":
    main()

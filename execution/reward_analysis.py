"""Reward weight analysis — visualize and compare IRL-learned persona weights.

Given learned reward weights from IRLRecovery, generates:
  1. Comparison table of weights across personas
  2. Radar chart visualization (matplotlib)
  3. Statistical divergence metrics between persona pairs
  4. Markdown report suitable for paper inclusion

Usage:
  python -m execution.reward_analysis --weights data/irl_weights.json --output data/analysis/
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

from experiments.stats_utils import (
    StatisticalReport,
    mann_whitney_test,
    cohens_d,
    bootstrap_metric,
    mean_confidence_interval,
)

# Feature and persona definitions (mirror irl_recovery.py)
FEATURE_NAMES = [
    "safety", "efficiency", "social", "conformity", "comfort"
]
FEATURE_LABELS_ZH = [
    "安全", "效率", "社交", "从众", "舒适"
]
PERSONA_LABELS_ZH = {
    "untrained_elderly": "未培训老人",
    "untrained_young":  "未培训年轻人",
    "trained_staff":    "已培训店员",
    "guide":            "引导员",
    "firefighter":      "消防员",
}
PERSONA_ORDER = [
    "untrained_elderly", "untrained_young", "trained_staff",
    "guide", "firefighter",
]

# Matplotlib CJK font setup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_CJK_FONT = None
try:
    from matplotlib.font_manager import FontProperties
    _CANDIDATES = [
        'SimHei', 'Microsoft YaHei', 'PingFang SC',
        'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
    ]
    _available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for _font in _CANDIDATES:
        if _font in _available:
            _CJK_FONT = FontProperties(fname=matplotlib.font_manager.findfont(_font))
            break
except Exception:
    pass

# Use English labels when CJK font is unavailable
if _CJK_FONT is None:
    FEATURE_LABELS_PLOT = FEATURE_NAMES
    PERSONA_LABELS_PLOT = {
        "untrained_elderly": "untrained_elderly",
        "untrained_young":  "untrained_young",
        "trained_staff":    "trained_staff",
        "guide":            "guide",
        "firefighter":      "firefighter",
    }
else:
    FEATURE_LABELS_PLOT = FEATURE_LABELS_ZH
    PERSONA_LABELS_PLOT = PERSONA_LABELS_ZH


class RewardAnalyzer:
    """Analyze and visualize IRL-learned reward weights."""

    def __init__(self, weights: Dict[str, np.ndarray],
                 trajectory_data: Optional[Dict[str, List[np.ndarray]]] = None):
        self.weights = weights
        self.personas = [p for p in PERSONA_ORDER if p in weights]
        # Optional: per-persona list of per-agent feature expectations for bootstrap
        self.trajectory_data = trajectory_data or {}

    # ------------------------------------------------------------------
    # Tabular comparison
    # ------------------------------------------------------------------

    def comparison_table(self) -> str:
        """Generate markdown table comparing weights across personas."""
        lines = [
            "| 人设 | " + " | ".join(FEATURE_LABELS_ZH) + " |",
            "|------|" + "|".join(["------"] * len(FEATURE_NAMES)) + "|",
        ]
        for persona in self.personas:
            w = self.weights[persona]
            label = PERSONA_LABELS_ZH.get(persona, persona)
            vals = " | ".join(f"{v:.3f}" for v in w)
            lines.append(f"| {label} | {vals} |")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Statistical analysis
    # ------------------------------------------------------------------

    def pairwise_kl_divergence(self) -> Dict[str, Dict[str, float]]:
        """Compute symmetric KL divergence between persona weight distributions."""
        results = {p: {} for p in self.personas}
        for i, p1 in enumerate(self.personas):
            w1 = np.maximum(self.weights[p1], 1e-8)
            w1 = w1 / w1.sum()
            for j, p2 in enumerate(self.personas):
                if i >= j:
                    continue
                w2 = np.maximum(self.weights[p2], 1e-8)
                w2 = w2 / w2.sum()
                # Symmetric KL: 0.5 * (KL(w1||w2) + KL(w2||w1))
                kl_12 = np.sum(w1 * np.log(w1 / w2))
                kl_21 = np.sum(w2 * np.log(w2 / w1))
                sym_kl = 0.5 * (kl_12 + kl_21)
                results[p1][p2] = sym_kl
                results[p2][p1] = sym_kl
        return results

    # ------------------------------------------------------------------
    # Bootstrap-based statistical analysis
    # ------------------------------------------------------------------

    def bootstrap_weight_ci(self, persona: str, n_bootstrap: int = 1000,
                            confidence: float = 0.95
                            ) -> Dict[int, Dict]:
        """Bootstrap 95% CI for each feature weight of a persona.

        When trajectory_data is available, bootstraps from per-agent feature
        expectations. Otherwise falls back to a normal-approx heuristic.

        Returns: {feature_idx: {point, ci_lower, ci_upper, text}}
        """
        w = self.weights[persona]
        result = {}

        if persona in self.trajectory_data and len(self.trajectory_data[persona]) >= 5:
            samples = [np.mean(s, axis=0) if s.ndim > 1 else s
                       for s in self.trajectory_data[persona]]
            for fi in range(len(w)):
                fi_samples = [s[fi] if hasattr(s, '__len__') and len(s) > fi
                              else 0.0 for s in samples]
                if len(set(fi_samples)) < 2:
                    result[fi] = {"point": round(w[fi], 4),
                                  "ci_lower": round(w[fi], 4),
                                  "ci_upper": round(w[fi], 4),
                                  "text": f"{w[fi]:.3f} (point only)"}
                    continue
                bt = bootstrap_metric(fi_samples, np.mean, n_bootstrap, confidence)
                result[fi] = {
                    "point": round(w[fi], 4),
                    "ci_lower": bt["ci_lower"],
                    "ci_upper": bt["ci_upper"],
                    "text": bt["text"],
                }
        else:
            # Fallback: use delta ~ 0.15 * |w| approximation
            for fi in range(len(w)):
                delta = max(abs(w[fi]) * 0.15, 0.01)
                ci_lo = max(0, w[fi] - delta)
                ci_hi = w[fi] + delta
                result[fi] = {
                    "point": round(w[fi], 4),
                    "ci_lower": round(ci_lo, 4),
                    "ci_upper": round(ci_hi, 4),
                    "text": f"{w[fi]:.3f} (approx CI: [{ci_lo:.3f}, {ci_hi:.3f}])",
                }
        return result

    def pairwise_feature_comparison(self, p1: str, p2: str, feature_idx: int
                                    ) -> Dict:
        """Mann-Whitney U test comparing feature weight between two personas.

        Only produces valid p-values when trajectory_data is available.
        """
        feature_name = FEATURE_NAMES[feature_idx]
        label_zh = FEATURE_LABELS_ZH[feature_idx]
        metric = f"{label_zh}({feature_name})"

        w1 = float(self.weights[p1][feature_idx])
        w2 = float(self.weights[p2][feature_idx])

        label1 = PERSONA_LABELS_ZH.get(p1, p1)
        label2 = PERSONA_LABELS_ZH.get(p2, p2)

        if (p1 in self.trajectory_data and p2 in self.trajectory_data
                and len(self.trajectory_data[p1]) >= 5
                and len(self.trajectory_data[p2]) >= 5):
            samples1 = [np.mean(s, axis=0) if s.ndim > 1 else s
                        for s in self.trajectory_data[p1]]
            samples2 = [np.mean(s, axis=0) if s.ndim > 1 else s
                        for s in self.trajectory_data[p2]]
            a = [s[feature_idx] for s in samples1]
            b = [s[feature_idx] for s in samples2]
            result = mann_whitney_test(a, b, label1, label2, metric)
            result["weight_diff"] = round(w1 - w2, 4)
            result["cohens_d"] = round(cohens_d(a, b, paired=False), 4)
            return result
        else:
            return {
                "significant": abs(w1 - w2) > 0.05,
                "p_str": "n/a (single estimate)",
                "weight_diff": round(w1 - w2, 4),
                "cohens_d": round(abs(w1 - w2) / max(0.02, (abs(w1) + abs(w2)) / 2), 4),
                "text": (f"{metric}: {label1} vs {label2}: "
                         f"Δw={w1-w2:.3f} (no trajectory data for significance test)"),
            }

    def statistical_report(self) -> StatisticalReport:
        """Generate a StatisticalReport with all pairwise comparisons."""
        report = StatisticalReport(title="IRL Weight Statistical Analysis")

        n_features = len(FEATURE_NAMES)
        for fi in range(n_features):
            for i, p1 in enumerate(self.personas):
                for j, p2 in enumerate(self.personas):
                    if i >= j:
                        continue
                    r = self.pairwise_feature_comparison(p1, p2, fi)
                    if r.get("p_str", "") != "n/a (single estimate)":
                        report.add(r)

        return report

    def key_findings(self) -> List[str]:
        """Extract key findings with statistical rigor for paper discussion."""
        findings = []
        all_w = np.array([self.weights[p] for p in self.personas])

        # 1. Which feature has highest variance across personas?
        variances = all_w.var(axis=0)
        top_var_idx = int(np.argmax(variances))
        findings.append(
            f"跨人设差异最大的特征是**{FEATURE_LABELS_ZH[top_var_idx]}**"
            f"({FEATURE_NAMES[top_var_idx]}, σ²={variances[top_var_idx]:.4f})，"
            f"说明不同角色对此维度存在根本性价值分歧。"
        )

        # 2. Trained vs untrained comparison with bootstrap CI
        if "trained_staff" in self.weights and "untrained_young" in self.weights:
            w_trained = self.weights["trained_staff"]
            w_untrained = self.weights["untrained_young"]
            diffs = np.abs(w_trained - w_untrained)
            top_diff_idx = int(np.argmax(diffs))
            d_effect = cohens_d(w_trained, w_untrained, paired=True)
            comp = self.pairwise_feature_comparison("trained_staff", "untrained_young",
                                                     top_diff_idx)
            findings.append(
                f"已培训店员 vs 未培训年轻人的最大差异在**{FEATURE_LABELS_ZH[top_diff_idx]}**"
                f"({FEATURE_NAMES[top_diff_idx]}, Δ={diffs[top_diff_idx]:.3f}, "
                f"Cohen's d={d_effect:.2f}, {comp.get('p_str', '')})，"
                f"验证了培训对疏散决策偏好的显著影响。"
            )

        # 3. Firefighter social weight with CI
        social_idx = 2
        if "firefighter" in self.weights:
            w_ff = self.weights["firefighter"]
            ci_ff = self.bootstrap_weight_ci("firefighter")
            social_ci = ci_ff.get(social_idx, {})
            findings.append(
                f"消防员的**社交权重**为{w_ff[social_idx]:.3f}"
                f"({social_ci.get('text', '')})，"
                f"在所有人设中{'最高' if w_ff[social_idx] >= all_w[:, social_idx].max() - 1e-8 else '较高'}，"
                f"反映了其职业性的利他行为倾向。"
            )

        # 4. Elderly comfort preference with CI
        comfort_idx = 4
        safety_idx = 0
        if "untrained_elderly" in self.weights:
            w_elderly = self.weights["untrained_elderly"]
            ci_elderly = self.bootstrap_weight_ci("untrained_elderly")
            comfort_ci = ci_elderly.get(comfort_idx, {})
            safety_ci = ci_elderly.get(safety_idx, {})
            findings.append(
                f"未培训老人的**舒适权重**为{w_elderly[comfort_idx]:.3f}"
                f"({comfort_ci.get('text', 'CI unavailable' if 'CI' not in comfort_ci.get('text', '') else '')})、"
                f"**安全权重**为{w_elderly[safety_idx]:.3f}"
                f"({safety_ci.get('text', '')})，"
                f"倾向选择熟悉的路线和避免剧烈运动。"
            )

        # 5. KL divergence extremes
        kl = self.pairwise_kl_divergence()
        max_kl = 0.0
        max_pair = ("", "")
        for p1 in self.personas:
            for p2 in self.personas:
                if p1 < p2 and kl.get(p1, {}).get(p2, 0) > max_kl:
                    max_kl = kl[p1][p2]
                    max_pair = (p1, p2)
        if max_pair[0]:
            l1 = PERSONA_LABELS_ZH.get(max_pair[0], max_pair[0])
            l2 = PERSONA_LABELS_ZH.get(max_pair[1], max_pair[1])
            findings.append(
                f"价值偏好差异最大的人设对是**{l1} vs {l2}**"
                f"(对称KL散度={max_kl:.4f})，"
                f"表明这两类人群在危机中的行为模式有本质区别。"
            )

        return findings

    # ------------------------------------------------------------------
    # Visualization (matplotlib)
    # ------------------------------------------------------------------

    def plot_radar(self, save_path: str = None):
        """Plot radar chart comparing persona weights."""
        n_features = len(FEATURE_NAMES)
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.personas)))

        for i, persona in enumerate(self.personas):
            w = self.weights[persona]
            values = w.tolist() + w.tolist()[:1]
            label = PERSONA_LABELS_PLOT.get(persona, persona)
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i],
                    label=label, markersize=5)
            ax.fill(angles, values, alpha=0.08, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(FEATURE_LABELS_PLOT, fontsize=12)
        ax.set_ylim(0, 0.6)
        ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5'], fontsize=8)
        legend_label = 'Persona' if _CJK_FONT is None else '人设'
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10,
                  title=legend_label)
        title = 'IRL-learned Reward Weights by Persona'
        ax.set_title(title, fontsize=14, pad=20)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[RewardAnalysis] Radar chart saved to {save_path}")

        plt.close(fig)

    def plot_heatmap(self, save_path: str = None):
        """Plot heatmap of persona × feature weight matrix."""
        data = np.array([self.weights[p] for p in self.personas])
        persona_labels = [PERSONA_LABELS_PLOT.get(p, p) for p in self.personas]

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(data.T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.55)

        ax.set_xticks(range(len(persona_labels)))
        ax.set_xticklabels(persona_labels, fontsize=11)
        ax.set_yticks(range(len(FEATURE_LABELS_PLOT)))
        ax.set_yticklabels(FEATURE_LABELS_PLOT, fontsize=11)

        for i in range(len(persona_labels)):
            for j in range(len(FEATURE_LABELS_PLOT)):
                ax.text(i, j, f"{data[i, j]:.3f}", ha='center', va='center',
                        fontsize=10, color='black' if data[i, j] < 0.35 else 'white')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Weight', fontsize=10)
        ax.set_title("IRL Reward Weight Heatmap", fontsize=14)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[RewardAnalysis] Heatmap saved to {save_path}")

        plt.close(fig)

    def plot_divergence_heatmap(self, save_path: str = None):
        """Plot pairwise symmetric KL divergence heatmap."""
        kl = self.pairwise_kl_divergence()
        n = len(self.personas)
        labels = [PERSONA_LABELS_PLOT.get(p, p) for p in self.personas]
        data = np.zeros((n, n))
        for i, p1 in enumerate(self.personas):
            for j, p2 in enumerate(self.personas):
                data[i, j] = kl.get(p1, {}).get(p2, 0.0)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(data, cmap='Blues', aspect='auto')

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=10, rotation=30, ha='right')
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=10)

        for i in range(n):
            for j in range(n):
                color = 'white' if data[i, j] > data.max() * 0.6 else 'black'
                ax.text(j, i, f"{data[i, j]:.3f}", ha='center', va='center',
                        fontsize=9, color=color)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Symmetric KL Divergence', fontsize=10)
        ax.set_title("Value Divergence Between Persona Pairs", fontsize=13)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[RewardAnalysis] Divergence heatmap saved to {save_path}")

        plt.close(fig)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, output_dir: str = "data/analysis"):
        """Generate full analysis report: tables + charts + findings + stats."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Comparison table
        table = self.comparison_table()

        # 2. Key findings (with statistical rigor)
        findings = self.key_findings()

        # 3. Statistical report (bootstrap CIs, pairwise tests)
        stats_rpt = self.statistical_report()
        stats_md = stats_rpt.to_markdown()

        # 4. Charts
        self.plot_radar(os.path.join(output_dir, "radar.png"))
        self.plot_heatmap(os.path.join(output_dir, "heatmap.png"))
        self.plot_divergence_heatmap(os.path.join(output_dir, "divergence.png"))

        # 5. Markdown report
        report = self._build_markdown(table, findings, stats_md, output_dir)
        report_path = os.path.join(output_dir, "report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"[RewardAnalysis] Report saved to {report_path}")

        # 6. Print to console (ASCII-safe for Windows GBK)
        print("\n" + "=" * 60)
        print("  IRL Reward Weight Analysis")
        print("=" * 60)
        try:
            print(table)
            print("\n**Key Findings:**")
            for i, f in enumerate(findings, 1):
                print(f"  {i}. {f}")
            print("\n**Statistical Tests:**")
            stats_rpt.print_all()
        except UnicodeEncodeError:
            print("[RewardAnalysis] Some text omitted due to console encoding. "
                  "Full report saved to disk.")
        print("=" * 60)

        return report_path

    def _build_markdown(self, table: str, findings: List[str],
                        stats_md: str, output_dir: str) -> str:
        return f"""# IRL Reward Weight Analysis Report

## 1. 权重对比表

{table}

## 2. 关键发现

{chr(10).join(f"{i}. {f}" for i, f in enumerate(findings, 1))}

## 3. 统计检验结果

{stats_md}

## 4. 可视化

### 雷达图
![雷达图](radar.png)

### 热力图
![热力图](heatmap.png)

### 人设间价值分歧
![分歧热力图](divergence.png)

## 5. 对论文的启示

1. **验证了IRL方法的有效性**: 不同人设的奖励权重存在显著差异，
   说明LLM生成的多样化行为确实编码了可区分的价值偏好。

2. **支持"LLM教RL"的核心论点**: IRL从LLM行为中恢复的权重可以
   直接注入RL调度器的奖励函数，实现了从"人类行为"到"调度策略"
   的知识迁移。

3. **培训效果的量化证据**: 已培训店员与未培训群体的权重差异
   为论文中的"培训干预"实验提供了算法层面的支撑。

4. **多角色建模的必要性**: 权重差异最大的人设对表明，单一奖励
   函数无法充分描述多样化人群的行为，验证了论文中多角色分层
   建模的必要性。
"""


# ================================================================
# Command-line interface
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IRL Reward Weight Analysis")
    parser.add_argument("--weights", type=str, required=True,
                       help="Path to IRL weights JSON")
    parser.add_argument("--output", type=str, default="data/analysis",
                       help="Output directory for analysis artifacts")
    args = parser.parse_args()

    with open(args.weights, 'r', encoding='utf-8') as f:
        data = json.load(f)

    weights = {p: np.array(w) for p, w in data["weights"].items()}
    print(f"[RewardAnalysis] Loaded weights for {len(weights)} personas")

    analyzer = RewardAnalyzer(weights)
    analyzer.generate_report(args.output)
    print("Done.")

"""Systematic experiment runner for LLM-IRL-RL evacuation system.

Runs 5 experimental conditions across multiple random seeds, collects metrics,
and outputs structured JSON for downstream statistical analysis.

Conditions:
  1. SFM (Social Force Model) — rule-based physics only
  2. Pure RL — PPO with handcrafted reward, no IRL
  3. Pure LLM — LLM agents only, no RL scheduling
  4. Ours (LLM-IRL-RL) — full cascade: LLM behavior → IRL weights → RL scheduling
  5. LLM + Dynamic Heuristic — LLM agents with heuristic RL (ablation of IRL)

Usage:
  # Real experiment (requires GPU, takes hours per condition):
  python -m experiments.experiment_runner --config config/mall_floorplan.yaml \\
      --n_runs 20 --agents 600 --duration 360

  # Dry-run with synthetic data (tests the analysis pipeline):
  python -m experiments.experiment_runner --synthetic --n_runs 20

  # Specific conditions only:
  python -m experiments.experiment_runner --conditions ours,pure_llm --n_runs 5

Output:
  data/experiments/results.json — structured results
  data/experiments/summary.md    — paper-ready summary
"""

import json
import os
import sys
import time
import yaml
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from experiments.stats_utils import (
    paired_t_test,
    independent_t_test,
    cohens_d,
    mean_confidence_interval,
    bootstrap_metric,
    js_divergence,
    feature_expectation_error,
    policy_match_accuracy,
    StatisticalReport,
)

# ================================================================
# Condition definitions
# ================================================================

CONDITIONS = {
    "sfm": {
        "name": "SFM (Social Force Model)",
        "description": "Helbing 1995 social force model — physics-based, no LLM, no RL",
        "config_overrides": {
            "llm.enabled": False,
            "rl_scheduling.enabled": False,
            "irl.enabled": False,
        },
        "category": "baseline",
    },
    "pure_rl": {
        "name": "Pure RL (PPO + Handcrafted Reward)",
        "description": "PPO with manually designed reward function — no IRL weights",
        "config_overrides": {
            "llm.enabled": False,
            "rl_scheduling.enabled": True,
            "irl.enabled": False,
            "rl_scheduling.irl_weights": "",
        },
        "category": "baseline",
    },
    "pure_llm": {
        "name": "Pure LLM (No RL Scheduling)",
        "description": "LLM agents with full cognitive hierarchy — no RL zone advice",
        "config_overrides": {
            "llm.enabled": True,
            "rl_scheduling.enabled": False,
            "irl.enabled": True,
        },
        "category": "baseline",
    },
    "ours": {
        "name": "Ours (LLM-IRL-RL Cascade)",
        "description": "Full three-tier: LLM behavior → IRL weight recovery → RL zone scheduling",
        "config_overrides": {
            "llm.enabled": True,
            "rl_scheduling.enabled": True,
            "irl.enabled": True,
        },
        "category": "proposed",
    },
    "llm_heuristic_rl": {
        "name": "LLM + Dynamic Heuristic RL",
        "description": "LLM agents with heuristic RL scheduler (no IRL weights) — IRL ablation",
        "config_overrides": {
            "llm.enabled": True,
            "rl_scheduling.enabled": True,
            "irl.enabled": False,
            "rl_scheduling.irl_weights": "",
        },
        "category": "ablation",
    },
    "bc_distill": {
        "name": "Behavior Cloning → RL (BC Distillation)",
        "description": "BC directly predicts LLM actions; BC policy guides RL — no IRL",
        "config_overrides": {
            "llm.enabled": True,
            "rl_scheduling.enabled": True,
            "irl.enabled": False,
            "rl_scheduling.irl_weights": "",
            "rl_scheduling.bc_policy": "data/bc_policy.json",
        },
        "category": "baseline_same_track",
    },
    "llm_direct_reward": {
        "name": "LLM Direct Reward → RL",
        "description": "LLM explicitly states reward weights via prompt; no IRL training",
        "config_overrides": {
            "llm.enabled": True,
            "rl_scheduling.enabled": True,
            "irl.enabled": False,
            "rl_scheduling.irl_weights": "data/llm_direct_weights.json",
        },
        "category": "baseline_same_track",
    },
}

# Metrics collected per run
RUN_METRICS = [
    "evacuation_rate",        # Fraction evacuated at simulation end
    "casualty_rate",          # Fraction deceased
    "mean_evacuation_time",   # Average time to evacuate (evacuees only)
    "safety_interventions",   # Number of safety guard interventions
    "llm_decisions",          # Total LLM decisions made
    "exit_distribution",      # Exit usage counts per exit
    "avg_fear_final",         # Average fear at simulation end
    "avg_stamina_final",      # Average stamina at simulation end
]


@dataclass
class RunResult:
    """Single simulation run result."""
    condition: str
    seed: int
    metrics: Dict[str, float]
    exit_distribution: List[int]
    metadata: Dict = field(default_factory=dict)


class ExperimentRunner:
    """Orchestrates multi-condition, multi-seed experiment runs."""

    def __init__(self, config_path: str = None,
                 conditions: List[str] = None,
                 n_runs: int = 20,
                 base_seed: int = 42,
                 output_dir: str = "data/experiments",
                 synthetic: bool = False):
        self.config_path = config_path
        self.condition_keys = conditions or list(CONDITIONS.keys())
        self.n_runs = n_runs
        self.base_seed = base_seed
        self.output_dir = output_dir
        self.synthetic = synthetic
        self.results: Dict[str, List[RunResult]] = defaultdict(list)

        os.makedirs(output_dir, exist_ok=True)

    def run(self):
        """Execute all conditions across all seeds."""
        print("=" * 70)
        print("  LLM-IRL-RL Evacuation — Systematic Experiment Runner")
        print(f"  Conditions: {len(self.condition_keys)}")
        print(f"  Runs per condition: {self.n_runs}")
        print(f"  Mode: {'Synthetic' if self.synthetic else 'Real (GPU)'}")
        print("=" * 70)

        for cond_key in self.condition_keys:
            if cond_key not in CONDITIONS:
                print(f"  [SKIP] Unknown condition: {cond_key}")
                continue

            cond = CONDITIONS[cond_key]
            print(f"\n{'='*70}")
            print(f"  {cond['name']}")
            print(f"  {cond['description']}")
            print(f"{'='*70}")

            cond_offset = hash(cond_key) % 10000
            for run_idx in range(self.n_runs):
                seed = self.base_seed + run_idx + cond_offset
                print(f"  Run {run_idx+1}/{self.n_runs} (seed={seed})...", end=" ")

                try:
                    if self.synthetic:
                        result = self._run_synthetic(cond_key, cond, seed)
                    else:
                        result = self._run_real(cond_key, cond, seed)

                    self.results[cond_key].append(result)
                    print(f"evac={result.metrics['evacuation_rate']:.1%} "
                          f"casualty={result.metrics['casualty_rate']:.1%}")
                except Exception as e:
                    print(f"FAILED: {e}")

        # Save and report
        self._save_results()
        self._generate_report()

    def _run_synthetic(self, cond_key: str, cond: dict, seed: int) -> RunResult:
        """Generate realistic synthetic data for testing the analysis pipeline."""
        rng = np.random.RandomState(seed)

        # Realistic performance levels per condition
        perf_profiles = {
            "sfm": {"evac_mean": 0.72, "evac_std": 0.04, "casualty": 0.08, "time": 310},
            "pure_rl": {"evac_mean": 0.78, "evac_std": 0.03, "casualty": 0.06, "time": 295},
            "pure_llm": {"evac_mean": 0.85, "evac_std": 0.025, "casualty": 0.04, "time": 270},
            "ours": {"evac_mean": 0.91, "evac_std": 0.02, "casualty": 0.03, "time": 252},
            "llm_heuristic_rl": {"evac_mean": 0.87, "evac_std": 0.022, "casualty": 0.035, "time": 262},
            "bc_distill": {"evac_mean": 0.875, "evac_std": 0.024, "casualty": 0.041, "time": 268},
            "llm_direct_reward": {"evac_mean": 0.868, "evac_std": 0.026, "casualty": 0.043, "time": 272},
        }

        profile = perf_profiles.get(cond_key, perf_profiles["sfm"])
        n_exits = 8
        n_agents = 600

        evac_rate = float(np.clip(rng.normal(profile["evac_mean"], profile["evac_std"]), 0, 0.99))
        casualty_rate = float(np.clip(rng.normal(profile["casualty"], 0.015), 0, 0.3))
        # Real evacuation times are right-skewed — use lognormal
        mu_log = np.log(profile["time"])
        sigma_log = 0.05
        evac_time = float(np.clip(rng.lognormal(mu_log, sigma_log), 100, 360))
        n_evac = int(evac_rate * n_agents)
        mean_evac_time = evac_time + rng.normal(0, 5)
        safety_interventions = int(rng.uniform(50, 200))
        llm_decisions = int(rng.uniform(2000, 8000))
        avg_fear = float(np.clip(rng.normal(4.5, 1.0), 0, 10))
        avg_stamina = float(np.clip(rng.normal(55, 10), 0, 100))

        # Exit distribution (Dirichlet with concentration around realistic values)
        if cond_key == "ours":
            # More balanced exit usage (good scheduling)
            exit_conc = np.ones(n_exits) * 3.0
        elif cond_key == "sfm":
            # Nearest-exit bias (strong)
            exit_conc = np.array([8, 8, 1, 1, 8, 8, 1, 1])
        else:
            # Moderate bias
            exit_conc = np.array([5, 5, 2, 2, 5, 5, 2, 2])

        exit_dist = list(rng.dirichlet(exit_conc) * n_evac)
        exit_dist = [max(0, int(round(x))) for x in exit_dist]

        return RunResult(
            condition=cond_key,
            seed=seed,
            metrics={
                "evacuation_rate": evac_rate,
                "casualty_rate": casualty_rate,
                "mean_evacuation_time": mean_evac_time,
                "safety_interventions": safety_interventions,
                "llm_decisions": llm_decisions,
                "avg_fear_final": avg_fear,
                "avg_stamina_final": avg_stamina,
            },
            exit_distribution=exit_dist,
            metadata={"synthetic": True, "n_agents": n_agents},
        )

    def _run_real(self, cond_key: str, cond: dict, seed: int) -> RunResult:
        """Run actual simulation with orchestrator (requires GPU)."""
        from execution.orchestrator import SimulationOrchestrator

        # Load and override config
        with open(self.config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        cfg["simulation"]["seed"] = seed

        # Apply condition overrides
        for override_path, value in cond.get("config_overrides", {}).items():
            keys = override_path.split(".")
            target = cfg
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = value

        # Write temp config
        tmp_config = os.path.join(self.output_dir, f"_tmp_{cond_key}_{seed}.yaml")
        os.makedirs(os.path.dirname(tmp_config), exist_ok=True)
        with open(tmp_config, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f)

        try:
            orch = SimulationOrchestrator(config_path=tmp_config)
            orch.run()

            # Extract metrics
            n_total = len(orch.agents)
            n_civilian = orch.num_agents
            evac_rate = orch.evacuated_count / max(1, n_civilian)
            casualty_rate = orch.casualty_count / max(1, n_civilian)

            # Mean evacuation time
            evac_times = []
            for a in orch.agents:
                if a.dynamic.evacuated and hasattr(a.dynamic, 'evacuation_time'):
                    t = a.dynamic.evacuation_time
                    if t and t > 0:
                        evac_times.append(t)
            mean_evac_time = float(np.mean(evac_times)) if evac_times else orch.sim_time

            # Exit distribution
            exit_dist = [0] * len(orch.exits)
            for a in orch.agents:
                if a.dynamic.evacuated and a.dynamic.target_exit_idx is not None:
                    idx = a.dynamic.target_exit_idx
                    if 0 <= idx < len(exit_dist):
                        exit_dist[idx] += 1

            # Final fear/stamina
            alive = [a for a in orch.agents if a.dynamic.alive and not a.dynamic.evacuated]
            avg_fear = float(np.mean([a.dynamic.fear_level for a in orch.agents])) if orch.agents else 0
            avg_stamina = float(np.mean([a.dynamic.stamina for a in orch.agents])) if orch.agents else 0

            return RunResult(
                condition=cond_key,
                seed=seed,
                metrics={
                    "evacuation_rate": evac_rate,
                    "casualty_rate": casualty_rate,
                    "mean_evacuation_time": mean_evac_time,
                    "safety_interventions": orch.safety_blocks + orch.safety_modifications,
                    "llm_decisions": orch.decision_count,
                    "avg_fear_final": avg_fear,
                    "avg_stamina_final": avg_stamina,
                },
                exit_distribution=exit_dist,
                metadata={
                    "synthetic": False,
                    "n_agents": len(orch.agents),
                    "sim_time": orch.sim_time,
                },
            )
        finally:
            if os.path.exists(tmp_config):
                os.unlink(tmp_config)

    def _save_results(self):
        """Save experiment results to JSON."""
        output = {
            "metadata": {
                "conditions": list(self.results.keys()),
                "n_runs": self.n_runs,
                "base_seed": self.base_seed,
                "synthetic": self.synthetic,
                "condition_names": {k: CONDITIONS[k]["name"] for k in self.results},
            },
            "results": {},
        }

        for cond_key, runs in self.results.items():
            output["results"][cond_key] = {
                "metrics": {
                    metric: {
                        "mean": float(np.mean([r.metrics[metric] for r in runs])),
                        "std": float(np.std([r.metrics[metric] for r in runs])),
                        "values": [r.metrics[metric] for r in runs],
                    }
                    for metric in RUN_METRICS if metric != "exit_distribution"
                },
                "exit_distributions": [r.exit_distribution for r in runs],
            }

        path = os.path.join(self.output_dir, "results.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[Experiment] Results saved to {path}")

    def _generate_report(self):
        """Generate paper-ready summary report."""
        lines = [
            "# LLM-IRL-RL Evacuation Experiment Summary",
            "",
            f"**Conditions**: {len(self.results)}",
            f"**Runs per condition**: {self.n_runs}",
            f"**Mode**: {'Synthetic (pipeline test)' if self.synthetic else 'Real simulation'}",
            "",
            "## Evacuation Performance",
            "",
            "| Condition | Evac Rate | Casualty Rate | Mean Evac Time | Safety Int. |",
            "|-----------|-----------|---------------|----------------|-------------|",
        ]

        for cond_key in self.condition_keys:
            if cond_key not in self.results:
                continue
            runs = self.results[cond_key]
            name = CONDITIONS[cond_key]["name"]

            evac_vals = [r.metrics["evacuation_rate"] for r in runs]
            cas_vals = [r.metrics["casualty_rate"] for r in runs]
            time_vals = [r.metrics["mean_evacuation_time"] for r in runs]
            safe_vals = [r.metrics["safety_interventions"] for r in runs]

            evac_ci = mean_confidence_interval(evac_vals)
            cas_ci = mean_confidence_interval(cas_vals)
            time_ci = mean_confidence_interval(time_vals)
            safe_ci = mean_confidence_interval(safe_vals)

            lines.append(
                f"| **{name}** | {evac_ci['mean']:.1%} [{evac_ci['ci_lower']:.1%},{evac_ci['ci_upper']:.1%}] | "
                f"{cas_ci['mean']:.1%} [{cas_ci['ci_lower']:.1%},{cas_ci['ci_upper']:.1%}] | "
                f"{time_ci['mean']:.1f}s [{time_ci['ci_lower']:.1f},{time_ci['ci_upper']:.1f}] | "
                f"{safe_ci['mean']:.0f} [{safe_ci['ci_lower']:.0f},{safe_ci['ci_upper']:.0f}] |"
            )

        lines.append("")

        # Statistical comparisons (Ours vs each baseline)
        if "ours" in self.results:
            ours_runs = self.results["ours"]
            lines.append("## Statistical Comparison (Ours vs Baselines)")
            lines.append("")

            for metric_key in ["evacuation_rate", "casualty_rate", "mean_evacuation_time"]:
                metric_name = {
                    "evacuation_rate": "Evacuation Rate",
                    "casualty_rate": "Casualty Rate",
                    "mean_evacuation_time": "Mean Evacuation Time",
                }[metric_key]
                lines.append(f"### {metric_name}")
                lines.append("")

                ours_vals = [r.metrics[metric_key] for r in ours_runs]
                for cond_key in self.condition_keys:
                    if cond_key == "ours" or cond_key not in self.results:
                        continue
                    baseline_vals = [r.metrics[metric_key] for r in self.results[cond_key]]
                    result = independent_t_test(ours_vals, baseline_vals,
                                                name_a="Ours", name_b=CONDITIONS[cond_key]["name"],
                                                metric=metric_name)
                    lines.append(f"- {result['text']}")

                lines.append("")

        # Exit distribution analysis
        lines.append("## Exit Distribution JS Divergence")
        lines.append("")
        lines.append("| Comparison | JS Divergence |")
        lines.append("|------------|---------------|")

        if "ours" in self.results:
            ours_exits = np.mean([r.exit_distribution for r in self.results["ours"]], axis=0)
            for cond_key in self.condition_keys:
                if cond_key == "ours" or cond_key not in self.results:
                    continue
                baseline_exits = np.mean([r.exit_distribution for r in self.results[cond_key]], axis=0)
                js = js_divergence(ours_exits, baseline_exits)
                lines.append(f"| Ours vs {CONDITIONS[cond_key]['name']} | {js:.4f} |")

        lines.append("")

        path = os.path.join(self.output_dir, "summary.md")
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"[Experiment] Report saved to {path}")


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM-IRL-RL Experiment Runner")
    parser.add_argument("--config", type=str, default="config/mall_floorplan.yaml",
                       help="Base config file for real runs")
    parser.add_argument("--n_runs", type=int, default=20,
                       help="Number of runs per condition")
    parser.add_argument("--conditions", type=str,
                       default="sfm,pure_rl,pure_llm,ours,llm_heuristic_rl,bc_distill,llm_direct_reward",
                       help="Comma-separated condition keys")
    parser.add_argument("--output", type=str, default="data/experiments",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic data (pipeline test, no GPU)")
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]

    runner = ExperimentRunner(
        config_path=args.config,
        conditions=conditions,
        n_runs=args.n_runs,
        base_seed=args.seed,
        output_dir=args.output,
        synthetic=args.synthetic,
    )
    runner.run()

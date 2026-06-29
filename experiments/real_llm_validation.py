"""Real LLM validation — small-scale verification that synthetic results hold.

Addresses the most critical reviewer concern: "Core results are synthetic."

This script runs a minimal but genuine LLM-in-the-loop experiment:
  - 50 agents (not 600 — fits within 24GB VRAM for one full run)
  - 120s simulation (not 360s — shorter but still meaningful)
  - 3 conditions: SFM, Pure LLM, Ours (LLM-IRL-RL)
  - 5 runs per condition (not 20 — keeps total time manageable at ~2-4 hours)

The goal is NOT to reproduce the full paper results, but to provide
a "smoke test" proving the system works with real LLM inference and
that the performance ordering (Ours > Pure LLM > SFM) is maintained.

Usage:
  # Full validation (requires GPU, ~2-4 hours):
  python -m experiments.real_llm_validation \
      --config config/mall_floorplan.yaml \
      --n_runs 5 --agents 50 --duration 120

  # Quick sanity check (1 run, ~30 min):
  python -m experiments.real_llm_validation \
      --config config/mall_floorplan.yaml \
      --n_runs 1 --agents 30 --duration 60
"""

import json
import os
import sys
import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class ValidationRun:
    """One real-LLM validation run."""
    condition: str
    run_idx: int
    seed: int
    evac_rate: float
    casualty_rate: float
    mean_evac_time: float
    n_agents: int
    duration: float
    wall_time: float
    success: bool
    error: str = ""


class RealLLMValidator:
    """Minimal real-LLM validation runner."""

    def __init__(self, config_path: str,
                 conditions: List[str] = None,
                 n_runs: int = 5,
                 n_agents: int = 50,
                 duration: float = 120.0,
                 output_dir: str = "data/experiments"):
        self.config_path = config_path
        self.conditions = conditions or ["sfm", "pure_llm", "ours"]
        self.n_runs = n_runs
        self.n_agents = n_agents
        self.duration = duration
        self.output_dir = output_dir
        self.results: List[ValidationRun] = []

    def run(self) -> List[ValidationRun]:
        """Execute validation experiments with real LLM inference."""

        print("=" * 70)
        print("  REAL LLM VALIDATION — Small-Scale Verification")
        print(f"  Conditions: {self.conditions}")
        print(f"  Runs/condition: {self.n_runs}")
        print(f"  Agents: {self.n_agents}")
        print(f"  Duration: {self.duration}s")
        print("=" * 70)
        print()
        print("  WARNING: This requires GPU + vLLM server running.")
        print("  Expected time: ~30 min per run (1 condition × 1 run)")
        print()

        import yaml

        for cond_key in self.conditions:
            print(f"\n{'='*60}")
            print(f"  Condition: {cond_key}")
            print(f"{'='*60}")

            cond_offset = hash(cond_key) % 10000

            for run_idx in range(self.n_runs):
                seed = 42 + run_idx + cond_offset
                print(f"  Run {run_idx+1}/{self.n_runs} (seed={seed})...")
                t_start = time.time()

                try:
                    result = self._run_single(cond_key, seed)
                    result.wall_time = time.time() - t_start
                    self.results.append(result)

                    if result.success:
                        print(f"    evac={result.evac_rate:.1%} "
                              f"casualty={result.casualty_rate:.1%} "
                              f"time={result.mean_evac_time:.0f}s "
                              f"wall={result.wall_time:.0f}s")
                    else:
                        print(f"    FAILED: {result.error}")

                except Exception as e:
                    print(f"    EXCEPTION: {e}")
                    self.results.append(ValidationRun(
                        condition=cond_key, run_idx=run_idx, seed=seed,
                        evac_rate=0, casualty_rate=0, mean_evac_time=0,
                        n_agents=self.n_agents, duration=self.duration,
                        wall_time=time.time() - t_start,
                        success=False, error=str(e),
                    ))

        self._save_results()
        self._print_summary()
        return self.results

    def _run_single(self, cond_key: str, seed: int) -> ValidationRun:
        """Run one validation episode with real LLM."""
        import yaml
        from execution.orchestrator import SimulationOrchestrator

        # Load config
        with open(self.config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        # Override for small-scale validation
        cfg["simulation"]["num_agents"] = self.n_agents
        cfg["simulation"]["duration"] = self.duration
        cfg["simulation"]["seed"] = seed
        cfg["simulation"]["enable_command_agents"] = False  # Disable role agents for clean comparison

        # Apply condition overrides
        condition_overrides = {
            "sfm": {
                "llm.enabled": False,
                "rl_scheduling.enabled": False,
                "irl.enabled": False,
            },
            "pure_llm": {
                "llm.enabled": True,
                "rl_scheduling.enabled": False,
                "irl.enabled": True,
            },
            "ours": {
                "llm.enabled": True,
                "rl_scheduling.enabled": True,
                "irl.enabled": True,
            },
        }

        overrides = condition_overrides.get(cond_key, {})
        for override_path, value in overrides.items():
            keys = override_path.split(".")
            target = cfg
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = value

        # If RL enabled but no pretrained weights, use heuristic
        if cfg.get("rl_scheduling", {}).get("enabled"):
            if not cfg["rl_scheduling"].get("pretrained_weights"):
                cfg["rl_scheduling"]["pretrained_weights"] = ""
                print("    (using heuristic RL — no pretrained weights)")

        # Write temp config
        tmp_config = os.path.join(
            self.output_dir, f"_val_{cond_key}_{seed}.yaml")
        os.makedirs(os.path.dirname(tmp_config), exist_ok=True)
        with open(tmp_config, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f)

        try:
            orch = SimulationOrchestrator(config_path=tmp_config)
            orch.run()

            # Extract metrics — use actual agent count (includes roles)
            n_total = len(orch.agents)
            evac_rate = orch.evacuated_count / max(1, n_total)
            casualty_rate = orch.casualty_count / max(1, n_total)

            evac_times = []
            for a in orch.agents:
                if a.dynamic.evacuated and hasattr(a.dynamic, 'evacuation_time'):
                    t = a.dynamic.evacuation_time
                    if t and t > 0:
                        evac_times.append(t)
            mean_evac_time = float(np.mean(evac_times)) if evac_times else orch.sim_time

            return ValidationRun(
                condition=cond_key, run_idx=0, seed=seed,
                evac_rate=evac_rate, casualty_rate=casualty_rate,
                mean_evac_time=mean_evac_time,
                n_agents=self.n_agents, duration=self.duration,
                wall_time=0, success=True,
            )

        finally:
            if os.path.exists(tmp_config):
                os.unlink(tmp_config)

    def _save_results(self):
        """Save validation results to JSON."""
        output = {
            "validation_type": "real_llm_small_scale",
            "n_agents": self.n_agents,
            "duration": self.duration,
            "n_runs": self.n_runs,
            "conditions": self.conditions,
            "results": [],
        }

        for r in self.results:
            output["results"].append({
                "condition": r.condition,
                "run_idx": r.run_idx,
                "seed": r.seed,
                "evac_rate": r.evac_rate,
                "casualty_rate": r.casualty_rate,
                "mean_evac_time": r.mean_evac_time,
                "wall_time": r.wall_time,
                "success": r.success,
                "error": r.error if not r.success else "",
            })

        path = os.path.join(self.output_dir, "real_llm_validation.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[Validator] Results saved to {path}")

    def _print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("  VALIDATION SUMMARY")
        print("=" * 60)

        by_condition = {}
        for r in self.results:
            by_condition.setdefault(r.condition, []).append(r)

        print(f"\n{'Condition':<20} {'Evac':>8} {'Casualty':>10} {'Time':>8} {'Success':>10}")
        print("-" * 60)

        for cond in self.conditions:
            runs = by_condition.get(cond, [])
            successful = [r for r in runs if r.success]
            if successful:
                evac = np.mean([r.evac_rate for r in successful])
                cas = np.mean([r.casualty_rate for r in successful])
                etime = np.mean([r.mean_evac_time for r in successful])
                print(f"{cond:<20} {evac:>7.1%} {cas:>9.1%} {etime:>7.0f}s "
                      f"{len(successful)}/{len(runs):>7}")
            else:
                print(f"{cond:<20} {'N/A':>8} {'N/A':>10} {'N/A':>8} "
                      f"0/{len(runs):>7}")

        print()
        print("Comparison to synthetic results (for ordering verification):")
        print("  Synthetic:  SFM=72.7% < PureLLM=85.6% < Ours=90.1%")
        print("  Real LLM should maintain this ordering (values may differ).")


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Real LLM Validation — Small-Scale Experiment")
    parser.add_argument("--config", type=str,
                       default="config/mall_floorplan.yaml",
                       help="Base config file")
    parser.add_argument("--conditions", type=str,
                       default="sfm,pure_llm,ours",
                       help="Comma-separated conditions")
    parser.add_argument("--n_runs", type=int, default=5,
                       help="Runs per condition (5 recommended)")
    parser.add_argument("--agents", type=int, default=50,
                       help="Number of agents (50 for fast validation)")
    parser.add_argument("--duration", type=float, default=120.0,
                       help="Simulation duration in seconds")
    parser.add_argument("--output", type=str,
                       default="data/experiments",
                       help="Output directory")
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]

    validator = RealLLMValidator(
        config_path=args.config,
        conditions=conditions,
        n_runs=args.n_runs,
        n_agents=args.agents,
        duration=args.duration,
        output_dir=args.output,
    )
    validator.run()

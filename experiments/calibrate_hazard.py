"""Batch calibration of the hazard-damage probabilities.

Runs a small headless simulation across a grid of
``GroupIntelligence`` damage-probability settings and reports
casualty / evacuation rates so the defaults can be tuned to a
reasonable operating band (e.g. 3-8% baseline casualties).

Usage:
    python -m experiments.calibrate_hazard \
        --config config/default.yaml \
        --agents 60 --duration 60 --seeds 101 202
"""

import argparse
import contextlib
import io
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from execution.orchestrator import SimulationOrchestrator


# Default grids over the two dominant lethal channels (fire & lethal smoke);
# heat and injury probabilities are held at their current defaults.
DEFAULT_FIRE_GRID = [0.02, 0.05, 0.10]
DEFAULT_SMOKE_GRID = [0.01, 0.02, 0.05]
HEAT_FIXED = 0.02
INJURY_FIXED = 0.002


def run_once(config_path: str, agents: int, duration: float, seed: int,
             fire_prob: float, smoke_prob: float,
             smoke_threshold: float = 0.75) -> dict:
    """Run one headless simulation with overridden hazard probabilities."""
    orch = SimulationOrchestrator(config_path=config_path)
    orch.cfg["simulation"]["num_agents"] = agents
    orch.cfg["simulation"]["duration"] = duration
    orch.cfg["simulation"]["seed"] = seed
    orch.cfg["llm"]["enabled"] = False
    orch.cfg["visualization"]["enabled"] = False
    orch.num_agents = agents
    orch.duration = duration

    gi = orch.group_intel
    gi.FIRE_DEATH_PROB_PER_S = fire_prob
    gi.SMOKE_LEATHAL_DEATH_PROB_PER_S = smoke_prob
    gi.SMOKE_LETHAL_THRESHOLD = smoke_threshold
    gi.HEAT_DEATH_PROB_PER_S = HEAT_FIXED
    gi.SMOKE_INJURY_PROB_PER_S = INJURY_FIXED

    with contextlib.redirect_stdout(io.StringIO()):
        orch.run()

    n = max(1, len(orch.agents))
    return {
        "seed": seed,
        "fire_prob": fire_prob,
        "smoke_prob": smoke_prob,
        "evac_rate": orch.evacuated_count / n,
        "casualty_rate": orch.casualty_count / n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--agents", type=int, default=60)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202])
    parser.add_argument("--fire-grid", type=float, nargs="+",
                        default=DEFAULT_FIRE_GRID)
    parser.add_argument("--smoke-grid", type=float, nargs="+",
                        default=DEFAULT_SMOKE_GRID)
    parser.add_argument("--smoke-threshold", type=float, default=0.75)
    parser.add_argument("--output", default="data/experiments/hazard_calibration_report.md")
    args = parser.parse_args()

    rows = []
    combos = list(itertools.product(args.fire_grid, args.smoke_grid))
    total = len(combos) * len(args.seeds)
    print(f"[Calibrate] {total} runs "
          f"({len(combos)} combos x {len(args.seeds)} seeds) ...")
    t0 = time.time()

    for fire_prob, smoke_prob in combos:
        for seed in args.seeds:
            r = run_once(args.config, args.agents, args.duration, seed,
                         fire_prob, smoke_prob, args.smoke_threshold)
            rows.append(r)

    elapsed = time.time() - t0
    print(f"[Calibrate] Done in {elapsed/60:.1f} min.")

    # Aggregate by combo
    grouped = {}
    for r in rows:
        key = (r["fire_prob"], r["smoke_prob"])
        grouped.setdefault(key, []).append(r)

    lines = [
        "# Hazard Damage Probability Calibration",
        "",
        f"- Scenario: `{args.config}`",
        f"- Agents/run: {args.agents}, duration: {args.duration:.0f}s, "
        f"seeds: {args.seeds}",
        f"- Heat death prob/s: {HEAT_FIXED}, smoke injury prob/s: {INJURY_FIXED}",
        f"- Lethal smoke threshold: {args.smoke_threshold}",
        "",
        "| fire/s | smoke/s | evac mean | casualty mean | casualty min-max |",
        "|-------|---------|-----------|---------------|------------------|",
    ]
    current = (0.05, 0.01)
    for key in sorted(grouped.keys()):
        runs = grouped[key]
        evac = np.mean([r["evac_rate"] for r in runs])
        cas = np.mean([r["casualty_rate"] for r in runs])
        cas_min = min(r["casualty_rate"] for r in runs)
        cas_max = max(r["casualty_rate"] for r in runs)
        marker = "  <- current" if key == current else ""
        lines.append(
            f"| {key[0]:.3f} | {key[1]:.3f} | {evac:.1%} | {cas:.1%} "
            f"| {cas_min:.1%}-{cas_max:.1%} |{marker}")

    lines.append("")
    lines.append(
        "Interpretation: pick the smallest fire/smoke probabilities that "
        "produce a meaningful baseline casualty band; then update "
        "`GroupIntelligence.FIRE_DEATH_PROB_PER_S` and "
        "`SMOKE_LEATHAL_DEATH_PROB_PER_S`.")
    report = "\n".join(lines)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()

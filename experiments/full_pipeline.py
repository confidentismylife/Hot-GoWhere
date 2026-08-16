"""Full LLM → IRL → RL cascade pipeline — trajectory collection to validation.

Runs the complete three-tier architecture experiment:
  Step 1: Collect LLM decision trajectories (N_COLLECT runs)
  Step 2: Fit IRL reward weights from trajectories
  Step 3: Train RL zone scheduler with IRL weights
  Step 4: Run 3-condition validation (SFM / Pure LLM / Ours)
  Step 5: Print final comparison table

Usage:
  # Full pipeline (overnight, ~3-5 hours):
  nohup python -m experiments.full_pipeline \
      --config config/mall_floorplan.yaml \
      --n_collect 3 --n_val 5 --agents 100 --duration 180 \
      --rl_episodes 500 \
      > pipeline.log 2>&1 &

  # Quick test (~2 hours):
  nohup python -m experiments.full_pipeline \
      --config config/mall_floorplan.yaml \
      --n_collect 2 --n_val 3 --agents 50 --duration 120 \
      --rl_episodes 200 \
      > pipeline.log 2>&1 &
"""

import os
import sys
import time
import json
import yaml
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def step_banner(step: int, title: str, total: int = 5):
    print(f"\n{'='*70}")
    print(f"  STEP {step}/{total}: {title}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


def run_collect_trajectories(config_path: str, n_runs: int,
                             n_agents: int, duration: float,
                             output_dir: str) -> List[str]:
    """Run N simulations with IRL trajectory collection enabled."""
    from execution.orchestrator import SimulationOrchestrator

    # Load base config and enable IRL
    with open(config_path, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)

    base_cfg["irl"]["enabled"] = True
    base_cfg["llm"]["enabled"] = True
    base_cfg["rl_scheduling"]["enabled"] = False
    base_cfg["simulation"]["num_agents"] = n_agents
    base_cfg["simulation"]["duration"] = duration
    base_cfg["simulation"]["enable_command_agents"] = False

    traj_files = []
    for run_i in range(n_runs):
        seed = 42 + run_i * 100
        base_cfg["simulation"]["seed"] = seed

        tmp_path = os.path.join(output_dir, f"_collect_{seed}.yaml")
        os.makedirs(output_dir, exist_ok=True)
        with open(tmp_path, 'w', encoding='utf-8') as f:
            yaml.dump(base_cfg, f)

        print(f"\n[Collect] Run {run_i+1}/{n_runs} (seed={seed})...")
        t0 = time.time()

        orch = SimulationOrchestrator(config_path=tmp_path)
        orch.run()

        elapsed = time.time() - t0
        print(f"[Collect] Run {run_i+1} done: "
              f"evac={orch.evacuated_count/max(1,len(orch.agents)):.1%} "
              f"wall={elapsed:.0f}s")

        # Find the generated trajectory file
        traj_dir = base_cfg["irl"].get("output_dir", "data/trajectories")
        if os.path.isdir(traj_dir):
            # Get most recent .jsonl file
            jsonl_files = sorted(
                [f for f in os.listdir(traj_dir) if f.endswith('.jsonl')],
                key=lambda x: os.path.getmtime(os.path.join(traj_dir, x)),
                reverse=True)
            if jsonl_files:
                latest = os.path.join(traj_dir, jsonl_files[0])
                traj_files.append(latest)

        # Cleanup temp config
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    print(f"\n[Collect] All {n_runs} runs complete. {len(traj_files)} trajectory files.")
    return traj_files


def run_irl_fitting(trajectory_dir: str, output_path: str):
    """Fit IRL reward weights from collected trajectories."""
    from execution.irl_recovery import TrajectoryCollector, IRLRecovery

    collector = TrajectoryCollector()
    trajectories = collector.load_all(trajectory_dir)
    print(f"[IRL] Loaded {len(trajectories)} trajectories")

    # Print persona distribution
    personas = {}
    for t in trajectories:
        personas[t.persona] = personas.get(t.persona, 0) + 1
    print(f"[IRL] Persona distribution: {json.dumps(personas, ensure_ascii=False)}")

    irl = IRLRecovery(learning_rate=0.01, max_iter=500)
    irl.fit(trajectories, verbose=True)
    irl.save(output_path)
    print(f"[IRL] Weights saved to {output_path}")


def run_rl_training(config_path: str, episodes: int,
                    irl_weights_path: str, output_path: str):
    """Train RL zone scheduler with IRL weights.

    The fast simulator is built from the SAME config/floorplan that will be
    used for validation (dims, exits, zones), so the trained policy matches
    the deployment environment instead of a hardcoded 150x80 / 4-exit map.
    """
    from execution.rl_scheduler import (
        RLZoneScheduler, FastTrainingSimulator, ZoneDefinition, DEFAULT_ZONES,
    )
    from execution.irl_recovery import IRLRecovery

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    env = cfg.get("environment", {})
    sim_cfg = cfg.get("simulation", {})

    # Resolve the actual environment from the floorplan + config overrides.
    if env.get("type") == "mall" or env.get("floorplan"):
        from perception.floorplan import get_floorplan
        fp = get_floorplan(env.get("floorplan", "chaoyang_joycity_1f"))
        width, height = fp.width, fp.height
        exit_positions = ([tuple(e) for e in env["exit_positions"]]
                          if env.get("exit_positions") else fp.exits)
    else:
        width = float(env.get("width", 100.0))
        height = float(env.get("height", 60.0))
        exit_positions = [tuple(e) for e in env.get("exit_positions", [])]
    num_exits = len(exit_positions)

    # Zone layout: use config zones when present, else DEFAULT_ZONES.
    zones_cfg = cfg.get("zones")
    if zones_cfg:
        zones = [
            ZoneDefinition(
                zone_id=z["id"],
                name=z.get("name", f"Zone{z['id']}"),
                x_min=z["x_min"], x_max=z["x_max"],
                y_min=z["y_min"], y_max=z["y_max"],
                primary_exits=z.get("primary_exits", list(range(num_exits))),
                description=z.get("description", ""),
            )
            for z in zones_cfg
        ]
    else:
        zones = DEFAULT_ZONES

    training_seed = int(sim_cfg.get("seed", 42))
    scheduler = RLZoneScheduler(
        zones=zones,
        num_exits=num_exits,
        seed=training_seed,
        blocked_exit_penalty=float(
            cfg.get("rl_scheduling", {}).get("blocked_exit_penalty", 0.0)),
        smoke_block_threshold=float(
            cfg.get("rl_scheduling", {}).get("smoke_block_threshold", 0.6)),
        outcome_reward_weight=float(
            cfg.get("rl_scheduling", {}).get("outcome_reward_weight", 0.0)),
    )
    scheduler.initialize()

    # Load IRL weights
    if os.path.exists(irl_weights_path):
        irl = IRLRecovery()
        irl.load(irl_weights_path)
        scheduler.load_irl_weights(irl.weights)
        print(f"[RL Train] Loaded IRL weights for {len(irl.weights)} personas")
    else:
        print(f"[RL Train] WARNING: IRL weights not found at {irl_weights_path}")
        print(f"[RL Train] Using default balanced weights")

    sim = FastTrainingSimulator(
        width=width, height=height,
        num_agents=int(sim_cfg.get("num_agents", 600)),
        num_exits=num_exits,
        zone_defs=zones,
        exit_positions=exit_positions,
        seed=training_seed,
        fire_sources=env.get("fire_sources"),
        spread_rate=env.get("disaster_spread_rate"),
        origin_jitter=float(
            cfg.get("rl_scheduling", {}).get("origin_jitter", 15.0)),
        smoke_block_threshold=float(
            cfg.get("rl_scheduling", {}).get("smoke_block_threshold", 0.6)),
    )

    print(f"[RL Train] Environment: {width:.0f}x{height:.0f}m, "
          f"{num_exits} exits, {len(zones)} zones")
    print(f"[RL Train] Starting {episodes} episodes...")
    t0 = time.time()
    steps_per_episode = int(float(sim_cfg.get("duration", 360.0)) / sim.dt)
    history = scheduler.train_offline(
        env_simulator=sim,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        save_path=output_path,
    )
    elapsed = time.time() - t0
    print(f"[RL Train] Done in {elapsed/60:.1f} min. "
          f"Final evac: {history['evacuation_rate'][-1]:.1%}")


def run_validation(config_path: str, n_runs: int, n_agents: int,
                   duration: float, rl_weights_path: str,
                   output_dir: str):
    """Run 3-condition validation with real LLM."""
    from experiments.real_llm_validation import RealLLMValidator

    # Load config and enable full cascade
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Enable RL scheduler with trained weights
    if os.path.exists(rl_weights_path):
        cfg["rl_scheduling"]["enabled"] = True
        cfg["rl_scheduling"]["pretrained_weights"] = rl_weights_path
        # Also set in the copied config for validation
    else:
        print(f"[Validate] WARNING: RL weights not found at {rl_weights_path}")

    # Write modified config
    tmp_config = os.path.join(output_dir, "_val_full_cascade.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(tmp_config, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f)

    validator = RealLLMValidator(
        config_path=tmp_config,
        conditions=["sfm", "pure_llm", "ours"],
        n_runs=n_runs,
        n_agents=n_agents,
        duration=duration,
        output_dir=output_dir,
    )
    results = validator.run()

    if os.path.exists(tmp_config):
        os.unlink(tmp_config)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Full LLM → IRL → RL Cascade Pipeline")
    parser.add_argument("--config", default="config/mall_floorplan.yaml")
    parser.add_argument("--n_collect", type=int, default=3,
                       help="Trajectory collection runs")
    parser.add_argument("--n_val", type=int, default=5,
                       help="Validation runs per condition")
    parser.add_argument("--agents", type=int, default=100,
                       help="Agents per run")
    parser.add_argument("--duration", type=float, default=180.0,
                       help="Simulation duration (seconds)")
    parser.add_argument("--rl_episodes", type=int, default=500,
                       help="RL training episodes")
    parser.add_argument("--output", default="data/experiments")
    parser.add_argument("--skip_collect", action="store_true",
                       help="Skip trajectory collection (use existing)")
    parser.add_argument("--skip_irl", action="store_true",
                       help="Skip IRL fitting (use existing weights)")
    parser.add_argument("--skip_rl", action="store_true",
                       help="Skip RL training (use existing policy)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    irl_weights_path = os.path.join(args.output, "irl_weights.json")
    rl_policy_path = os.path.join(args.output, "rl_policy.json")
    traj_dir = "data/trajectories"

    total_start = time.time()
    total_steps = 5 - sum([args.skip_collect, args.skip_irl, args.skip_rl])

    # ================================================================
    # Step 1: Collect LLM trajectories
    # ================================================================
    if not args.skip_collect:
        step_banner(1, "Collect LLM Trajectories", total_steps)
        traj_files = run_collect_trajectories(
            args.config, args.n_collect, args.agents, args.duration,
            args.output)
        if not traj_files:
            print("[ERROR] No trajectory files generated. Aborting.")
            sys.exit(1)
    else:
        step_banner(1, "SKIP Collect (using existing)", total_steps)

    # ================================================================
    # Step 2: Fit IRL weights
    # ================================================================
    if not args.skip_irl:
        step_banner(2, "Fit IRL Reward Weights", total_steps)
        run_irl_fitting(traj_dir, irl_weights_path)
    else:
        step_banner(2, "SKIP IRL (using existing)", total_steps)

    # ================================================================
    # Step 3: Train RL scheduler
    # ================================================================
    if not args.skip_rl:
        step_banner(3, "Train RL Zone Scheduler", total_steps)
        run_rl_training(args.config, args.rl_episodes,
                        irl_weights_path, rl_policy_path)
    else:
        step_banner(3, "SKIP RL (using existing)", total_steps)

    # ================================================================
    # Step 4: Run validation
    # ================================================================
    step_banner(4, "Run 3-Condition Validation", total_steps)
    results = run_validation(
        args.config, args.n_val, args.agents, args.duration,
        rl_policy_path, args.output)

    # ================================================================
    # Step 5: Final summary
    # ================================================================
    step_banner(5, "Pipeline Complete — Final Summary", total_steps)
    total_elapsed = time.time() - total_start
    print(f"\nTotal wall time: {total_elapsed/3600:.1f} hours")
    print(f"Output directory: {args.output}")
    print(f"IRL weights: {irl_weights_path}")
    print(f"RL policy: {rl_policy_path}")
    print(f"Validation: {os.path.join(args.output, 'real_llm_validation.json')}")


if __name__ == "__main__":
    main()

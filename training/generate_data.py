"""Training data generator — runs simulation, captures oracle decisions.

The safety guard acts as the "oracle": for each agent at decision time,
we compute the optimal exit/speed using rule-based heuristics and capture
the full prompt context. This produces (instruction, output) pairs for
supervised fine-tuning.

Usage:
    python -m training.generate_data --num-scenarios 50 --output data/train.jsonl
"""

import sys
import os
import json
import random
import argparse
import time
import numpy as np
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.orchestrator import SimulationOrchestrator
from decision.agent_state import Agent, AgentProfile, AgentDynamic, Speed, Cooperation
from decision.prompt_manager import PromptManager
from decision.safety_guard import SafetyGuard
from perception.environment import EnvironmentSnapshot, DisasterSimulator


def _generate_reasoning(agent: Agent, env: EnvironmentSnapshot,
                        best_idx: int, speed: Speed,
                        exit_smoke: List[float]) -> str:
    """Generate natural-language reasoning text from oracle decision logic."""
    exit_num = best_idx + 1
    exit_pos = env.exits[best_idx]
    dist = float(np.linalg.norm(np.array(exit_pos) - agent.position))

    smoke_val = exit_smoke[best_idx]
    if smoke_val < 0.3:
        smoke_desc = "通畅"
    elif smoke_val < 0.6:
        smoke_desc = "有轻微烟雾但可通行"
    else:
        smoke_desc = "烟雾较重但尚未封锁"

    speed_reason = ""
    if speed == Speed.RUN:
        speed_reason = "体力充足，选择奔跑以争取时间"
    elif speed == Speed.WALK:
        speed_reason = "保持步行以节省体力"
    elif speed == Speed.CRAWL:
        speed_reason = "体力不足，只能缓慢爬行"
    else:
        speed_reason = "原地等待更安全"

    smoke_at_pos = env.smoke_at(agent.position)
    risk_note = ""
    if smoke_at_pos > 0.5:
        risk_note = f"当前位置烟雾浓度{smoke_at_pos:.0%}，需尽快离开。"
    if agent.dynamic.injured:
        risk_note += "因受伤行动受限。"

    return (f"当前烟雾浓度较高，选择出口{exit_num}(距离{int(dist)}m,{smoke_desc})。"
            f"{speed_reason}。{risk_note}")


def generate_scenario_data(orchestrator: SimulationOrchestrator,
                           scenario_id: int) -> List[dict]:
    """Run one simulation scenario and capture all decision points.

    Returns list of {instruction, output, metadata} dicts.
    """
    orchestrator.generate_agents()
    orchestrator._spawn_command_agents()

    # Override: skip real LLM, use safety guard as oracle
    orchestrator.llm_engine._ready = False
    orchestrator.cfg["visualization"]["enabled"] = False

    prompt_mgr = PromptManager()
    safety = orchestrator.safety_guard
    kb = orchestrator.knowledge_base

    total_ticks = int(orchestrator.duration / orchestrator.dt)
    samples = []

    for tick in range(total_ticks):
        orchestrator.disaster.step(orchestrator.dt)
        env_snapshot = orchestrator.disaster.snapshot(
            orchestrator.tick, orchestrator.sim_time,
            orchestrator.exits, orchestrator.obstacles,
            official_broadcast=orchestrator._get_broadcast(),
        )

        # Find agents needing a decision
        agents_to_decide = [
            a for a in orchestrator.agents
            if (a.dynamic.alive and not a.dynamic.evacuated
                and (a.dynamic.has_new_info
                     or orchestrator.tick - a.dynamic.last_decision_tick
                     >= orchestrator.decision_ticks))
        ]

        for agent in agents_to_decide:
            if agent.profile.role != "civilian":
                continue  # Only train on civilian decisions for now

            # Compute oracle decision via safety guard
            fb = safety.fallback_decision(agent, env_snapshot)
            optimal_exit_idx = fb["target_exit_idx"]
            optimal_speed = fb["speed"]
            optimal_coop = fb["cooperation"]

            # Compute exit smoke for reasoning generation
            exit_smokes = [
                env_snapshot.smoke_at(np.array(ep, dtype=np.float64))
                for ep in env_snapshot.exits
            ]

            # Build system + user prompt
            kdocs = kb.query("火灾疏散决策", env_snapshot.disaster_type, top_k=3)
            system = prompt_mgr.build_system(
                env_snapshot.disaster_type, kdocs, role="civilian")
            user = prompt_mgr.build_user(agent, env_snapshot)

            instruction = system + "\n\n" + user
            reasoning = _generate_reasoning(
                agent, env_snapshot, optimal_exit_idx, optimal_speed, exit_smokes)

            # Build output JSON
            output = {
                "risk_assessment": (f"出口{optimal_exit_idx+1}方向{exit_smokes[optimal_exit_idx]:.0%}烟雾"
                                    if exit_smokes[optimal_exit_idx] > 0.1
                                    else "环境相对安全"),
                "target_exit": f"出口{optimal_exit_idx + 1}",
                "route_reasoning": reasoning.split("。")[0],
                "speed": optimal_speed.value,
                "cooperation": optimal_coop.value,
                "reasoning": reasoning,
            }

            samples.append({
                "instruction": instruction,
                "output": json.dumps(output, ensure_ascii=False),
                "metadata": {
                    "scenario_id": scenario_id,
                    "tick": tick,
                    "disaster_type": env_snapshot.disaster_type,
                    "agent_stamina": float(agent.dynamic.stamina),
                    "agent_position": agent.position.tolist(),
                    "smoke_coverage": float((env_snapshot.grid[:, :, 0] > 0.1).mean()),
                },
            })

            # Apply the optimal decision so simulation proceeds correctly
            agent.dynamic.target_exit = np.array(
                env_snapshot.exits[optimal_exit_idx], dtype=np.float64)
            agent.dynamic.speed_choice = optimal_speed
            agent.dynamic.cooperation_choice = optimal_coop
            agent.dynamic.reasoning_text = reasoning
            agent.dynamic.last_decision_tick = orchestrator.tick
            agent.dynamic.has_new_info = False

        # Run physics
        orchestrator.group_intel.propagate(
            orchestrator.agents, env_snapshot.official_broadcast, orchestrator.dt)
        orchestrator.group_intel.update_fear_levels(
            orchestrator.agents, env_snapshot, orchestrator.dt)
        orchestrator.group_intel.update_stamina(orchestrator.agents, orchestrator.dt)
        orchestrator.physics.step_all(orchestrator.agents, orchestrator.dt)

        orchestrator.evacuated_count = sum(
            1 for a in orchestrator.agents if a.dynamic.evacuated)
        orchestrator.casualty_count = sum(
            1 for a in orchestrator.agents if not a.dynamic.alive)

        orchestrator.tick += 1
        orchestrator.sim_time += orchestrator.dt

        # Early termination
        remaining = (orchestrator.num_agents
                     - orchestrator.evacuated_count
                     - orchestrator.casualty_count)
        if remaining <= 0:
            break

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate LoRA fine-tuning data from simulation")
    parser.add_argument("--num-scenarios", type=int, default=50,
                        help="Number of simulation scenarios")
    parser.add_argument("--agents-per-scenario", type=int, default=100,
                        help="Agents per scenario")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Seconds per scenario")
    parser.add_argument("--output", type=str, default="data/train_lora.jsonl",
                        help="Output file path")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    all_samples = []
    disaster_types = ["fire", "earthquake", "flood"]

    print("=" * 60)
    print("  Generating LoRA fine-tuning data")
    print(f"  Scenarios: {args.num_scenarios}")
    print(f"  Agents/scenario: {args.agents_per_scenario}")
    print(f"  Duration/scenario: {args.duration}s")
    print("=" * 60)

    for sid in range(args.num_scenarios):
        dt = random.choice(disaster_types)
        origin_x = random.uniform(10, 80)
        origin_y = random.uniform(10, 50)

        orch = SimulationOrchestrator(config_path="config/default.yaml")
        orch.num_agents = args.agents_per_scenario
        orch.duration = args.duration
        orch.cfg["environment"]["disaster"] = dt
        orch.cfg["environment"]["disaster_origin"] = [origin_x, origin_y]

        t0 = time.perf_counter()
        samples = generate_scenario_data(orch, sid)
        elapsed = time.perf_counter() - t0

        all_samples.extend(samples)
        print(f"  Scenario {sid+1}/{args.num_scenarios}: "
              f"{len(samples)} samples in {elapsed:.1f}s "
              f"({dt}, origin=({origin_x:.0f},{origin_y:.0f}))")

    # Train/val split
    random.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * args.val_split))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    train_path = args.output
    val_path = args.output.replace(".jsonl", "_val.jsonl")

    for path, samples in [(train_path, train_samples), (val_path, val_samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"  Done: {len(train_samples)} train / {len(val_samples)} val")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

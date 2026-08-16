"""Generate Oracle expert trajectories for IRL training.

The Oracle decision function scores exits with a weight configuration
(safety / efficiency / crowd / fire-distance), producing trajectories with
distinct preference structures for IRL sensitivity analysis.

Note: the current implementation uses *current* smoke at each exit, not a
true arrival-time prediction; treat the Oracle as a synthetic expert used
to validate the IRL pipeline, not as a fire-dynamics model.

Sensitivity analysis: multiple weight configurations are provided to test whether
IRL can recover different preference structures. Each config represents a distinct
behavioral strategy (safety-first, efficiency-first, balanced).

Usage:
  python -m experiments.generate_oracle_trajectories \
      --config config/mall_floorplan.yaml \
      --weights all \
      --n_runs 10 --agents 400 --duration 360 \
      --output data/trajectories
"""

import json
import os
import sys
import time
import argparse
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from decision.agent_state import Agent, AgentProfile, AgentDynamic, Speed, Cooperation
from decision.policy import (
    OracleWeights,
    OraclePolicy,
    ORACLE_WEIGHT_CONFIGS,
    ORACLE_WEIGHTS_LEGACY,
)
from perception.environment import DisasterSimulator, EnvironmentSnapshot
from execution.batched_physics import BatchedPhysics
from execution.agent_factory import random_profile, create_family_groups
from execution.irl_recovery import TrajectoryCollector
from group_intel.propagation import GroupIntelligence


def oracle_decide(agent: Agent, env: EnvironmentSnapshot,
                  exits: List[Tuple[float, float]],
                  agent_crowd_map: Dict[int, int],
                  weights: OracleWeights = None) -> "DecisionResult":
    """Backward-compatible wrapper around :class:`OraclePolicy`."""
    return OraclePolicy().decide(
        agent, env, exits,
        crowd_map=agent_crowd_map, weights=weights)


# ================================================================
# Oracle trajectory generator
# ================================================================

class OracleTrajectoryGenerator:
    """Run simulations with Oracle decision logic to collect IRL training data.

    Supports multiple weight configurations for sensitivity analysis:
    IRL should be able to recover distinct preference structures regardless
    of which configuration generated the data.
    """

    def __init__(self, config_path: str, n_agents: int = None,
                 duration: float = None, seed: int = None,
                 weight_configs: List[OracleWeights] = None,
                 spread_rate: float = None,
                 fire_origin: Tuple[float, float] = None):
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        sim = self.cfg["simulation"]
        env = self.cfg["environment"]
        agent_cfg = self.cfg["agents"]

        self.num_agents = n_agents or sim["num_agents"]
        self.duration = duration or sim["duration"]
        self.dt = sim["dt"]
        self.decision_interval = sim["decision_interval"]
        self.decision_ticks = int(self.decision_interval / self.dt)
        self._base_seed = seed if seed is not None else sim["seed"]
        self.seed = self._base_seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Weight configs for sensitivity analysis
        self.weight_configs = weight_configs or [ORACLE_WEIGHTS_LEGACY]

        # Floor plan
        self.floorplan = None
        if env.get("type") == "mall" or env.get("floorplan"):
            from perception.floorplan import get_floorplan
            fp_name = env.get("floorplan", "chaoyang_joycity_1f")
            self.floorplan = get_floorplan(fp_name)
            print(f"[OracleGen] Loaded floor plan: {self.floorplan.name}")

        if self.floorplan:
            self.width = self.floorplan.width
            self.height = self.floorplan.height
            self.exits = ([tuple(e) for e in env["exit_positions"]]
                          if env.get("exit_positions")
                          else self.floorplan.exits)
            self.obstacles = self.floorplan.obstacles + env.get("obstacles", [])
        else:
            self.width = env["width"]
            self.height = env["height"]
            self.exits = [tuple(e) for e in env["exit_positions"]]
            self.obstacles = env.get("obstacles", [])

        # Disaster — CLI overrides take precedence over config
        disaster_origin = env.get("disaster_origin", (self.width * 0.2, self.height * 0.5))
        if self.floorplan and env.get("disaster_origin") is None:
            disaster_origin = self.floorplan.disaster_origin_default
        if fire_origin is not None:
            disaster_origin = fire_origin
            print(f"[OracleGen] Fire origin override: {disaster_origin}")
        self.fire_origin = tuple(disaster_origin)

        self.spread_rate = (spread_rate if spread_rate is not None
                            else env.get("disaster_spread_rate", 0.8))
        if spread_rate is not None:
            print(f"[OracleGen] Spread rate override: {self.spread_rate}")

        self.agent_cfg = agent_cfg
        self._crowd_map: Dict[int, int] = {}
        self._current_weights: OracleWeights = self.weight_configs[0]
        self._oracle_policy = OraclePolicy()

    def generate(self, output_dir: str = "data/trajectories",
                 run_label: str = "",
                 weights: OracleWeights = None) -> str:
        """Run a single simulation with oracle decisions, collect trajectories.

        Args:
            output_dir: Directory for trajectory files.
            run_label: Suffix for the filename (e.g. "run_001").
            weights: OracleWeights config for this run.

        Returns the path to the saved trajectory file.
        """
        if weights is not None:
            self._current_weights = weights

        import datetime

        w = self._current_weights
        print(f"\n[OracleGen] Weight config: {w.description}")

        # Create disaster simulator
        wall_grid = self.floorplan.grid if self.floorplan else None
        disaster = DisasterSimulator(
            width=self.width, height=self.height,
            disaster_type=self.cfg["environment"]["disaster"],
            origin=self.fire_origin,
            spread_rate=self.spread_rate,
            resolution=0.5,
            wall_mask=wall_grid,
        )

        # Physics
        physics = BatchedPhysics(
            width=self.width, height=self.height,
            obstacles=self.obstacles, wall_grid=wall_grid,
        )

        # Group intelligence
        group_intel = GroupIntelligence(width=self.width, height=self.height)

        # Generate agents
        agents = self._generate_agents()
        self._create_family_groups(agents)

        # Trajectory collector
        collector = TrajectoryCollector()

        total_ticks = int(self.duration / self.dt)
        tick = 0
        sim_time = 0.0
        evacuated = 0
        casualties = 0

        print(f"[OracleGen] Starting: {len(agents)} agents, "
              f"{total_ticks} ticks, {self.duration}s, dt={self.dt}s")
        print(f"[OracleGen] Fire origin: {self.fire_origin}, "
              f"spread: {self.spread_rate} m/s")
        print(f"[OracleGen] {len(self.exits)} exits, {len(self.obstacles)} obstacles")

        t_start = time.time()

        while tick < total_ticks:
            # 1. Disaster step
            disaster.step(self.dt)

            # 2. Build crowd map (which exits are agents heading to)
            self._rebuild_crowd_map(agents)

            # 3. Environment snapshot
            env_snapshot = disaster.snapshot(
                tick, sim_time, self.exits, self.obstacles)

            # 4. Agent decisions at decision interval
            if tick % self.decision_ticks == 0:
                for agent in agents:
                    if not agent.dynamic.alive or agent.dynamic.evacuated:
                        continue

                    # Oracle decision with current weight config
                    d = self._oracle_policy.decide(
                        agent, env_snapshot, self.exits,
                        crowd_map=self._crowd_map, weights=w, tick=tick)
                    # Apply
                    agent.dynamic.target_exit = np.array(
                        d.target_exit_pos, dtype=np.float64)
                    agent.dynamic.target_exit_idx = d.target_exit_idx
                    agent.dynamic.speed_choice = d.speed
                    agent.dynamic.last_decision_tick = tick
                    agent.dynamic.has_new_info = False
                    agent.dynamic.reasoning_text = d.reasoning

                    # Record trajectory
                    collector.record_decision(agent, d, env_snapshot, False, False)

            # 5. Tactical adjustments
            self._tactical_adjust(agents, env_snapshot, tick)

            # 5.1 Group intelligence (fear/stamina/hazard) — matches orchestrator
            group_intel.propagate(agents, "", self.dt)
            group_intel.update_fear_levels(agents, env_snapshot, self.dt)
            group_intel.update_stamina(agents, self.dt)
            group_intel.update_hazard_damage(agents, env_snapshot, self.dt)

            # 6. Physics
            physics.step_all(agents, self.dt)

            # 7. Stats
            evac = dead = 0
            for a in agents:
                if a.dynamic.evacuated:
                    evac += 1
                elif not a.dynamic.alive:
                    dead += 1
            evacuated = evac
            casualties = dead

            # 8. Progress
            if tick % 100 == 0:
                alive_count = len(agents) - evac - dead
                print(f"[Tick {tick:4d}] t={sim_time:6.1f}s | "
                      f"alive={alive_count:4d} evac={evac:4d} dead={dead:4d}")

            # 9. Termination check
            remaining = len(agents) - evac - dead
            if remaining <= 0:
                print(f"[OracleGen] All agents resolved at t={sim_time:.1f}s")
                break

            tick += 1
            sim_time += self.dt

        # Finalize and save — filename includes weight config for traceability
        collector.finalize(agents, dt=self.dt)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        label = run_label or f"run_{ts}"
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"oracle_{w.name}_{label}.jsonl")
        collector.save(path)

        elapsed = time.time() - t_start
        n_trajs = len(collector._trajectories)
        total_decisions = sum(len(t.decisions) for t in collector._trajectories.values())
        print(f"[OracleGen] Done in {elapsed:.1f}s: "
              f"{n_trajs} trajectories, {total_decisions} decisions → {path}")
        print(f"[OracleGen] Evac: {evacuated}/{len(agents)} "
              f"({100*evacuated/len(agents):.1f}%), "
              f"Dead: {casualties}/{len(agents)} "
              f"({100*casualties/len(agents):.1f}%)")

        return path

    def generate_multiple(self, n_runs: int, output_dir: str = "data/trajectories"):
        """Run simulations across all weight configs with different seeds.

        For each weight config, runs n_runs simulations with different random seeds.
        Total runs = len(weight_configs) * n_runs.
        """
        paths = []
        for w in self.weight_configs:
            # Reset seed for each config to keep comparisons fair
            self.seed = self._base_seed
            random.seed(self.seed)
            np.random.seed(self.seed)

            print(f"\n{'─' * 50}")
            print(f"  Weight config: {w.name} — {w.label}")
            print(f"  {w.description}")
            print(f"{'─' * 50}")

            for i in range(n_runs):
                self.seed = self.seed + i + 1
                random.seed(self.seed)
                np.random.seed(self.seed)
                path = self.generate(
                    output_dir=output_dir,
                    run_label=f"run_{i:03d}",
                    weights=w,
                )
                paths.append(path)

        return paths

    def _generate_agents(self) -> List[Agent]:
        ac = self.agent_cfg
        agents = []
        for idx in range(self.num_agents):
            profile = self._random_profile(idx)
            dynamic = self._random_dynamic(profile)
            agents.append(Agent(profile=profile, dynamic=dynamic))
        return agents

    def _random_profile(self, idx: int) -> AgentProfile:
        """Generate one agent profile from config-driven distributions."""
        return random_profile(self.agent_cfg)

    def _random_dynamic(self, profile: AgentProfile) -> AgentDynamic:
        for _ in range(1000):
            x = random.uniform(5, self.width - 5)
            y = random.uniform(5, self.height - 5)
            pos = np.array([x, y], dtype=np.float64)
            if self.floorplan and not self.floorplan.is_walkable(x, y):
                continue
            blocked = False
            for obs in self.obstacles:
                oc = np.array(obs["center"], dtype=np.float64)
                if np.linalg.norm(pos - oc) < obs["radius"] + 0.5:
                    blocked = True
                    break
            if not blocked:
                break

        num_known = max(1, int(profile.familiarity * len(self.exits)))
        known = random.sample(self.exits, num_known)

        # Oracle initial exit choice
        fire_origin = np.array(self.fire_origin, dtype=np.float64)
        best_exit = known[0]
        best_score = -float('inf')
        for ex in known:
            ex_arr = np.array(ex, dtype=np.float64)
            dist = float(np.linalg.norm(pos - ex_arr))
            fire_dist_to_exit = float(np.linalg.norm(fire_origin - ex_arr))
            # Prefer exits far from fire
            safety = min(1.0, fire_dist_to_exit / 100.0)
            score = safety * 2.0 - dist / 100.0
            if score > best_score:
                best_score = score
                best_exit = ex

        return AgentDynamic(
            position=pos,
            target_exit=np.array(best_exit, dtype=np.float64),
            speed_choice=Speed.WALK,
            cooperation_choice=Cooperation.NONE,
            stamina=random.uniform(60, 100),
            trust_official_now=profile.trust_authority,
            known_exit_positions=known,
            has_new_info=False,
        )

    def _create_family_groups(self, agents: List[Agent]):
        new_members = create_family_groups(agents, self.agent_cfg)
        if new_members:
            agents.extend(new_members)

    def _rebuild_crowd_map(self, agents: List[Agent]):
        self._crowd_map.clear()
        for a in agents:
            if not a.dynamic.alive or a.dynamic.evacuated:
                continue
            if a.dynamic.target_exit is not None:
                for i, ex in enumerate(self.exits):
                    d = np.linalg.norm(a.dynamic.target_exit - np.array(ex))
                    if d < 3.0:
                        self._crowd_map[i] = self._crowd_map.get(i, 0) + 1
                        break

    def _tactical_adjust(self, agents: List[Agent],
                         env: EnvironmentSnapshot, tick: int):
        """Per-tick tactical adjustments (same as TacticalLayer)."""
        for agent in agents:
            if not agent.dynamic.alive or agent.dynamic.evacuated:
                continue
            pos = agent.dynamic.position
            smoke = float(env.smoke_at(pos))
            stamina = agent.dynamic.stamina
            fear = agent.dynamic.fear_level

            # Speed adjustment
            if env.is_on_fire(pos):
                agent.dynamic.speed_choice = Speed.RUN
            elif stamina < 15:
                agent.dynamic.speed_choice = Speed.CRAWL
            elif smoke > 0.6:
                agent.dynamic.speed_choice = Speed.CRAWL
            elif fear > 8 and stamina > 40:
                agent.dynamic.speed_choice = Speed.RUN
            elif stamina < 30:
                agent.dynamic.speed_choice = Speed.WALK
            elif smoke > 0.3:
                agent.dynamic.speed_choice = Speed.WALK

            # Exit switch if blocked — uses Oracle weights for consistency
            target = agent.dynamic.target_exit
            if target is not None:
                ts = float(env.smoke_at(target))
                if ts > 0.85:
                    # Score alternatives with the SAME weights as the Oracle,
                    # not a uniform heuristic. This preserves the behavioral
                    # difference between weight configs even under duress.
                    best_idx = None
                    best_score = -float('inf')
                    w = self._current_weights
                    for i, ex in enumerate(self.exits):
                        ex_arr = np.array(ex, dtype=np.float64)
                        s = float(env.smoke_at(ex_arr))
                        if s > 0.85:
                            continue
                        dist = float(np.linalg.norm(pos - ex_arr))
                        safety = 1.0 - s
                        efficiency = 1.0 / (1.0 + dist / 40.0)
                        # Match oracle scoring (minus crowd/fire_risk — unavailable per-tick)
                        score = safety * w.w_safety + efficiency * w.w_efficiency
                        if score > best_score:
                            best_score = score
                            best_idx = i
                    if best_idx is not None:
                        agent.dynamic.target_exit = np.array(
                            self.exits[best_idx], dtype=np.float64)
                        agent.dynamic.target_exit_idx = best_idx


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Oracle Expert Trajectories for IRL "
                    "(with sensitivity analysis)")
    parser.add_argument("--config", default="config/mall_floorplan.yaml",
                       help="Base config file")
    parser.add_argument("--weights", type=str, default="all",
                       choices=["all", "legacy", "safety_first", "balanced",
                                "efficiency_first"],
                       help="Weight configuration: 'all' runs all 3 sensitivity "
                            "configs; specific names run just one; 'legacy' uses "
                            "the original 0.45/0.25/0.15/0.15")
    parser.add_argument("--n_runs", type=int, default=10,
                       help="Number of simulation runs PER weight config")
    parser.add_argument("--agents", type=int, default=None,
                       help="Number of agents (override config)")
    parser.add_argument("--duration", type=float, default=None,
                       help="Simulation duration in seconds")
    parser.add_argument("--output", default="data/trajectories",
                       help="Output directory for trajectory files")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed")
    parser.add_argument("--spread-rate", type=float, default=None,
                       help="Override fire spread rate (m/s). "
                            "Higher = faster fire = more safety/efficiency conflict. "
                            "Config default is 0.1; try 0.5 for hard scenarios.")
    parser.add_argument("--fire-origin", type=float, nargs=2, default=None,
                       metavar=("X", "Y"),
                       help="Override fire origin coordinates. "
                            "Asymmetric (near one exit) creates more conflict.")
    args = parser.parse_args()

    # Resolve weight configs
    if args.weights == "all":
        weight_configs = ORACLE_WEIGHT_CONFIGS
    elif args.weights == "legacy":
        weight_configs = [ORACLE_WEIGHTS_LEGACY]
    else:
        weight_configs = [
            next(w for w in ORACLE_WEIGHT_CONFIGS if w.name == args.weights)
        ]

    print("=" * 60)
    print("  ORACLE TRAJECTORY GENERATOR (Sensitivity Analysis)")
    print(f"  Config: {args.config}")
    print(f"  Weight configs: {len(weight_configs)} ({', '.join(w.name for w in weight_configs)})")
    if len(weight_configs) > 1:
        print(f"  Sensitivity: IRL must recover {len(weight_configs)} distinct "
              f"preference structures")
    print(f"  Runs per config: {args.n_runs} "
         f"(total: {len(weight_configs) * args.n_runs})")
    print(f"  Output: {args.output}")
    print("=" * 60)
    print()

    generator = OracleTrajectoryGenerator(
        config_path=args.config,
        n_agents=args.agents,
        duration=args.duration,
        seed=args.seed,
        weight_configs=weight_configs,
        spread_rate=args.spread_rate,
        fire_origin=(tuple(args.fire_origin) if args.fire_origin else None),
    )

    t_start = time.time()
    paths = generator.generate_multiple(
        n_runs=args.n_runs, output_dir=args.output)
    elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"  Generated {len(paths)} trajectory files in {elapsed:.1f}s")
    print(f"  Output directory: {args.output}")
    print()
    print(f"  Next: run IRL per config to compare recovered weights:")
    for w in weight_configs:
        print(f"    python -m execution.irl_recovery \\")
        print(f"        --trajectory_dir {args.output} \\")
        print(f"        --weight_config {w.name} \\")
        print(f"        --output data/irl_weights_{w.name}.json")
    print(f"{'=' * 60}")

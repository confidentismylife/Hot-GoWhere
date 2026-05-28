"""Main simulation orchestrator — the event loop that ties everything together.

Runs on a single machine (4090 GPU). Core loop:
  1. Perception: update disaster + sample environment
  2. Cognition: submit LLM batch for agents that need re-decision
  3. Group Intel: propagate information, update fear/stamina
  4. Physics: Social Force Model step for every agent
  5. Collect: gather LLM results, apply decisions
  6. Visualize: render frame
"""

import time
import random
import yaml
import numpy as np
from typing import List, Dict, Optional

from decision.agent_state import Agent, AgentProfile, AgentDynamic, Speed, Cooperation
from decision.cognitive_engine import LLMCognitiveEngine, DecisionResult
from decision.knowledge_base import DisasterKnowledgeBase
from perception.environment import DisasterSimulator, EnvironmentSnapshot
from execution.batched_physics import BatchedPhysics
from group_intel.propagation import GroupIntelligence


class SimulationOrchestrator:
    """Coordinates the full simulation pipeline."""

    def __init__(self, config_path: str = "config/default.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        sim = self.cfg["simulation"]
        env = self.cfg["environment"]
        llm_cfg = self.cfg["llm"]
        agent_cfg = self.cfg["agents"]

        # Simulation parameters
        self.num_agents = sim["num_agents"]
        self.duration = sim["duration"]
        self.dt = sim["dt"]
        self.decision_interval = sim["decision_interval"]
        self.decision_ticks = int(self.decision_interval / self.dt)
        self.seed = sim["seed"]
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Environment dimensions
        self.width = env["width"]
        self.height = env["height"]
        self.exits = [tuple(e) for e in env["exit_positions"]]
        self.obstacles = env.get("obstacles", [])

        # Disaster
        self.disaster = DisasterSimulator(
            width=self.width, height=self.height,
            disaster_type=env["disaster"],
            origin=tuple(env["disaster_origin"]),
            spread_rate=env["disaster_spread_rate"],
            resolution=0.5,
        )

        # LLM Engine
        self.llm_engine = LLMCognitiveEngine(config=llm_cfg)
        self.knowledge_base = DisasterKnowledgeBase(
            persist_dir=self.cfg.get("knowledge_base", {}).get("persist_dir")
        )

        # Physics (batched, all agents in one JIT call)
        self.physics = BatchedPhysics(
            width=self.width, height=self.height, obstacles=self.obstacles
        )

        # Group intelligence
        self.group_intel = GroupIntelligence(width=self.width, height=self.height)

        # Agents
        self.agents: List[Agent] = []
        self.agent_cfg = agent_cfg

        # Stats
        self.tick = 0
        self.sim_time = 0.0
        self.evacuated_count = 0
        self.casualty_count = 0
        self.decision_count = 0
        self.total_llm_time = 0.0

    # ================================================================
    # Agent Generation
    # ================================================================

    def generate_agents(self):
        """Spawn agents with realistic demographic distribution."""
        print(f"[Orchestrator] Generating {self.num_agents} agents...")

        ac = self.agent_cfg

        for i in range(self.num_agents):
            profile = self._random_profile(i)
            dynamic = self._random_dynamic(profile)
            agent = Agent(profile=profile, dynamic=dynamic)
            self.agents.append(agent)

        # Create family groups
        self._create_family_groups()

        print(f"[Orchestrator] Generated {len(self.agents)} agents "
              f"across {self.width}×{self.height}m environment.")

    def _random_profile(self, idx: int) -> AgentProfile:
        ac = self.agent_cfg

        # Age
        age_roll = random.random()
        if age_roll < 0.35:
            age = random.randint(18, 35)
        elif age_roll < 0.80:
            age = random.randint(36, 55)
        else:
            age = random.randint(56, 80)

        # Familiarity
        fam_roll = random.random()
        if fam_roll < 0.3:
            familiarity = random.uniform(0.0, 0.3)
        elif fam_roll < 0.8:
            familiarity = random.uniform(0.3, 0.7)
        else:
            familiarity = random.uniform(0.7, 1.0)

        # Max speed (age-dependent)
        if age < 35:
            max_speed = random.uniform(1.2, 2.0)
        elif age < 55:
            max_speed = random.uniform(1.0, 1.6)
        else:
            max_speed = random.uniform(0.6, 1.2)

        return AgentProfile(
            age=age,
            gender=random.choice(["male", "female"]),
            occupation=random.choice(["office_worker", "student", "shopkeeper",
                                       "tourist", "security_guard", "retiree"]),
            familiarity=familiarity,
            max_speed=max_speed,
            risk_aversion=random.uniform(0.2, 0.9),
            altruism=random.uniform(0.1, 0.8),
            trust_authority=random.uniform(0.3, 0.95),
            conformity=random.uniform(0.1, 0.9),
        )

    def _random_dynamic(self, profile: AgentProfile) -> AgentDynamic:
        # Random starting position (avoid obstacles and exits)
        while True:
            x = random.uniform(5, self.width - 5)
            y = random.uniform(5, self.height - 5)
            pos = np.array([x, y], dtype=np.float64)

            # Check not inside obstacle
            blocked = False
            for obs in self.obstacles:
                oc = np.array(obs["center"], dtype=np.float64)
                if np.linalg.norm(pos - oc) < obs["radius"] + 0.5:
                    blocked = True
                    break
            if not blocked:
                break

        # Known exits (familiar people know more exits)
        num_known = max(1, int(profile.familiarity * len(self.exits)))
        known = random.sample(self.exits, num_known)

        return AgentDynamic(
            position=pos,
            stamina=random.uniform(60, 100),
            trust_official_now=profile.trust_authority,
            known_exit_positions=known,
            has_new_info=True,  # Will trigger initial decision
        )

    def _create_family_groups(self):
        """Group some agents into family units."""
        prob = self.agent_cfg.get("family_group_probability", 0.3)
        if prob <= 0:
            return

        # Find agents eligible for family grouping
        eligible = [a for a in self.agents if a.profile.age < 60]
        random.shuffle(eligible)

        family_count = int(len(eligible) * prob / 2)
        for _ in range(family_count):
            if len(eligible) < 2:
                break
            a1 = eligible.pop()
            a2 = eligible.pop()

            # Link them
            a1.dynamic.family_member_ids.append(a2.id)
            a2.dynamic.family_member_ids.append(a1.id)

            # Maybe add a child or elderly
            if random.random() < 0.3:
                a1.dynamic.has_children = True
            if random.random() < 0.3:
                a2.dynamic.has_elderly = a2.profile.age > 60

    # ================================================================
    # Main Simulation Loop
    # ================================================================

    def run(self):
        """Execute the full simulation."""
        print("\n" + "=" * 60)
        print("   LLM-Powered Crowd Evacuation Simulation")
        print("   Single GPU (4090 24GB) Edition")
        print("=" * 60 + "\n")

        # Initialize
        self.generate_agents()
        self.llm_engine.initialize()

        total_ticks = int(self.duration / self.dt)
        vis = None

        vis_cfg = self.cfg.get("visualization", {})
        if vis_cfg.get("mode") == "headless":
            from visualization.headless_renderer import HeadlessRenderer
            vis = HeadlessRenderer(
                frame_interval=vis_cfg.get("frame_interval", 10)
            )
            vis.initialize(self.width, self.height)
        elif vis_cfg.get("enabled", True):
            from visualization.renderer import PygameRenderer
            vis = PygameRenderer(vis_cfg, self.width, self.height)
            vis.initialize()

        print(f"[Orchestrator] Starting simulation: "
              f"{total_ticks} ticks, {self.duration}s, dt={self.dt}s")
        print(f"[Orchestrator] Agents re-decide every {self.decision_interval}s "
              f"({self.decision_ticks} ticks)")

        running = True
        tick_times = []

        while running and self.tick < total_ticks:
            tick_start = time.perf_counter()

            # ---- 1. Perception ----
            self.disaster.step(self.dt)
            env_snapshot = self.disaster.snapshot(
                self.tick, self.sim_time, self.exits, self.obstacles,
                official_broadcast=self._get_broadcast()
            )

            # ---- 2. Cognition (async LLM inference) ----
            agents_to_decide = [
                a for a in self.agents
                if (a.dynamic.alive and not a.dynamic.evacuated and
                    (a.dynamic.has_new_info or
                     self.tick - a.dynamic.last_decision_tick >= self.decision_ticks))
            ]

            if agents_to_decide:
                # Prepare knowledge docs for this batch
                kdocs = self.knowledge_base.query(
                    f"{self.cfg['environment']['disaster']}疏散决策",
                    disaster_type=self.cfg['environment']['disaster'],
                    top_k=3
                )
                kdocs_map = {self.cfg['environment']['disaster']: kdocs}

                self.llm_engine.submit_batch(agents_to_decide, env_snapshot, kdocs_map)
            # ---- 3. Collect LLM results ----
            decisions = self.llm_engine.collect_results()
            if decisions:
                self.decision_count += len(decisions)  # Only count actual results
                self.total_llm_time += sum(d.compute_time for d in decisions.values())
                self._apply_decisions(decisions)

            # ---- 4. Group Intelligence ----
            self.group_intel.propagate(
                self.agents,
                env_snapshot.official_broadcast, self.dt)
            self.group_intel.update_fear_levels(self.agents, env_snapshot, self.dt)
            self.group_intel.update_stamina(self.agents, self.dt)

            # ---- 5. Physics (batched — all agents in one JIT call) ----
            self.physics.step_all(self.agents, self.dt)

            # ---- 6. Stats update ----
            self.evacuated_count = sum(
                1 for a in self.agents if a.dynamic.evacuated)
            self.casualty_count = sum(
                1 for a in self.agents if not a.dynamic.alive)

            # ---- 7. Visualization ----
            if vis:
                running = vis.render(
                    self.agents, env_snapshot,
                    self.tick, self.sim_time,
                    self.evacuated_count, self.casualty_count,
                    self.decision_count
                )

            # ---- 8. Progress logging ----
            tick_time = (time.perf_counter() - tick_start) * 1000
            tick_times.append(tick_time)

            if self.tick % 100 == 0:
                avg_tt = np.mean(tick_times[-100:])
                print(f"[Tick {self.tick:5d}] "
                      f"Time: {self.sim_time:6.1f}s | "
                      f"Tick: {tick_time:5.1f}ms avg: {avg_tt:5.1f}ms | "
                      f"Alive: {self.num_agents - self.evacuated_count - self.casualty_count:4d} | "
                      f"Evac: {self.evacuated_count:4d} | "
                      f"Dead: {self.casualty_count:4d} | "
                      f"Decisions: {self.decision_count:5d}")

            # ---- 9. Termination check ----
            remaining = self.num_agents - self.evacuated_count - self.casualty_count
            if remaining <= 0:
                print(f"\n[Orchestrator] All agents evacuated or deceased at "
                      f"t={self.sim_time:.1f}s")
                running = False

            self.tick += 1
            self.sim_time += self.dt

        # ---- Cleanup ----
        if vis:
            vis.close()

        self.llm_engine.shutdown()
        self._print_summary(tick_times)

    def _apply_decisions(self, decisions: Dict[str, DecisionResult]):
        """Apply LLM decisions to agent states."""
        for agent in self.agents:
            if agent.id in decisions:
                d = decisions[agent.id]
                agent.dynamic.target_exit = np.array(d.target_exit_pos, dtype=np.float64)
                agent.dynamic.speed_choice = d.speed
                agent.dynamic.cooperation_choice = d.cooperation
                agent.dynamic.reasoning_text = d.reasoning
                agent.dynamic.last_decision_tick = self.tick
                agent.dynamic.has_new_info = False

                # Add to memory
                agent.dynamic.memory_events.append({
                    "time": f"{self.sim_time:.0f}s",
                    "desc": f"决定前往出口{d.target_exit_idx+1}: {d.reasoning[:60]}",
                    "credibility": 0.95,
                })
                if len(agent.dynamic.memory_events) > 20:
                    agent.dynamic.memory_events = agent.dynamic.memory_events[-20:]

    def _get_broadcast(self) -> str:
        """Generate official broadcast messages at specific times."""
        if 5 < self.sim_time < 6:
            return "请注意,西南方向发生火灾,请从北侧和东侧出口有序撤离。"
        if 30 < self.sim_time < 31:
            return "东出口出现拥堵,请考虑使用西出口。"
        if 60 < self.sim_time < 61:
            return "消防人员已到达,请保持冷静,听从指挥。"
        return ""

    def _print_summary(self, tick_times: List[float]):
        print("\n" + "=" * 60)
        print("   SIMULATION COMPLETE")
        print("=" * 60)
        print(f"  Duration:         {self.sim_time:.1f}s")
        print(f"  Ticks:            {self.tick}")
        print(f"  Avg tick time:    {np.mean(tick_times):.1f}ms")
        print(f"  Max tick time:    {np.max(tick_times):.1f}ms")
        print(f"  Total agents:     {self.num_agents}")
        print(f"  Evacuated:        {self.evacuated_count} "
              f"({self.evacuated_count/self.num_agents*100:.1f}%)")
        print(f"  Casualties:       {self.casualty_count} "
              f"({self.casualty_count/self.num_agents*100:.1f}%)")
        print(f"  LLM decisions:    {self.decision_count}")
        avg_llm = (self.total_llm_time / self.decision_count * 1000
                   if self.decision_count > 0 else 0)
        print(f"  Avg LLM latency:  {avg_llm:.0f}ms/decision")
        print(f"  Total LLM time:   {self.total_llm_time:.1f}s")
        print("=" * 60)

"""Main simulation orchestrator — the event loop that ties everything together.

Runs on a single machine (4090 GPU). Core loop:
  1. Perception: update disaster + sample environment
  2. Cognition: submit LLM batch for agents that need re-decision
  3. Group Intel: propagate information, update fear/stamina
  4. Physics: Social Force Model step for every agent
  5. Collect: gather LLM results, apply decisions
  6. Visualize: render frame
"""

import os
import time
import random
import yaml
import numpy as np
from typing import List, Dict, Optional

from decision.agent_state import Agent, AgentProfile, AgentDynamic, Speed, Cooperation
from decision.cognitive_engine import LLMCognitiveEngine, DecisionResult
from decision.knowledge_base import DisasterKnowledgeBase
from decision.safety_guard import SafetyGuard
from decision.agent_roles import AgentRole, define_zones
from decision.policy import HeuristicPolicy
from perception.environment import DisasterSimulator, EnvironmentSnapshot
from execution.batched_physics import BatchedPhysics
from execution.diffusion_policy import build_scene_map
from execution.agent_factory import random_profile, create_family_groups
from execution.irl_recovery import TrajectoryCollector
from execution.rl_scheduler import RLZoneScheduler, inject_rl_preferences, DEFAULT_ZONES
from execution.tactical_layer import TacticalLayer
from group_intel.propagation import GroupIntelligence


class SimulationOrchestrator:
    """Coordinates the full simulation pipeline."""

    def __init__(self, config_path: str = "config/default.yaml",
                 llm_engine: Optional[LLMCognitiveEngine] = None,
                 keep_llm_engine: bool = False):
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
        self.enable_command_agents = sim.get("enable_command_agents", True)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Floor plan support (mall layout)
        self.floorplan = None
        if env.get("type") == "mall" or env.get("floorplan"):
            from perception.floorplan import get_floorplan
            fp_name = env.get("floorplan", "chaoyang_joycity_1f")
            self.floorplan = get_floorplan(fp_name)
            print(f"[Orchestrator] Loaded floor plan: {self.floorplan.name}")

        # Environment dimensions
        if self.floorplan:
            self.width = self.floorplan.width
            self.height = self.floorplan.height
            # Config exit_positions can override floorplan exits (for experiments)
            if env.get("exit_positions"):
                self.exits = [tuple(e) for e in env["exit_positions"]]
                print(f"[Orchestrator] Using config exit overrides: {len(self.exits)} exits")
            else:
                self.exits = self.floorplan.exits
            # Merge floor plan obstacles with config obstacles
            self.obstacles = self.floorplan.obstacles + env.get("obstacles", [])
        else:
            self.width = env["width"]
            self.height = env["height"]
            self.exits = [tuple(e) for e in env["exit_positions"]]
            self.obstacles = env.get("obstacles", [])

        # Disaster — supports one or more fire origins
        disaster_origin = env.get("disaster_origin", (self.width * 0.2, self.height * 0.5))
        if self.floorplan and env.get("disaster_origin") is None:
            disaster_origin = self.floorplan.disaster_origin_default
        fire_sources = env.get("fire_sources")
        self.fire_origins = (
            [tuple(o) for o in fire_sources]
            if fire_sources else [tuple(disaster_origin)]
        )
        self.disaster = DisasterSimulator(
            width=self.width, height=self.height,
            disaster_type=env["disaster"],
            origin=self.fire_origins,
            spread_rate=env["disaster_spread_rate"],
            resolution=0.5,
            wall_mask=(self.floorplan.grid if self.floorplan else None),
        )

        # LLM Engine — may be injected from outside so multiple simulation
        # runs share one loaded vLLM engine (load once, run all seeds).
        self._external_llm_engine = llm_engine
        self._keep_llm_engine = keep_llm_engine or (llm_engine is not None)
        if llm_engine is not None:
            self.llm_engine = llm_engine
        else:
            self.llm_engine = LLMCognitiveEngine(config=llm_cfg)
        self.knowledge_base = DisasterKnowledgeBase(
            persist_dir=self.cfg.get("knowledge_base", {}).get("persist_dir")
        )

        # Physics (batched, all agents in one JIT call)
        wall_grid = self.floorplan.grid if self.floorplan else None
        self.physics = BatchedPhysics(
            width=self.width, height=self.height,
            obstacles=self.obstacles, wall_grid=wall_grid,
        )

        # Group intelligence
        self.group_intel = GroupIntelligence(width=self.width, height=self.height)

        # Safety guard (hard constraints on LLM output)
        self.safety_guard = SafetyGuard()
        self.heuristic_policy = HeuristicPolicy()

        # Agents
        self.agents: List[Agent] = []
        self._agent_lookup: Dict[str, Agent] = {}  # O(1) lookup
        self.agent_cfg = agent_cfg

        # Stats
        self.tick = 0
        self.sim_time = 0.0
        self.evacuated_count = 0
        self.casualty_count = 0
        self.decision_count = 0
        self.total_llm_time = 0.0
        self.safety_blocks = 0       # Count of LLM decisions blocked by safety guard
        self.safety_modifications = 0  # Count of decisions modified by safety guard
        self._command_broadcasts = []  # Commander-generated broadcasts (injected into civilian prompts)
        self._kb_cache: Dict[str, List[str]] = {}  # Cache KB queries
        self._kb_cache_tick: int = -999

        # IRL/Reward data collection (LLM → IRL → RL cascade)
        irl_cfg = self.cfg.get("irl", {})
        self.enable_irl_collection = irl_cfg.get("enabled", False)
        self.trajectory_collector: Optional[TrajectoryCollector] = None
        self.irl_output_dir = irl_cfg.get("output_dir", "data/trajectories")

        # RL zone scheduling
        rl_cfg = self.cfg.get("rl_scheduling", {})
        self.enable_rl_scheduling = rl_cfg.get("enabled", False)
        self._rl_smoke_block = float(
            rl_cfg.get("smoke_block_threshold", 0.6))
        self._rl_fire_path_block = bool(
            rl_cfg.get("fire_path_block", True))
        self.zone_scheduler: Optional[RLZoneScheduler] = None
        self._rl_zones = DEFAULT_ZONES  # May be overridden by YAML config
        self._zone_actions = None  # Cached per-tick zone actions
        self._rl_preferences_cache: Dict[str, str] = {}  # agent_id → NL advice
        self.rl_policy_loaded = False  # True only if pretrained MLP weights loaded
        self.role_stats = None  # Filled by _compute_role_stats() at end of run
        self.avg_tick_ms = 0.0  # Filled at end of run
        self.max_tick_ms = 0.0  # Filled at end of run

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
        self._rebuild_lookup()

        print(f"[Orchestrator] Generated {len(self.agents)} agents "
              f"across {self.width}×{self.height}m environment.")

    def _random_profile(self, idx: int) -> AgentProfile:
        """Generate one agent profile from config-driven distributions."""
        return random_profile(self.agent_cfg)

    def _random_dynamic(self, profile: AgentProfile) -> AgentDynamic:
        # Random starting position (avoid walls, obstacles and exits)
        for _ in range(1000):  # Safety limit
            x = random.uniform(5, self.width - 5)
            y = random.uniform(5, self.height - 5)
            pos = np.array([x, y], dtype=np.float64)

            # Check not inside floor plan wall
            if self.floorplan and not self.floorplan.is_walkable(x, y):
                continue

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

        # Seed initial target: score known exits by distance + fire risk.
        # Exits near the fire origin get heavily penalized because they'll
        # be smoked soon. Agents start moving immediately with a reasonable
        # target while waiting for first LLM response.
        fire_origin = np.array(
            self.cfg["environment"].get("disaster_origin", (self.width/2, self.height/2)),
            dtype=np.float64)
        best_exit = known[0]
        best_score = float('inf')
        for ex in known:
            ex_arr = np.array(ex, dtype=np.float64)
            dist = float(np.linalg.norm(pos - ex_arr))
            dist_to_fire = float(np.linalg.norm(ex_arr - fire_origin))
            fire_risk = max(0.0, 1.0 - dist_to_fire / 40.0)
            score = dist * (1.0 + fire_risk * 4.0)
            if score < best_score:
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
            has_new_info=True,  # Will trigger initial LLM decision
        )

    def _create_family_groups(self):
        """Group some agents into family units."""
        new_members = create_family_groups(self.agents, self.agent_cfg)
        if new_members:
            self.agents.extend(new_members)
            self._rebuild_lookup()

    def _spawn_command_agents(self):
        """Spawn multi-role command agents (commander, firefighter, guide).

        These agents use role-specific LLM prompts and do NOT participate
        in the social-force physics engine. Their decisions influence
        civilian behavior through broadcasts and guidance.
        """
        zones = define_zones(self.exits, self.width, self.height)
        disaster_origin = self.cfg["environment"]["disaster_origin"]
        num_command_agents = 0

        # --- 1 Global Commander ---
        profile = AgentProfile(
            role=AgentRole.GLOBAL_COMMANDER.value,
            age=45, occupation="fire_chief",
            familiarity=1.0, max_speed=0.0,
            trust_authority=1.0, altruism=0.9,
            equipment=["对讲机", "建筑蓝图", "监控终端"],
        )
        # Position at a safe command post (center-top of map)
        dynamic = AgentDynamic(
            position=np.array([self.width / 2, self.height - 2], dtype=np.float64),
            stamina=100.0, known_exit_positions=self.exits,
            has_new_info=True,
        )
        self.agents.append(Agent(profile=profile, dynamic=dynamic))
        num_command_agents += 1

        # --- Area Commanders (one per exit) ---
        for i, exit_pos in enumerate(self.exits):
            profile = AgentProfile(
                role=AgentRole.AREA_COMMANDER.value,
                age=35, occupation="station_staff",
                familiarity=0.9, max_speed=1.2,
                trust_authority=0.9, altruism=0.8,
                equipment=["对讲机", "扩音器"],
            )
            # Position near their assigned exit
            ex, ey = exit_pos
            sx = max(2, min(self.width - 2, ex + random.uniform(-5, 5)))
            sy = max(2, min(self.height - 2, ey + random.uniform(-5, 5)))
            dynamic = AgentDynamic(
                position=np.array([sx, sy], dtype=np.float64),
                stamina=100.0, known_exit_positions=[exit_pos],
                has_new_info=True,
            )
            self.agents.append(Agent(profile=profile, dynamic=dynamic))
            num_command_agents += 1

        # --- Firefighters ---
        firefighter_count = max(2, self.num_agents // 100)  # ~2% of civilians
        for _ in range(firefighter_count):
            profile = AgentProfile(
                role=AgentRole.FIREFIGHTER.value,
                age=random.randint(25, 40), occupation="firefighter",
                familiarity=0.7, max_speed=1.6,
                risk_aversion=0.3, altruism=0.95, trust_authority=0.9,
                equipment=["呼吸器", "灭火器", "对讲机", "热成像仪"],
            )
            # Start near disaster origin but at safe distance
            ox, oy = disaster_origin
            sx = max(5, min(self.width - 5, ox + random.uniform(-15, 15)))
            sy = max(5, min(self.height - 5, oy + random.uniform(-15, 15)))
            dynamic = AgentDynamic(
                position=np.array([sx, sy], dtype=np.float64),
                stamina=100.0, known_exit_positions=self.exits,
                has_new_info=True,
            )
            self.agents.append(Agent(profile=profile, dynamic=dynamic))
            num_command_agents += 1

        # --- Guides ---
        guide_count = min(len(self.exits) * 2, 8)
        for i in range(guide_count):
            assigned_exit = self.exits[i % len(self.exits)]
            profile = AgentProfile(
                role=AgentRole.GUIDE.value,
                age=random.randint(25, 45), occupation="station_staff",
                familiarity=0.9, max_speed=1.3,
                altruism=0.9, trust_authority=0.8,
                equipment=["反光背心", "手电筒", "扩音器"],
            )
            ex, ey = assigned_exit
            sx = max(5, min(self.width - 5, ex + random.uniform(-10, 10)))
            sy = max(5, min(self.height - 5, ey - random.uniform(5, 15)))
            dynamic = AgentDynamic(
                position=np.array([sx, sy], dtype=np.float64),
                stamina=100.0, known_exit_positions=[assigned_exit],
                has_new_info=True,
            )
            self.agents.append(Agent(profile=profile, dynamic=dynamic))
            num_command_agents += 1

        print(f"[Orchestrator] Spawned {num_command_agents} command agents "
              f"(1 global, {len(self.exits)} area, "
              f"{firefighter_count} firefighter, {guide_count} guide)")
        self._rebuild_lookup()

    def _build_command_context(self, agent: Agent,
                               env: EnvironmentSnapshot) -> str:
        """Build NL context for command-role agents (global view)."""
        from perception.nl_converter import NLConverter

        # Commander sees aggregate stats, not just personal position
        alive = sum(1 for a in self.agents if a.dynamic.alive and not a.dynamic.evacuated)
        evac = sum(1 for a in self.agents if a.dynamic.evacuated)
        dead = sum(1 for a in self.agents if not a.dynamic.alive)

        # Smoke coverage estimate
        smoke_grid = env.grid[:, :, 0]
        smoke_coverage = float((smoke_grid > 0.3).mean())
        fire_coverage = float((env.grid[:, :, 3] > 0.5).mean())

        exit_lines = []
        for i, ep in enumerate(env.exits):
            e_smoke = env.smoke_at(np.array(ep, dtype=np.float64))
            status = "通畅" if e_smoke < 0.3 else ("有烟雾" if e_smoke < 0.6 else "浓烟封锁")
            exit_lines.append(f"  出口{i+1}({ep[0]:.0f},{ep[1]:.0f}): {status}")

        return f"""[全局态势]
时间: {env.timestamp:.0f}秒
灾害类型: {env.disaster_type}
烟雾覆盖: {smoke_coverage:.0%} 区域
火灾覆盖: {fire_coverage:.0%} 区域

[人员统计]
总人数: {len(self.agents)}
已疏散: {evac}
伤亡: {dead}
仍在现场: {alive}

[出口状态]
{chr(10).join(exit_lines)}

[官方广播记录]
{env.official_broadcast or '暂无'}

[指挥角色]
{agent.profile.role}: {agent.profile.occupation}
位置: ({agent.position[0]:.0f}, {agent.position[1]:.0f})"""

    def _get_broadcast(self) -> str:
        """Return latest commander-generated broadcast, with cleanup."""
        # Remove broadcasts older than 30 seconds
        cutoff = self.sim_time - 30.0
        self._command_broadcasts = [
            b for b in self._command_broadcasts
            if b.get("tick", 0) * self.dt > cutoff
        ]
        if self._command_broadcasts:
            # Return the most recent 3 messages, joined
            recent = self._command_broadcasts[-3:]
            return " | ".join(b["message"] for b in recent)
        return ""

    # ================================================================
    # Main Simulation Loop
    # ================================================================

    def run(self):
        """Execute the full simulation with multi-role agents."""
        self.use_llm = self.cfg.get("llm", {}).get("enabled", True)
        self.use_vlm = self.cfg.get("vlm", {}).get("enabled", False)
        self.use_diffusion = self.cfg.get("diffusion", {}).get("enabled", False)

        print("\n" + "=" * 60)
        print("   LLM-Powered Crowd Evacuation — Multi-Role v2.1")
        if self.use_vlm or self.use_diffusion:
            print(f"   VLM: {'ON' if self.use_vlm else 'OFF'}  "
                  f"Trajectory: {'DIFFUSION' if self.use_diffusion else 'SOCIAL FORCE'}")
        print("=" * 60 + "\n")

        # Initialize
        self.generate_agents()
        if self.enable_command_agents:
            self._spawn_command_agents()
        if self.use_llm:
            self.llm_engine.initialize()
            if self.cfg.get("llm", {}).get("fixed_seed", False):
                self.llm_engine.set_seed(self.seed)

        # IRL trajectory collection
        if self.enable_irl_collection:
            self.trajectory_collector = TrajectoryCollector()
            print(f"[Orchestrator] IRL trajectory collection ENABLED → "
                  f"{self.irl_output_dir}")

        # v2: VLM 感知器 + YOLO 检测器 (双通道，独立开关)
        self.vlm = None
        self.yolo = None
        self.use_yolo = self.cfg.get("yolo", {}).get("enabled", False)

        # VLM 感知通道
        vlm_mock = self.cfg.get("vlm", {}).get("mock", False)
        if self.use_vlm:
            from perception.vlm_perceiver import VLMPerceiver, MockVLMPerceiver

            if vlm_mock:
                self.vlm = MockVLMPerceiver(
                    call_interval=self.cfg["vlm"].get("call_interval", 30),
                )
                self.vlm.initialize()
            else:
                try:
                    self.vlm = VLMPerceiver(
                        model_name=self.cfg["vlm"].get("model", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"),
                        call_interval=self.cfg["vlm"].get("call_interval", 30),
                    )
                    self.vlm.initialize()
                except Exception as e:
                    print(f"[Orch] VLM load failed ({e}), falling back to mock.")
                    self.vlm = MockVLMPerceiver(
                        call_interval=self.cfg["vlm"].get("call_interval", 30),
                    )
                    self.vlm.initialize()

        # YOLO 检测通道 (独立于VLM)
        if self.use_yolo:
            from perception.yolo_detector import YOLODetector
            yolo_cfg = self.cfg.get("yolo", {})
            self.yolo = YOLODetector(
                model_name=yolo_cfg.get("model", "yolov8n.pt"),
                confidence_threshold=yolo_cfg.get("conf_threshold", 0.35),
                call_interval=yolo_cfg.get("call_interval", 10),
            )
            self.yolo.initialize()
            self.yolo.set_calibration(
                img_w=640, img_h=480,
                world_w=self.width, world_h=self.height,
            )

        # v2: 扩散模型轨迹生成
        self.diffusion_policy = None
        if self.use_diffusion:
            from execution.diffusion_policy import DiffusionPolicy
            self.diffusion_policy = DiffusionPolicy(self.cfg)
            self.diffusion_policy.initialize()

        # v2.1: RL zone scheduler (LLM → IRL → RL cascade)
        if self.enable_rl_scheduling:
            rl_cfg = self.cfg.get("rl_scheduling", {})
            self._rl_zones = self._load_zones_from_config()
            self.zone_scheduler = RLZoneScheduler(
                zones=self._rl_zones,
                num_exits=len(self.exits),
            )
            pretrained = rl_cfg.get("pretrained_weights")
            self.zone_scheduler.initialize(
                pretrained_path=pretrained if pretrained else None
            )
            self.rl_policy_loaded = self.zone_scheduler._pretrained_loaded
            # Load IRL weights if available
            irl_weights_path = rl_cfg.get("irl_weights")
            if irl_weights_path and os.path.exists(irl_weights_path):
                from execution.irl_recovery import IRLRecovery
                irl = IRLRecovery()
                irl.load(irl_weights_path)
                self.zone_scheduler.load_irl_weights(irl.weights)
            print(f"[Orchestrator] RL zone scheduling ENABLED "
                  f"({len(self._rl_zones)} zones, {len(self.exits)} exits)")

        total_ticks = int(self.duration / self.dt)
        vis = None

        vis_cfg = self.cfg.get("visualization", {})
        if vis_cfg.get("mode") == "headless":
            try:
                from visualization.headless_renderer import HeadlessRenderer
                vis = HeadlessRenderer(
                    frame_interval=vis_cfg.get("frame_interval", 10),
                    floorplan=self.floorplan,
                )
                vis.initialize(self.width, self.height)
            except ImportError:
                vis = None
        elif vis_cfg.get("enabled", True):
            try:
                from visualization.renderer import PygameRenderer
                vis = PygameRenderer(vis_cfg, self.width, self.height,
                                     floorplan=self.floorplan)
                vis.initialize()
            except ImportError:
                vis = None

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

            # Compute exit crowd counts for LLM strategic context (v2.2)
            exit_crowd_counts = [0] * len(self.exits)
            for a in self.agents:
                if (a.dynamic.alive and not a.dynamic.evacuated
                        and a.dynamic.target_exit is not None):
                    tgt = a.dynamic.target_exit
                    best_i = 0
                    best_d = float('inf')
                    for i, ex in enumerate(self.exits):
                        d = float(np.linalg.norm(tgt - np.array(ex, dtype=np.float64)))
                        if d < best_d:
                            best_d = d
                            best_i = i
                    if best_d < 2.0:  # within 2m of an exit → counted as heading there
                        exit_crowd_counts[best_i] += 1

            env_snapshot = self.disaster.snapshot(
                self.tick, self.sim_time, self.exits, self.obstacles,
                official_broadcast=self._get_broadcast(),
                exit_crowd_counts=exit_crowd_counts,
            )

            # ---- 1.5 RL Zone Scheduling (LLM → IRL → RL cascade) ----
            if self.enable_rl_scheduling and self.zone_scheduler is not None:
                self._zone_actions = self.zone_scheduler.infer(
                    env_snapshot, self.agents, self.tick)
            else:
                self._zone_actions = None
                self._rl_preferences_cache.clear()

            # ---- 2. Collect LLM results from previous batch FIRST ----
            decisions = self.llm_engine.collect_results() if self.use_llm else {}
            if decisions:
                # Separate civilian vs command decisions
                civilian_decisions = {}
                command_decisions = {}
                for aid, d in decisions.items():
                    agent = self._find_agent(aid)
                    if agent and agent.profile.role != AgentRole.CIVILIAN.value:
                        command_decisions[aid] = d
                    else:
                        civilian_decisions[aid] = d

                self.decision_count += len(decisions)
                self.total_llm_time += sum(d.compute_time for d in decisions.values())
                if self.tick % 50 == 0:
                    print(f"[LLM collect] tick={self.tick} collected={len(decisions)} "
                          f"total_llm_time={self.total_llm_time:.1f}s")

                # Apply civilian decisions normally
                if civilian_decisions:
                    self._apply_decisions(civilian_decisions, env_snapshot)

                # Process command decisions → generate broadcasts
                if command_decisions:
                    self._apply_command_decisions(command_decisions, env_snapshot)

            # ---- 3. Cognition: compute agents that need re-decision ----
            agents_to_decide = [
                a for a in self.agents
                if (a.dynamic.alive and not a.dynamic.evacuated and
                    (a.dynamic.has_new_info or
                     self.tick - a.dynamic.last_decision_tick >= self.decision_ticks))
            ]

            # ---- 3.1 Criticality Filter: Brain-Torso split ----
            # ALL agents get heuristic decisions immediately (no waiting).
            # Critical agents are ALSO submitted to LLM for strategic refinement;
            # when the LLM response arrives later, it overrides the heuristic.
            # NOTE: is_critical must be checked BEFORE _apply_decisions clears has_new_info.
            if self.use_llm and agents_to_decide:
                critical = [a for a in agents_to_decide
                           if TacticalLayer.is_critical(a, env_snapshot, self.tick, self.decision_ticks)]

                # Give every agent an immediate heuristic decision
                heur_decisions = {}
                for agent in agents_to_decide:
                    heur_decisions[agent.id] = self.heuristic_policy.decide(
                        agent, env_snapshot, self.exits, tick=self.tick)
                if heur_decisions:
                    self._apply_decisions(heur_decisions, env_snapshot)
                    self.decision_count += len(heur_decisions)

                # Only critical agents get LLM strategic refinement
                agents_to_decide = critical

            # ---- 3.5 Heuristic Fallback (when LLM disabled) ----
            if not self.use_llm and agents_to_decide:
                heur_decisions = {}
                for agent in agents_to_decide:
                    heur_decisions[agent.id] = self.heuristic_policy.decide(
                        agent, env_snapshot, self.exits, tick=self.tick)
                if heur_decisions:
                    self._apply_decisions(heur_decisions, env_snapshot)

            # ---- 3.6 RL advice for agents that will decide ----
            if self._zone_actions is not None and agents_to_decide:
                self._rl_preferences_cache.clear()
                for agent in agents_to_decide:
                    advice = inject_rl_preferences(
                        self._zone_actions, self._rl_zones,
                        agent, env_snapshot,
                        smoke_block_threshold=self._rl_smoke_block,
                        fire_path_block=self._rl_fire_path_block)
                    if advice:
                        self._rl_preferences_cache[agent.id] = advice

            # ---- 4. Cognition: submit new agents for async LLM inference ----
            if agents_to_decide and self.use_llm:
                if self.tick - self._kb_cache_tick > 30:
                    d_type = self.cfg['environment']['disaster']
                    self._kb_cache["professional"] = \
                        self.knowledge_base.query(
                            f"{d_type}疏散决策", disaster_type=d_type, top_k=3)
                    self._kb_cache["civilian"] = \
                        self.knowledge_base.query(
                            "通用安全常识", disaster_type="general", top_k=3)
                    self._kb_cache_tick = self.tick
                kdocs_map = {
                    "professional": self._kb_cache.get("professional", []),
                    "civilian": self._kb_cache.get("civilian", []),
                }

                self.llm_engine.submit_batch(
                    agents_to_decide, env_snapshot, kdocs_map,
                    rl_preferences=(self._rl_preferences_cache
                                    if self.enable_rl_scheduling else None))

            # ---- 3.5 VLM + YOLO 双通道感知 (v2.0) ----
            frame = None
            vlm_desc = ""
            yolo_res = None
            dirty = False

            # 通道A: VLM 语义理解 (on its own interval)
            if self.vlm is not None and self.tick % self.vlm.call_interval == 0:
                frame = self._render_cctv_frame()
                vlm_desc = self.vlm.perceive(frame, self.tick, env_snapshot)
                dirty = True

            # 通道B: YOLO 人员检测 (independent, on its own interval)
            yolo_interval = self.yolo.call_interval if self.yolo is not None else 10
            if self.yolo is not None and self.tick % yolo_interval == 0:
                if frame is None:
                    frame = self._render_cctv_frame()
                yolo_res = self.yolo.detect(frame)
                dirty = True

            if dirty and self.use_llm:
                self.llm_engine.set_perception_context(vlm_desc, yolo_res)

            # ---- 4. Group Intelligence ----
            self.group_intel.propagate(
                self.agents,
                env_snapshot.official_broadcast, self.dt)
            self.group_intel.update_fear_levels(self.agents, env_snapshot, self.dt)
            self.group_intel.update_stamina(self.agents, self.dt)
            self.group_intel.update_hazard_damage(self.agents, env_snapshot, self.dt)

            # ---- 4.5 Tactical Layer (Brain-Torso: per-tick reactive adjustments) ----
            TacticalLayer.adjust_all(self.agents, env_snapshot, self.exits, self.tick)

            # ---- 5. Physics ----
            if self.use_diffusion and self.diffusion_policy is not None:
                self._step_diffusion(env_snapshot)
                self._mark_evacuated_timestamps()
            else:
                self.physics.step_all(self.agents, self.dt)
                self._mark_evacuated_timestamps()

            # ---- 6. Stats update (single pass) ----
            evac = 0; dead = 0
            for a in self.agents:
                if a.dynamic.evacuated: evac += 1
                elif not a.dynamic.alive: dead += 1
            self.evacuated_count = evac
            self.casualty_count = dead

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
                total = len(self.agents)
                alive = total - self.evacuated_count - self.casualty_count
                print(f"[Tick {self.tick:5d}] "
                      f"Time: {self.sim_time:6.1f}s | "
                      f"Tick: {tick_time:5.1f}ms avg: {avg_tt:5.1f}ms | "
                      f"Alive: {alive:4d} | "
                      f"Evac: {self.evacuated_count:4d} | "
                      f"Dead: {self.casualty_count:4d} | "
                      f"LLM: {self.decision_count:5d} | "
                      f"Blocked: {self.safety_blocks:3d}/{self.safety_modifications:3d}")

            # ---- 9. Termination check ----
            remaining = len(self.agents) - self.evacuated_count - self.casualty_count
            if remaining <= 0:
                print(f"\n[Orchestrator] All agents evacuated or deceased at "
                      f"t={self.sim_time:.1f}s")
                running = False

            self.tick += 1
            self.sim_time += self.dt

        # ---- Drain pending LLM results before shutdown ----
        if self.use_llm:
            final_decisions = self.llm_engine.drain(timeout=30.0)
            if final_decisions:
                self.decision_count += len(final_decisions)
                self.total_llm_time += sum(d.compute_time for d in final_decisions.values())
                print(f"[Orchestrator] Drained {len(final_decisions)} final LLM results")

        # ---- Cleanup ----
        if vis:
            vis.close()

        # v2.0 模块清理
        if self.vlm is not None:
            self.vlm.shutdown()
        if self.yolo is not None:
            self.yolo.shutdown()
        if self.diffusion_policy is not None:
            self.diffusion_policy.shutdown()

        if self.use_llm and not self._keep_llm_engine:
            self.llm_engine.shutdown()

        # Save IRL trajectory data
        if self.trajectory_collector is not None:
            self.trajectory_collector.finalize(self.agents, dt=self.dt)
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            traj_path = os.path.join(self.irl_output_dir, f"run_{ts}.jsonl")
            self.trajectory_collector.save(traj_path)
            print(f"[Orchestrator] Trajectories saved to {traj_path}")

        self.avg_tick_ms = float(np.mean(tick_times)) if tick_times else 0.0
        self.max_tick_ms = float(np.max(tick_times)) if tick_times else 0.0
        self._compute_role_stats()
        self.safety_rule_counts = dict(
            getattr(self.safety_guard, "counters", {}))
        self._print_summary(tick_times)

    def _apply_decisions(self, decisions: Dict[str, DecisionResult],
                         env_snapshot=None):
        """Apply LLM decisions to agent states, with safety guard filtering."""
        for agent in self.agents:
            if agent.id not in decisions:
                continue

            d = decisions[agent.id]
            safety = None

            # --- Safety guard check ---
            if env_snapshot is not None:
                safety = self.safety_guard.check(d, agent, env_snapshot)

                if not safety.passed:
                    # Decision blocked — use safety fallback
                    self.safety_blocks += 1
                    fb = self.safety_guard.fallback_decision(agent, env_snapshot)
                    target_exit = np.array(fb["target_exit_pos"], dtype=np.float64)
                    speed = fb["speed"]
                    cooperation = fb["cooperation"]
                    reasoning = fb["reasoning"]
                    target_idx = fb["target_exit_idx"]
                    agent.dynamic.memory_events.append({
                        "time": f"{self.sim_time:.0f}s",
                        "desc": f"⛔ 决策被安全约束拦截: {safety.block_reason}",
                        "credibility": 1.0,
                    })
                elif safety.modified:
                    # Decision modified — apply corrected version
                    self.safety_modifications += 1
                    target_exit = np.array(
                        env_snapshot.exits[safety.final_exit_idx],
                        dtype=np.float64
                    )
                    speed = Speed(safety.final_speed)
                    cooperation = d.cooperation
                    reasoning = d.reasoning
                    target_idx = safety.final_exit_idx
                    for w in safety.warnings:
                        agent.dynamic.memory_events.append({
                            "time": f"{self.sim_time:.0f}s",
                            "desc": f"⚠ 安全修正: {w}",
                            "credibility": 1.0,
                        })
                else:
                    # Clean — apply original
                    target_exit = np.array(d.target_exit_pos, dtype=np.float64)
                    speed = d.speed
                    cooperation = d.cooperation
                    reasoning = d.reasoning
                    target_idx = d.target_exit_idx
            else:
                # No env snapshot — apply as-is
                target_exit = np.array(d.target_exit_pos, dtype=np.float64)
                speed = d.speed
                cooperation = d.cooperation
                reasoning = d.reasoning
                target_idx = d.target_exit_idx

            # Apply final decision to agent
            agent.dynamic.target_exit = target_exit
            agent.dynamic.target_exit_idx = target_idx
            agent.dynamic.speed_choice = speed
            agent.dynamic.cooperation_choice = cooperation
            agent.dynamic.reasoning_text = reasoning
            agent.dynamic.last_decision_tick = self.tick
            agent.dynamic.has_new_info = False

            # Record decision in memory (always, regardless of safety outcome)
            safety_tag = ""
            if safety is not None and not safety.passed:
                safety_tag = " [安全约束兜底]"
            agent.dynamic.memory_events.append({
                "time": f"{self.sim_time:.0f}s",
                "desc": f"决定前往出口{target_idx+1}{safety_tag}: {reasoning[:60]}",
                "credibility": 0.95 if (safety is None or safety.passed) else 0.7,
            })

            # Trim memory
            if len(agent.dynamic.memory_events) > 20:
                agent.dynamic.memory_events = agent.dynamic.memory_events[-20:]

            # Record trajectory for IRL (if collection enabled)
            if self.trajectory_collector is not None and env_snapshot is not None:
                blocked = (safety is not None and not safety.passed)
                modified = (safety is not None and safety.modified)
                self.trajectory_collector.record_decision(
                    agent, d, env_snapshot, blocked, modified)

    def _find_agent(self, agent_id: str):
        """Find agent by ID. Returns None if not found. O(1) via lookup dict."""
        return self._agent_lookup.get(agent_id)

    def _mark_evacuated_timestamps(self):
        """Stamp the real sim time/tick on agents evacuated this step."""
        for a in self.agents:
            d = a.dynamic
            if d.evacuated and d.evacuation_time < 0:
                d.evacuation_time = self.sim_time
                d.evacuation_tick = self.tick

    def _load_zones_from_config(self):
        """Load zone definitions from YAML config, falling back to DEFAULT_ZONES."""
        zones_cfg = self.cfg.get("zones")
        if not zones_cfg or not isinstance(zones_cfg, list):
            return DEFAULT_ZONES
        from execution.rl_scheduler import ZoneDefinition
        zones = []
        for z in zones_cfg:
            zones.append(ZoneDefinition(
                zone_id=z["id"],
                name=z.get("name", f"Zone{z['id']}"),
                x_min=z["x_min"], x_max=z["x_max"],
                y_min=z["y_min"], y_max=z["y_max"],
                primary_exits=z.get("primary_exits", list(range(len(self.exits)))),
                description=z.get("description", ""),
            ))
        print(f"[Orchestrator] Loaded {len(zones)} zone definitions from config")
        return zones

    def _rebuild_lookup(self):
        """Rebuild agent ID → agent dict. Call after spawning agents."""
        self._agent_lookup = {a.id: a for a in self.agents}

    def _apply_command_decisions(self, decisions: Dict[str, DecisionResult],
                                 env_snapshot: EnvironmentSnapshot):
        """Parse LLM commander outputs into broadcasts and guidance.

        Commander decisions use the raw LLM JSON (d.raw_data) which preserves
        role-specific fields like broadcast_message, action, target_position.
        """
        for agent_id, d in decisions.items():
            agent = self._find_agent(agent_id)
            if agent is None:
                continue

            role = agent.profile.role

            # Use raw_data — the full parsed LLM JSON, not just reasoning text
            data = d.raw_data or {}

            if role == AgentRole.GLOBAL_COMMANDER.value:
                # Extract broadcast message for civilians
                broadcast = data.get("broadcast_message", "")
                if broadcast:
                    self._command_broadcasts.append({
                        "tick": self.tick,
                        "message": broadcast,
                        "source": "global_commander",
                    })

                # Store area priorities
                priorities = data.get("area_priorities", [])
                if priorities:
                    env_snapshot.official_broadcast = broadcast

            elif role == AgentRole.AREA_COMMANDER.value:
                broadcast = data.get("broadcast_message", "")
                if broadcast:
                    self._command_broadcasts.append({
                        "tick": self.tick,
                        "message": broadcast,
                        "source": f"area_commander_{agent_id[:6]}",
                    })

            elif role == AgentRole.FIREFIGHTER.value:
                # Parse firefighter action
                action = data.get("action", "move")
                target = data.get("target_position", agent.position.tolist())
                agent.dynamic.reasoning_text = d.reasoning

                # If rescuing, set cooperation to help_family equivalent
                if action == "rescue":
                    agent.dynamic.cooperation_choice = Cooperation.LEAD_OTHERS

                # Apply target position if valid
                if isinstance(target, list) and len(target) == 2:
                    agent.dynamic.target_exit = np.array(target, dtype=np.float64)

            elif role == AgentRole.GUIDE.value:
                # Parse guide decision
                exit_str = data.get("target_exit", "")
                route = data.get("route_description", "")
                call = data.get("call_for_followers", True)

                # Determine which exit the guide is leading to
                # Iterate descending to avoid substring matches (出口1 ≠ 出口12)
                for i in range(len(env_snapshot.exits), 0, -1):
                    if f"出口{i}" in exit_str:
                        agent.dynamic.target_exit = np.array(
                            env_snapshot.exits[i - 1], dtype=np.float64
                        )
                        agent.dynamic.target_exit_idx = i - 1
                        break

                if call and route:
                    self._command_broadcasts.append({
                        "tick": self.tick,
                        "message": f"引导员: {route}",
                        "source": f"guide_{agent_id[:6]}",
                    })

            # Common: apply speed decision
            speed_str = data.get("speed", "walk")
            try:
                agent.dynamic.speed_choice = Speed(speed_str)
            except ValueError:
                agent.dynamic.speed_choice = Speed.WALK

            agent.dynamic.last_decision_tick = self.tick
            agent.dynamic.has_new_info = False

            # Record command agent trajectory for IRL
            if self.trajectory_collector is not None:
                self.trajectory_collector.record_decision(
                    agent, d, env_snapshot, False, False)

    # ================================================================
    # v2.0: 扩散模型轨迹播放
    # ================================================================

    def _step_diffusion(self, env_snapshot):
        """v2.0: 播放预生成的扩散轨迹, 需要时重新生成."""

        for agent in self.agents:
            if agent.dynamic.evacuated or not agent.dynamic.alive:
                continue

            # 判断是否需要重新生成轨迹
            needs_new = (
                agent.dynamic.future_trajectory is None or
                agent.dynamic.traj_step >= len(agent.dynamic.future_trajectory) or
                agent.dynamic.has_new_info
            )

            if needs_new:
                if agent.dynamic.target_exit is not None:
                    scene = build_scene_map(
                        agent.position, env_snapshot, resolution=64
                    )
                    traj = self.diffusion_policy.generate_one(
                        start=agent.position,
                        target=agent.dynamic.target_exit,
                        llm_reasoning=agent.dynamic.reasoning_text or "",
                        scene_map=scene,
                        num_steps=31,
                    )
                    agent.dynamic.future_trajectory = traj
                    agent.dynamic.traj_step = 0
                else:
                    # 还没有LLM决策 — 原地不动
                    agent.dynamic.future_trajectory = None
                    continue

            # 播放当前帧
            traj = agent.dynamic.future_trajectory
            if traj is not None and agent.dynamic.traj_step < len(traj):
                idx = agent.dynamic.traj_step
                new_pos = traj[idx]
                agent.dynamic.velocity = new_pos - agent.dynamic.position
                agent.dynamic.position = new_pos
                agent.dynamic.traj_step += 1

                # 出口检查
                if agent.dynamic.target_exit is not None:
                    dist = np.linalg.norm(new_pos - agent.dynamic.target_exit)
                    if dist < 1.5:
                        agent.dynamic.evacuated = True

    def _render_cctv_frame(self) -> np.ndarray:
        """渲染当前场景为RGB图像 (模拟CCTV画面). 用于VLM输入."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_facecolor('#1a1a2e')

        # 烟雾
        snap = self.disaster.snapshot(
            self.tick, self.sim_time, self.exits, self.obstacles
        )
        for r in range(0, snap.grid.shape[0], 4):
            for c in range(0, snap.grid.shape[1], 4):
                s = snap.grid[r, c, 0]
                if s > 0.05:
                    rect = plt.Rectangle(
                        (c * snap.grid_resolution, r * snap.grid_resolution),
                        snap.grid_resolution * 4, snap.grid_resolution * 4,
                        facecolor='gray', alpha=min(0.6, s * 0.7), edgecolor='none'
                    )
                    ax.add_patch(rect)

        # Agent位置
        active = [a for a in self.agents
                  if a.dynamic.alive and not a.dynamic.evacuated]
        if active:
            pos = np.array([a.position for a in active])
            ax.scatter(pos[:, 0], pos[:, 1], c='cyan', s=3, alpha=0.8)

        ax.set_xticks([])
        ax.set_yticks([])

        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        return frame

    # ================================================================
    # RL Training Mode (offline, no LLM)
    # ================================================================

    def train_rl_scheduler(self, episodes: int = 500,
                           irl_weights_path: str = "data/irl_weights.json",
                           output_path: str = "data/rl_policy.json"):
        """Run offline RL training using a fast rule-based simulator.

        This trains the RL zone scheduler without running the full LLM simulation.
        Uses IRL-learned weights as the reward function.

        Args:
            episodes: Number of training episodes.
            irl_weights_path: Path to IRL-learned reward weights.
            output_path: Where to save trained policy weights.
        """
        from execution.rl_scheduler import (
            RLZoneScheduler, FastTrainingSimulator, DEFAULT_ZONES,
        )
        from execution.irl_recovery import IRLRecovery

        print("\n" + "=" * 60)
        print("   RL Zone Scheduler — Offline Training (Zone-PPO)")
        print("=" * 60)

        zones = self._load_zones_from_config()

        training_seed = int(self.cfg.get("simulation", {}).get("seed", 42))
        scheduler = RLZoneScheduler(
            zones=zones,
            num_exits=len(self.exits),
            seed=training_seed,
            blocked_exit_penalty=float(
                self.cfg.get("rl_scheduling", {}).get(
                    "blocked_exit_penalty", 0.0)),
            smoke_block_threshold=float(
                self.cfg.get("rl_scheduling", {}).get(
                    "smoke_block_threshold", 0.6)),
            outcome_reward_weight=float(
                self.cfg.get("rl_scheduling", {}).get(
                    "outcome_reward_weight", 0.0)),
        )
        scheduler.initialize()

        # Load IRL weights
        if os.path.exists(irl_weights_path):
            irl = IRLRecovery()
            irl.load(irl_weights_path)
            scheduler.load_irl_weights(irl.weights)
            print(f"[TrainRL] Loaded IRL weights for {len(irl.weights)} personas")
        else:
            print(f"[TrainRL] IRL weights not found at {irl_weights_path}, "
                  f"using default balanced weights")

        # Create fast training simulator with real mall floor plan exits.
        # In extreme mode it uses the deployment fire sources / spread rate
        # and jitters the origins every episode so the policy generalizes.
        train_env = self.cfg.get("environment", {})
        train_rl_cfg = self.cfg.get("rl_scheduling", {})
        sim = FastTrainingSimulator(
            width=self.width,
            height=self.height,
            num_agents=self.num_agents,
            num_exits=len(self.exits),
            zone_defs=zones,
            exit_positions=[(float(e[0]), float(e[1])) for e in self.exits],
            seed=training_seed,
            fire_sources=train_env.get("fire_sources"),
            spread_rate=train_env.get("disaster_spread_rate"),
            origin_jitter=float(train_rl_cfg.get("origin_jitter", 15.0)),
            smoke_block_threshold=float(
                train_rl_cfg.get("smoke_block_threshold", 0.6)),
            advice_accept_rate=float(
                train_rl_cfg.get("advice_accept_rate", 0.6)),
        )

        print(f"[TrainRL] Simulator: {self.width}×{self.height}m, "
              f"{self.num_agents} agents, {len(self.exits)} exits, "
              f"{len(zones)} zones")
        print(f"[TrainRL] Training: {episodes} episodes")

        history = scheduler.train_offline(
            env_simulator=sim,
            episodes=episodes,
            steps_per_episode=int(self.duration / sim.dt),
            save_path=output_path,
        )

        print(f"\n[TrainRL] Training complete. "
              f"Final evacuation rate: {history['evacuation_rate'][-1]:.1%}")
        print(f"[TrainRL] Policy weights saved to {output_path}")
        print("=" * 60)

    # ================================================================
    # Summary
    # ================================================================

    def _compute_role_stats(self):
        """Break down evacuation/casualty counts by civilian vs command agents.

        Command agents (commander / area commander / firefighter / guide) are
        spawned on top of the CLI ``--agents`` count, so including them in the
        denominator dilutes the civilian evacuation rate. This split is stored
        on ``self.role_stats`` for the summary and for external comparison
        scripts.
        """
        civilians = [a for a in self.agents
                     if a.profile.role == AgentRole.CIVILIAN.value]
        commands = [a for a in self.agents
                    if a.profile.role != AgentRole.CIVILIAN.value]

        def split_stats(group):
            n = len(group)
            evac = sum(1 for a in group if a.dynamic.evacuated)
            dead = sum(1 for a in group if not a.dynamic.alive)
            return {
                "total": n,
                "evacuated": evac,
                "casualties": dead,
                "remaining": n - evac - dead,
                "evac_rate": evac / n if n else 0.0,
                "casualty_rate": dead / n if n else 0.0,
            }

        self.role_stats = {
            "civilian": split_stats(civilians),
            "command": split_stats(commands),
            "all": split_stats(self.agents),
        }

    def _print_summary(self, tick_times: List[float]):
        print("\n" + "=" * 60)
        print("   SIMULATION COMPLETE")
        print("=" * 60)
        print(f"  Duration:         {self.sim_time:.1f}s")
        print(f"  Ticks:            {self.tick}")
        print(f"  Avg tick time:    {np.mean(tick_times):.1f}ms")
        print(f"  Max tick time:    {np.max(tick_times):.1f}ms")
        total_n = max(1, self.evacuated_count + self.casualty_count
                      + sum(1 for a in self.agents
                            if a.dynamic.alive and not a.dynamic.evacuated))
        print(f"  Total agents:     {len(self.agents)} (active: {total_n})")
        print(f"  Evacuated:        {self.evacuated_count} "
              f"({self.evacuated_count/total_n*100:.1f}%)")
        print(f"  Casualties:       {self.casualty_count} "
              f"({self.casualty_count/total_n*100:.1f}%)")
        if getattr(self, "role_stats", None):
            for key, label in (("civilian", "Civilians"),
                               ("command", "Command")):
                s = self.role_stats[key]
                if not s["total"]:
                    continue
                print(f"  [{label:9s}] {s['total']:4d} agents | "
                      f"evac {s['evacuated']:4d} ({s['evac_rate']*100:5.1f}%) | "
                      f"casualty {s['casualties']:3d} "
                      f"({s['casualty_rate']*100:5.1f}%) | "
                      f"remaining {s['remaining']:4d}")
        if getattr(self, "enable_rl_scheduling", False):
            loaded = self.rl_policy_loaded
            print(f"  RL policy:        "
                  f"{'LOADED (trained weights)' if loaded else 'NOT LOADED (heuristic advice fallback)'}")
        print(f"  LLM decisions:    {self.decision_count}")
        print(f"  Safety blocked:   {self.safety_blocks}")
        print(f"  Safety modified:  {self.safety_modifications}")
        rule_counts = getattr(self, "safety_rule_counts", {})
        if rule_counts:
            detail = " ".join(f"{k}={v}" for k, v in rule_counts.items())
            print(f"  Safety rules:    {detail}")
        avg_llm = (self.total_llm_time / self.decision_count * 1000
                   if self.decision_count > 0 else 0)
        print(f"  Avg LLM latency:  {avg_llm:.0f}ms/decision")
        print(f"  Total LLM time:   {self.total_llm_time:.1f}s")
        if self.use_llm:
            fb_rate = self.llm_engine.fallback_rate
            fb_pct = fb_rate * 100
            print(f"  LLM parse fails:  {self.llm_engine.parse_failures}/{self.llm_engine.total_llm_decisions} "
                  f"({fb_pct:.1f}% fallback)")
            if fb_pct > 20:
                print(f"  !! WARNING: High LLM fallback rate — check max_tokens and prompt format")
        print("=" * 60)

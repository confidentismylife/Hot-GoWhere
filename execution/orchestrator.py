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
from decision.safety_guard import SafetyGuard
from decision.agent_roles import AgentRole, define_zones
from perception.environment import DisasterSimulator, EnvironmentSnapshot
from execution.batched_physics import BatchedPhysics
from execution.diffusion_policy import build_scene_map
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

        # Safety guard (hard constraints on LLM output)
        self.safety_guard = SafetyGuard()

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
        self.safety_blocks = 0       # Count of LLM decisions blocked by safety guard
        self.safety_modifications = 0  # Count of decisions modified by safety guard
        self._command_broadcasts = []  # Commander-generated broadcasts (injected into civilian prompts)

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
总人数: {self.num_agents}
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
        self._spawn_command_agents()
        self.llm_engine.initialize()

        # v2: VLM 感知器 + YOLO 检测器 (双通道)
        self.vlm = None
        self.yolo = None
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

            # YOLO 检测通道 (与VLM互补)
            from perception.yolo_detector import YOLODetector
            self.yolo = YOLODetector(
                model_name=self.cfg.get("yolo", {}).get("model", "yolov8n.pt"),
                confidence_threshold=self.cfg.get("yolo", {}).get("conf_threshold", 0.35),
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

                # Apply civilian decisions normally
                if civilian_decisions:
                    self._apply_decisions(civilian_decisions, env_snapshot)

                # Process command decisions → generate broadcasts
                if command_decisions:
                    self._apply_command_decisions(command_decisions, env_snapshot)

            # ---- 3.5 VLM + YOLO 双通道感知 (v2.0) ----
            if self.vlm is not None and self.tick % self.vlm.call_interval == 0:
                frame = self._render_cctv_frame()
                # 通道A: VLM 语义理解
                vlm_desc = self.vlm.perceive(frame, self.tick, env_snapshot)
                # 通道B: YOLO 人员检测
                yolo_res = self.yolo.detect(frame) if self.yolo is not None else None
                # 只在有新数据时更新 LLM 引擎的感知上下文 (持久化缓存)
                self.llm_engine.set_perception_context(vlm_desc, yolo_res)

            # ---- 4. Group Intelligence ----
            self.group_intel.propagate(
                self.agents,
                env_snapshot.official_broadcast, self.dt)
            self.group_intel.update_fear_levels(self.agents, env_snapshot, self.dt)
            self.group_intel.update_stamina(self.agents, self.dt)

            # ---- 5. Physics ----
            if self.use_diffusion and self.diffusion_policy is not None:
                self._step_diffusion(env_snapshot)
            else:
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
                      f"LLM: {self.decision_count:5d} | "
                      f"Blocked: {self.safety_blocks:3d}/{self.safety_modifications:3d}")

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

        # v2.0 模块清理
        if self.vlm is not None:
            self.vlm.shutdown()
        if self.yolo is not None:
            self.yolo.shutdown()
        if self.diffusion_policy is not None:
            self.diffusion_policy.shutdown()

        self.llm_engine.shutdown()
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

    def _find_agent(self, agent_id: str):
        """Find agent by ID. Returns None if not found."""
        for a in self.agents:
            if a.id == agent_id:
                return a
        return None

    def _apply_command_decisions(self, decisions: Dict[str, DecisionResult],
                                 env_snapshot: EnvironmentSnapshot):
        """Parse LLM commander outputs into broadcasts and guidance.

        Commander decisions are parsed from their role-specific JSON output
        and converted into messages that influence civilian behavior.
        """
        for agent_id, d in decisions.items():
            agent = self._find_agent(agent_id)
            if agent is None:
                continue

            role = agent.profile.role

            # Parse commander JSON from reasoning text (which contains the raw LLM output)
            try:
                import json
                data = json.loads(d.reasoning) if d.reasoning else {}
            except (json.JSONDecodeError, TypeError):
                data = {}

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
                for i in range(1, len(env_snapshot.exits) + 1):
                    if f"出口{i}" in exit_str:
                        agent.dynamic.target_exit = np.array(
                            env_snapshot.exits[i - 1], dtype=np.float64
                        )
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
    # Summary
    # ================================================================

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
        print(f"  Safety blocked:   {self.safety_blocks}")
        print(f"  Safety modified:  {self.safety_modifications}")
        avg_llm = (self.total_llm_time / self.decision_count * 1000
                   if self.decision_count > 0 else 0)
        print(f"  Avg LLM latency:  {avg_llm:.0f}ms/decision")
        print(f"  Total LLM time:   {self.total_llm_time:.1f}s")
        print("=" * 60)

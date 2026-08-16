"""End-to-end v2 pipeline test — VLM mock + multi-role + safety guard.

Validates the complete v2.0 data flow without requiring a real GPU:
  1. Mock VLM generates NL descriptions from env state
  2. Descriptions are injected into LLM prompt context
  3. Multi-role agents (commander, firefighter, guide) spawn correctly
  4. Commander broadcasts are generated and flow to civilian prompts
  5. Safety guard intercepts dangerous decisions
  6. Simulation runs to completion without crashes

Usage:
    python tests/test_v2_pipeline.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.orchestrator import SimulationOrchestrator


def test_vlm_mock_synthesis():
    """Test 1: MockVLM generates realistic NL from env snapshot."""
    from perception.vlm_perceiver import MockVLMPerceiver
    from perception.environment import DisasterSimulator

    disaster = DisasterSimulator(
        width=100, height=60, disaster_type="fire",
        origin=(15, 30), spread_rate=0.15, resolution=0.5,
    )

    # Let fire spread a bit
    for _ in range(100):
        disaster.step(0.1)

    snapshot = disaster.snapshot(
        tick=100, timestamp=10.0,
        exits=[(5, 5), (95, 30), (5, 55), (50, 60)],
        obstacles=[{"center": (25, 25), "radius": 1.5}],
    )

    vlm = MockVLMPerceiver(call_interval=10)
    vlm.initialize()

    desc = vlm.perceive(None, tick=0, env_snapshot=snapshot)
    assert desc, "MockVLM should produce non-empty description"
    assert "烟雾" in desc, f"Description should mention smoke: {desc}"
    assert "出口" in desc or "exit" in desc.lower(), \
        f"Description should mention exits: {desc}"
    print(f"  [PASS] MockVLM synthesis: {desc[:100]}...")

    # Cache test: same tick should return cached result
    desc2 = vlm.perceive(None, tick=0, env_snapshot=snapshot)
    assert desc2 == desc, "Same tick should return cached description"

    # New tick beyond interval should generate new description
    desc3 = vlm.perceive(None, tick=20, env_snapshot=snapshot)
    assert desc3, "New tick should generate new description"
    print(f"  [PASS] MockVLM caching: interval={vlm.call_interval}")

    vlm.shutdown()


def test_orchestrator_multi_role_spawn():
    """Test 2: Orchestrator spawns command agents with correct roles."""
    orch = SimulationOrchestrator(config_path="config/default.yaml")
    orch.num_agents = 50
    orch.generate_agents()
    orch._spawn_command_agents()

    roles = {}
    for a in orch.agents:
        r = a.profile.role
        roles[r] = roles.get(r, 0) + 1

    assert roles.get("civilian", 0) >= 50, \
        f"Expected at least 50 civilians (family dependents may be added), " \
        f"got {roles.get('civilian', 0)}"
    assert roles.get("global_commander", 0) >= 1, \
        f"Expected at least 1 global commander"
    assert roles.get("area_commander", 0) >= 1, \
        f"Expected at least 1 area commander"
    assert roles.get("firefighter", 0) >= 1, \
        f"Expected at least 1 firefighter"
    assert roles.get("guide", 0) >= 1, \
        f"Expected at least 1 guide"

    total = sum(roles.values())
    print(f"  [PASS] Multi-role spawn: {total} total agents, roles={roles}")


def test_commander_broadcast_flow():
    """Test 3: Commander broadcasts flow into civilian prompt context."""
    orch = SimulationOrchestrator(config_path="config/default.yaml")
    orch.num_agents = 20
    orch.generate_agents()
    orch._spawn_command_agents()

    # Simulate a broadcast being stored
    orch._command_broadcasts.append({
        "tick": 10,
        "message": "请所有人员立即从东出口撤离！",
        "source": "global_commander",
    })
    orch.sim_time = 1.0

    broadcast = orch._get_broadcast()
    assert "东出口" in broadcast, \
        f"Broadcast should contain the message: {broadcast}"
    print(f"  [PASS] Commander broadcast: {broadcast}")

    # Test expiry: broadcasts older than 30s should be removed
    orch.sim_time = 35.0
    expired = orch._get_broadcast()
    assert expired == "", \
        f"Expired broadcast should return empty, got: {expired}"
    print(f"  [PASS] Broadcast expiry at t=35s: cleared")


def test_safety_guard_blocks_dangerous_exit():
    """Test 4: Safety guard blocks LLM decision choosing smoke-blocked exit."""
    from decision.cognitive_engine import DecisionResult
    from decision.agent_state import Agent, AgentProfile, AgentDynamic, Speed, Cooperation
    from perception.environment import DisasterSimulator

    # Setup disaster with heavy smoke at exit 0
    disaster = DisasterSimulator(
        width=100, height=60, disaster_type="fire",
        origin=(5, 5), spread_rate=0.5, resolution=0.5,
    )
    for _ in range(200):
        disaster.step(0.1)

    exits = [(5, 5), (95, 30), (5, 55), (50, 60)]
    snapshot = disaster.snapshot(
        tick=200, timestamp=20.0,
        exits=exits,
        obstacles=[],
    )

    # Verify exit 0 has heavy smoke
    exit0_smoke = snapshot.smoke_at(np.array(exits[0], dtype=np.float64))
    assert exit0_smoke > 0.5, \
        f"Exit 0 should have heavy smoke, got {exit0_smoke:.2f}"

    # Create agent and bogus decision pointing to smoked exit
    agent = Agent(
        profile=AgentProfile(
            max_speed=1.5,
            risk_aversion=0.5,
        ),
        dynamic=AgentDynamic(
            position=np.array([50.0, 30.0], dtype=np.float64),
            stamina=80.0,
        ),
    )

    decision = DecisionResult(
        agent_id=agent.id,
        target_exit_idx=0,
        target_exit_pos=exits[0],
        speed=Speed.RUN,
        cooperation=Cooperation.NONE,
        reasoning="选最近出口",
        risk_assessment="安全",
        compute_time=0.0,
    )

    orch = SimulationOrchestrator(config_path="config/default.yaml")
    orch.tick = 100
    orch.sim_time = 10.0
    orch.agents = [agent]  # Register agent so _apply_decisions finds it

    # Apply via orchestrator's safety-checked path
    orch._apply_decisions({agent.id: decision}, snapshot)

    # Agent should have been redirected away from smoke-blocked exit 0
    assert agent.dynamic.target_exit is not None
    redirected = not np.allclose(agent.dynamic.target_exit, exits[0], atol=1.0)
    assert redirected, \
        f"Agent should be redirected from exit 0 (smoke={exit0_smoke:.2f})"
    assert orch.safety_blocks + orch.safety_modifications >= 1, \
        f"Safety guard should intercept dangerous exit: b={orch.safety_blocks} m={orch.safety_modifications}"
    print(f"  [PASS] Safety guard redirected from exit 0 (smoke={exit0_smoke:.2f}), "
          f"blocks={orch.safety_blocks} mods={orch.safety_modifications}")


def test_full_simulation_mini():
    """Test 5: Run a mini simulation end-to-end (no LLM, headless).

    Validates the full orchestrator loop without crashing:
    - Agent spawning
    - Multi-role agents
    - Physics stepping
    - Safety guard
    - Cleanup
    """
    orch = SimulationOrchestrator(config_path="config/default.yaml")
    orch.num_agents = 30
    orch.duration = 5.0  # Short run
    orch.cfg["visualization"]["enabled"] = False
    orch.cfg["visualization"]["mode"] = "none"

    # Skip LLM — test pure physics + safety fallback path
    orch.llm_engine._ready = False

    # Patch run() to skip LLM init and just test the loop
    orch.generate_agents()
    orch._spawn_command_agents()
    # Don't initialize LLM (would try to load model)

    total_ticks = int(orch.duration / orch.dt)
    tick_times = []

    # Manual mini-loop without LLM calls
    for tick in range(total_ticks):
        t0 = time.perf_counter()

        orch.disaster.step(orch.dt)
        env_snapshot = orch.disaster.snapshot(
            orch.tick, orch.sim_time, orch.exits, orch.obstacles,
            official_broadcast=orch._get_broadcast(),
        )

        # Apply fallback decisions to all agents needing one
        for agent in orch.agents:
            if (agent.dynamic.alive and not agent.dynamic.evacuated
                    and agent.dynamic.has_new_info):
                fb = orch.safety_guard.fallback_decision(agent, env_snapshot)
                agent.dynamic.target_exit = np.array(
                    fb["target_exit_pos"], dtype=np.float64)
                agent.dynamic.speed_choice = fb["speed"]
                agent.dynamic.cooperation_choice = fb["cooperation"]
                agent.dynamic.reasoning_text = fb["reasoning"]
                agent.dynamic.last_decision_tick = tick
                agent.dynamic.has_new_info = False

        orch.group_intel.propagate(
            orch.agents, env_snapshot.official_broadcast, orch.dt)
        orch.group_intel.update_fear_levels(orch.agents, env_snapshot, orch.dt)
        orch.group_intel.update_stamina(orch.agents, orch.dt)
        orch.physics.step_all(orch.agents, orch.dt)

        orch.evacuated_count = sum(
            1 for a in orch.agents if a.dynamic.evacuated)
        orch.casualty_count = sum(
            1 for a in orch.agents if not a.dynamic.alive)

        orch.tick += 1
        orch.sim_time += orch.dt
        tick_times.append((time.perf_counter() - t0) * 1000)

    remaining = len(orch.agents) - orch.evacuated_count - orch.casualty_count
    assert remaining >= 0, f"Agent count should be non-negative, got {remaining}"
    assert len(tick_times) == total_ticks

    avg_ms = np.mean(tick_times)
    print(f"  [PASS] Mini simulation: {total_ticks} ticks, "
          f"avg {avg_ms:.1f}ms/tick, "
          f"evac={orch.evacuated_count}, dead={orch.casualty_count}, "
          f"active={remaining}")


def test_vlm_dual_channel_context():
    """Test 6: NL converter correctly embeds VLM + YOLO context in prompt."""
    from perception.nl_converter import NLConverter
    from decision.agent_state import Agent, AgentProfile, AgentDynamic
    from perception.environment import DisasterSimulator

    disaster = DisasterSimulator(
        width=100, height=60, disaster_type="fire",
        origin=(15, 30), spread_rate=0.15, resolution=0.5,
    )
    for _ in range(50):
        disaster.step(0.1)

    exits = [(5, 5), (95, 30)]
    snapshot = disaster.snapshot(
        tick=50, timestamp=5.0, exits=exits, obstacles=[],
    )

    agent = Agent(
        profile=AgentProfile(familiarity=0.7, max_speed=1.5),
        dynamic=AgentDynamic(
            position=np.array([50.0, 30.0], dtype=np.float64),
            stamina=80.0, known_exit_positions=exits,
        ),
    )

    # Without VLM/YOLO
    ctx_no_vlm = NLConverter.full_context(agent, snapshot)
    assert "[监控画面分析]" not in ctx_no_vlm, \
        "Should not have VLM section when disabled"
    assert "[人群检测]" not in ctx_no_vlm, \
        "Should not have YOLO section when disabled"

    # With VLM description
    vlm_desc = "西南角有浓烟，东出口附近人群密集，约30人正在移动。"
    ctx_vlm = NLConverter.full_context(agent, snapshot, vlm_description=vlm_desc)
    assert "[监控画面分析]" in ctx_vlm, \
        f"Should have VLM section: {ctx_vlm}"
    assert "西南角有浓烟" in ctx_vlm, \
        f"VLM content should appear in context: {ctx_vlm}"

    # With mock YOLO result
    from perception.yolo_detector import YOLOResult, DetectionBox
    yolo = YOLOResult(
        boxes=[DetectionBox(x=50, y=30, w=1, h=2, confidence=0.9, cls=0)],
        person_count=35,
        density_hotspots=[{"center": (50, 30), "count": 25, "grid_size": 5}],
        abnormal_events=["疑似摔倒: 2人"],
    )
    ctx_full = NLConverter.full_context(agent, snapshot,
                                        vlm_description=vlm_desc,
                                        yolo_result=yolo)
    assert "[人群检测]" in ctx_full, f"Should have YOLO section: {ctx_full}"
    assert "35人" in ctx_full, f"Should mention person count: {ctx_full}"
    assert "疑似摔倒" in ctx_full, f"Should mention abnormal events: {ctx_full}"
    assert "[监控画面分析]" in ctx_full, f"Should also have VLM section: {ctx_full}"

    print(f"  [PASS] Dual-channel NL context: VLM + YOLO both embedded")


if __name__ == "__main__":
    print("=" * 60)
    print("  v2.0 Pipeline Integration Tests")
    print("=" * 60)

    tests = [
        ("MockVLM synthesis + cache", test_vlm_mock_synthesis),
        ("Multi-role agent spawning", test_orchestrator_multi_role_spawn),
        ("Commander broadcast flow + expiry", test_commander_broadcast_flow),
        ("Safety guard blocks dangerous exit", test_safety_guard_blocks_dangerous_exit),
        ("Full mini simulation (no LLM)", test_full_simulation_mini),
        ("Dual-channel NL context (VLM+YOLO)", test_vlm_dual_channel_context),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            print(f"\n--- {name} ---")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{passed+failed} passed")
    if failed > 0:
        print(f"  {failed} test(s) FAILED")
        sys.exit(1)
    else:
        print(f"  All tests passed!")
    print(f"{'=' * 60}")

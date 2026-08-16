"""Integration test — feeds dangerous LLM decisions directly to verify
SafetyGuard catches and corrects them. No GPU needed, no LLM loading.

Usage:
    python tests/test_safety_integration.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import numpy as np

from decision.agent_state import Agent, AgentProfile, AgentDynamic, Speed, Cooperation
from decision.cognitive_engine import DecisionResult
from decision.safety_guard import SafetyGuard
from perception.environment import DisasterSimulator


def load_config():
    with open("config/default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # Deterministic fire spread; otherwise the 60s scenario randomly
    # blocks all viable exits and makes this test flaky.
    np.random.seed(42)

    cfg = load_config()
    env_cfg = cfg["environment"]

    # ---- 1. Create disaster simulator ----
    disaster = DisasterSimulator(
        width=env_cfg["width"],
        height=env_cfg["height"],
        disaster_type=env_cfg["disaster"],
        origin=tuple(env_cfg["disaster_origin"]),
        spread_rate=env_cfg["disaster_spread_rate"],
    )
    exits = [tuple(e) for e in env_cfg["exit_positions"]]
    obstacles = env_cfg.get("obstacles", [])

    # Let fire spread for 60s so smoke reaches some exits
    print("Simulating 60s of fire spread to create realistic hazard...")
    for _ in range(600):
        disaster.step(0.1)

    env_snap = disaster.snapshot(600, 60.0, exits, obstacles,
                                 official_broadcast="请注意,西南方向发生火灾!")

    # Check exit smoke levels
    for i, e in enumerate(exits):
        s = env_snap.smoke_at(np.array(e, dtype=np.float64))
        print(f"  出口{i+1} ({e[0]:.0f},{e[1]:.0f}): 烟雾 {s:.0%}")

    # ---- 2. Create guard and test agents ----
    guard = SafetyGuard()

    # Helper
    def make_agent(name, x, y, stamina=80, injured=False, age=30):
        p = AgentProfile(id=name, age=age, max_speed=1.5)
        d = AgentDynamic(position=np.array([x, y], dtype=np.float64), stamina=stamina,
                        injured=injured, speed_choice=Speed.WALK)
        return Agent(profile=p, dynamic=d)

    def make_decision(exit_idx, speed, cooperation=Cooperation.NONE):
        return DecisionResult(
            agent_id="", target_exit_idx=exit_idx,
            target_exit_pos=exits[exit_idx],
            speed=speed, cooperation=cooperation,
            reasoning="LLM raw decision", risk_assessment="test",
            compute_time=0.0,
        )

    tests_passed = 0
    tests_total = 0

    def check(name, agent, decision, expect_blocked, expect_modified, expect_speed=None):
        nonlocal tests_passed, tests_total
        tests_total += 1
        result = guard.check(decision, agent, env_snap)

        ok = True
        issues = []

        if expect_blocked and result.passed:
            ok = False
            issues.append("应该被拦截但通过了")
        if not expect_blocked and not result.passed:
            ok = False
            issues.append("不应该被拦截但拦截了")
        if expect_modified and not result.modified:
            ok = False
            issues.append("应该被修正但未修正")
        if expect_speed and result.final_speed != expect_speed:
            ok = False
            issues.append(f"速度应为{expect_speed}实际为{result.final_speed}")

        status = "PASS" if ok else "FAIL"
        if ok:
            tests_passed += 1

        flags = []
        if not result.passed:
            flags.append(f"拦截: {result.block_reason}")
        if result.modified:
            flags.append(f"修正: {result.warnings}")

        print(f"  [{status}] {name}")
        if flags:
            print(f"         {' | '.join(flags)}")
        if issues:
            print(f"         ❌ {'; '.join(issues)}")

    print("\n--- 测试1: 烟雾封锁出口 ---")
    # Fire is at (15, 30). Exit 1 is at (5, 5) — close to fire
    # After 60s fire spread, some exits may be smokey
    # Agent near exit 1, LLM chooses smoke-blocked exit
    agent = make_agent("A", x=20, y=20)
    smoke_e0 = env_snap.smoke_at(np.array(exits[0], dtype=np.float64))
    smoke_e2 = env_snap.smoke_at(np.array(exits[2], dtype=np.float64))

    # Choose the smokiest exit
    bad_exit = 0 if smoke_e0 > smoke_e2 else 2
    decision = make_decision(bad_exit, Speed.WALK)
    # If the chosen exit has heavy smoke, it should be blocked or modified
    exit_smoke = env_snap.smoke_at(np.array(exits[bad_exit], dtype=np.float64))
    if exit_smoke > 0.6:
        check("选择浓烟封锁出口 → 应被拦截",
              agent, decision, expect_blocked=True, expect_modified=True)
    else:
        check("选择烟雾未超限出口 → 正常通过",
              agent, decision, expect_blocked=False, expect_modified=False)

    print("\n--- 测试2: 体力不足奔跑 ---")
    agent = make_agent("B", x=50, y=30, stamina=15)
    decision = make_decision(1, Speed.RUN)  # exit 2 is at (95, 30) — far from fire
    check("体力15选run → 应降级为walk",
          agent, decision, expect_blocked=False, expect_modified=True, expect_speed="walk")

    agent2 = make_agent("C", x=50, y=30, stamina=5)
    check("体力5选run → 应降级为crawl",
          agent2, decision, expect_blocked=False, expect_modified=True, expect_speed="crawl")

    print("\n--- 测试3: 受伤奔跑 ---")
    agent = make_agent("D", x=50, y=30, stamina=90, injured=True)
    decision = make_decision(1, Speed.RUN)
    check("受伤选run → 应降级为walk",
          agent, decision, expect_blocked=False, expect_modified=True, expect_speed="walk")

    print("\n--- 测试4: 正常决策通过 ---")
    agent = make_agent("E", x=50, y=30, stamina=80)
    decision = make_decision(1, Speed.WALK)  # exit 2 at (95, 30)
    check("正常体力+安全出口 → 无修改通过",
          agent, decision, expect_blocked=False, expect_modified=False)

    print("\n--- 测试5: 站在火中 ---")
    # Place agent near fire origin
    fire_origin = env_cfg["disaster_origin"]
    agent = make_agent("F", x=fire_origin[0] + 2, y=fire_origin[1] + 2, stamina=80)
    decision = make_decision(1, Speed.WAIT)
    on_fire = env_snap.is_on_fire(agent.position)
    check(f"火源附近选wait (位置是否着火: {on_fire})",
          agent, decision,
          expect_blocked=False,
          expect_modified=on_fire,
          expect_speed=("run" if on_fire else None))

    print(f"\n{'='*50}")
    print(f"  结果: {tests_passed}/{tests_total} 通过")
    print(f"{'='*50}")

    if tests_passed == tests_total:
        print("\n安全约束模块验证通过 — 可以部署到GPU服务器。")
    else:
        print(f"\n{tests_total - tests_passed} 个测试失败，请检查。")

    return tests_passed == tests_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

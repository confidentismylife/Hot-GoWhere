"""Unit tests for SafetyGuard per-rule counters.

Run directly:
    python tests/test_safety_counters.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from decision.agent_state import Agent, AgentProfile, AgentDynamic, Speed
from decision.cognitive_engine import DecisionResult
from decision.safety_guard import SafetyGuard
from perception.environment import EnvironmentSnapshot


def make_env(exits, smoke_at_exit=None, on_fire_pos=None):
    res = 1.0
    grid = np.zeros((10, 10, 5), dtype=np.float32)
    for idx, s in (smoke_at_exit or {}).items():
        x, y = exits[idx]
        grid[int(y), int(x), 0] = s
    if on_fire_pos is not None:
        x, y = on_fire_pos
        grid[int(y), int(x), 3] = 1.0
    return EnvironmentSnapshot(
        tick=0, timestamp=0.0, width=10.0, height=10.0,
        grid=grid, grid_resolution=res, exits=exits, obstacles=[],
        official_broadcast="", disaster_type="fire",
        fire_origin=(0.0, 0.0), spread_rate=0.05,
    )


def make_agent(stamina=100.0, injured=False, pos=(5.0, 5.0)):
    profile = AgentProfile(role="civilian", age=30, occupation="visitor",
                           familiarity=0.5, max_speed=1.5,
                           trust_authority=0.5, altruism=0.5)
    dynamic = AgentDynamic(position=np.array(pos, dtype=np.float64),
                           stamina=stamina, injured=injured)
    return Agent(profile=profile, dynamic=dynamic)


def make_decision(exit_idx, speed=Speed.WALK):
    return DecisionResult(
        agent_id="a1", target_exit_idx=exit_idx,
        target_exit_pos=(0.0, 0.0), speed=speed,
        cooperation=None, reasoning="", risk_assessment="low",
        compute_time=0.0, tick=0, source="llm",
    )


def test_exit_smoke_swap_counter():
    exits = [(2.0, 2.0), (4.0, 4.0), (6.0, 6.0)]
    env = make_env(exits, smoke_at_exit={1: 0.9})  # exit 2 blocked
    guard = SafetyGuard()
    result = guard.check(make_decision(1), make_agent(pos=(2.0, 6.0)), env)
    assert result.modified
    assert guard.counters["exit_smoke_swap"] == 1
    assert guard.counters["stamina_speed"] == 0


def test_stamina_speed_counter():
    exits = [(2.0, 2.0), (4.0, 4.0), (6.0, 6.0)]
    env = make_env(exits)
    guard = SafetyGuard()
    result = guard.check(
        make_decision(0, speed=Speed.RUN), make_agent(stamina=5.0), env)
    assert result.modified
    assert guard.counters["stamina_speed"] == 1
    assert guard.counters["exit_smoke_swap"] == 0


def test_agent_on_fire_counter():
    exits = [(2.0, 2.0), (4.0, 4.0), (6.0, 6.0)]
    env = make_env(exits, on_fire_pos=(5.0, 5.0))
    guard = SafetyGuard()
    result = guard.check(make_decision(0, speed=Speed.WALK),
                         make_agent(pos=(5.0, 5.0)), env)
    assert result.modified
    assert guard.counters["agent_on_fire"] == 1


def test_reset_counters():
    guard = SafetyGuard()
    guard.counters["stamina_speed"] = 3
    guard.reset_counters()
    assert all(v == 0 for v in guard.counters.values())
    assert set(guard.counters) == set(SafetyGuard.RULE_NAMES)


if __name__ == "__main__":
    test_exit_smoke_swap_counter()
    test_stamina_speed_counter()
    test_agent_on_fire_counter()
    test_reset_counters()
    print("OK: SafetyGuard per-rule counters work.")

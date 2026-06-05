"""Quick unit tests for SafetyGuard — validates hard constraints work correctly."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from decision.safety_guard import SafetyGuard
from decision.agent_state import Agent, AgentProfile, AgentDynamic, Speed, Cooperation
from perception.environment import EnvironmentSnapshot


def make_agent(x=30.0, y=30.0, stamina=80.0, injured=False, age=30):
    """Helper to create a test agent."""
    profile = AgentProfile(age=age, max_speed=1.5)
    dynamic = AgentDynamic(
        position=np.array([x, y], dtype=np.float64),
        stamina=stamina,
        injured=injured,
    )
    return Agent(profile=profile, dynamic=dynamic)


def make_env(smoke_levels=None, fire_cells=None):
    """Helper to create a test environment snapshot.

    smoke_levels: dict mapping exit_index -> smoke value (0-1)
    fire_cells: list of (grid_r, grid_c) tuples that are on fire
    """
    grid = np.zeros((12, 20, 4), dtype=np.float32)  # 60m / 0.5 = 120 rows → use 12 for test
    grid[:, :, 1] = 25.0  # ambient temp
    grid[:, :, 2] = 1.0   # structural

    exits = [(5.0, 5.0), (95.0, 30.0), (5.0, 55.0), (50.0, 60.0)]

    if smoke_levels:
        for idx, smoke_val in smoke_levels.items():
            ex, ey = exits[idx]
            gr = min(int(ey / 5.0), 11)
            gc = min(int(ex / 5.0), 19)
            grid[gr, gc, 0] = smoke_val

    if fire_cells:
        for gr, gc in fire_cells:
            if 0 <= gr < 12 and 0 <= gc < 20:
                grid[gr, gc, 3] = 1.0
                grid[gr, gc, 0] = 0.8
                grid[gr, gc, 1] = 400.0

    return EnvironmentSnapshot(
        tick=0, timestamp=0.0,
        width=100.0, height=60.0,
        grid=grid, grid_resolution=5.0,  # 5m cells for test
        exits=exits,
        obstacles=[],
        disaster_type="fire",
    )


# Mock DecisionResult for testing
class MockDecision:
    def __init__(self, target_exit_idx, speed, cooperation=Cooperation.NONE, reasoning=""):
        self.target_exit_idx = target_exit_idx
        self.speed = speed
        self.cooperation = cooperation
        self.reasoning = reasoning
        self.target_exit_pos = (0, 0)  # Not used in safety check
        self.risk_assessment = ""


def test_exit_smoke_block():
    """Blocked: exit has heavy smoke → switch to better exit."""
    guard = SafetyGuard()
    agent = make_agent(x=50, y=30)
    env = make_env(smoke_levels={0: 0.8, 1: 0.1, 2: 0.2, 3: 0.1})

    decision = MockDecision(target_exit_idx=0, speed=Speed.WALK)  # exit 1 = heavy smoke
    result = guard.check(decision, agent, env)

    assert result.passed, "Should pass (modified to different exit)"
    assert result.modified, "Should be modified"
    assert result.final_exit_idx != 0, f"Should not choose exit 0, got {result.final_exit_idx}"
    print(f"  PASS test_exit_smoke_block: switched to exit {result.final_exit_idx+1}, "
          f"warnings: {result.warnings}")


def test_stamina_run_block():
    """Blocked: low stamina agent tries to run → downgrade."""
    guard = SafetyGuard()
    agent = make_agent(stamina=15.0)  # Below RUN_MIN(20), above CRAWL_MAX(10)
    env = make_env()

    decision = MockDecision(target_exit_idx=1, speed=Speed.RUN)
    result = guard.check(decision, agent, env)

    assert result.passed, "Should pass (modified speed)"
    assert result.modified, "Should be modified"
    assert result.final_speed == Speed.WALK.value, f"Should walk, got {result.final_speed}"
    print(f"  PASS test_stamina_run_block: {result.warnings}")


def test_stamina_crawl_force():
    """Blocked: very low stamina → force crawl."""
    guard = SafetyGuard()
    agent = make_agent(stamina=8.0)  # Below CRAWL_MAX(10)
    env = make_env()

    decision = MockDecision(target_exit_idx=1, speed=Speed.RUN)
    result = guard.check(decision, agent, env)

    assert result.passed
    assert result.modified
    assert result.final_speed == Speed.CRAWL.value, f"Should crawl, got {result.final_speed}"
    print(f"  PASS test_stamina_crawl_force: {result.warnings}")


def test_clean_decision():
    """Clean: good decision passes through unchanged."""
    guard = SafetyGuard()
    agent = make_agent(stamina=80.0)
    env = make_env(smoke_levels={0: 0.1, 1: 0.1, 2: 0.2, 3: 0.1})

    decision = MockDecision(target_exit_idx=1, speed=Speed.WALK)
    result = guard.check(decision, agent, env)

    assert result.passed, "Should pass unchanged"
    assert not result.modified, "Should not be modified"
    print(f"  PASS test_clean_decision: no modifications")


def test_fire_path_block():
    """Blocked: path to exit crosses fire — no better exit available → blocked."""
    guard = SafetyGuard()
    # Agent at (50, 30), exit 4 at (50, 60) — fire directly on this path
    agent = make_agent(x=50, y=30)
    env = make_env(
        smoke_levels={0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1},
        fire_cells=[(9, 10)]  # grid (9,10) at 5m resolution = world (52.5, 47.5)
    )

    decision = MockDecision(target_exit_idx=3, speed=Speed.WALK)  # exit 4 at (50, 60)
    result = guard.check(decision, agent, env)

    # Fire on path → check catches it
    # Exit 4 is still the "best" exit (shortest distance, all smoke equal)
    # So it passes=False (no better alternative to switch to)
    assert not result.passed or result.modified, \
        f"Should either block or modify. passed={result.passed}, modified={result.modified}"
    print(f"  PASS test_fire_path_block: passed={result.passed}, "
          f"modified={result.modified}, reason={result.block_reason}")


def test_injured_run_block():
    """Blocked: injured agent tries to run."""
    guard = SafetyGuard()
    agent = make_agent(stamina=80.0, injured=True)
    env = make_env()

    decision = MockDecision(target_exit_idx=1, speed=Speed.RUN)
    result = guard.check(decision, agent, env)

    assert result.passed, "Should pass (modified to walk)"
    assert result.modified, "Should be modified"
    assert result.final_speed != Speed.RUN.value, f"Should not run, got {result.final_speed}"
    print(f"  PASS test_injured_run_block: {result.warnings}")


def test_wait_in_smoke():
    """Blocked: agent waiting in heavy smoke → forced to move."""
    guard = SafetyGuard()
    agent = make_agent(x=50, y=30)
    # Put heavy smoke at agent's position
    env = make_env(smoke_levels={})
    # Set agent's grid cell smoke to heavy
    gr = min(int(30 / 5.0), 11)
    gc = min(int(50 / 5.0), 19)
    env.grid[gr, gc, 0] = 0.7  # heavy smoke

    decision = MockDecision(target_exit_idx=1, speed=Speed.WAIT)
    result = guard.check(decision, agent, env)

    assert result.passed, "Should pass (modified from wait to walk)"
    assert result.modified, "Should be modified"
    assert result.final_speed != Speed.WAIT.value, f"Should not wait, got {result.final_speed}"
    print(f"  PASS test_wait_in_smoke: {result.warnings}")


if __name__ == "__main__":
    print("=== SafetyGuard Unit Tests ===\n")
    tests = [
        test_exit_smoke_block,
        test_stamina_run_block,
        test_stamina_crawl_force,
        test_clean_decision,
        test_fire_path_block,
        test_injured_run_block,
        test_wait_in_smoke,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL {test.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== {passed}/{len(tests)} tests passed ===")

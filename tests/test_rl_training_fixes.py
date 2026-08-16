"""Unit tests for the v5 training fixes.

Run directly:
    python tests/test_rl_training_fixes.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from execution.features import blocked_exit_weight
from execution.rl_scheduler import (
    FastTrainingSimulator,
    RLZoneScheduler,
    ZoneAction,
    ZoneDefinition,
    _vectorized_blocked_exits,
)
from perception.environment import EnvironmentSnapshot


def _env(exits, smoke_at=None, fire_at=None):
    res = 1.0
    grid = np.zeros((10, 10, 5), dtype=np.float32)
    for idx, s in (smoke_at or {}).items():
        x, y = exits[idx]
        grid[int(y), int(x), 0] = s
    for (x, y) in (fire_at or []):
        grid[int(y), int(x), 3] = 1.0
    return EnvironmentSnapshot(
        tick=0, timestamp=0.0, width=10.0, height=10.0,
        grid=grid, grid_resolution=res, exits=exits, obstacles=[],
        official_broadcast="", disaster_type="fire",
        fire_origin=(0.0, 0.0), spread_rate=0.05,
    )


def test_vectorized_blocked_exits():
    exits = [(2.0, 2.0), (4.0, 4.0), (6.0, 6.0), (8.0, 8.0)]
    env = _env(exits, smoke_at={0: 0.8}, fire_at=[(5.0, 5.0)])
    smoke_b, path_b = _vectorized_blocked_exits(
        np.array([[1.0, 1.0]]), env, smoke_block_threshold=0.6)
    assert smoke_b[0] and not smoke_b[1]
    # paths to exits 2 and 3 (indices 1,2,3?) cross (5,5)
    assert path_b[0, 2] and path_b[0, 3]      # (6,6),(8,8) behind fire
    assert not path_b[0, 0]                    # (2,2) in front of fire


def test_blocked_exit_weight_with_precomputed_set():
    exits = [(2.0, 2.0), (4.0, 4.0), (6.0, 6.0)]
    env = _env(exits)
    w = blocked_exit_weight(
        env, ZoneAction([1.0, 0.0, 0.0]), blocked_exits=[0])
    assert w > 0.75
    w2 = blocked_exit_weight(
        env, ZoneAction([0.0, 1.0, 0.0]), blocked_exits=[0])
    assert w2 < 0.25


def test_training_rollout_masks_fire_path_exit():
    exits = [(2.0, 2.0), (4.0, 4.0), (6.0, 6.0), (8.0, 8.0),
             (2.0, 8.0), (8.0, 2.0)]
    zone = ZoneDefinition(0, "test", 0, 10, 0, 10, list(range(6)))
    def make_sim(seed):
        return FastTrainingSimulator(
            width=10.0, height=10.0, num_agents=1, num_exits=6,
            zone_defs=[zone], exit_positions=exits, seed=seed,
            fire_sources=[(5.0, 5.0)], spread_rate=0.0, origin_jitter=0.0)

    # Pick a seed whose initial random target is the safe exit 0 (2,2),
    # so the agent stays alive until the first re-evaluation at tick 3.
    seed = next(s for s in range(1, 500)
                if int(make_sim(s).agent_target_exits[0]) == 0)
    sim = make_sim(seed)
    # Strongly recommend exits 3/4 (paths cross fire at (5,5))
    actions = {0: ZoneAction([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])}
    for _ in range(6):
        env, agents = sim.step(1.0, zone_actions=actions)
    target = int(agents.target_exits[0])
    assert target in (0, 4, 5), \
        f"agent chose fire-path exit {target} despite masking"


def test_outcome_shaping_orders_close_agents_higher():
    class A:
        def __init__(self, pos, stamina=100.0):
            self.position = np.array(pos, dtype=np.float64)
            self.dynamic = type("D", (), {"stamina": stamina})()

    env = _env([(2.0, 2.0)], fire_at=[])
    sched = RLZoneScheduler(zones=[], num_exits=1)
    sched.outcome_reward_weight = 1.0
    close = sched._outcome_shaping([A((1.5, 1.5))], env)
    far = sched._outcome_shaping([A((9.0, 9.0))], env)
    assert close > far


if __name__ == "__main__":
    test_vectorized_blocked_exits()
    test_blocked_exit_weight_with_precomputed_set()
    test_training_rollout_masks_fire_path_exit()
    test_outcome_shaping_orders_close_agents_higher()
    print("OK: v5 training fixes work.")

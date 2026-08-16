"""Unit tests for the deployment-like FastTrainingSimulator (extreme mode).

Run directly:
    python tests/test_fast_sim_extreme.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from execution.features import blocked_exit_weight
from execution.rl_scheduler import (
    FastTrainingSimulator,
    ZoneAction,
)
from perception.environment import EnvironmentSnapshot
from perception.floorplan import get_floorplan


def _extreme_sim(jitter=0.0, num_agents=20, steps=0):
    fp = get_floorplan("wuhan_baoli_1f")
    sim = FastTrainingSimulator(
        width=fp.width,
        height=fp.height,
        num_agents=num_agents,
        num_exits=len(fp.exits),
        exit_positions=fp.exits,
        seed=42,
        fire_sources=[(165.0, 55.0), (85.0, 105.0)],
        spread_rate=0.22,
        origin_jitter=jitter,
        smoke_block_threshold=0.6,
    )
    for _ in range(steps):
        sim.step(1.0)
    return sim, fp


def test_dual_fire_ignition_and_exit_blocking():
    sim, fp = _extreme_sim(jitter=0.0, steps=200)
    fire = sim.grid[:, :, 3] > 0.5
    assert fire.sum() > 0
    # Both base origins should have produced fire cells nearby.
    for ox, oy in [(165.0, 55.0), (85.0, 105.0)]:
        fx, fy = int(ox / sim.grid_res), int(oy / sim.grid_res)
        assert fire[max(0, fy - 8):fy + 9, max(0, fx - 8):fx + 9].any(), \
            f"no fire near origin ({ox},{oy})"
    # In the deployment-like scenario exits must become smoke-blocked.
    exit_smoke = [
        float(sim.grid[int(y / sim.grid_res), int(x / sim.grid_res), 0])
        for x, y in fp.exits
    ]
    assert max(exit_smoke) > 0.6, f"no exit blocked after 200s: {exit_smoke}"


def test_origin_jitter_changes_episodes():
    sim, _ = _extreme_sim(jitter=15.0)
    mask_a = sim.grid[:, :, 3] > 0.5
    sim.reset()
    mask_b = sim.grid[:, :, 3] > 0.5
    assert not np.array_equal(mask_a, mask_b), \
        "fire origins did not change between episodes with jitter"


def test_blocked_exit_penalty_weight():
    res = 1.0
    grid = np.zeros((10, 10, 5), dtype=np.float32)
    exits = [(2.0, 2.0), (4.0, 4.0), (6.0, 6.0)]
    grid[2, 2, 0] = 0.8  # exit 1 smoke-blocked
    env = EnvironmentSnapshot(
        tick=0, timestamp=0.0, width=10.0, height=10.0,
        grid=grid, grid_resolution=res, exits=exits, obstacles=[],
        official_broadcast="", disaster_type="fire",
        fire_origin=(0.0, 0.0), spread_rate=0.05,
    )
    # Softmax puts ~0.79 weight on the top exit -> penalty weight > 0.75
    w1 = blocked_exit_weight(env, ZoneAction([1.0, 0.0, 0.0]))
    assert w1 > 0.75, w1
    # Top exit safe -> penalty weight < 0.25
    w2 = blocked_exit_weight(env, ZoneAction([0.0, 1.0, 0.0]))
    assert w2 < 0.25, w2


if __name__ == "__main__":
    test_dual_fire_ignition_and_exit_blocking()
    test_origin_jitter_changes_episodes()
    test_blocked_exit_penalty_weight()
    print("OK: extreme fast-simulator works.")

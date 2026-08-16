"""Unit test for the RL advice smoke filter (hypothesis A).

Verifies that exits whose smoke exceeds the safety-guard threshold are
hard-masked out of the RL recommendation text, so the LLM is never pushed
toward a smoke-blocked exit.

Run directly (pytest optional):
    python tests/test_rl_smoke_filter.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from execution.rl_scheduler import (
    ZoneDefinition,
    ZoneAction,
    inject_rl_preferences,
)
from perception.environment import EnvironmentSnapshot


def make_env():
    """10x10 m environment; exits 3 and 4 (1-based) are smoke-blocked."""
    res = 1.0
    grid = np.zeros((10, 10, 5), dtype=np.float32)
    exits = [
        (2.0, 2.0),   # 出口1
        (4.0, 4.0),   # 出口2
        (6.0, 6.0),   # 出口3 — smoke 0.85 (blocked)
        (8.0, 8.0),   # 出口4 — smoke 0.75 (blocked)
        (2.0, 8.0),   # 出口5
        (8.0, 2.0),   # 出口6
    ]
    grid[6, 6, 0] = 0.85
    grid[8, 8, 0] = 0.75
    return EnvironmentSnapshot(
        tick=0, timestamp=0.0, width=10.0, height=10.0,
        grid=grid, grid_resolution=res, exits=exits, obstacles=[],
        official_broadcast="", disaster_type="fire",
        fire_origin=(5.0, 5.0), spread_rate=0.05,
    )


class _Agent:
    def __init__(self):
        self.position = np.array([3.0, 3.0])


def test_smoke_filter():
    env = make_env()
    zone = ZoneDefinition(0, "测试区", 0, 10, 0, 10, list(range(6)))
    action = ZoneAction(exit_preferences=[0.9, 0.8, 0.7, -0.9, 0.4, 0.2])

    text = inject_rl_preferences({0: action}, [zone], _Agent(), env)
    print(text)

    # Blocked exits are announced once under 封锁 and never recommended.
    assert "出口3（浓烟封锁" in text
    assert "出口4（浓烟封锁" in text
    assert "推荐 出口3" not in text
    assert "推荐 出口4" not in text
    assert "避免前往 出口3" not in text
    assert "避免前往 出口4" not in text

    # Safe exits still recommended normally.
    assert "强烈推荐 出口1" in text
    assert "强烈推荐 出口2" in text
    assert "建议考虑 出口5" in text
    assert "建议考虑 出口6" in text

    # The original preference vector must not be mutated.
    assert action.exit_preferences == [0.9, 0.8, 0.7, -0.9, 0.4, 0.2]

    # Below-threshold smoke (0.55) must NOT be blocked.
    env.grid[6, 6, 0] = 0.55
    text2 = inject_rl_preferences({0: action}, [zone], _Agent(), env)
    assert "出口3（浓烟封锁" not in text2

    print("OK: RL advice smoke filter works.")


def test_fire_path_filter():
    """Exits whose straight-line path crosses fire must also be blocked."""
    res = 1.0
    grid = np.zeros((10, 10, 5), dtype=np.float32)
    exits = [
        (2.0, 2.0),   # exit 1
        (4.0, 4.0),   # exit 2
        (6.0, 6.0),   # exit 3 - path crosses fire at (5,5)
        (8.0, 8.0),   # exit 4 - path crosses fire at (5,5)
        (2.0, 8.0),   # exit 5
        (8.0, 2.0),   # exit 6
    ]
    grid[5, 5, 3] = 1.0  # fire on the path
    env = EnvironmentSnapshot(
        tick=0, timestamp=0.0, width=10.0, height=10.0,
        grid=grid, grid_resolution=res, exits=exits, obstacles=[],
        official_broadcast="", disaster_type="fire",
        fire_origin=(5.0, 5.0), spread_rate=0.05,
    )
    zone = ZoneDefinition(0, "测试区", 0, 10, 0, 10, list(range(6)))
    action = ZoneAction(exit_preferences=[0.0, 0.0, 0.9, 0.9, 0.0, 0.0])

    class Agent:
        def __init__(self):
            self.position = np.array([1.0, 1.0])

    text = inject_rl_preferences({0: action}, [zone], Agent(), env)
    print(text)
    assert "\u5c01\u9501" in text  # 封锁
    assert "\u706b\u8def\u963b\u65ad" in text  # 火路阻断
    assert "\u63a8\u8350 \u51fa\u53e33" not in text  # 推荐 出口3
    assert "\u63a8\u8350 \u51fa\u53e34" not in text  # 推荐 出口4

    # With fire-path filtering disabled, the same env must NOT block.
    text2 = inject_rl_preferences(
        {0: action}, [zone], Agent(), env, fire_path_block=False)
    assert "\u5c01\u9501" not in text2


if __name__ == "__main__":
    test_smoke_filter()
    test_fire_path_filter()

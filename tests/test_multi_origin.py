"""Unit tests for multi-fire-source support in DisasterSimulator.

Run directly:
    python tests/test_multi_origin.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from perception.environment import DisasterSimulator


def test_multi_origin():
    sim = DisasterSimulator(
        width=20.0, height=20.0, disaster_type="fire",
        origin=[(2.0, 2.0), (16.0, 16.0)],
        spread_rate=0.1, resolution=2.0,
    )
    # origins (x=2,y=2)->grid(1,1) and (x=16,y=16)->grid(8,8)
    assert sim.grid[1, 1, 3] == 1.0, "first origin not ignited"
    assert sim.grid[8, 8, 3] == 1.0, "second origin not ignited"
    assert len(sim.origins) == 2

    snap = sim.snapshot(tick=0, timestamp=0.0, exits=[], obstacles=[])
    assert snap.fire_origins == [(2.0, 2.0), (16.0, 16.0)]
    assert snap.fire_origin == (2.0, 2.0)  # backward-compatible primary


def test_single_origin_backward_compat():
    sim = DisasterSimulator(
        width=20.0, height=20.0, disaster_type="fire",
        origin=(2.0, 2.0), spread_rate=0.1, resolution=2.0,
    )
    assert len(sim.origins) == 1
    assert sim.grid[1, 1, 3] == 1.0
    snap = sim.snapshot(tick=0, timestamp=0.0, exits=[], obstacles=[])
    assert snap.fire_origins == [(2.0, 2.0)]


if __name__ == "__main__":
    test_multi_origin()
    test_single_origin_backward_compat()
    print("OK: multi-origin fire support works.")

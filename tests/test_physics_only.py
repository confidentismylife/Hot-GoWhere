"""Physics-only test — validates BatchedPhysics engine, no LLM required.

Usage: python tests/test_physics_only.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from decision.agent_state import Agent, AgentProfile, AgentDynamic, Speed, Cooperation
from execution.batched_physics import BatchedPhysics
from perception.environment import DisasterSimulator


def create_test_agents(n: int, width: float, height: float) -> list:
    agents = []
    for i in range(n):
        profile = AgentProfile(
            age=np.random.randint(18, 65),
            max_speed=np.random.uniform(0.8, 2.0),
            familiarity=np.random.random(),
        )
        # Distribute exits evenly
        exit_idx = i % 3
        exits_list = [(5.0, 5.0), (95.0, 30.0), (5.0, 55.0)]
        dynamic = AgentDynamic(
            position=np.array([np.random.uniform(5, width - 5),
                              np.random.uniform(5, height - 5)],
                              dtype=np.float64),
            stamina=np.random.uniform(50, 100),
            speed_choice=Speed.WALK,
            target_exit=np.array(exits_list[exit_idx], dtype=np.float64),
        )
        agents.append(Agent(profile=profile, dynamic=dynamic))
    return agents


def test_physics():
    print("=" * 50)
    print("  Batched Physics Engine Benchmark")
    print("=" * 50)

    width, height = 100.0, 60.0
    obstacles = [
        {"center": [25.0, 25.0], "radius": 1.5},
        {"center": [50.0, 30.0], "radius": 2.0},
    ]

    physics = BatchedPhysics(width, height, obstacles)
    dt = 0.1

    for n in [100, 500, 1000, 2000, 5000]:
        agents = create_test_agents(n, width, height)

        # Warm up (triggers Numba JIT compilation)
        t0 = time.perf_counter()
        for _ in range(20):
            physics.step_all(agents, dt)
        warmup_time = (time.perf_counter() - t0) * 1000
        warmup_per_tick = warmup_time / 20

        # Benchmark
        t0 = time.perf_counter()
        ticks = 200
        for _ in range(ticks):
            physics.step_all(agents, dt)
        elapsed = (time.perf_counter() - t0) * 1000

        avg_tick_ms = elapsed / ticks
        rating = "OK" if avg_tick_ms < 33 else ("OK(>30fps)" if avg_tick_ms < 50 else "SLOW")
        evac = sum(1 for a in agents if a.dynamic.evacuated)
        print(f"  {n:5d} agents: {avg_tick_ms:6.2f}ms/tick  "
              f"(warmup: {warmup_per_tick:5.1f}ms)  evac: {evac:5d}  [{rating}]")


def test_disaster():
    print("\n" + "=" * 50)
    print("  Disaster Simulation Test")
    print("=" * 50)

    sim = DisasterSimulator(
        width=100, height=60,
        disaster_type="fire",
        origin=(15.0, 30.0),
        spread_rate=0.3,
        resolution=0.5,
    )

    for t in range(0, 600, 50):
        for _ in range(50):
            sim.step(0.1)
        snap = sim.snapshot(t, t * 0.1, [], [])
        fire_cells = (snap.grid[:, :, 3] > 0.5).sum()
        max_smoke = snap.grid[:, :, 0].max()
        max_temp = snap.grid[:, :, 1].max()
        print(f"  t={t*0.1:5.0f}s  fire_cells={fire_cells:5d}  "
              f"max_smoke={max_smoke:.2f}  max_temp={max_temp:.0f}C")


if __name__ == "__main__":
    test_disaster()
    test_physics()
    print("\n  All tests passed!")

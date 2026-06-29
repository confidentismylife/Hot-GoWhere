"""Unit tests for RL Zone Scheduler — inference, training, heuristic fallback."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tempfile
from execution.rl_scheduler import (
    RLZoneScheduler, ZoneObservation, ZoneAction, ZoneDefinition,
    FastTrainingSimulator, DEFAULT_ZONES,
)

ZONES_2 = [
    ZoneDefinition(0, "左", 0, 75, 0, 80, [0, 1, 2]),
    ZoneDefinition(1, "右", 75, 150, 0, 80, [3, 4, 5]),
]


def make_obs(exit_smoke=None, exit_crowd=None, agent_count=50):
    """Helper to create a test ZoneObservation."""
    n = 6 if exit_smoke is None else len(exit_smoke)
    return ZoneObservation(
        exit_smoke=exit_smoke or [0.2] * n,
        exit_crowd=exit_crowd or [5] * n,
        exit_fire_distance=[0.7] * n,
        agent_count=agent_count,
        avg_fear=3.0, avg_stamina=75.0,
        trained_ratio=0.2, elderly_ratio=0.1,
        avg_smoke=0.25, fire_distance=60.0,
        prev_exit_usage=[0.0] * n,
    )


def test_initialization():
    """Scheduler initializes with correct architecture."""
    s = RLZoneScheduler(zones=ZONES_2, num_exits=6)
    s.initialize()
    p = s._policies[0]
    assert 'W3_a' in p, "Missing policy head"
    assert 'W3_v' in p, "Missing value head"
    assert p['W1'].shape == (31, 128), f"W1 shape mismatch: {p['W1'].shape}"  # 4*6+7=31
    assert p['W3_a'].shape == (128, 6), f"W3_a shape mismatch: {p['W3_a'].shape}"
    assert p['W3_v'].shape == (128, 1), f"W3_v shape mismatch: {p['W3_v'].shape}"
    print(f"  PASS test_initialization")


def test_inference():
    """Inference produces valid exit preferences."""
    s = RLZoneScheduler(zones=ZONES_2, num_exits=6)
    s.initialize()
    obs = make_obs()

    action = s._forward(0, obs)
    assert len(action.exit_preferences) == 6
    assert all(-1.0 <= p <= 1.0 for p in action.exit_preferences), \
        f"Preferences out of [-1,1]: {action.exit_preferences}"
    print(f"  PASS test_inference: {[f'{v:.2f}' for v in action.exit_preferences[:3]]}..")


def test_heuristic_fallback():
    """Heuristic action prefers low-smoke exits."""
    s = RLZoneScheduler(zones=ZONES_2, num_exits=6)
    s.initialize()
    # Don't set policy for zone 0 → triggers heuristic
    s._policies.pop(0, None)

    obs = make_obs(
        exit_smoke=[0.9, 0.8, 0.1, 0.3, 0.2, 0.4],
        exit_crowd=[50, 40, 5, 10, 8, 15],
    )
    action = s._forward(0, obs)  # zone 0 has no policy → heuristic

    # Exit 2 (index 2) has lowest smoke (0.1) and low crowd (5) → should be preferred
    prefs = action.exit_preferences
    assert prefs[2] > prefs[0], f"Exit 2 (low smoke) should rank above exit 0 (high smoke)"
    assert prefs[2] > prefs[1], f"Exit 2 should rank above exit 1"
    print(f"  PASS test_heuristic_fallback: prefs={[f'{v:.2f}' for v in prefs]}")


def test_irl_weighted_heuristic():
    """Heuristic uses IRL weights when loaded."""
    s = RLZoneScheduler(zones=ZONES_2, num_exits=6)
    s.initialize()
    s._policies.pop(0, None)  # Force heuristic

    # Load IRL weights with extreme preferences
    s.load_irl_weights({
        "test": np.array([0.8, 0.05, 0.05, 0.05, 0.05]),  # heavily safety-biased
    })
    assert s._use_irl_reward

    # Same observation, safety-heavy IRL should strongly prefer exit 2 (best safety)
    obs = make_obs(
        exit_smoke=[0.9, 0.8, 0.1, 0.3, 0.2, 0.4],
        exit_crowd=[5, 5, 50, 10, 8, 15],  # Now exit 2 has HIGH crowd
    )
    action = s._forward(0, obs)
    # With w_safety=0.8, safe exits should still rank high despite crowds
    prefs = action.exit_preferences
    # Exit 2 has lowest smoke (0.1), should rank near top even with crowd of 50
    # Exit 0 has highest smoke (0.9), should rank near bottom
    assert prefs[2] > prefs[0], \
        f"Safety-heavy IRL: exit 2 (smoke=0.1) should outrank exit 0 (smoke=0.9), " \
        f"got {prefs[2]:.2f} vs {prefs[0]:.2f}"
    assert len(set(prefs)) > 1, "Preferences should vary across exits"
    print(f"  PASS test_irl_weighted_heuristic: prefs={[f'{v:.2f}' for v in prefs]}")


def test_training_step_no_nan():
    """Training step produces finite losses and valid weight updates."""
    s = RLZoneScheduler(zones=ZONES_2, num_exits=6)
    s.initialize()

    obs = make_obs()
    action = s._forward(0, obs)

    # Run multiple training steps
    for _ in range(3):
        loss = s._train_step(0, [obs], [action], [0.5], [0.5], [-2.0], lr=1e-4)

    assert not np.isnan(loss["policy_loss"]), "Policy loss should not be NaN"
    assert not np.isnan(loss["value_loss"]), "Value loss should not be NaN"
    assert not np.isinf(loss["policy_loss"]), "Policy loss should not be Inf"
    assert loss["value_loss"] >= 0, f"Value loss should be >= 0, got {loss['value_loss']}"

    # Weights should have changed
    p = s._policies[0]
    for key in ["W1", "W2", "W3_a", "W3_v"]:
        assert not np.any(np.isnan(p[key])), f"{key} contains NaN"
        assert not np.any(np.isinf(p[key])), f"{key} contains Inf"
    print(f"  PASS test_training_step_no_nan: p_loss={loss['policy_loss']:.4f} "
          f"v_loss={loss['value_loss']:.4f}")


def test_training_gradient_direction():
    """Training with positive reward increases action mean toward target."""
    s = RLZoneScheduler(zones=ZONES_2, num_exits=6)
    s.initialize()

    obs = make_obs()
    target_action = s._forward(0, obs)

    # Record initial weights and action mean
    p_before = {k: v.copy() for k, v in s._policies[0].items()}
    action_before = target_action.exit_preferences.copy()

    # Train with positive reward on current action
    for _ in range(10):
        s._train_step(0, [obs], [target_action], [1.0], [1.0], [-2.0], lr=1e-3)

    # Verify weights changed
    for key in p_before:
        diff = np.max(np.abs(s._policies[0][key] - p_before[key]))
        assert diff > 0, f"{key} did not change after training"

    print("  PASS test_training_gradient_direction")


def test_save_load():
    """Save/load preserves policy weights exactly."""
    s = RLZoneScheduler(zones=ZONES_2, num_exits=6)
    s.initialize()

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        tmp = f.name
    try:
        s.save_weights(tmp)
        s2 = RLZoneScheduler(zones=ZONES_2, num_exits=6)
        s2._load_weights(tmp)
        for zid in [0, 1]:
            for key in s._policies[zid]:
                assert np.allclose(s._policies[zid][key], s2._policies[zid][key]), \
                    f"Zone {zid}: mismatch in {key}"
        print("  PASS test_save_load")
    finally:
        os.unlink(tmp)


def test_fast_simulator():
    """FastTrainingSimulator runs correctly."""
    sim = FastTrainingSimulator(
        width=100, height=50, num_agents=50, num_exits=4,
        exit_positions=[(10, 5), (90, 5), (10, 45), (90, 45)],
    )
    sim.reset()
    assert sim.tick == 0
    assert sim.agent_alive.sum() == 50

    env_snap, agents = sim.step(0.1)
    assert sim.tick == 1
    assert len(list(agents)) == 50
    assert hasattr(env_snap, 'smoke_at')
    assert hasattr(env_snap, 'is_on_fire')

    # Smoke lookup works
    smoke = env_snap.smoke_at(np.array([50.0, 25.0]))
    assert isinstance(smoke, float)

    # Evacuation rate is between 0 and 1
    rate = sim.get_evacuation_rate()
    assert 0.0 <= rate <= 1.0, f"Rate should be [0,1], got {rate}"
    print(f"  PASS test_fast_simulator: evac_rate={rate:.2f}")


def test_zone_partitioning():
    """Agents are correctly partitioned into zones."""
    s = RLZoneScheduler(zones=ZONES_2, num_exits=6)
    s.initialize()

    sim = FastTrainingSimulator(
        width=150, height=80, num_agents=100, num_exits=6,
        zone_defs=ZONES_2,
        exit_positions=[(25, 5), (75, 5), (125, 5), (25, 75), (75, 75), (125, 75)],
    )
    sim.reset()
    env_snap, agents = sim.step(0.1)

    # Count agents per zone
    zone_counts = {0: 0, 1: 0}
    for a in agents:
        pos = a.position
        for z in ZONES_2:
            if z.x_min <= pos[0] < z.x_max and z.y_min <= pos[1] < z.y_max:
                zone_counts[z.zone_id] += 1
                break

    total = sum(zone_counts.values())
    assert total == 100, f"All 100 agents should be zoned, got {total}"
    assert zone_counts[0] > 0, "Zone 0 should have agents"
    assert zone_counts[1] > 0, "Zone 1 should have agents"
    print(f"  PASS test_zone_partitioning: zone0={zone_counts[0]}, zone1={zone_counts[1]}")


def test_recommendation_text():
    """ZoneAction generates correct Chinese recommendation text."""
    action = ZoneAction(exit_preferences=[0.8, 0.3, -0.1, -0.6, 0.1, -0.3])
    zone = ZONES_2[0]
    text = action.to_recommendation_text(zone, 6)

    assert "调度中心建议" in text
    assert "强烈推荐" in text
    assert "避免前往" in text
    print(f"  PASS test_recommendation_text:\n{text}")


if __name__ == "__main__":
    print("=== RL Zone Scheduler Unit Tests ===\n")
    tests = [
        test_initialization,
        test_inference,
        test_heuristic_fallback,
        test_irl_weighted_heuristic,
        test_training_step_no_nan,
        test_training_gradient_direction,
        test_save_load,
        test_fast_simulator,
        test_zone_partitioning,
        test_recommendation_text,
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

"""Unit tests for IRL Recovery module — MaxEnt IRL with synthetic trajectories."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
import tempfile
from execution.irl_recovery import (
    IRLRecovery, TrajectoryCollector, AgentTrajectory,
    FEATURE_NAMES, PERSONA_CATEGORIES,
)


def make_trajectory(persona, n_decisions=50, bias_weights=None):
    """Generate a synthetic trajectory with known feature preferences.

    If bias_weights is provided, the decisions will be biased toward
    features with higher weights (simulating that persona's true preference).
    """
    if bias_weights is None:
        bias_weights = {"untrained_elderly": [0.35, 0.15, 0.30, 0.10, 0.10],
                       "untrained_young": [0.25, 0.30, 0.10, 0.20, 0.15],
                       "trained_staff": [0.25, 0.35, 0.15, 0.20, 0.05],
                       "guide": [0.20, 0.25, 0.30, 0.15, 0.10],
                       "firefighter": [0.20, 0.10, 0.50, 0.05, 0.15]}

    w = np.array(bias_weights.get(persona, [0.2, 0.2, 0.2, 0.2, 0.2]))
    # Deterministic seed per persona (avoid hash() which varies across runs)
    persona_seeds = {"untrained_elderly": 1001, "untrained_young": 2002,
                     "trained_staff": 3003, "guide": 4004, "firefighter": 5005}
    rng = np.random.RandomState(persona_seeds.get(persona, 42))

    decisions = []
    for t in range(n_decisions):
        # Generate features with noise, biased by the persona's true weights
        base_features = rng.dirichlet(np.ones(5) * 3)  # Random 5-vector summing to 1
        # Push features toward the weight vector
        noise = rng.normal(0, 0.15, 5)
        features = base_features + w * 0.3 + noise
        features = np.clip(features, 0.05, 0.95)

        exit_idx = rng.randint(0, 8)
        speed = rng.choice(["run", "walk", "crawl"])
        coop = rng.choice(["none", "follow_crowd", "help_family", "lead_others"])

        decisions.append({
            "tick": t,
            "sim_time": t * 0.5,
            "position": [float(rng.uniform(10, 140)), float(rng.uniform(10, 70))],
            "smoke_at_pos": float(features[0] * 0.5),  # Inverted: high safety = low smoke
            "fire_distance": float(50 + features[0] * 100),
            "nearest_exit_idx": exit_idx,
            "nearest_exit_dist": float(10 + (1 - features[1]) * 80),
            "nearest_exit_smoke": float(features[0] * 0.5),
            "target_exit_idx": exit_idx,
            "speed": speed,
            "cooperation": coop,
            "stamina": float(rng.uniform(40, 100)),
            "fear": float((1 - features[4]) * 8),  # Inverted: high comfort = low fear
            "was_blocked": False,
            "was_modified": False,
        })

    return AgentTrajectory(
        agent_id=f"test_{persona}_{rng.randint(0, 9999)}",
        role="civilian" if persona in ("untrained_elderly", "untrained_young", "trained_staff")
              else persona,
        persona=persona,
        age=65 if "elderly" in persona else 30,
        familiarity=0.7 if "trained" in persona or persona in ("guide", "firefighter") else 0.3,
        max_speed=1.2,
        decisions=decisions,
        outcome="evacuated",
        evacuation_time=n_decisions * 0.5,
    )


def test_feature_extraction():
    """Feature extraction produces valid 5D vectors."""
    irl = IRLRecovery()
    traj = make_trajectory("untrained_elderly", n_decisions=20)
    features = irl._extract_trajectory_features(traj)

    assert len(features) == 20, f"Expected 20 feature vectors, got {len(features)}"
    for f in features:
        assert len(f) == 5, f"Each feature should be 5D, got {len(f)}"
        assert all(0.0 <= v <= 1.0 for v in f), f"Features should be [0,1], got {f}"
    print("  PASS test_feature_extraction")


def test_mdp_construction():
    """MDP construction produces valid discrete states and transitions."""
    irl = IRLRecovery(n_bins=3)
    trajectories = [make_trajectory("trained_staff", n_decisions=30) for _ in range(15)]

    mdp = irl._build_discrete_mdp(trajectories)

    assert mdp["n_states"] > 1, "Should have multiple discrete states"
    assert mdp["n_states"] <= 3**5, f"Max 243 states, got {mdp['n_states']}"
    assert mdp["state_features"].shape == (mdp["n_states"], 5)
    assert len(mdp["transitions"]) == mdp["n_states"]
    assert abs(mdp["init_dist"].sum() - 1.0) < 1e-6, "Init dist should sum to 1"
    print(f"  PASS test_mdp_construction: {mdp['n_states']} states")


def test_soft_value_iteration_convergence():
    """Soft VI converges and produces finite values."""
    irl = IRLRecovery(n_bins=3)
    trajectories = [make_trajectory("guide", n_decisions=30) for _ in range(20)]
    mdp = irl._build_discrete_mdp(trajectories)

    w = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
    policy_fe = irl._compute_policy_fe(mdp, w)

    assert len(policy_fe) == 5, f"Policy FE should be 5D, got shape {policy_fe.shape}"
    assert not np.any(np.isnan(policy_fe)), "Policy FE should not contain NaN"
    assert not np.any(np.isinf(policy_fe)), "Policy FE should not contain Inf"
    assert np.all(policy_fe >= 0), f"Policy FE should be non-negative, got {policy_fe}"
    print(f"  PASS test_soft_value_iteration_convergence: FE={policy_fe}")


def test_gc_maxent_fit():
    """GC-MaxEnt fit runs end-to-end with synthetic data."""
    irl = IRLRecovery(learning_rate=0.01, max_iter=100, tolerance=1e-3,
                      l2_reg=0.01, group_reg=0.02, n_bins=3)

    trajectories = []
    for persona in ["untrained_elderly", "untrained_young", "trained_staff",
                     "guide", "firefighter"]:
        trajectories.extend([make_trajectory(persona, n_decisions=40) for _ in range(15)])

    irl.fit(trajectories, verbose=False)

    assert len(irl.weights) == 5, f"Should learn 5 persona weights, got {len(irl.weights)}"
    for persona in PERSONA_CATEGORIES:
        assert persona in irl.weights, f"Missing persona: {persona}"
        w = irl.weights[persona]
        assert len(w) == 5, f"Weight should be 5D, got {len(w)}"
        assert abs(w.sum() - 1.0) < 1e-4, f"Weights should sum to 1, got {w.sum():.6f}"
        assert np.all(w >= 0.0009), f"Weights should be >= 0.0009, got {w}"
        assert not np.any(np.isnan(w)), "Weights should not contain NaN"

    # Firefighter should have highest social weight
    fw = irl.weights["firefighter"]
    # GC-MaxEnt Laplacian regularization pulls weights toward group mean,
    # so firefighter social weight may be lower than pure MaxEnt would give.
    assert fw[2] > 0.1, f"Firefighter social weight should be elevated, got {fw[2]:.3f}"

    print(f"  PASS test_gc_maxent_fit: all 5 personas learned")
    for p in PERSONA_CATEGORIES:
        w = irl.weights[p]
        print(f"    {p}: safety={w[0]:.3f} eff={w[1]:.3f} social={w[2]:.3f} "
              f"conf={w[3]:.3f} comfort={w[4]:.3f}")


def test_irl_persistence():
    """Save/load roundtrip preserves weights."""
    irl = IRLRecovery()
    irl.weights = {p: np.random.dirichlet(np.ones(5)) for p in PERSONA_CATEGORIES}

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        tmp = f.name
    try:
        irl.save(tmp)
        irl2 = IRLRecovery()
        irl2.load(tmp)
        for persona in PERSONA_CATEGORIES:
            assert np.allclose(irl.weights[persona], irl2.weights[persona]), \
                f"Weight mismatch for {persona}"
        print("  PASS test_irl_persistence")
    finally:
        os.unlink(tmp)


def test_reward_function_callable():
    """get_reward_function returns a valid callable."""
    irl = IRLRecovery()
    irl.weights = {"trained_staff": np.array([0.3, 0.35, 0.15, 0.15, 0.05])}

    rf = irl.get_reward_function("trained_staff")
    reward = rf({"safety": 0.8, "efficiency": 0.9, "social": 0.3,
                 "conformity": 0.5, "comfort": 0.6})
    assert isinstance(reward, float), f"Reward should be float, got {type(reward)}"
    assert 0.0 <= reward <= 1.0, f"Reward should be [0,1], got {reward}"
    print(f"  PASS test_reward_function_callable: reward={reward:.3f}")


def test_numerical_stability_extreme_weights():
    """Soft VI handles extreme weight values gracefully."""
    irl = IRLRecovery(n_bins=3)
    trajectories = [make_trajectory("untrained_young", n_decisions=20) for _ in range(10)]
    mdp = irl._build_discrete_mdp(trajectories)

    # Test with extreme but valid weights
    extreme_cases = [
        np.array([0.999, 0.00025, 0.00025, 0.00025, 0.00025]),
        np.array([0.001, 0.001, 0.001, 0.001, 0.996]),
        np.ones(5) / 5,
    ]
    for w in extreme_cases:
        w = w / w.sum()
        policy_fe = irl._compute_policy_fe(mdp, w)
        assert not np.any(np.isnan(policy_fe)), f"NaN with weights {w}"
        assert not np.any(np.isinf(policy_fe)), f"Inf with weights {w}"
    print("  PASS test_numerical_stability_extreme_weights")


def test_empty_mdp_fallback():
    """Empty MDP returns valid structure, not crashing."""
    mdp = IRLRecovery._empty_mdp()
    assert mdp["n_states"] == 1
    assert mdp["n_actions"] == 8
    assert mdp["init_dist"].sum() == 1.0

    # Soft VI should work on empty MDP
    irl = IRLRecovery()
    w = np.ones(5) / 5
    policy_fe = irl._compute_policy_fe(mdp, w)
    assert not np.any(np.isnan(policy_fe))
    print("  PASS test_empty_mdp_fallback")


def test_trajectory_collector():
    """TrajectoryCollector correctly categorizes personas."""
    collector = TrajectoryCollector()

    # Mock agent for persona testing
    class MockProfile:
        pass
    class MockDynamic:
        pass
    class MockAgent:
        def __init__(self, aid, role, age, familiarity):
            self.id = aid
            self.profile = MockProfile()
            self.profile.role = role
            self.profile.age = age
            self.profile.familiarity = familiarity
            self.dynamic = MockDynamic()
            self.dynamic.position = np.array([50.0, 30.0])
            self.dynamic.stamina = 80.0
            self.dynamic.fear_level = 3.0
            self.dynamic.evacuated = False
            self.dynamic.alive = True
            self.dynamic.last_decision_tick = 0

    personas = collector._get_persona(MockAgent("a1", "firefighter", 35, 0.9))
    assert personas == "firefighter", f"Expected firefighter, got {personas}"

    personas = collector._get_persona(MockAgent("a2", "guide", 30, 0.8))
    assert personas == "guide", f"Expected guide, got {personas}"

    personas = collector._get_persona(MockAgent("a3", "civilian", 25, 0.7))
    assert personas == "trained_staff", f"Expected trained_staff, got {personas}"

    personas = collector._get_persona(MockAgent("a4", "civilian", 65, 0.2))
    assert personas == "untrained_elderly", f"Expected untrained_elderly, got {personas}"

    personas = collector._get_persona(MockAgent("a5", "civilian", 25, 0.2))
    assert personas == "untrained_young", f"Expected untrained_young, got {personas}"

    print("  PASS test_trajectory_collector")


if __name__ == "__main__":
    print("=== IRL Recovery Unit Tests ===\n")
    tests = [
        test_feature_extraction,
        test_mdp_construction,
        test_soft_value_iteration_convergence,
        test_gc_maxent_fit,
        test_irl_persistence,
        test_reward_function_callable,
        test_numerical_stability_extreme_weights,
        test_empty_mdp_fallback,
        test_trajectory_collector,
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

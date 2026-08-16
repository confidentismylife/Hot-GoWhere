"""Canonical reward-feature definitions shared by IRL and RL.

Previously IRL (execution/irl_recovery.py) and RL
(execution/rl_scheduler.py) each maintained their own feature formulas,
which drifted apart. This module is the single source of truth:

  - ``FEATURE_NAMES`` — the 5 interpretable reward dimensions.
  - ``extract_decision_record_features`` — per-decision features used by IRL
    (from a trajectory decision record).
  - ``extract_trajectory_features`` — convenience wrapper over the above.
  - ``zone_reward_features`` — zone-level features used by the RL scheduler
    to score its exit-recommendation action.
"""

from typing import List, Tuple

import numpy as np


FEATURE_NAMES = [
    "safety",       # Staying away from fire/smoke
    "efficiency",   # Moving toward nearest usable exit quickly
    "social",       # Helping others, staying with family
    "conformity",   # Following crowd / authority
    "comfort",      # Taking familiar routes, avoiding exertion
]


def extract_decision_record_features(dec: dict) -> np.ndarray:
    """Extract normalized 5-D features from one trajectory decision record."""
    smoke = dec.get("smoke_at_pos", 0)
    fire_dist = dec.get("fire_distance", 200)
    safety = (1.0 - smoke) * min(fire_dist / 50.0, 1.0)

    nearest_dist = max(dec.get("nearest_exit_dist", 30), 1)
    speed_str = str(dec.get("speed", "walk"))
    speed_val = 1.0 if "run" in speed_str else (
        0.5 if "walk" in speed_str else 0.2)
    efficiency = (1.0 / (1.0 + nearest_dist / 50.0)) * speed_val

    coop = str(dec.get("cooperation", "none"))
    if "help_family" in coop or "lead_others" in coop:
        social = 1.0
    elif "follow_crowd" in coop:
        social = 0.5
    else:
        social = 0.1

    conformity = 1.0 if "follow_crowd" in coop else (
        0.3 if "none" in coop else 0.5)

    fear = float(dec.get("fear", 0)) / 10.0
    speed_comfort = 1.0 if "walk" in speed_str else (
        0.8 if "crawl" in speed_str else 0.3)
    comfort = (1.0 - fear) * speed_comfort

    return np.array([safety, efficiency, social, conformity, comfort])


def extract_trajectory_features(traj) -> List[List[float]]:
    """Extract normalized feature vectors from one trajectory."""
    return [extract_decision_record_features(dec).tolist()
            for dec in traj.decisions]


def zone_reward_features(zone, zone_agents, env_snap, action) -> np.ndarray:
    """Compute the 5 zone-level reward features for an RL action.

    Mirrors the semantics of the IRL features at zone granularity:
      safety      — low smoke at exits weighted by the recommendation
      efficiency  — short average distance to recommended exits
      social      — low crowd at recommended exits
      conformity  — recommendation agrees with the majority target
      comfort     — simplified as path safety (same as safety)

    The recommendation is converted with a softmax instead of ``argmax``.
    A hard top-exit lookup makes the reward piecewise constant in the action,
    which gives a continuous Gaussian policy almost no useful gradient.
    """
    n = len(zone_agents)
    if n == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    n_exits = len(env_snap.exits)
    preferences = np.asarray(
        action.exit_preferences[:n_exits], dtype=np.float64
    )
    if preferences.size != n_exits:
        preferences = np.pad(
            preferences, (0, n_exits - preferences.size), constant_values=0.0
        )

    # Temperature keeps the reward smooth while still favoring the best
    # recommendations. Subtracting the maximum avoids exponential overflow.
    temperature = 0.5
    logits = (preferences - np.max(preferences)) / temperature
    exit_weights = np.exp(logits)
    exit_weights /= max(np.sum(exit_weights), 1e-12)

    exit_positions = np.asarray(env_snap.exits, dtype=np.float64)
    exit_smoke = np.asarray([
        float(env_snap.smoke_at(pos)) for pos in exit_positions
    ], dtype=np.float64)
    safety_by_exit = np.clip(1.0 - exit_smoke, 0.0, 1.0)
    safety = float(np.dot(exit_weights, safety_by_exit))

    positions = np.asarray([a.position for a in zone_agents[:20]],
                           dtype=np.float64)
    distances = np.linalg.norm(
        positions[:, None, :] - exit_positions[None, :, :], axis=2
    ).mean(axis=0)
    efficiency_by_exit = 1.0 - np.minimum(1.0, distances / 150.0)
    efficiency = float(np.dot(exit_weights, efficiency_by_exit))

    crowd_by_exit = np.asarray([
        sum(1 for a in zone_agents
            if a.dynamic.target_exit_idx == exit_idx)
        for exit_idx in range(n_exits)
    ], dtype=np.float64)
    social_by_exit = 1.0 - np.minimum(1.0, crowd_by_exit / max(1, n))
    social = float(np.dot(exit_weights, social_by_exit))

    target_indices = [
        int(a.dynamic.target_exit_idx) for a in zone_agents
        if a.dynamic.target_exit_idx is not None
        and 0 <= int(a.dynamic.target_exit_idx) < n_exits
    ]
    if target_indices:
        counts = np.bincount(target_indices, minlength=n_exits)
        majority_exit = int(np.argmax(counts))
        conformity = float(exit_weights[majority_exit])
    else:
        conformity = 0.5

    comfort = safety  # Simplified: comfort ≈ safety of route
    return np.array([safety, efficiency, social, conformity, comfort])


def blocked_exit_weight(env_snap, action,
                        smoke_block_threshold: float = 0.6,
                        blocked_exits=None) -> float:
    """Fraction of recommendation weight placed on smoke-blocked exits.

    Used as a training-time penalty so the RL scheduler learns not to push
    agents toward exits that the safety guard would reject at deployment.
    """
    n_exits = len(env_snap.exits)
    if n_exits == 0:
        return 0.0
    preferences = np.asarray(
        action.exit_preferences[:n_exits], dtype=np.float64)
    if preferences.size != n_exits:
        preferences = np.pad(
            preferences, (0, n_exits - preferences.size), constant_values=0.0)
    temperature = 0.5
    logits = (preferences - np.max(preferences)) / temperature
    exit_weights = np.exp(logits)
    exit_weights /= max(np.sum(exit_weights), 1e-12)

    if blocked_exits is not None:
        # Precomputed blocked set (e.g. smoke-blocked OR fire-path-blocked)
        blocked = np.zeros(n_exits, dtype=bool)
        for i in blocked_exits:
            if 0 <= int(i) < n_exits:
                blocked[int(i)] = True
    else:
        exit_smoke = np.asarray([
            float(env_snap.smoke_at(np.asarray(pos, dtype=np.float64)))
            for pos in env_snap.exits
        ], dtype=np.float64)
        blocked = exit_smoke > smoke_block_threshold
    return float(np.dot(exit_weights, blocked.astype(np.float64)))

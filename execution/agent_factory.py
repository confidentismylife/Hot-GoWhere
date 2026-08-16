"""Config-driven agent profile generation.

Previously both ``orchestrator._random_profile`` and the oracle trajectory
generator hardcoded the same age/familiarity distributions, silently ignoring
``agents.age_groups`` / ``agents.familiarity_distribution`` /
``agents.max_speed_range`` in the YAML configs (and never producing children
even when a config defined a child group). This module is the single,
config-driven implementation used by both callers.
"""

import random
from typing import Dict, List, Tuple

import numpy as np

from decision.agent_state import Agent, AgentProfile, AgentDynamic


DEFAULT_AGE_GROUPS = {
    "young":   {"min": 18, "max": 35, "weight": 0.35},
    "middle":  {"min": 36, "max": 55, "weight": 0.45},
    "elderly": {"min": 56, "max": 80, "weight": 0.20},
}

DEFAULT_FAMILIARITY = {
    "low":    {"range": [0.0, 0.3], "weight": 0.3},
    "medium": {"range": [0.3, 0.7], "weight": 0.5},
    "high":   {"range": [0.7, 1.0], "weight": 0.2},
}

DEFAULT_MAX_SPEED_RANGE = [0.8, 2.0]

# Speed multiplier per age-group label; falls back to an age-based rule.
AGE_SPEED_FACTORS = {
    "child": 0.70,
    "teen": 0.90,
    "young": 1.00,
    "middle": 0.85,
    "elderly": 0.65,
}


def _age_speed_factor(group_name: str, age: int) -> float:
    if group_name in AGE_SPEED_FACTORS:
        return AGE_SPEED_FACTORS[group_name]
    if age < 18:
        return 0.70
    if age < 35:
        return 1.00
    if age < 55:
        return 0.85
    return 0.65


def random_profile(agent_cfg: dict, rng=random) -> AgentProfile:
    """Generate one AgentProfile from config-driven distributions."""
    age_groups = agent_cfg.get("age_groups") or DEFAULT_AGE_GROUPS
    names = list(age_groups.keys())
    weights = [float(g.get("weight", 1.0)) for g in age_groups.values()]
    group_name = rng.choices(names, weights=weights, k=1)[0]
    group = age_groups[group_name]
    lo, hi = int(group.get("min", 18)), int(group.get("max", 35))
    if lo > hi:
        lo, hi = hi, lo
    age = rng.randint(lo, hi) if hi > lo else lo

    fam_dist = agent_cfg.get("familiarity_distribution") or DEFAULT_FAMILIARITY
    fam_names = list(fam_dist.keys())
    fam_weights = [float(f.get("weight", 1.0)) for f in fam_dist.values()]
    fam_name = rng.choices(fam_names, weights=fam_weights, k=1)[0]
    fam_range = fam_dist[fam_name]["range"]
    familiarity = rng.uniform(fam_range[0], fam_range[1])

    speed_range = agent_cfg.get("max_speed_range", DEFAULT_MAX_SPEED_RANGE)
    base_speed = rng.uniform(speed_range[0], speed_range[1])
    max_speed = base_speed * _age_speed_factor(group_name, age)

    return AgentProfile(
        age=age,
        gender=rng.choice(["male", "female"]),
        occupation=rng.choice(["office_worker", "student", "shopkeeper",
                               "tourist", "security_guard", "retiree"]),
        familiarity=familiarity,
        max_speed=max_speed,
        risk_aversion=rng.uniform(0.2, 0.9),
        altruism=rng.uniform(0.1, 0.8),
        trust_authority=rng.uniform(0.3, 0.95),
        conformity=rng.uniform(0.1, 0.9),
    )


def _make_dependent(age_range: Tuple[int, int], occupation: str,
                    anchor_position: np.ndarray, rng=random) -> Agent:
    """Create a child/elderly family member anchored near a parent."""
    profile = AgentProfile(
        age=rng.randint(age_range[0], age_range[1]),
        gender=rng.choice(["male", "female"]),
        occupation=occupation,
        familiarity=rng.uniform(0.4, 0.7),
        max_speed=rng.uniform(0.6, 1.0),
        risk_aversion=rng.uniform(0.3, 0.8),
        altruism=rng.uniform(0.2, 0.7),
        trust_authority=rng.uniform(0.3, 0.9),
        conformity=rng.uniform(0.2, 0.8),
    )
    dynamic = AgentDynamic(
        position=anchor_position + np.array(
            [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)],
            dtype=np.float64),
        stamina=rng.uniform(60, 100),
        known_exit_positions=[],
        has_new_info=True,
    )
    return Agent(profile=profile, dynamic=dynamic)


def create_family_groups(agents: List[Agent], agent_cfg: dict,
                         rng=random) -> List[Agent]:
    """Link agents into family units; optionally append dependents.

    Previously the flags ``has_children`` / ``has_elderly`` were set without
    creating actual child/elderly members (and ``has_elderly`` could never be
    True because the second parent was always < 60). Now a family pair may
    gain real child (5-16) and elderly (65-85) members linked to both parents,
    and both parents carry the corresponding flags.

    Returns newly created dependent agents; the caller must extend its agent
    list with them.
    """
    prob = agent_cfg.get("family_group_probability", 0.3)
    if prob <= 0:
        return []

    eligible = [a for a in agents if a.profile.age < 60]
    rng.shuffle(eligible)
    family_count = int(len(eligible) * prob / 2)

    child_prob = agent_cfg.get("family_child_probability", 0.25)
    elderly_prob = agent_cfg.get("family_elderly_probability", 0.25)
    new_members = []

    for _ in range(family_count):
        if len(eligible) < 2:
            break
        a1 = eligible.pop()
        a2 = eligible.pop()

        a1.dynamic.family_member_ids.append(a2.id)
        a2.dynamic.family_member_ids.append(a1.id)

        if rng.random() < child_prob:
            child = _make_dependent((5, 16), "student", a1.position, rng)
            child.dynamic.family_member_ids += [a1.id, a2.id]
            a1.dynamic.has_children = True
            a2.dynamic.has_children = True
            new_members.append(child)

        if rng.random() < elderly_prob:
            elder = _make_dependent((65, 85), "retiree", a2.position, rng)
            elder.dynamic.family_member_ids += [a1.id, a2.id]
            a1.dynamic.has_elderly = True
            a2.dynamic.has_elderly = True
            new_members.append(elder)

    return new_members

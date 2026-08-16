"""Unified decision policy interface.

Every decision source (heuristic, oracle, LLM) should implement
``DecisionPolicy`` so the orchestrator, tactical layer and experiment
generators share one exit-scoring contract instead of maintaining
several subtly-different copies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from decision.agent_state import Agent, Cooperation, Speed
from decision.cognitive_engine import DecisionResult
from perception.environment import EnvironmentSnapshot


class DecisionPolicy:
    """Base interface for decision sources."""

    name: str = "base"

    def decide(self, agent: Agent, env_snapshot: EnvironmentSnapshot,
               exits: List[Tuple[float, float]], tick: int = 0,
               **ctx) -> DecisionResult:
        raise NotImplementedError


class HeuristicPolicy(DecisionPolicy):
    """Distance + smoke-penalty exit choice.

    Unified scoring used by the orchestrator fallback, the tactical layer,
    and the non-LLM simulation path:
        score(exit) = dist * (1 + smoke * smoke_penalty)

    Exits with smoke above ``smoke_block`` are skipped unless every exit
    is blocked (in which case the least-smoky one is chosen).
    """

    name = "heuristic"

    def __init__(self, smoke_block: float = 0.7,
                 smoke_penalty: float = 3.0):
        self.smoke_block = smoke_block
        self.smoke_penalty = smoke_penalty

    def decide(self, agent: Agent, env_snapshot: EnvironmentSnapshot,
               exits: List[Tuple[float, float]], tick: int = 0,
               **ctx) -> DecisionResult:
        pos = agent.dynamic.position
        best_idx = 0
        best_score = float('inf')
        all_blocked = True
        min_smoke_idx = 0
        min_smoke = float('inf')

        for i, ex in enumerate(exits):
            ex_arr = np.array(ex, dtype=np.float64)
            dist = float(np.linalg.norm(ex_arr - pos))
            smoke = float(env_snapshot.smoke_at(ex_arr))
            if smoke < min_smoke:
                min_smoke = smoke
                min_smoke_idx = i
            if smoke > self.smoke_block:
                continue
            all_blocked = False
            score = dist * (1.0 + smoke * self.smoke_penalty)
            if score < best_score:
                best_score = score
                best_idx = i

        if all_blocked and exits:
            best_idx = min_smoke_idx

        local_smoke = float(env_snapshot.smoke_at(pos))
        if local_smoke > 0.6:
            spd = Speed.CRAWL
        elif local_smoke > 0.3:
            spd = Speed.WALK
        else:
            # Safe conditions: allow normal walking (kept conservative to
            # preserve prior behavior; tactical layer may escalate to RUN).
            spd = Speed.WALK

        return DecisionResult(
            agent_id=agent.id,
            target_exit_idx=best_idx,
            target_exit_pos=exits[best_idx] if exits else (0.0, 0.0),
            speed=spd,
            cooperation=agent.dynamic.cooperation_choice,
            reasoning=f"[{self.name}] exit {best_idx + 1}",
            risk_assessment="low",
            compute_time=0.0,
            tick=tick,
            source=self.name,
        )


# ================================================================
# Oracle policy (synthetic expert used for IRL sensitivity analysis)
# ================================================================

@dataclass(frozen=True)
class OracleWeights:
    """One set of Oracle scoring weights representing a behavioral strategy.

    Weights are for features: safety, efficiency, crowd_penalty, fire_risk.
    These are synthetic parameterizations used to test whether IRL can
    recover distinct preference structures from behavioral data alone.
    """
    name: str
    label: str
    w_safety: float
    w_efficiency: float
    w_crowd: float       # penalty, applied with negative sign
    w_fire_risk: float   # penalty, applied with negative sign

    @property
    def description(self) -> str:
        return (f"{self.label}: safety={self.w_safety:.2f}, "
                f"efficiency={self.w_efficiency:.2f}, "
                f"crowd={self.w_crowd:.2f}, fire_risk={self.w_fire_risk:.2f}")


ORACLE_WEIGHT_CONFIGS = [
    OracleWeights("safety_first", "极度惜命型",
                  w_safety=0.70, w_efficiency=0.10, w_crowd=0.10, w_fire_risk=0.10),
    OracleWeights("balanced", "均衡型",
                  w_safety=0.35, w_efficiency=0.35, w_crowd=0.15, w_fire_risk=0.15),
    OracleWeights("efficiency_first", "追求效率型",
                  w_safety=0.10, w_efficiency=0.70, w_crowd=0.10, w_fire_risk=0.10),
]

ORACLE_WEIGHTS_LEGACY = OracleWeights(
    "legacy", "原始设定(0.45/0.25/0.15/0.15)",
    w_safety=0.45, w_efficiency=0.25, w_crowd=0.15, w_fire_risk=0.15)


class OraclePolicy(DecisionPolicy):
    """Oracle expert decision function.

    Scores each exit with a weight configuration and returns a
    DecisionResult directly (source="oracle").
    """

    name = "oracle"

    def decide(self, agent: Agent, env_snapshot: EnvironmentSnapshot,
               exits: List[Tuple[float, float]], tick: int = 0,
               crowd_map: Optional[Dict[int, int]] = None,
               weights: Optional[OracleWeights] = None,
               **ctx) -> DecisionResult:
        if weights is None:
            weights = ORACLE_WEIGHTS_LEGACY
        crowd_map = crowd_map or {}

        pos = agent.dynamic.position
        fire_origin = np.array(env_snapshot.fire_origin, dtype=np.float64)

        best_idx = 0
        best_score = -float('inf')
        scores = []

        for i, ex in enumerate(exits):
            ex_arr = np.array(ex, dtype=np.float64)
            dist = float(np.linalg.norm(pos - ex_arr))

            smoke_now = float(env_snapshot.smoke_at(ex_arr))
            safety = max(0.0, 1.0 - smoke_now)
            efficiency = 1.0 / (1.0 + dist / 40.0)

            crowd = crowd_map.get(i, 0)
            max_crowd_per_exit = max(
                1, sum(crowd_map.values()) / max(1, len(exits)))
            crowd_ratio = crowd / max(1, max_crowd_per_exit * 2)
            crowd_penalty = min(0.5, crowd_ratio * 0.5)

            fire_dist_to_agent = float(np.linalg.norm(fire_origin - pos))
            fire_risk = 1.0 / (1.0 + fire_dist_to_agent / 30.0)

            score = (safety * weights.w_safety +
                     efficiency * weights.w_efficiency -
                     crowd_penalty * weights.w_crowd -
                     fire_risk * weights.w_fire_risk)
            scores.append((i, score, smoke_now, dist, safety, efficiency))
            if score > best_score:
                best_score = score
                best_idx = i

        urgency = 1.0 - best_score
        if urgency > 0.7 or scores[best_idx][2] > 0.5:
            spd = Speed.RUN
        else:
            spd = Speed.WALK

        return DecisionResult(
            agent_id=agent.id,
            target_exit_idx=best_idx,
            target_exit_pos=exits[best_idx],
            speed=spd,
            cooperation=Cooperation.NONE,
            reasoning=(f"[Oracle-{weights.name}] exit {best_idx + 1}: "
                       f"safety={scores[best_idx][4]:.2f} "
                       f"smoke={scores[best_idx][2]:.2f}"),
            risk_assessment="low" if best_score > 0.6 else "moderate",
            compute_time=0.0,
            raw_data={"weight_config": weights.name},
            tick=tick,
            source=self.name,
        )

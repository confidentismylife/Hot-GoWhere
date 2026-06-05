"""Safety guard — hard constraints that LLM decisions must satisfy.

Sits between LLM output and agent state mutation. Every decision passes through
this filter before being applied. Blocked decisions fall back to heuristic.

Design:
  Hard constraints (BLOCK) — physically impossible or fatal actions
  Soft constraints (WARN)  — unreasonable but not immediately fatal
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from decision.agent_state import Agent, Speed, Cooperation
from perception.environment import EnvironmentSnapshot


@dataclass
class SafetyResult:
    passed: bool
    modified: bool
    original_exit_idx: int = 0
    final_exit_idx: int = 0
    original_speed: str = ""
    final_speed: str = ""
    warnings: List[str] = field(default_factory=list)
    block_reason: str = ""


class SafetyGuard:
    """Hard safety constraints for LLM evacuation decisions.

    Usage:
        guard = SafetyGuard()
        for decision, agent, env in decisions:
            result = guard.check(decision, agent, env)
            if result.passed:
                apply(decision)  # possibly modified
            else:
                fallback(agent, env)
    """

    # ------------------------------------------------------------------
    # Hard constraint thresholds
    # ------------------------------------------------------------------
    EXIT_SMOKE_BLOCK = 0.6       # Exit blocked if smoke > this
    EXIT_SMOKE_WARN = 0.3        # Warn if smoke > this
    STAMINA_RUN_MIN = 20.0       # Must have >= this to run
    STAMINA_CRAWL_MAX = 10.0     # Force crawl if stamina <= this
    FIRE_SAFE_DISTANCE = 8.0     # Meters from any fire cell
    FIRE_ESCAPE_URGENCY = 2.0    # If standing on fire, force RUN
    EXIT_DISTANCE_MAX = 150.0    # Sanity: exit farther than this is likely an error
    SPEED_STAMINA_COST = {       # Stamina cost per meter traveled
        "run": 0.08,
        "walk": 0.02,
        "crawl": 0.01,
        "wait": 0.0,
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, decision, agent: Agent, env: EnvironmentSnapshot) -> SafetyResult:
        """Validate an LLM decision against all safety rules.

        Args:
            decision: DecisionResult from cognitive engine
            agent: The agent making the decision
            env: Current environment snapshot

        Returns:
            SafetyResult with pass/fail and any modifications
        """
        result = SafetyResult(
            passed=True,
            modified=False,
            original_exit_idx=decision.target_exit_idx,
            final_exit_idx=decision.target_exit_idx,
            original_speed=decision.speed.value,
            final_speed=decision.speed.value,
        )

        # --- Rule execution order matters ---
        self._check_exit_smoke(decision, agent, env, result)
        self._check_fire_proximity(decision, agent, env, result)
        self._check_stamina_speed(decision, agent, result)
        self._check_injured_speed(decision, agent, result)
        self._check_agent_on_fire(decision, agent, env, result)
        self._check_distance_sanity(decision, agent, env, result)
        self._check_wait_on_fire(decision, agent, env, result)

        # If exit was changed, validate the new one
        if result.modified and result.final_exit_idx != result.original_exit_idx:
            new_exit_pos = env.exits[result.final_exit_idx]
            new_smoke = env.smoke_at(np.array(new_exit_pos, dtype=np.float64))
            if new_smoke > self.EXIT_SMOKE_BLOCK:
                result.passed = False
                result.block_reason = "所有出口均被浓烟封锁"

        return result

    # ------------------------------------------------------------------
    # Individual constraint checks
    # ------------------------------------------------------------------

    def _check_exit_smoke(self, decision, agent: Agent,
                          env: EnvironmentSnapshot, result: SafetyResult):
        """Block exits with heavy smoke. Downgrade to best available."""
        exit_pos = env.exits[decision.target_exit_idx]
        smoke = env.smoke_at(np.array(exit_pos, dtype=np.float64))

        if smoke > self.EXIT_SMOKE_BLOCK:
            # Find best alternative
            best_idx = self._best_exit(agent.position, env)
            if best_idx != decision.target_exit_idx:
                result.warnings.append(
                    f"出口{decision.target_exit_idx+1}烟雾{smoke:.0%}>"
                    f"{self.EXIT_SMOKE_BLOCK:.0%}, 切换至出口{best_idx+1}"
                )
                result.final_exit_idx = best_idx
                result.modified = True

        elif smoke > self.EXIT_SMOKE_WARN:
            result.warnings.append(
                f"出口{decision.target_exit_idx+1}有烟雾({smoke:.0%}), 请注意安全"
            )

    def _check_fire_proximity(self, decision, agent: Agent,
                              env: EnvironmentSnapshot, result: SafetyResult):
        """Block if target exit requires crossing within FIRE_SAFE_DISTANCE of fire."""
        exit_pos = env.exits[decision.target_exit_idx]
        agent_pos = agent.position

        # Sample points along the path and check fire proximity
        steps = 8
        for t in range(steps + 1):
            alpha = t / steps
            px = agent_pos[0] + alpha * (exit_pos[0] - agent_pos[0])
            py = agent_pos[1] + alpha * (exit_pos[1] - agent_pos[1])
            point = np.array([px, py], dtype=np.float64)

            if env.is_on_fire(point):
                # Path crosses fire — find alternative
                best_idx = self._best_exit(agent.position, env)
                if best_idx != decision.target_exit_idx:
                    result.warnings.append(
                        f"前往出口{decision.target_exit_idx+1}的路径经过火源, "
                        f"切换至出口{best_idx+1}"
                    )
                    result.final_exit_idx = best_idx
                    result.modified = True
                else:
                    result.passed = False
                    result.block_reason = "路径经过火源且无替代出口"
                return

    def _check_stamina_speed(self, decision, agent: Agent, result: SafetyResult):
        """Enforce stamina-based speed limits."""
        stamina = agent.dynamic.stamina
        speed = decision.speed

        if speed == Speed.RUN and stamina < self.STAMINA_RUN_MIN:
            if stamina <= self.STAMINA_CRAWL_MAX:
                result.final_speed = Speed.CRAWL.value
                result.warnings.append(f"体力仅{stamina:.0f}, run→crawl")
            else:
                result.final_speed = Speed.WALK.value
                result.warnings.append(f"体力仅{stamina:.0f}, run→walk")
            result.modified = True

    def _check_injured_speed(self, decision, agent: Agent, result: SafetyResult):
        """Injured agents cannot run."""
        if agent.dynamic.injured and decision.speed == Speed.RUN:
            result.final_speed = Speed.WALK.value
            result.warnings.append("受伤状态禁止奔跑, run→walk")
            result.modified = True

    def _check_agent_on_fire(self, decision, agent: Agent,
                             env: EnvironmentSnapshot, result: SafetyResult):
        """Agent standing on fire must move immediately."""
        if env.is_on_fire(agent.position):
            if decision.speed == Speed.WAIT:
                result.final_speed = Speed.RUN.value
                result.warnings.append("所在位置已着火, wait→run")
                result.modified = True
            elif decision.speed == Speed.CRAWL:
                result.final_speed = Speed.RUN.value
                result.warnings.append("所在位置已着火, crawl→run")
                result.modified = True

    def _check_distance_sanity(self, decision, agent: Agent,
                               env: EnvironmentSnapshot, result: SafetyResult):
        """Flag unreasonably far exits (likely LLM hallucination)."""
        exit_pos = env.exits[decision.target_exit_idx]
        dist = float(np.linalg.norm(np.array(exit_pos) - agent.position))

        if dist > self.EXIT_DISTANCE_MAX and len(env.exits) > 1:
            best_idx = self._best_exit(agent.position, env)
            result.warnings.append(
                f"出口{decision.target_exit_idx+1}距离{dist:.0f}m超过合理范围, "
                f"切换至出口{best_idx+1}"
            )
            result.final_exit_idx = best_idx
            result.modified = True

    def _check_wait_on_fire(self, decision, agent: Agent,
                            env: EnvironmentSnapshot, result: SafetyResult):
        """Don't wait if smoke at current position is heavy."""
        if decision.speed != Speed.WAIT:
            return

        smoke = env.smoke_at(agent.position)
        temp = env.temperature_at(agent.position)

        if smoke > 0.5 or temp > 60:
            result.final_speed = Speed.WALK.value
            result.warnings.append(
                f"当前位置烟雾{smoke:.0%}/温度{temp:.0f}°C, 不应原地等待"
            )
            result.modified = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _best_exit(self, agent_pos: np.ndarray,
                   env: EnvironmentSnapshot) -> int:
        """Find the best exit: closest balance of distance and smoke."""
        best_idx = 0
        best_score = float("inf")

        for i, exit_pos in enumerate(env.exits):
            dist = float(np.linalg.norm(np.array(exit_pos) - agent_pos))
            smoke = env.smoke_at(np.array(exit_pos, dtype=np.float64))

            # Smoke penalty grows exponentially after warn threshold
            if smoke > self.EXIT_SMOKE_BLOCK:
                continue  # Skip blocked exits entirely

            smoke_penalty = 0.0
            if smoke > self.EXIT_SMOKE_WARN:
                smoke_penalty = ((smoke - self.EXIT_SMOKE_WARN) /
                                 (self.EXIT_SMOKE_BLOCK - self.EXIT_SMOKE_WARN)) * 300

            score = dist + smoke_penalty
            if score < best_score:
                best_score = score
                best_idx = i

        return best_idx

    def check_exit_safety(self, exit_idx: int,
                           env: EnvironmentSnapshot) -> dict:
        """Quick check: is this exit safe? Used by comparison framework."""
        if exit_idx < 0 or exit_idx >= len(env.exits):
            return {"blocked": True, "reason": "Invalid exit index"}
        smoke = env.smoke_at(np.array(env.exits[exit_idx], dtype=np.float64))
        blocked = smoke > self.EXIT_SMOKE_BLOCK
        return {
            "blocked": blocked,
            "smoke": float(smoke),
            "exit_idx": exit_idx,
            "reason": "浓烟封锁" if blocked else "安全",
        }

    def fallback_decision(self, agent: Agent,
                          env: EnvironmentSnapshot) -> dict:
        """Generate a safe fallback decision when LLM output is unrecoverable."""
        best_idx = self._best_exit(agent.position, env)
        exit_pos = env.exits[best_idx]

        stamina = agent.dynamic.stamina
        if stamina > self.STAMINA_RUN_MIN and not agent.dynamic.injured:
            speed = Speed.RUN
        elif stamina > self.STAMINA_CRAWL_MAX:
            speed = Speed.WALK
        else:
            speed = Speed.CRAWL

        smoke_at_pos = env.smoke_at(agent.position)
        if env.is_on_fire(agent.position):
            speed = Speed.RUN

        return {
            "target_exit_idx": best_idx,
            "target_exit_pos": exit_pos,
            "speed": speed,
            "cooperation": Cooperation.NONE,
            "reasoning": f"[安全约束兜底] 选择出口{best_idx+1}",
            "risk_assessment": f"当前位置烟雾{smoke_at_pos:.0%}",
        }

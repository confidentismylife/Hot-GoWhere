"""Fast tactical layer — reactive adjustments every tick for all agents.

Implements the "Torso" in the Brain-Torso cognitive dual-process architecture:
  - Slow (LLM): strategic exit selection, cooperation mode — infrequent (~10-30s)
  - Fast (Tactical): speed adjustment, exit switching, local reactions — every tick

This decoupling means LLM latency no longer leaves agents frozen — the fast
layer continuously adapts while waiting for the next strategic update.

Also provides criticality scoring to filter which agents actually need LLM
re-decisions, avoiding wasted inference on agents in safe, stable situations.

Reference: AgileThinker (2024), UAV Swarm Brain-Torso (Frontiers in
Neurorobotics 2026), ScaleSim sparse activation (2025).
"""

import numpy as np
from typing import List, Tuple, Optional
from decision.agent_state import Agent, Speed, Cooperation
from decision.policy import HeuristicPolicy
from perception.environment import EnvironmentSnapshot


class TacticalLayer:
    """Per-tick reactive adjustments for all agents.

    Usage (in orchestrator main loop, every tick, before physics):
        TacticalLayer.adjust_all(agents, env_snapshot, exits)

    Then when building the LLM submission list:
        critical = [a for a in candidates if TacticalLayer.is_critical(a, env, ...)]
        non_critical get heuristic decisions instead.
    """

    # Thresholds
    EXIT_SMOKE_BLOCK = 0.7    # Switch exit if target exit smoke > this
    EXIT_SMOKE_WARN = 0.4     # Flag exit as risky
    FIRE_CRITICAL_DIST = 30.0 # Within this distance → agent is "critical"
    SMOKE_CRITICAL = 0.5       # Above this → agent is "critical"
    FORCED_REFRESH_MULT = 2    # Re-decide after N × decision_interval (LLM refresh)

    # ==================================================================
    # Public API
    # ==================================================================

    @classmethod
    def adjust_all(cls, agents: List[Agent], env: EnvironmentSnapshot,
                   exits: List[Tuple[float, float]], tick: int = 0):
        """Run tactical adjustments for all alive, non-evacuated agents.

        Called every tick. Must complete in < 1ms for 600 agents.
        """
        for agent in agents:
            if not agent.dynamic.alive or agent.dynamic.evacuated:
                continue

            cls._adjust_speed(agent, env)
            cls._check_exit_switch(agent, env, exits, tick)

    @classmethod
    def is_critical(cls, agent: Agent, env: EnvironmentSnapshot,
                    tick: int, decision_ticks: int) -> bool:
        """Determine if this agent needs LLM re-decision (vs heuristic).

        Returns True if agent is in a dangerous or rapidly-changing situation
        that warrants expensive LLM reasoning. Returns False for stable,
        safe agents that can continue with heuristic updates.

        Critical conditions (any one triggers LLM):
          1. First decision pending (has_new_info)
          2. In heavy smoke (> SMOKE_CRITICAL)
          3. Near fire (< FIRE_CRITICAL_DIST meters)
          4. Target exit is smoke-blocked
          5. No LLM decision for > FORCED_REFRESH_MULT × decision_interval
             (periodic strategic refresh)
        """
        # First decision always goes to LLM (initial strategic orientation)
        if agent.dynamic.has_new_info:
            return True

        pos = agent.dynamic.position
        smoke = float(env.smoke_at(pos))

        # Heavy smoke → need LLM to pick best escape route
        if smoke > cls.SMOKE_CRITICAL:
            return True

        # Near fire → dangerous, LLM may find better exit
        fire_dist = cls._fire_distance(pos, env)
        if fire_dist < cls.FIRE_CRITICAL_DIST:
            return True

        # Target exit blocked → need LLM to re-plan
        target = agent.dynamic.target_exit
        if target is not None:
            target_smoke = float(env.smoke_at(target))
            if target_smoke > cls.EXIT_SMOKE_BLOCK:
                return True

        # Periodic forced refresh: even safe agents re-check with LLM
        # every FORCED_REFRESH_MULT × decision_interval
        ticks_since = tick - agent.dynamic.last_decision_tick
        if ticks_since >= decision_ticks * cls.FORCED_REFRESH_MULT:
            return True

        return False

    # ==================================================================
    # Per-agent adjustments
    # ==================================================================

    @classmethod
    def _adjust_speed(cls, agent: Agent, env: EnvironmentSnapshot):
        """Adjust speed based on local smoke, stamina, fear, and fire.

        Rules (in priority order):
          1. Standing on fire → RUN (survival reflex)
          2. Stamina < 15 → CRAWL (physically can't do more)
          3. Smoke > 0.6 → CRAWL (stay low)
          4. Fear > 8 AND stamina > 40 → RUN (panic)
          5. Stamina < 30 → WALK (conserve energy)
          6. Smoke > 0.3 → WALK (caution)
          7. Otherwise keep current speed
        """
        pos = agent.dynamic.position
        smoke = float(env.smoke_at(pos))
        stamina = agent.dynamic.stamina
        fear = agent.dynamic.fear_level

        if env.is_on_fire(pos):
            agent.dynamic.speed_choice = Speed.RUN
            return

        if stamina < 15:
            agent.dynamic.speed_choice = Speed.CRAWL
            return

        if smoke > 0.6:
            agent.dynamic.speed_choice = Speed.CRAWL
            return

        if fear > 8 and stamina > 40:
            agent.dynamic.speed_choice = Speed.RUN
            return

        if stamina < 30:
            agent.dynamic.speed_choice = Speed.WALK
            return

        if smoke > 0.3:
            agent.dynamic.speed_choice = Speed.WALK
            return

        # Safe conditions: keep current speed (LLM-decided or seed)

    @classmethod
    def _check_exit_switch(cls, agent: Agent, env: EnvironmentSnapshot,
                           exits: List[Tuple[float, float]], tick: int = 0):
        """If target exit is smoke-blocked, switch to best available exit.

        This is the key safety mechanism: even if LLM hasn't responded yet,
        agents won't walk into a blocked exit.

        Grace period: within 2s (20 ticks) of a new decision, only switch
        if exit is critically blocked (>0.9). This prevents oscillation
        between the tactical layer and LLM/heuristic decisions.
        """
        target = agent.dynamic.target_exit
        if target is None:
            return

        target_smoke = float(env.smoke_at(target))

        # Grace period: recent decision → higher threshold for switching
        ticks_since_decision = tick - agent.dynamic.last_decision_tick
        block_threshold = cls.EXIT_SMOKE_BLOCK  # 0.7
        if ticks_since_decision < 20:
            block_threshold = 0.9  # Only switch if critically blocked

        if target_smoke <= block_threshold:
            return  # Exit still viable

        # Find best unblocked alternative
        pos = agent.dynamic.position
        best_idx = None
        best_score = float('inf')

        for i, ex in enumerate(exits):
            ex_arr = np.array(ex, dtype=np.float64)
            s = float(env.smoke_at(ex_arr))
            if s > cls.EXIT_SMOKE_BLOCK:
                continue  # Skip blocked
            dist = float(np.linalg.norm(pos - ex_arr))
            score = dist * (1.0 + s * 3.0)
            if score < best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            agent.dynamic.target_exit = np.array(exits[best_idx], dtype=np.float64)
            agent.dynamic.target_exit_idx = best_idx
            # Don't change target_exit_idx here — it's not stored on AgentDynamic
            # (kept in sync above; the heuristic decision block refreshes it
            #  at the next decision time)

    # ==================================================================
    # Heuristic decision (for non-critical agents at decision boundary)
    # ==================================================================

    @classmethod
    def heuristic_decision(cls, agent: Agent, env: EnvironmentSnapshot,
                           exits: List[Tuple[float, float]],
                           tick: int = 0) -> dict:
        """Generate a heuristic exit choice for non-critical agents.

        Delegates to the unified HeuristicPolicy so all decision sources
        share one scoring formula.
        """
        return HeuristicPolicy(
            smoke_block=cls.EXIT_SMOKE_BLOCK
        ).decide(agent, env, exits, tick=tick)

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _fire_distance(pos: np.ndarray, env: EnvironmentSnapshot) -> float:
        """Distance from position to nearest fire cell."""
        fire_mask = env.grid[:, :, 3] > 0.3
        if not fire_mask.any():
            return 200.0
        rows, cols = np.where(fire_mask)
        fire_x = cols * env.grid_resolution
        fire_y = rows * env.grid_resolution
        dists = np.sqrt((fire_x - pos[0])**2 + (fire_y - pos[1])**2)
        return float(dists.min())

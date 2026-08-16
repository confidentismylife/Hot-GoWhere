"""RL Zone Scheduler — MAPPO-style (CTDE) zone-level evacuation scheduling.

Receives IRL-learned reward weights from LLM behavior, then optimizes
zone-level exit recommendations with a multi-agent PPO setup:

Architecture (CTDE):
  - Decentralized actors: one policy MLP per zone. At inference each zone
    acts only on its own observation (< 1ms/tick per zone).
  - Centralized critic: a shared state-value MLP over the concatenated
    observations of ALL zones, used only during offline training.
  - Team reward: mean of the per-zone IRL-weighted rewards. Each zone's
    policy is updated with its own GAE advantage that shares the central
    value function.
  - The joint policy is factorized Gaussian (product of per-zone policies),
    so the joint log-probability is the sum of per-zone log-probs.

Historical naming: earlier docs called this "P-MAPPO". With the shared
central critic and decentralized actors below, the CTDE label is now
accurate, though the policy remains factorized rather than a full joint
action model.

The key insight: RL schedulers don't control individual agents directly.
They output natural-language "advice" that gets injected into LLM prompts —
LLM agents still make the final decision, but with global-optimal guidance.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from perception.environment import EnvironmentSnapshot
from execution.features import (
    FEATURE_NAMES,
    blocked_exit_weight,
    zone_reward_features,
)

# ================================================================
# Training constants (overridable via config)
# ================================================================
GAMMA = 0.99               # Discount factor for returns
LAMBDA_GAE = 0.95          # GAE trace decay parameter
CLIP_EPSILON = 0.2          # PPO clipping range
VALUE_COEF = 0.5            # Value loss coefficient in total loss
ENTROPY_COEF = 0.01         # Entropy bonus coefficient
# NOTE: v4 analysis (rl_policy_v4 weights) showed the policy head W3_a
# never left its init scale (max dev ~0.01-0.02 after 500 episodes)
# while the value head moved ~0.5 — the actor was optimization-starved,
# not reward-misaligned alone. lr 3e-4 with per-sample normalization and
# max_norm=1.0 clipping gave per-step updates of ~1e-6. Raised defaults;
# a local 10-episode smoke at lr=1e-3 reached only delta ~0.003/ep
# (would need ~500 eps to matter), so lr was raised further to 3e-3.
LEARNING_RATE = 3e-3        # Default learning rate (was 3e-4)
GRAD_MAX_NORM = 5.0         # Policy gradient clip (was 1.0)
CRITIC_EPOCHS = 4           # Critic SGD passes per episode (was 1)
BATCH_SIZE = 32             # Mini-batch size for PPO updates
POLICY_SIGMA = 0.5          # Fixed Gaussian policy standard deviation
PPO_EPOCHS = 4              # Reuse each on-policy rollout for stable PPO updates
HIDDEN_DIM = 64             # Hidden layer dimension for dual-head MLP
NUM_EPISODES = 200          # Default training episodes
STEPS_PER_EPISODE = 200     # Default steps per episode
RL_EXIT_SMOKE_BLOCK = 0.6   # Same as SafetyGuard.EXIT_SMOKE_BLOCK

# ================================================================
# Zone definitions
# ================================================================

@dataclass
class ZoneDefinition:
    """One zone (quadrant) of the mall floorplan."""
    zone_id: int
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    primary_exits: List[int]    # Exit indices most relevant to this zone
    description: str = ""       # Chinese description for NL output

# Default zone layout for Chaoyang Joycity 1F (150m × 80m)
DEFAULT_ZONES = [
    ZoneDefinition(0, "西北区", 0,   75,  40, 80,  [0, 1, 4, 5],
                   "商场西北区域，靠近ZARA、H&M，主要使用北侧和西侧出口"),
    ZoneDefinition(1, "东北区", 75,  150, 40, 80,  [1, 2, 3, 7],
                   "商场东北区域，靠近优衣库、海底捞，主要使用北侧和东侧出口"),
    ZoneDefinition(2, "西南区", 0,   75,  0,  40,  [4, 5, 6],
                   "商场西南区域，靠近星巴克、Apple，主要使用西侧和南侧出口"),
    ZoneDefinition(3, "东南区", 75,  150, 0,  40,  [3, 6, 7],
                   "商场东南区域，靠近华为、餐饮区，主要使用东侧和南侧出口"),
]


# ================================================================
# Zone observation and action spaces
# ================================================================

@dataclass
class ZoneObservation:
    """What a zone scheduler sees at each tick."""
    # Global exit states
    exit_smoke: List[float]        # Smoke at each exit [0-1]
    exit_crowd: List[int]          # Agents heading to each exit
    exit_fire_distance: List[float]  # Fire distance to each exit (m)

    # Zone-local state
    agent_count: int               # Total agents in zone
    avg_fear: float                # Mean fear level in zone
    avg_stamina: float             # Mean stamina in zone
    trained_ratio: float           # Fraction of trained staff in zone
    elderly_ratio: float           # Fraction of elderly (>55) in zone
    avg_smoke: float               # Mean smoke level in zone
    fire_distance: float           # Distance to nearest fire (m)

    # History encoding (for GRU)
    prev_exit_usage: List[float]   # Exit usage from previous tick

    def to_vector(self) -> np.ndarray:
        """Flatten observation to a fixed-size vector for RL input."""
        return np.array(
            self.exit_smoke +                          # N_exits floats
            [c / max(1, sum(self.exit_crowd) / len(self.exit_crowd))
             for c in self.exit_crowd] +               # N_exits normalized
            self.exit_fire_distance +                  # N_exits floats
            [self.agent_count / 200.0,                 # normalized by ~max per zone
             self.avg_fear / 10.0,
             self.avg_stamina / 100.0,
             self.trained_ratio,
             self.elderly_ratio,
             self.avg_smoke,
             self.fire_distance / 100.0] +
            self.prev_exit_usage,
            dtype=np.float32,
        )


@dataclass
class ZoneAction:
    """What a zone scheduler outputs — exit preference scores."""
    exit_preferences: List[float]  # [-1, 1] for each exit, -1=avoid, +1=recommend

    def to_recommendation_text(self, zone: ZoneDefinition,
                                exit_count: int,
                                blocked_exits=None) -> str:
        """Convert action to natural language for LLM prompt injection.

        ``blocked_exits`` is a list of exit indices (0-based) that must not be
        recommended — typically exits whose smoke exceeds the safety-guard
        threshold. Their preference is hard-masked to -1.0 so the RL advice
        never pushes the LLM toward a smoke-blocked exit.
        """
        prefs = list(self.exit_preferences[:exit_count])
        blocked = sorted(
            int(i) for i in (blocked_exits or [])
            if 0 <= int(i) < len(prefs)
        )
        for i in blocked:
            prefs[i] = -1.0

        lines = [f"【{zone.name}调度中心建议】"]

        # Sort exits by preference
        ranked = sorted(
            enumerate(prefs),
            key=lambda x: x[1], reverse=True
        )

        recommend = []
        avoid = []
        for exit_idx, pref in ranked:
            if exit_idx in blocked:
                continue  # blocked exits are announced once in the 封锁 line
            exit_label = f"出口{exit_idx + 1}"
            if pref > 0.5:
                recommend.append(f"强烈推荐 {exit_label}")
            elif pref > 0:
                recommend.append(f"建议考虑 {exit_label}")
            elif pref < -0.5:
                avoid.append(f"避免前往 {exit_label}（严重拥堵或危险）")
            elif pref < -0.2:
                avoid.append(f"{exit_label} 较为拥堵")

        if recommend:
            lines.append("  [推荐] " + "、".join(recommend))
        if avoid:
            lines.append("  [警告] " + "、".join(avoid))
        if blocked:
            lines.append("  [封锁] " + "、".join(
                f"出口{i + 1}（浓烟封锁或火路阻断，请勿前往）" for i in blocked))
        if not recommend and not avoid and not blocked:
            lines.append("  暂无特别建议，各出口均可通行")

        return "\n".join(lines)


# ================================================================
# Centralized critic (MAPPO / CTDE)
# ================================================================

class CentralizedCritic:
    """Shared state-value network used only during centralized training.

    Input: concatenated observations of ALL zones (one vector per time step).
    Output: scalar state value V(s). The critic is not used at execution
    time — each zone's actor acts on its own observation only (decentralized
    execution), which is the CTDE pattern.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / max(1, obs_dim))
        scale2 = np.sqrt(2.0 / max(1, hidden_dim))
        self.W1 = rng.randn(obs_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        self.W3 = rng.randn(hidden_dim, 1).astype(np.float32) * 0.01
        self.b3 = np.zeros(1, dtype=np.float32)

    def forward(self, obs_concat: np.ndarray) -> float:
        """Value estimate for a concatenated global observation."""
        h1 = np.maximum(0, obs_concat.reshape(1, -1) @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        return float((h2 @ self.W3 + self.b3).item())

    def update(self, obs_list: List[np.ndarray], returns: List[float],
               lr: float = 3e-4, max_norm: float = 10.0) -> float:
        """One SGD step minimizing 0.5 * (V(s) - R)^2 (gradient clipping)."""
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)
        dW3 = np.zeros_like(self.W3)
        db3 = np.zeros_like(self.b3)
        total_loss = 0.0

        for obs, ret in zip(obs_list, returns):
            x = obs.reshape(1, -1)
            h1_pre = x @ self.W1 + self.b1
            h1 = np.maximum(0, h1_pre)
            h2_pre = h1 @ self.W2 + self.b2
            h2 = np.maximum(0, h2_pre)
            v = float((h2 @ self.W3 + self.b3).item())

            err = v - ret
            total_loss += 0.5 * err * err

            dL_dv = err
            dW3 += h2.T * dL_dv
            db3 += dL_dv
            dL_dh2 = dL_dv * self.W3.T
            dL_dh2_pre = dL_dh2 * (h2_pre > 0)
            dW2 += h1.T @ dL_dh2_pre
            db2 += dL_dh2_pre.sum(axis=0)
            dL_dh1 = dL_dh2_pre @ self.W2.T
            dL_dh1_pre = dL_dh1 * (h1_pre > 0)
            dW1 += x.T @ dL_dh1_pre
            db1 += dL_dh1_pre.sum(axis=0)

        n = max(1, len(obs_list))
        # The loss is averaged over the batch, so the gradient must be too.
        # Without this normalization, the effective learning rate grows with
        # episode length and the clip threshold hides the resulting instability.
        all_grads = [dW1, db1, dW2, db2, dW3, db3]
        for grad in all_grads:
            grad /= n
        total_norm = float(np.sqrt(sum(np.sum(g * g) for g in all_grads)))
        scale = min(1.0, max_norm / (total_norm + 1e-8))
        self.W1 -= lr * dW1 * scale
        self.b1 -= lr * db1 * scale
        self.W2 -= lr * dW2 * scale
        self.b2 -= lr * db2 * scale
        self.W3 -= lr * dW3 * scale
        self.b3 -= lr * db3 * scale
        return total_loss / n


# ================================================================
# RL Zone Scheduler (zone-level PPO, lightweight MLP)
# ================================================================

class RLZoneScheduler:
    """Zone-level scheduler using MAPPO-style CTDE.

    Decentralized actors: one lightweight MLP policy per zone.
    Centralized critic: shared value MLP over all zone observations
    (training only). Offline training with PPO + GAE.

    Runtime: < 1ms per zone (4 × MLP forward pass).
    """

    def __init__(self, zones: List[ZoneDefinition] = None,
                 num_exits: int = 8,
                 obs_dim: int = None,
                 hidden_dim: int = 128,
                 seed: int = 42,
                 blocked_exit_penalty: float = 0.0,
                 smoke_block_threshold: float = 0.6,
                 outcome_reward_weight: float = 0.0):
        self.zones = zones or DEFAULT_ZONES
        self.num_exits = num_exits
        self.obs_dim = obs_dim or (4 * num_exits + 7)  # smoke + crowd + fire_dist + prev_usage + 7 scalars
        self.hidden_dim = hidden_dim
        self.seed = int(seed)
        self.blocked_exit_penalty = blocked_exit_penalty
        self.smoke_block_threshold = smoke_block_threshold
        self.outcome_reward_weight = outcome_reward_weight
        self._rng = np.random.RandomState(self.seed)

        # Policy networks (one per zone, or shared with zone-specific head)
        # For now: shared backbone + zone-specific output heads
        self._policies: Dict[int, dict] = {}  # zone_id → {W1, b1, W2, b2, W3, b3}
        self._initialized = False
        self._prev_exit_usage: Dict[int, List[float]] = {}  # zone_id → [counts]

        # IRL reward weights (loaded from IRLRecovery)
        self._irl_weights: Dict[str, np.ndarray] = {}
        self._use_irl_reward = False

        # Pretrained flag: only use MLP if pretrained weights loaded
        self._pretrained_loaded = False

        # Centralized critic (MAPPO / CTDE); created during training.
        self._critic: Optional[CentralizedCritic] = None

    def initialize(self, pretrained_path: str = None):
        """Initialize policy networks.

        Args:
            pretrained_path: Optional path to load pretrained weights.
        """
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_weights(pretrained_path)
            self._initialized = True
            print(f"[RLScheduler] Loaded pretrained weights from {pretrained_path}")
        else:
            # Initialize with random weights (heuristic baselines)
            for zone in self.zones:
                # Give zones independent initial noise so identical early
                # observations do not force permanently identical actors.
                self._policies[zone.zone_id] = self._init_network(
                    seed=self.seed + zone.zone_id
                )
                self._prev_exit_usage[zone.zone_id] = [0.0] * self.num_exits
            self._initialized = True
            if pretrained_path:
                print(f"[RLScheduler] WARNING: pretrained weights NOT FOUND at "
                      f"{pretrained_path} — using heuristic baselines")
            else:
                print(f"[RLScheduler] Initialized with heuristic baselines "
                      f"({len(self.zones)} zones, {self.num_exits} exits)")

    def load_irl_weights(self, irl_weights: Dict[str, np.ndarray]):
        """Load IRL-learned reward weights to guide scheduling decisions."""
        self._irl_weights = irl_weights
        self._use_irl_reward = True
        print(f"[RLScheduler] Loaded IRL weights for {len(irl_weights)} personas")

    def _init_network(self, seed: int = None) -> dict:
        """Initialize a small 3-layer MLP with separate policy and value heads.

        Shared backbone:
          Layer 1: obs_dim → hidden_dim  (ReLU)
          Layer 2: hidden_dim → hidden_dim (ReLU)

        Policy head (action means for Gaussian policy):
          Layer 3a: hidden_dim → num_exits (no activation during training)

        Value head (state-value estimate):
          Layer 3v: hidden_dim → 1 (scalar output)

        During inference: action_mean → tanh → [-1, 1] preferences.
        During training: action_mean used directly as Gaussian mean.
        """
        rng = np.random.RandomState(self.seed if seed is None else seed)
        scale1 = np.sqrt(2.0 / self.obs_dim)
        scale2 = np.sqrt(2.0 / self.hidden_dim)
        # Policy head: small init to start near zero (unbiased)
        scale3a = 0.01
        # Value head: small init
        scale3v = 0.01

        return {
            "W1": rng.randn(self.obs_dim, self.hidden_dim).astype(np.float32) * scale1,
            "b1": np.zeros(self.hidden_dim, dtype=np.float32),
            "W2": rng.randn(self.hidden_dim, self.hidden_dim).astype(np.float32) * scale2,
            "b2": np.zeros(self.hidden_dim, dtype=np.float32),
            # Policy head
            "W3_a": rng.randn(self.hidden_dim, self.num_exits).astype(np.float32) * scale3a,
            "b3_a": np.zeros(self.num_exits, dtype=np.float32),
            # Value head
            "W3_v": rng.randn(self.hidden_dim, 1).astype(np.float32) * scale3v,
            "b3_v": np.zeros(1, dtype=np.float32),
        }

    def infer(self, env_snapshot: EnvironmentSnapshot, agents: List,
              tick: int = 0) -> Dict[int, ZoneAction]:
        """Run inference for all zones. Returns zone_id → ZoneAction.

        This is called every tick during simulation. Must complete in < 1ms.

        Args:
            env_snapshot: Current environment state.
            agents: List of all agents (to compute zone statistics).
            tick: Current simulation tick.

        Returns:
            Dict mapping zone_id → ZoneAction (exit preferences).
        """
        if not self._initialized:
            self.initialize()

        # Partition agents into zones
        zone_agents = defaultdict(list)
        for a in agents:
            if not a.dynamic.alive or a.dynamic.evacuated:
                continue
            pos = a.position
            for zone in self.zones:
                if (zone.x_min <= pos[0] < zone.x_max and
                    zone.y_min <= pos[1] < zone.y_max):
                    zone_agents[zone.zone_id].append(a)
                    break

        results = {}
        for zone in self.zones:
            obs = self._build_observation(zone, zone_agents.get(zone.zone_id, []),
                                          env_snapshot)
            action = self._forward(zone.zone_id, obs)

            # Smooth with previous usage to avoid oscillations
            prev = self._prev_exit_usage.get(zone.zone_id, [0.0] * self.num_exits)
            smoothed = [0.7 * a + 0.3 * p for a, p in
                        zip(action.exit_preferences, prev)]
            action.exit_preferences = smoothed
            self._prev_exit_usage[zone.zone_id] = list(smoothed)

            results[zone.zone_id] = action

        return results

    def _build_observation(self, zone: ZoneDefinition,
                           zone_agents: List,
                           env: EnvironmentSnapshot) -> ZoneObservation:
        """Build observation vector for a zone."""
        n_exits = len(env.exits)

        # Exit states
        exit_smoke = []
        exit_crowd = []
        exit_fire_dist = []
        for i, ep in enumerate(env.exits):
            s = float(env.smoke_at(np.array(ep, dtype=np.float64)))
            exit_smoke.append(s)
            # Count agents heading to this exit from this zone
            crowd = sum(1 for a in zone_agents
                       if a.dynamic.target_exit is not None
                       and np.linalg.norm(a.dynamic.target_exit - np.array(ep)) < 2.0)
            exit_crowd.append(crowd)
            # Fire distance
            fd = self._fire_distance(np.array(ep), env)
            exit_fire_dist.append(fd / 100.0)

        # Zone-local stats
        n = len(zone_agents)
        if n > 0:
            avg_fear = np.mean([a.dynamic.fear_level for a in zone_agents])
            avg_stamina = np.mean([a.dynamic.stamina for a in zone_agents])
            trained = sum(1 for a in zone_agents
                         if a.profile.role in ("guide", "firefighter") or
                         a.profile.familiarity > 0.6)
            elderly = sum(1 for a in zone_agents if a.profile.age > 55)
            trained_ratio = trained / n
            elderly_ratio = elderly / n
            # Average smoke in zone (sample positions)
            avg_smoke = np.mean([
                float(env.smoke_at(a.position)) for a in zone_agents[:20]
            ])
            # Fire distance from zone center
            zone_center = np.array([
                (zone.x_min + zone.x_max) / 2,
                (zone.y_min + zone.y_max) / 2,
            ], dtype=np.float64)
            fire_dist = self._fire_distance(zone_center, env)
        else:
            avg_fear = 0
            avg_stamina = 100
            trained_ratio = 0
            elderly_ratio = 0
            avg_smoke = 0
            fire_dist = 100

        prev_usage = self._prev_exit_usage.get(zone.zone_id, [0.0] * n_exits)

        return ZoneObservation(
            exit_smoke=exit_smoke,
            exit_crowd=exit_crowd,
            exit_fire_distance=exit_fire_dist,
            agent_count=n,
            avg_fear=avg_fear,
            avg_stamina=avg_stamina,
            trained_ratio=trained_ratio,
            elderly_ratio=elderly_ratio,
            avg_smoke=avg_smoke,
            fire_distance=fire_dist,
            prev_exit_usage=prev_usage,
        )

    def _forward(self, zone_id: int, obs: ZoneObservation) -> ZoneAction:
        """MLP forward pass: observation → exit preferences [-1, 1].

        If no pretrained weights loaded, uses heuristic (random MLP would give
        garbage recommendations that mislead the LLM).
        """
        params = self._policies.get(zone_id)
        if params is None or not self._pretrained_loaded:
            return self._heuristic_action(obs)

        x = obs.to_vector().reshape(1, -1)

        # Layer 1
        h1 = np.maximum(0, x @ params["W1"] + params["b1"])  # ReLU
        # Layer 2
        h2 = np.maximum(0, h1 @ params["W2"] + params["b2"])  # ReLU
        # Policy head → action preferences (tanh → [-1, 1])
        action_mean = h2 @ params["W3_a"] + params["b3_a"]
        out = np.tanh(action_mean)

        preferences = out[0].tolist()[:self.num_exits]
        return ZoneAction(exit_preferences=preferences)

    def _sample_action(self, zone_id: int, obs: ZoneObservation,
                       sigma: float = POLICY_SIGMA) -> Tuple[ZoneAction, float]:
        """Sample an action from the Gaussian policy (on-policy training).

        Samples u ~ N(action_mean, sigma^2), applies tanh, and returns the
        action together with its log-probability (including the tanh
        Jacobian correction). Used by train_offline so the data actually
        comes from the policy being optimized.
        """
        params = self._policies[zone_id]
        x = obs.to_vector().reshape(1, -1)
        h1 = np.maximum(0, x @ params["W1"] + params["b1"])
        h2 = np.maximum(0, h1 @ params["W2"] + params["b2"])
        action_mean = h2 @ params["W3_a"] + params["b3_a"]

        noise = self._rng.randn(1, self.num_exits).astype(np.float32) * sigma
        u = action_mean + noise
        a = np.tanh(u)

        n_dims = self.num_exits
        log_prob = (
            -0.5 * n_dims * np.log(2 * np.pi * sigma ** 2)
            - 0.5 * np.sum((u - action_mean) ** 2) / sigma ** 2
            - float(np.sum(np.log(1.0 - a ** 2 + 1e-8)))
        )
        return ZoneAction(exit_preferences=a[0].tolist()), float(log_prob)

    def sample_actions(self, env_snapshot, agents, tick: int = 0,
                       sigma: float = POLICY_SIGMA) -> Tuple[Dict[int, ZoneAction], Dict[int, float]]:
        """On-policy rollout: sample one action + log-prob per zone."""
        zone_agents = self._partition_agents(agents)

        actions, log_probs = {}, {}
        for zone in self.zones:
            obs = self._build_observation(
                zone, zone_agents.get(zone.zone_id, []), env_snapshot)
            act, lp = self._sample_action(zone.zone_id, obs, sigma)
            actions[zone.zone_id] = act
            log_probs[zone.zone_id] = lp
            # The next observation should contain the action actually taken,
            # not a stale value from a previous episode or deployment run.
            self._prev_exit_usage[zone.zone_id] = list(act.exit_preferences)
        return actions, log_probs

    def _reset_action_history(self):
        """Reset recurrent-like recommendation features at episode start."""
        self._prev_exit_usage = {
            zone.zone_id: [0.0] * self.num_exits for zone in self.zones
        }

    def _partition_agents(self, agents) -> Dict[int, list]:
        """Group alive/non-evacuated agents into their zone."""
        zone_agents = defaultdict(list)
        for a in agents:
            if not a.dynamic.alive or a.dynamic.evacuated:
                continue
            pos = a.position
            for zone in self.zones:
                if (zone.x_min <= pos[0] < zone.x_max and
                    zone.y_min <= pos[1] < zone.y_max):
                    zone_agents[zone.zone_id].append(a)
                    break
        return zone_agents

    def _heuristic_action(self, obs: ZoneObservation) -> ZoneAction:
        """Heuristic fallback: prefer exits with low smoke and low crowd.

        If IRL weights are loaded, use them to weight the features.
        Otherwise use equal weighting.
        """
        n = len(obs.exit_smoke)
        prefs = []

        # If we have IRL weights, use the mean across personas for zone-level decisions
        if self._use_irl_reward and self._irl_weights:
            # Average weights across all personas for zone-level decision
            all_w = np.mean([w for w in self._irl_weights.values()], axis=0)
            # Extract the safety+efficiency components (most relevant to exit choice)
            w_safety = all_w[0]   # safety weight
            w_efficiency = all_w[1]  # efficiency weight
        else:
            w_safety = 0.4
            w_efficiency = 0.4

        for i in range(n):
            # Score: high safety (low smoke, far from fire) + low crowd
            safety_score = (1.0 - obs.exit_smoke[i]) * 0.5 + obs.exit_fire_distance[i] * 0.5
            crowd_penalty = obs.exit_crowd[i] / max(1, sum(obs.exit_crowd) / n)
            # Normalize crowd penalty to [-0.5, 0.5] range
            crowd_norm = min(crowd_penalty / 3.0, 1.0)

            score = (w_safety * safety_score +
                     w_efficiency * (1.0 - crowd_norm))
            # Map [0, 1] → [-1, 1]
            prefs.append(2.0 * score - 1.0)

        return ZoneAction(exit_preferences=prefs)

    @staticmethod
    def _fire_distance(pos: np.ndarray, env: EnvironmentSnapshot) -> float:
        """Distance from position to nearest fire cell."""
        fire_mask = env.grid[:, :, 3] > 0.3
        if not fire_mask.any():
            return 100.0
        rows, cols = np.where(fire_mask)
        fire_x = cols * env.grid_resolution
        fire_y = rows * env.grid_resolution
        dists = np.sqrt((fire_x - pos[0])**2 + (fire_y - pos[1])**2)
        return float(dists.min())

    # ---- Persistence ----

    def save_weights(self, path: str):
        """Save policy network weights (NaN-safe)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {}
        nan_count = 0
        for zid, params in self._policies.items():
            zone_data = {}
            for k, v in params.items():
                arr = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
                nan_count += int(np.sum(np.isnan(v)))
                zone_data[k] = arr.tolist()
            data[str(zid)] = zone_data
        if nan_count > 0:
            print(f"[RLScheduler] WARNING: {nan_count} NaN values replaced with 0.0 "
                  f"during save — training may have diverged")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[RLScheduler] Policy weights saved to {path}")

    def _load_weights(self, path: str):
        """Load policy network weights, with backward compatibility for old format."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for zid_str, params_dict in data.items():
            zid = int(zid_str)
            loaded = {}
            # Backward compat: old format had "W3"/"b3" (single head)
            if "W3" in params_dict and "W3_a" not in params_dict:
                loaded["W3_a"] = np.array(params_dict["W3"], dtype=np.float32)
                loaded["b3_a"] = np.array(params_dict["b3"], dtype=np.float32)
                rng = np.random.RandomState(42)
                loaded["W3_v"] = rng.randn(self.hidden_dim, 1).astype(np.float32) * 0.01
                loaded["b3_v"] = np.zeros(1, dtype=np.float32)
                for key in ["W1", "b1", "W2", "b2"]:
                    if key in params_dict:
                        loaded[key] = np.array(params_dict[key], dtype=np.float32)
            else:
                loaded = {k: np.array(v, dtype=np.float32) for k, v in params_dict.items()}
            self._policies[zid] = loaded
            self._prev_exit_usage[zid] = [0.0] * self.num_exits
        self._pretrained_loaded = True

# ---- Training interface (offline, PPO-style) ----

    def train_offline(self, env_simulator, episodes: int = 500,
                      steps_per_episode: int = 360,
                      lr: float = LEARNING_RATE, save_path: str = None,
                      log_interval: int = 5,
                      history_path: str = None,
                      ppo_epochs: int = PPO_EPOCHS) -> dict:
        """Complete offline MAPPO (CTDE) training loop.

        Each episode:
          1. For every step, build per-zone observations from the PRE-step
             state, sample actions from the decentralized policies, and step
             the fast simulator with those actions.
          2. Compute per-zone IRL-weighted rewards; the team reward is their
             mean.
          3. At episode end, compute GAE advantages for every zone using the
             SHARED central critic (V over concatenated global observations),
             update each zone's policy with its own advantage, and update the
             critic with the mean per-zone returns.
        """
        history = {"episode": [], "mean_reward": [], "policy_loss": [],
                   "value_loss": [], "critic_loss": [], "evacuation_rate": [],
                   "casualty_rate": [], "episode_return": [],
                   "policy_head_delta": []}

        # On-policy + CTDE: network rollouts, shared central critic.
        self._pretrained_loaded = True
        self._rng = np.random.RandomState(self.seed)
        critic_obs_dim = self.obs_dim * len(self.zones)
        self._critic = CentralizedCritic(
            critic_obs_dim, hidden_dim=self.hidden_dim)
        low_delta_streak = 0

        for ep in range(episodes):
            # Reset simulator state
            self._reset_action_history()
            env_simulator.reset()
            policy_head_before = {
                zid: params["W3_a"].copy()
                for zid, params in self._policies.items()
            }

            # Experience buffers per zone
            buffers: Dict[int, dict] = {
                zid: {"obs": [], "obs_next": [], "acts": [],
                      "rewards": [], "log_probs": []}
                for zid in [z.zone_id for z in self.zones]
            }
            global_states = []       # concatenated pre-step obs (critic input)
            global_states_next = []  # concatenated post-step obs
            team_rewards = []

            total_reward = 0.0

            for step in range(steps_per_episode):
                evac_before = float(env_simulator.get_evacuation_rate())
                survival_before = float(
                    env_simulator.get_survival_rate()
                    if hasattr(env_simulator, "get_survival_rate") else 1.0
                )

                # Build pre-step snapshot from current simulator state
                pre_snap = _FastEnvSnapshot(
                    grid=env_simulator.grid,
                    grid_resolution=env_simulator.grid_res,
                    exits=env_simulator.exit_positions,
                    timestamp=env_simulator.tick * env_simulator.dt,
                    disaster_type="fire",
                    official_broadcast="",
                )
                pre_agents = _FastAgentList(
                    positions=env_simulator.agent_positions,
                    target_exits=env_simulator.agent_target_exits,
                    exit_positions=env_simulator.exit_positions,
                    alive=env_simulator.agent_alive,
                    evacuated=env_simulator.agent_evacuated,
                    fear=env_simulator.agent_fear,
                    stamina=env_simulator.agent_stamina,
                    age=env_simulator.agent_age,
                    trained=env_simulator.agent_trained,
                )

                zone_groups = self._partition_agents(pre_agents)
                pre_obs = {
                    z.zone_id: self._build_observation(
                        z, zone_groups.get(z.zone_id, []), pre_snap)
                    for z in self.zones
                }

                # RL inference for all zones (sampled, with log-probs)
                zone_actions, log_probs = self.sample_actions(
                    pre_snap, pre_agents, step)

                # Step simulator WITH zone actions so RL affects agent behavior
                env_snap, agents = env_simulator.step(env_simulator.dt, zone_actions)

                post_groups = self._partition_agents(agents)
                post_obs_all = {}
                step_rewards = []
                for zone in self.zones:
                    zid = zone.zone_id
                    z_agents = post_groups.get(zid, [])
                    post_obs = self._build_observation(zone, z_agents, env_snap)
                    post_obs_all[zid] = post_obs
                    act = zone_actions[zid]
                    reward = self._compute_zone_reward(zone, z_agents, env_snap, act)
                    # Log-prob was computed at sampling time (same policy).
                    old_lp = log_probs[zid]

                    buffers[zid]["obs"].append(pre_obs[zid])
                    buffers[zid]["obs_next"].append(post_obs)
                    buffers[zid]["acts"].append(act)
                    buffers[zid]["rewards"].append(reward)
                    buffers[zid]["log_probs"].append(old_lp)
                    step_rewards.append(reward)

                evac_after = float(env_simulator.get_evacuation_rate())
                survival_after = float(
                    env_simulator.get_survival_rate()
                    if hasattr(env_simulator, "get_survival_rate") else survival_before
                )
                # The feature reward describes exit quality. Add a bounded
                # transition term so the policy is also trained on the actual
                # evacuation objective and casualty risk.
                transition_bonus = (
                    5.0 * (evac_after - evac_before)
                    + 5.0 * (survival_after - survival_before)
                )
                team_reward = (
                    float(np.mean(step_rewards)) if step_rewards else 0.0
                ) + transition_bonus
                total_reward += team_reward * len(buffers)

                global_states.append(np.concatenate(
                    [pre_obs[z.zone_id].to_vector() for z in self.zones]))
                global_states_next.append(np.concatenate(
                    [post_obs_all[z.zone_id].to_vector() for z in self.zones]))
                team_rewards.append(team_reward)

            # End of episode: shared-critic GAE, then update policies + critic
            ep_policy_loss = 0.0
            ep_value_loss = 0.0
            n_policy_samples = 0
            T = steps_per_episode

            # Central critic values for every stored state
            V_t = [self._critic.forward(s) for s in global_states]
            V_next = [self._critic.forward(s) for s in global_states_next]

            # Team-level returns (critic target): GAE over the cooperative
            # reward. Every actor uses this same advantage so the actor and
            # centralized critic optimize the same objective.
            gae_team = 0.0
            team_advantages = [0.0] * T
            team_returns = [0.0] * T
            for t in reversed(range(T)):
                next_val = V_next[t] if t + 1 < T else 0.0
                delta = team_rewards[t] + GAMMA * next_val - V_t[t]
                gae_team = delta + GAMMA * LAMBDA_GAE * gae_team
                team_advantages[t] = gae_team
                team_returns[t] = gae_team + V_t[t]

            # Divide by std only — do NOT subtract the mean. Zero-mean
            # normalization cancels the (already weak) directional team
            # signal and was one of the causes of the v4 policy stall.
            adv_std = float(np.std(team_advantages)) + 1e-8
            normalized_advantages = [
                adv / adv_std for adv in team_advantages
            ]

            for zid in buffers:
                buf = buffers[zid]
                if not buf["obs"]:
                    continue

                n_steps = len(buf["rewards"])

                # PPO reuses this fixed on-policy buffer for a few epochs.
                # The stored old log-probabilities remain unchanged across
                # epochs; the ratio therefore provides the PPO trust region.
                for _ in range(max(1, int(ppo_epochs))):
                    indices = self._rng.permutation(n_steps)
                    for start in range(0, n_steps, BATCH_SIZE):
                        batch_idx = indices[start:start + BATCH_SIZE]
                        batch_obs = [buf["obs"][i] for i in batch_idx]
                        batch_acts = [buf["acts"][i] for i in batch_idx]
                        batch_adv = [normalized_advantages[i] for i in batch_idx]
                        batch_ret = [team_returns[i] for i in batch_idx]
                        batch_lp = [buf["log_probs"][i] for i in batch_idx]

                        loss = self._train_step(
                            zid, batch_obs, batch_acts,
                            batch_adv, batch_ret, batch_lp, lr
                        )
                        batch_size = len(batch_idx)
                        ep_policy_loss += loss["policy_loss"] * batch_size
                        ep_value_loss += loss["value_loss"] * batch_size
                        n_policy_samples += batch_size

            # Update the shared critic with team-level returns. A single
            # SGD step per episode under-fits V (value loss stalls ~10),
            # which keeps the GAE advantages too noisy to move the actor.
            critic_loss = 0.0
            for _ in range(CRITIC_EPOCHS):
                critic_loss = self._critic.update(
                    global_states, team_returns, lr=lr)

            n_zones = len(buffers)
            mean_r = total_reward / max(1, steps_per_episode * n_zones)
            evac_rate = env_simulator.get_evacuation_rate()

            history["episode"].append(ep)
            history["mean_reward"].append(float(mean_r))
            history["policy_loss"].append(
                float(ep_policy_loss / max(1, n_policy_samples))
            )
            history["value_loss"].append(
                float(ep_value_loss / max(1, n_policy_samples))
            )
            history["critic_loss"].append(critic_loss)
            history["evacuation_rate"].append(float(evac_rate))
            history["casualty_rate"].append(float(
                env_simulator.get_casualty_rate()
                if hasattr(env_simulator, "get_casualty_rate") else 0.0
            ))
            history["episode_return"].append(float(total_reward))
            history["policy_head_delta"].append(float(max(
                np.max(np.abs(self._policies[zid]["W3_a"]
                              - policy_head_before[zid]))
                for zid in policy_head_before
            )))

            # Optimization-stall monitor: if the policy head barely moves
            # for 100 consecutive episodes the reward/advantage signal is
            # too weak to matter (the v4 failure mode). Warn early instead
            # of burning the full budget and discovering it post-hoc.
            if history["policy_head_delta"][-1] < 1e-3:
                low_delta_streak += 1
                if low_delta_streak == 100:
                    print("[RL Train] WARNING: policy_head_delta < 1e-3 for "
                          "100 consecutive episodes — optimization likely "
                          "stalled. Consider raising lr or loosening "
                          "gradient clipping.")
            else:
                low_delta_streak = 0

            if ep % log_interval == 0 or ep == episodes - 1:
                print(f"[RL Train] Ep {ep:4d}/{episodes} | "
                      f"MeanR: {mean_r:+.4f} | "
                      f"PolicyLoss: {history['policy_loss'][-1]:.4f} | "
                      f"ValueLoss: {history['value_loss'][-1]:.4f} | "
                      f"CriticLoss: {critic_loss:.4f} | "
                      f"PolicyDelta: {history['policy_head_delta'][-1]:.6f} | "
                      f"Evac: {evac_rate:.1%}")

        if save_path:
            self.save_weights(save_path)
            history_path = history_path or f"{save_path}.history.json"

        if history_path:
            os.makedirs(
                os.path.dirname(history_path) or ".", exist_ok=True
            )
            payload = {
                "metadata": {
                    "format_version": 2,
                    "algorithm": "zone-ppo-ctde",
                    "seed": self.seed,
                    "episodes": episodes,
                    "steps_per_episode": steps_per_episode,
                    "learning_rate": lr,
                    "ppo_epochs": int(ppo_epochs),
                    "gamma": GAMMA,
                    "gae_lambda": LAMBDA_GAE,
                    "clip_epsilon": CLIP_EPSILON,
                    "policy_sigma": POLICY_SIGMA,
                    "obs_dim": self.obs_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_exits": self.num_exits,
                    "zone_ids": [zone.zone_id for zone in self.zones],
                    "simulator_seed": getattr(env_simulator, "seed", None),
                },
                "history": history,
            }
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        # After training completes, the network is the deployed policy:
        # enable MLP inference for the remainder of this process.
        self._pretrained_loaded = True

        return history

    def _train_step(self, zone_id: int, observations: List[ZoneObservation],
                    actions: List[ZoneAction],
                    advantages: List[float],
                    returns: List[float],
                    old_log_probs: List[float],
                    lr: float = LEARNING_RATE) -> dict:
        """Single PPO update for one zone with analytical backpropagation.

        Network (shared backbone + dual head):
          h1 = ReLU(x @ W1 + b1)
          h2 = ReLU(h1 @ W2 + b2)
          action_mean = h2 @ W3_a + b3_a   (Gaussian policy mean)
          value       = h2 @ W3_v + b3_v   (scalar state-value)

        Loss:
          policy_loss = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
          value_loss  = (R - V)^2
          total = mean(policy_loss + vf_coef * value_loss)

        Gradients computed analytically through all layers.
        """
        clip_epsilon = CLIP_EPSILON
        value_coef = VALUE_COEF
        sigma = POLICY_SIGMA
        sigma_sq = sigma ** 2

        params = self._policies[zone_id]
        W1, b1 = params["W1"], params["b1"]
        W2, b2 = params["W2"], params["b2"]
        W3_a, b3_a = params["W3_a"], params["b3_a"]
        W3_v, b3_v = params["W3_v"], params["b3_v"]

        batch_size = len(observations)
        if batch_size == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Accumulate gradients
        dW1 = np.zeros_like(W1);  db1 = np.zeros_like(b1)
        dW2 = np.zeros_like(W2);  db2 = np.zeros_like(b2)
        dW3_a = np.zeros_like(W3_a); db3_a = np.zeros_like(b3_a)
        dW3_v = np.zeros_like(W3_v); db3_v = np.zeros_like(b3_v)

        total_policy_loss = 0.0
        total_value_loss = 0.0

        for obs, act, adv, ret, old_lp in zip(
                observations, actions, advantages, returns, old_log_probs):
            x = obs.to_vector().reshape(1, -1)          # (1, obs_dim)
            target_action = np.array(act.exit_preferences[:self.num_exits],
                                     dtype=np.float32).reshape(1, -1)  # (1, n_exits)

            # ---- Forward pass (save intermediates for backprop) ----
            h1_pre = x @ W1 + b1                        # (1, hidden)
            h1 = np.maximum(0, h1_pre)                   # ReLU
            h2_pre = h1 @ W2 + b2                        # (1, hidden)
            h2 = np.maximum(0, h2_pre)                   # ReLU

            action_mean = h2 @ W3_a + b3_a               # (1, n_exits), pre-tanh

            # ---- Tanh correction: inference applies tanh(action_mean) → [-1,1] ----
            # target_action is post-tanh (collected from _forward output).
            # To compute correct log-prob, we invert tanh to get the pre-tanh
            # value that would produce target_action, then compute Gaussian
            # log-prob in the pre-tanh (unbounded) space.
            # a = tanh(u)  ⇒  u = arctanh(a) = 0.5 * log((1+a)/(1-a))
            target_clipped = np.clip(target_action, -0.9999, 0.9999)
            target_pre_tanh = np.arctanh(target_clipped)   # (1, n_exits)
            diff = target_pre_tanh - action_mean            # both pre-tanh
            # Tanh Jacobian correction: log p(a) = log N(u|μ,σ²) - Σ log(1 - tanh²(u))
            tanh_correction = float(np.sum(np.log(1.0 - target_clipped ** 2 + 1e-8)))

            value = float((h2 @ W3_v + b3_v).item())    # scalar

            # ---- Loss computation ----
            # Gaussian log prob in pre-tanh space + tanh correction
            n_dims = self.num_exits
            new_lp = float(-0.5 * n_dims * np.log(2 * np.pi * sigma_sq)
                          - 0.5 * np.sum(diff ** 2) / sigma_sq
                          - tanh_correction)

            # Keep the ratio numerically bounded. With a correct on-policy
            # rollout it should be close to 1 before the update; a large
            # value indicates a broken buffer or an overly large step.
            ratio = float(np.exp(np.clip(new_lp - old_lp, -10.0, 10.0)))
            # Use pre-computed GAE advantage (not raw reward)
            adv_scalar = float(adv)

            # PPO clipped objective: L = min(ratio * A, clip(ratio) * A)
            surr1 = ratio * adv_scalar
            surr2 = np.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_scalar
            policy_loss = -min(surr1, surr2)

            value_loss = (ret - value) ** 2

            total_policy_loss += policy_loss
            total_value_loss += value_loss

            # ---- Backward: gradient of policy loss w.r.t. action_mean ----
            # d(new_lp)/d(μ) = (u - μ) / σ²  where u = arctanh(a_target)
            # tanh_correction = -Σ log(1-a²) is independent of μ, so gradient is zero
            d_new_lp = diff / sigma_sq                     # (1, n_exits)

            # d(ratio)/d(μ) = ratio * d(new_lp)/d(μ)
            d_ratio = ratio * d_new_lp                     # (1, n_exits)

            # Determine which PPO branch is active
            # The unclipped branch is active when it is the minimum PPO
            # surrogate. This explicit condition handles both signs of A.
            use_unclipped = (
                (adv_scalar >= 0.0 and ratio <= 1.0 + clip_epsilon)
                or (adv_scalar < 0.0 and ratio >= 1.0 - clip_epsilon)
            )
            d_policy = (
                -adv_scalar * d_ratio
                if use_unclipped else np.zeros_like(action_mean)
            )

            dL_daction = d_policy / batch_size             # normalize by batch

            # ---- Backward: gradient of value loss w.r.t. value ----
            dL_dvalue = -2.0 * (ret - value) * value_coef / batch_size  # scalar

            # ---- Backprop through shared layers ----
            # dL/dh2 from action head
            dL_dh2_a = dL_daction @ W3_a.T                # (1, hidden)
            # dL/dh2 from value head
            dL_dh2_v = dL_dvalue * W3_v.T                 # (1, hidden)
            dL_dh2 = dL_dh2_a + dL_dh2_v

            # dL/d(h2_pre) through ReLU
            dL_dh2_pre = dL_dh2 * (h2_pre > 0)            # (1, hidden)

            # dL/dh1
            dL_dh1 = dL_dh2_pre @ W2.T                    # (1, hidden)
            dL_dh1_pre = dL_dh1 * (h1_pre > 0)            # (1, hidden)

            # ---- Accumulate parameter gradients ----
            # Policy head
            dW3_a += h2.T @ dL_daction                     # (hidden, n_exits)
            db3_a += dL_daction.sum(axis=0)                # (n_exits,)

            # Value head
            dW3_v += h2.T * dL_dvalue                      # (hidden, 1)
            db3_v += dL_dvalue                             # scalar → (1,)

            # Layer 2
            dW2 += h1.T @ dL_dh2_pre                       # (hidden, hidden)
            db2 += dL_dh2_pre.sum(axis=0)                  # (hidden,)

            # Layer 1
            dW1 += x.T @ dL_dh1_pre                        # (obs_dim, hidden)
            db1 += dL_dh1_pre.sum(axis=0)                  # (hidden,)

        # ---- Gradient clipping (prevent NaN from exploding gradients) ----
        all_grads = [dW1, db1, dW2, db2, dW3_a, db3_a, dW3_v, db3_v]
        total_norm = float(np.sqrt(sum(np.sum(g * g) for g in all_grads)))
        max_norm = GRAD_MAX_NORM
        scale = min(1.0, max_norm / (total_norm + 1e-8))
        dW1 *= scale; db1 *= scale; dW2 *= scale; db2 *= scale
        dW3_a *= scale; db3_a *= scale; dW3_v *= scale; db3_v *= scale

        # ---- Apply gradients (SGD) ----
        params["W1"] -= lr * dW1
        params["b1"] -= lr * db1
        params["W2"] -= lr * dW2
        params["b2"] -= lr * db2
        params["W3_a"] -= lr * dW3_a
        params["b3_a"] -= lr * db3_a
        params["W3_v"] -= lr * dW3_v
        params["b3_v"] -= lr * db3_v

        n = batch_size
        return {
            "policy_loss": float(total_policy_loss / n),
            "value_loss": float(total_value_loss / n),
            "entropy": float(0.5 * n_dims * np.log(2 * np.pi * np.e * sigma_sq)),
        }

    def _compute_zone_reward(self, zone: ZoneDefinition,
                             zone_agents: List,
                             env_snap, act: ZoneAction) -> float:
        """Compute reward for a zone's action using IRL-learned weights.

        Reward = Σ w_i * feature_i, where w_i are from IRL recovery.
        If no IRL weights loaded, uses default balanced weights.
        """
        n = len(zone_agents)
        if n == 0:
            return 0.0

        # Get IRL weights (mean across personas for zone-level)
        if self._use_irl_reward and self._irl_weights:
            all_w = np.mean([w for w in self._irl_weights.values()], axis=0)
        else:
            all_w = np.array([0.30, 0.35, 0.15, 0.10, 0.10])  # default balanced

        # Single source of truth: execution.features
        features = zone_reward_features(zone, zone_agents, env_snap, act)
        reward = float(np.dot(all_w, features))
        if self.blocked_exit_penalty > 0:
            sample_pos = np.asarray(
                [a.position for a in zone_agents[:40]], dtype=np.float64)
            smoke_b, path_b = _vectorized_blocked_exits(
                sample_pos, env_snap, self.smoke_block_threshold)
            blocked_any = smoke_b | path_b.any(axis=0)
            reward -= self.blocked_exit_penalty * blocked_exit_weight(
                env_snap, act, self.smoke_block_threshold,
                blocked_exits=np.where(blocked_any)[0].tolist())
        if self.outcome_reward_weight > 0:
            reward += self.outcome_reward_weight * self._outcome_shaping(
                zone_agents, env_snap, act)
        return reward

    def _outcome_shaping(self, zone_agents, env_snap,
                         act: ZoneAction = None) -> float:
        """Dense outcome-oriented shaping: closeness + survival + stamina.

        Ties the zone reward to actual evacuation progress instead of only
        the immediate quality of the recommendation vector.

        Closeness is measured against the softmax-weighted target point of
        the RECOMMENDED exits (not the arbitrary nearest exit), so the
        shaping assigns credit to the action itself. Stamina is scored as
        the fraction of agents with stamina > 30 (a survival indicator);
        using raw mean stamina would reward keeping agents idle, since
        stamina drains while moving.
        """
        sample = zone_agents[:20]
        m = len(sample)
        if m == 0:
            return 0.0
        exits = np.asarray(
            [(float(e[0]), float(e[1])) for e in env_snap.exits],
            dtype=np.float64)
        if len(exits) == 0:
            return 0.0
        grid_h, grid_w = env_snap.grid.shape[:2]
        maxd = max(1.0, np.hypot(grid_w * env_snap.grid_resolution,
                                 grid_h * env_snap.grid_resolution) / 2.0)

        # Softmax over the recommendation → weighted target point.
        n_exits = len(exits)
        if act is not None:
            prefs = np.asarray(
                act.exit_preferences[:n_exits], dtype=np.float64)
            if prefs.size < n_exits:
                prefs = np.pad(prefs, (0, n_exits - prefs.size),
                               constant_values=0.0)
            logits = (prefs - np.max(prefs)) / 0.5
            exit_weights = np.exp(logits)
            exit_weights /= max(exit_weights.sum(), 1e-12)
        else:
            exit_weights = np.ones(n_exits) / n_exits
        target = (exit_weights[:, None] * exits).sum(axis=0)

        dists = []
        smoke_sum = 0.0
        stamina_ok = 0
        for a in sample:
            p = np.asarray(a.position, dtype=np.float64)
            dists.append(float(np.linalg.norm(target - p)))
            smoke_sum += float(env_snap.smoke_at(p))
            if getattr(a.dynamic, "stamina", 100.0) > 30.0:
                stamina_ok += 1
        closeness = 1.0 - (float(np.mean(dists)) / maxd)
        return (0.4 * closeness
                + 0.3 * (1.0 - smoke_sum / m)
                + 0.3 * (stamina_ok / m))

    @staticmethod
    def _gaussian_log_prob(mean: np.ndarray, action: np.ndarray,
                           sigma: float = 1.0) -> float:
        """Log probability under isotropic Gaussian."""
        n = len(mean)
        diff = action - mean[:len(action)] if len(mean) >= len(action) else action - mean
        return float(-0.5 * n * np.log(2 * np.pi * sigma**2) -
                     np.sum(diff**2) / (2 * sigma**2))

    def train_step(self, zone_id: int, observations: List[ZoneObservation],
                   actions: List[ZoneAction], rewards: List[float],
                   old_log_probs: List[float]) -> dict:
        """Single training step — compatibility wrapper that treats
        rewards as both advantages and returns (no GAE pre-computation)."""
        return self._train_step(zone_id, observations, actions,
                                rewards, rewards, old_log_probs)


# ================================================================
# RL preference injector — converts zone actions to LLM prompt text
# ================================================================

def inject_rl_preferences(zone_actions: Dict[int, ZoneAction],
                          zones: List[ZoneDefinition],
                          agent,
                          env_snapshot: EnvironmentSnapshot,
                          smoke_block_threshold: float = RL_EXIT_SMOKE_BLOCK,
                          fire_path_block: bool = True) -> str:
    """Generate RL scheduling advice for a specific agent's LLM prompt.

    Finds which zone the agent is in and returns the zone's recommendation
    as natural language text that can be appended to the LLM context.

    Args:
        zone_actions: Zone ID → ZoneAction mapping from RL scheduler.
        zones: Zone definitions.
        agent: The specific agent receiving this advice.
        env_snapshot: Current environment (for zone localization).

    Returns:
        Natural language recommendation text, or empty string if agent
        is not in any zone.
    """
    pos = agent.position

    for zone in zones:
        if (zone.x_min <= pos[0] < zone.x_max and
            zone.y_min <= pos[1] < zone.y_max):
            if zone.zone_id in zone_actions:
                # Hard-mask exits that the safety guard would block later:
                # (1) exit itself is smoke-blocked, or
                # (2) the straight-line path to the exit crosses fire
                #     (mirrors SafetyGuard._check_fire_proximity).
                blocked = []
                for i, ep in enumerate(env_snapshot.exits):
                    ep_arr = np.array(ep, dtype=np.float64)
                    smoke_blocked = (
                        float(env_snapshot.smoke_at(ep_arr))
                        > smoke_block_threshold)
                    path_blocked = (
                        fire_path_block
                        and _path_crosses_fire(pos, ep_arr, env_snapshot))
                    if smoke_blocked or path_blocked:
                        blocked.append(i)
                return zone_actions[zone.zone_id].to_recommendation_text(
                    zone, len(env_snapshot.exits), blocked_exits=blocked)

    return ""


def _path_crosses_fire(agent_pos, exit_pos, env_snapshot,
                       max_steps: int = 400) -> bool:
    """Sample the straight line agent→exit and return True if it hits fire.

    Uses the same sampling density as SafetyGuard._check_fire_proximity
    (at least 2 samples per meter, capped at ``max_steps``) so the RL advice
    filter and the safety guard agree on what counts as a fire-blocked route.
    """
    dist = float(np.linalg.norm(exit_pos - agent_pos))
    steps = min(max(8, int(dist * 2)), max_steps)
    for t in range(steps + 1):
        alpha = t / steps
        px = agent_pos[0] + alpha * (exit_pos[0] - agent_pos[0])
        py = agent_pos[1] + alpha * (exit_pos[1] - agent_pos[1])
        if env_snapshot.is_on_fire(np.array([px, py], dtype=np.float64)):
            return True
    return False


def _vectorized_blocked_exits(positions, env_snapshot,
                              smoke_block_threshold: float = 0.6,
                              samples: int = 16):
    """Vectorized smoke + fire-path blocked-exit mask.

    Returns ``(smoke_blocked[E], path_blocked[M, E])``:
      - smoke_blocked: exit itself has smoke above the threshold;
      - path_blocked:  the straight line from an agent to that exit crosses
                       fire (same semantics as the inference-time filter and
                       SafetyGuard._check_fire_proximity).
    """
    exits = np.asarray(
        [(float(e[0]), float(e[1])) for e in env_snapshot.exits],
        dtype=np.float64)
    e_count = len(exits)
    smoke_blocked = np.asarray([
        float(env_snapshot.smoke_at(
            np.asarray(e, dtype=np.float64))) > smoke_block_threshold
        for e in env_snapshot.exits
    ], dtype=bool)

    m_count = len(positions)
    path_blocked = np.zeros((m_count, e_count), dtype=bool)
    if m_count == 0 or e_count == 0:
        return smoke_blocked, path_blocked

    pos = np.asarray(positions, dtype=np.float64).reshape(m_count, 2)
    seg = exits[None, :, :] - pos[:, None, :]              # (M,E,2)
    ts = np.linspace(0.0, 1.0, samples + 1)[None, None, :]
    pts = (pos[:, None, None, :] + seg[:, :, None, :] * ts[..., None])
    res = env_snapshot.grid_resolution
    cols = (pts[..., 0] / res).astype(np.int64)
    rows = (pts[..., 1] / res).astype(np.int64)
    h, w = env_snapshot.grid.shape[:2]
    valid = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    rr = np.clip(rows, 0, h - 1)
    cc = np.clip(cols, 0, w - 1)
    on_fire = env_snapshot.grid[rr, cc, 3] > 0.5
    on_fire = on_fire & valid
    path_blocked = on_fire.any(axis=2)
    return smoke_blocked, path_blocked


# ================================================================
# Fast training simulator (rule-based agents, no LLM)
# ================================================================

class FastTrainingSimulator:
    """Minimal simulation for offline RL training. Uses rule-based agents
    with heuristic exit choices — runs ~1000× faster than full LLM simulation.

    Provides the interface that RLZoneScheduler.train_offline() expects:
      reset() → None
      step(dt) → (EnvironmentSnapshot, List[Agent])
      get_evacuation_rate() → float
    """

    def __init__(self, width: float = 150.0, height: float = 80.0,
                 num_agents: int = 600, num_exits: int = 8,
                 zone_defs: List[ZoneDefinition] = None,
                 exit_positions: List[Tuple[float, float]] = None,
                 seed: int = 42,
                 fire_sources: List[Tuple[float, float]] = None,
                 spread_rate: float = None,
                 origin_jitter: float = 15.0,
                 smoke_block_threshold: float = 0.6,
                 advice_accept_rate: float = 0.6):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.num_exits = num_exits
        self.zones = zone_defs or DEFAULT_ZONES
        self.dt = 1.0   # Coarse step for fast RL training (10× fewer steps)
        self.tick = 0
        self.seed = int(seed)
        self._episode_index = 0
        # Extreme-mode parameters (None keeps the legacy spread behavior).
        self.spread_rate = spread_rate
        self.origin_jitter = origin_jitter
        self.smoke_block_threshold = smoke_block_threshold
        # Deployment-realism knob (route-A): in the real pipeline the RL
        # advice is injected into LLM prompts and only adopted with some
        # probability; SafetyGuard further overrides decisions (~31% in
        # measured runs). The training simulator must model this broken
        # action→outcome link, otherwise the learned policy assumes a
        # causal chain that does not exist at deployment time.
        self.advice_accept_rate = float(np.clip(advice_accept_rate, 0.0, 1.0))

        rng = np.random.RandomState(self.seed)
        self._rng = rng

        # Use provided exit positions, or generate evenly-spaced perimeter positions
        if exit_positions is not None:
            self.exit_positions = [(float(e[0]), float(e[1])) for e in exit_positions]
            self.num_exits = len(self.exit_positions)
        else:
            self.exit_positions = []
            margin = 5.0
            for i in range(num_exits):
                if i < num_exits / 2:
                    x = margin + (width - 2 * margin) * i / max(1, num_exits / 2 - 1)
                    y = margin
                else:
                    x = margin + (width - 2 * margin) * (i - num_exits / 2) / max(1, num_exits / 2 - 1)
                    y = height - margin
                self.exit_positions.append((x, y))

        # Simple grid for smoke/fire simulation
        self.grid_res = 0.5
        self.grid_w = int(width / self.grid_res) + 1
        self.grid_h = int(height / self.grid_res) + 1
        self.grid = np.zeros((self.grid_h, self.grid_w, 5), dtype=np.float32)

        # Fire origins — place near 2 exits to create genuine pressure
        if fire_sources is not None:
            self.fire_sources = [
                (float(ox), float(oy)) for ox, oy in fire_sources
            ]
            self.fire_origins = self.fire_sources
        elif len(self.exit_positions) >= 4:
            ex0 = np.array(self.exit_positions[0])
            ex3 = np.array(self.exit_positions[3])
            self.fire_sources = [
                (ex0[0] + 15.0, ex0[1] + 15.0),
                (ex3[0] - 15.0, ex3[1] - 15.0),
            ]
            self.fire_origins = self.fire_sources
        else:
            self.fire_sources = [(width / 2, height / 2)]
            self.fire_origins = self.fire_sources
        self._init_fire(rng)

        # Agent states (simple Position-Velocity agents)
        self.agent_positions = np.zeros((num_agents, 2), dtype=np.float32)
        self.agent_velocities = np.zeros((num_agents, 2), dtype=np.float32)
        self.agent_target_exits = np.zeros(num_agents, dtype=np.int32)
        self.agent_alive = np.ones(num_agents, dtype=bool)
        self.agent_evacuated = np.zeros(num_agents, dtype=bool)
        self.agent_fear = rng.uniform(2.0, 8.0, num_agents).astype(np.float32)
        self.agent_stamina = rng.uniform(60.0, 100.0, num_agents).astype(np.float32)
        self.agent_age = rng.uniform(18, 70, num_agents).astype(np.float32)
        self.agent_trained = rng.random(num_agents) < 0.1
        # Per-agent advice receptiveness: an agent only blends the zone
        # preference into its exit choice if it "adopts" the RL advice
        # (mimics the LLM following or ignoring prompt-injected guidance).
        # Drawn once per episode so the adoption set is stable within it.
        self.agent_accepts_advice = (
            rng.random(num_agents) < self.advice_accept_rate)

        # Initial positions (random, avoiding walls near center)
        for i in range(num_agents):
            self.agent_positions[i] = [
                rng.uniform(10, width - 10),
                rng.uniform(10, height - 10),
            ]
            self.agent_target_exits[i] = rng.randint(0, num_exits)

        self._initial_positions = self.agent_positions.copy()
        self._initial_stamina = self.agent_stamina.copy()
        self._initial_fear = self.agent_fear.copy()
        self._initial_target_exits = self.agent_target_exits.copy()

    def _init_fire(self, rng):
        # Fire and smoke are episode state. Clear the previous episode before
        # placing the new sources; otherwise reset() accumulates hazards.
        self.grid.fill(0.0)
        for ox, oy in self._sample_fire_origins(rng):
            fx = int(ox / self.grid_res)
            fy = int(oy / self.grid_res)
            for dy in range(-6, 7):
                for dx in range(-6, 7):
                    py, px = fy + dy, fx + dx
                    if 0 <= py < self.grid_h and 0 <= px < self.grid_w:
                        if dx*dx + dy*dy <= 36:
                            self.grid[py, px, 3] = 0.8 + rng.random() * 0.2
                            self.grid[py, px, 0] = 0.3 + rng.random() * 0.2

    def _sample_fire_origins(self, rng):
        """Jitter the base fire sources each episode (extreme mode)."""
        if self.origin_jitter <= 0:
            return list(self.fire_sources)
        return [
            (ox + rng.uniform(-self.origin_jitter, self.origin_jitter),
             oy + rng.uniform(-self.origin_jitter, self.origin_jitter))
            for ox, oy in self.fire_sources
        ]

    def reset(self):
        self._episode_index += 1
        self._rng = np.random.RandomState(self.seed + self._episode_index)
        self.tick = 0
        self.agent_positions = self._initial_positions.copy()
        self.agent_velocities.fill(0)
        self.agent_alive.fill(True)
        self.agent_evacuated.fill(False)
        self.agent_stamina = self._initial_stamina.copy()
        self.agent_fear = self._initial_fear.copy()
        # Re-sample the advice-adoption set each episode (uses the
        # episode-scoped rng, so adoption varies across episodes).
        self.agent_accepts_advice = (
            self._rng.random(self.num_agents) < self.advice_accept_rate)
        self.agent_target_exits = self._initial_target_exits.copy()
        self._init_fire(self._rng)

    def step(self, dt: float, zone_actions: dict = None):
        """Run one tick: spread fire/smoke, move agents.

        If zone_actions is provided (from RL scheduler), agents blend the zone's
        exit preferences with their own heuristic, creating the feedback loop
        that allows RL to influence evacuation outcomes.
        """
        self.tick += 1

        fire = self.grid[:, :, 3]
        if self.spread_rate is None:
            # Legacy fast spread (backward compatibility)
            sources = np.where(fire > 0.5, fire * 0.85, 0.0).astype(np.float32)
            up = np.zeros_like(fire); up[:-1, :] = sources[1:, :]
            down = np.zeros_like(fire); down[1:, :] = sources[:-1, :]
            left = np.zeros_like(fire); left[:, :-1] = sources[:, 1:]
            right = np.zeros_like(fire); right[:, 1:] = sources[:, :-1]
            rand_mask = self._rng.random(fire.shape).astype(np.float32) < 0.40
            incoming = np.where(
                rand_mask, np.maximum.reduce([up, down, left, right]), 0.0)
            self.grid[:, :, 3] = np.maximum(fire, incoming)
            self.grid[:, :, 0] = np.maximum(
                self.grid[:, :, 0] * 0.85, self.grid[:, :, 3] * 0.6)
        else:
            # Deployment-like spread: same 8-neighbour CA ignition
            # probability as DisasterSimulator (spread_rate m/s).
            ignite_prob = min(
                1.0, self.spread_rate * dt / (3.0 * self.grid_res))
            fire_mask = fire > 0.5
            h, w = fire_mask.shape
            padded = np.pad(fire_mask, 1, mode="constant",
                            constant_values=False)
            new_fire = fire_mask.copy()
            for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1),
                           (1,-1), (1,0), (1,1)]:
                neighbor = padded[1+dr:1+dr+h, 1+dc:1+dc+w]
                candidates = neighbor & ~fire_mask
                new_fire |= candidates & (
                    self._rng.random(fire_mask.shape) < ignite_prob)
            self.grid[:, :, 3] = new_fire.astype(np.float32)

            # Smoke: diffusion first, then production (matches
            # DisasterSimulator; production-before-diffusion would saturate
            # burning neighborhoods at ~0.5 instead of 1.0).
            smoke = self.grid[:, :, 0]
            smoke_pad = np.pad(smoke, 1, mode="edge")
            smoke = (
                smoke * 0.6 +
                0.1 * (smoke_pad[2:, 1:-1] + smoke_pad[:-2, 1:-1] +
                       smoke_pad[1:-1, 2:] + smoke_pad[1:-1, :-2])
            )
            smoke = smoke + self.grid[:, :, 3] * 0.05 * dt
            self.grid[:, :, 0] = np.clip(smoke, 0.0, 1.0)

        # Deployment-like advice masking: when zone advice is active, block
        # exits whose smoke exceeds the threshold OR whose straight-line path
        # crosses fire — exactly what the inference-time advice filter does.
        blocked = None
        if zone_actions is not None:
            pre_snap = _FastEnvSnapshot(
                grid=self.grid,
                grid_resolution=self.grid_res,
                exits=self.exit_positions,
                timestamp=self.tick * dt,
                disaster_type="fire",
                official_broadcast="",
            )
            blocked = _vectorized_blocked_exits(
                self.agent_positions, pre_snap, self.smoke_block_threshold)

        # Move agents toward their chosen exits
        for i in range(self.num_agents):
            if not self.agent_alive[i] or self.agent_evacuated[i]:
                continue

            target = np.array(self.exit_positions[self.agent_target_exits[i]],
                             dtype=np.float32)
            direction = target - self.agent_positions[i]
            dist = float(np.linalg.norm(direction))

            if dist < 2.0:
                self.agent_evacuated[i] = True
                continue

            # Speed depends on stamina and age
            base_speed = 1.0
            if self.agent_stamina[i] < 30:
                base_speed = 0.5
            elif self.agent_age[i] > 55:
                base_speed = 0.7

            velocity = direction / dist * base_speed
            self.agent_velocities[i] = velocity
            self.agent_positions[i] += velocity * dt

            # Stamina drain (scaled for dt=1.0)
            self.agent_stamina[i] -= 0.2 * base_speed

            # Check fire proximity + lethal smoke
            gx = int(self.agent_positions[i, 0] / self.grid_res)
            gy = int(self.agent_positions[i, 1] / self.grid_res)
            if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                if self.grid[gy, gx, 3] > 0.5 or self.grid[gy, gx, 0] > 0.7:
                    self.agent_alive[i] = False

            # Re-evaluate exit choice every 3 ticks
            if self.tick % 3 == 0:
                best_exit = self.agent_target_exits[i]
                best_score = float("inf")
                all_blocked = True
                min_smoke_idx = best_exit
                min_smoke_val = float("inf")
                for e in range(self.num_exits):
                    epos = np.array(self.exit_positions[e], dtype=np.float32)
                    d = float(np.linalg.norm(self.agent_positions[i] - epos))
                    # Smoke at exit
                    ex = int(epos[0] / self.grid_res)
                    ey = int(epos[1] / self.grid_res)
                    smoke = 0.0
                    if 0 <= ex < self.grid_w and 0 <= ey < self.grid_h:
                        smoke = float(self.grid[ey, ex, 0])
                    if smoke < min_smoke_val:
                        min_smoke_val = smoke
                        min_smoke_idx = e
                    path_blocked = (
                        blocked is not None and blocked[1][i, e])
                    if smoke > self.smoke_block_threshold or path_blocked:
                        continue  # Smoke-blocked exit: skip unless all blocked
                    all_blocked = False
                    heuristic_score = d + smoke * 200

                    # Blend with zone recommendation if available — but only
                    # for agents that adopt the advice (route-A realism).
                    zone_pref = 0.0
                    accepts = self.agent_accepts_advice[i]
                    if zone_actions is not None and accepts:
                        for zone in self.zones:
                            if (zone.x_min <= self.agent_positions[i, 0] < zone.x_max and
                                zone.y_min <= self.agent_positions[i, 1] < zone.y_max):
                                act = zone_actions.get(zone.zone_id)
                                if act is not None:
                                    zone_pref = act.exit_preferences[e]
                                break
                    if accepts and zone_actions is not None:
                        # Blend: 50% heuristic + 50% zone preference
                        score = heuristic_score * 0.5 - zone_pref * 60.0
                    else:
                        # Non-adopters (and no-advice runs) follow pure
                        # heuristic — the deployment fallback behavior.
                        score = heuristic_score
                    if score < best_score:
                        best_score = score
                        best_exit = e
                if all_blocked:
                    best_exit = min_smoke_idx
                self.agent_target_exits[i] = best_exit

        # Build a minimal EnvironmentSnapshot-compatible object
        env_snap = _FastEnvSnapshot(
            grid=self.grid,
            grid_resolution=self.grid_res,
            exits=self.exit_positions,
            timestamp=self.tick * dt,
            disaster_type="fire",
            official_broadcast="",
        )

        # Build lightweight Agent-like objects for zone observation
        agents = _FastAgentList(
            positions=self.agent_positions,
            target_exits=self.agent_target_exits,
            exit_positions=self.exit_positions,
            alive=self.agent_alive,
            evacuated=self.agent_evacuated,
            fear=self.agent_fear,
            stamina=self.agent_stamina,
            age=self.agent_age,
            trained=self.agent_trained,
        )

        return env_snap, agents

    def get_evacuation_rate(self) -> float:
        n_evac = int(self.agent_evacuated.sum())
        return n_evac / max(1, self.num_agents)

    def get_survival_rate(self) -> float:
        """Fraction of agents still alive, including evacuated agents."""
        return float(self.agent_alive.sum()) / max(1, self.num_agents)

    def get_casualty_rate(self) -> float:
        """Fraction of agents killed by the simulated hazard."""
        casualties = (~self.agent_alive & ~self.agent_evacuated).sum()
        return float(casualties) / max(1, self.num_agents)


class _FastEnvSnapshot:
    """Minimal EnvironmentSnapshot-compatible object for training."""
    def __init__(self, grid, grid_resolution, exits, timestamp,
                 disaster_type, official_broadcast):
        self.grid = grid
        self.grid_resolution = grid_resolution
        self.exits = exits
        self.timestamp = timestamp
        self.disaster_type = disaster_type
        self.official_broadcast = official_broadcast

    def smoke_at(self, pos: np.ndarray) -> float:
        x = int(pos[0] / self.grid_resolution)
        y = int(pos[1] / self.grid_resolution)
        if 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]:
            return float(self.grid[y, x, 0])
        return 0.0

    def temperature_at(self, pos: np.ndarray) -> float:
        return 25.0

    def is_on_fire(self, pos: np.ndarray) -> bool:
        x = int(pos[0] / self.grid_resolution)
        y = int(pos[1] / self.grid_resolution)
        if 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]:
            return bool(self.grid[y, x, 3] > 0.5)
        return False


class _FastAgentList:
    """Minimal Agent list for training — duck-types the attributes RLZoneScheduler reads."""
    def __init__(self, positions, target_exits, exit_positions, alive,
                 evacuated, fear, stamina, age, trained):
        self.positions = positions
        self.target_exits = target_exits
        self._exit_positions = exit_positions
        self.alive = alive
        self.evacuated = evacuated
        self.fear = fear
        self.stamina = stamina
        self.age = age
        self.trained = trained

    def __iter__(self):
        for i in range(len(self.positions)):
            yield _FastAgent(
                i, self.positions[i], self.target_exits[i],
                self._exit_positions, self.alive[i], self.evacuated[i],
                self.fear[i], self.stamina[i], self.age[i], self.trained[i],
            )

    def __len__(self):
        return len(self.positions)


class _FastAgent:
    """Single duck-typed agent for RLZoneScheduler observation building."""
    __slots__ = ('idx', 'position', 'dynamic', 'profile')

    def __init__(self, idx, pos, target_exit, exit_positions, alive,
                 evacuated, fear, stamina, age, trained):
        self.idx = idx
        self.position = pos
        self.dynamic = _FastDynamic(target_exit, exit_positions, alive,
                                    evacuated, fear, stamina)
        self.profile = _FastProfile(age, trained)


class _FastDynamic:
    __slots__ = ('target_exit_idx', 'target_exit', 'alive', 'evacuated',
                 'fear_level', 'stamina')
    def __init__(self, target_exit_idx, exit_positions, alive, evacuated,
                 fear, stamina):
        self.target_exit_idx = target_exit_idx
        self.target_exit = np.array(exit_positions[target_exit_idx], dtype=np.float64)
        self.alive = alive
        self.evacuated = evacuated
        self.fear_level = fear
        self.stamina = stamina


class _FastProfile:
    __slots__ = ('age', 'familiarity', 'role')
    def __init__(self, age, trained):
        self.age = age
        self.familiarity = 0.7 if trained else 0.3
        self.role = "guide" if trained else "civilian"


# ================================================================
# Command-line interface
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RL Zone Scheduler")
    parser.add_argument("--mode", choices=["init", "train", "infer"], default="init",
                       help="Mode: init (create baseline), train (offline), infer (test)")
    parser.add_argument("--irl_weights", type=str, default=None,
                       help="Path to IRL-learned weights JSON")
    parser.add_argument("--output", type=str, default="data/rl_policy.json",
                       help="Output path for policy weights")
    parser.add_argument("--episodes", type=int, default=500,
                       help="Training episodes")
    parser.add_argument("--num_agents", type=int, default=600,
                       help="Agents for training simulator")
    parser.add_argument("--num_exits", type=int, default=8,
                       help="Number of exits")
    args = parser.parse_args()

    scheduler = RLZoneScheduler(zones=DEFAULT_ZONES, num_exits=args.num_exits)
    scheduler.initialize()

    if args.irl_weights and os.path.exists(args.irl_weights):
        from execution.irl_recovery import IRLRecovery
        irl = IRLRecovery()
        irl.load(args.irl_weights)
        scheduler.load_irl_weights(irl.weights)

    if args.mode == "init":
        scheduler.save_weights(args.output)

    elif args.mode == "train":
        print(f"[RLScheduler] Starting offline training ({args.episodes} episodes)...")
        sim = FastTrainingSimulator(
            num_agents=args.num_agents,
            num_exits=args.num_exits,
            zone_defs=DEFAULT_ZONES,
        )
        history = scheduler.train_offline(
            env_simulator=sim,
            episodes=args.episodes,
            steps_per_episode=360,
            save_path=args.output,
        )
        print(f"[RLScheduler] Training complete. Final evac rate: "
              f"{history['evacuation_rate'][-1]:.1%}")

    elif args.mode == "infer":
        print("[RLScheduler] Inference test — run from orchestrator with --rl-scheduling")

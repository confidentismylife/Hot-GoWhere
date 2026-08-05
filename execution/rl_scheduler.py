"""RL Zone Scheduler — P-MAPPO multi-agent zone-level evacuation scheduling.

Receives IRL-learned reward weights from LLM behavior, then optimizes
zone-level exit recommendations using MAPPO-style multi-agent RL.

Architecture:
  - 4 zone schedulers, each responsible for one quadrant of the mall
  - Each scheduler observes zone state (smoke, crowd, exits) and outputs
    exit preference scores [-1, 1] for each exit
  - Reward function uses IRL-learned weights per persona category
  - Training: offline (CTDE — centralized training, decentralized execution)
  - Inference: < 1ms per tick (lightweight MLP forward pass)

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

# ================================================================
# Training constants (overridable via config)
# ================================================================
GAMMA = 0.99               # Discount factor for returns
LAMBDA_GAE = 0.95          # GAE trace decay parameter
CLIP_EPSILON = 0.2          # PPO clipping range
VALUE_COEF = 0.5            # Value loss coefficient in total loss
ENTROPY_COEF = 0.01         # Entropy bonus coefficient
LEARNING_RATE = 3e-4        # Default learning rate
BATCH_SIZE = 32             # Mini-batch size for PPO updates
POLICY_SIGMA = 1.0          # Fixed Gaussian policy standard deviation
HIDDEN_DIM = 64             # Hidden layer dimension for dual-head MLP
NUM_EPISODES = 200          # Default training episodes
STEPS_PER_EPISODE = 200     # Default steps per episode

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
                                exit_count: int) -> str:
        """Convert action to natural language for LLM prompt injection."""
        lines = [f"【{zone.name}调度中心建议】"]

        # Sort exits by preference
        ranked = sorted(
            enumerate(self.exit_preferences[:exit_count]),
            key=lambda x: x[1], reverse=True
        )

        recommend = []
        avoid = []
        for exit_idx, pref in ranked:
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
        if not recommend and not avoid:
            lines.append("  暂无特别建议，各出口均可通行")

        return "\n".join(lines)


# ================================================================
# RL Zone Scheduler (P-MAPPO style, lightweight MLP)
# ================================================================

class RLZoneScheduler:
    """Zone-level scheduler using trained policy network.

    Uses a lightweight MLP policy (not a full transformer) for fast inference.
    Training is done offline with MAPPO-style centralized critic.

    Runtime: < 1ms per zone (4 × MLP forward pass).
    """

    def __init__(self, zones: List[ZoneDefinition] = None,
                 num_exits: int = 8,
                 obs_dim: int = None,
                 hidden_dim: int = 128):
        self.zones = zones or DEFAULT_ZONES
        self.num_exits = num_exits
        self.obs_dim = obs_dim or (4 * num_exits + 7)  # smoke + crowd + fire_dist + prev_usage + 7 scalars
        self.hidden_dim = hidden_dim

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

    def initialize(self, pretrained_path: str = None):
        """Initialize policy networks.

        Args:
            pretrained_path: Optional path to load pretrained weights.
        """
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_weights(pretrained_path)
        else:
            # Initialize with random weights (heuristic baselines)
            for zone in self.zones:
                self._policies[zone.zone_id] = self._init_network()
                self._prev_exit_usage[zone.zone_id] = [0.0] * self.num_exits

        self._initialized = True
        if pretrained_path:
            print(f"[RLScheduler] Loaded pretrained weights from {pretrained_path}")
        else:
            print(f"[RLScheduler] Initialized with heuristic baselines "
                  f"({len(self.zones)} zones, {self.num_exits} exits)")

    def load_irl_weights(self, irl_weights: Dict[str, np.ndarray]):
        """Load IRL-learned reward weights to guide scheduling decisions."""
        self._irl_weights = irl_weights
        self._use_irl_reward = True
        print(f"[RLScheduler] Loaded IRL weights for {len(irl_weights)} personas")

    def _init_network(self) -> dict:
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
        rng = np.random.RandomState(42)
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

    # ---- Training interface (offline, MAPPO-style) ----

    def train_offline(self, env_simulator, episodes: int = 500,
                      steps_per_episode: int = 360,
                      lr: float = 1e-4, save_path: str = None,
                      log_interval: int = 5) -> dict:
        """Complete offline training loop using rule-based fast simulator.

        Runs many episodes without LLM — uses heuristic agents for fast rollout.
        Each episode: simulate `steps_per_episode` ticks, collect experiences,
        compute IRL-weighted rewards, update policy via PPO.

        Args:
            env_simulator: A fast simulation environment (not the full orchestrator)
            episodes: Number of training episodes
            steps_per_episode: Simulation ticks per episode
            lr: Learning rate for weight updates
            save_path: Where to save final policy weights
            log_interval: Print loss stats every N episodes

        Returns:
            dict with training history (mean rewards, losses per episode)
        """
        history = {"episode": [], "mean_reward": [], "policy_loss": [],
                   "value_loss": [], "evacuation_rate": []}

        for ep in range(episodes):
            # Reset simulator state
            env_simulator.reset()

            # Experience buffers per zone
            buffers: Dict[int, dict] = {
                zid: {"obs": [], "acts": [], "rewards": [], "log_probs": []}
                for zid in [z.zone_id for z in self.zones]
            }

            total_reward = 0.0

            for step in range(steps_per_episode):
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

                # RL inference for all zones
                zone_actions = self.infer(pre_snap, pre_agents, step)

                # Step simulator WITH zone actions so RL affects agent behavior
                env_snap, agents = env_simulator.step(env_simulator.dt, zone_actions)

                # Compute rewards from IRL weights and agent outcomes
                for zone in self.zones:
                    zid = zone.zone_id
                    z_agents = [a for a in agents
                                if (a.dynamic.alive and not a.dynamic.evacuated and
                                    zone.x_min <= a.position[0] < zone.x_max and
                                    zone.y_min <= a.position[1] < zone.y_max)]

                    obs = self._build_observation(zone, z_agents, env_snap)
                    act = zone_actions[zid]
                    reward = self._compute_zone_reward(zone, z_agents, env_snap, act)

                    # Log prob of current action under behavior policy
                    # action_mean is pre-tanh; act.exit_preferences is post-tanh.
                    # Convert post-tanh action back to pre-tanh for correct log-prob.
                    params = self._policies[zid]
                    x = obs.to_vector().reshape(1, -1)
                    h1 = np.maximum(0, x @ params["W1"] + params["b1"])
                    h2 = np.maximum(0, h1 @ params["W2"] + params["b2"])
                    action_mean = h2 @ params["W3_a"] + params["b3_a"]
                    post_tanh_action = np.array(act.exit_preferences[:self.num_exits])
                    clipped = np.clip(post_tanh_action, -0.9999, 0.9999)
                    pre_tanh_action = np.arctanh(clipped)
                    old_lp = self._gaussian_log_prob(
                        action_mean[0], pre_tanh_action, sigma=POLICY_SIGMA)
                    # Tanh Jacobian correction
                    old_lp -= float(np.sum(np.log(1.0 - clipped ** 2 + 1e-8)))

                    buffers[zid]["obs"].append(obs)
                    buffers[zid]["acts"].append(act)
                    buffers[zid]["rewards"].append(reward)
                    buffers[zid]["log_probs"].append(old_lp)
                    total_reward += reward

            # End of episode: compute GAE advantages, then update policy
            ep_policy_loss = 0.0
            ep_value_loss = 0.0
            gamma_gae = GAMMA
            lambda_gae = LAMBDA_GAE

            for zid in buffers:
                buf = buffers[zid]
                if not buf["obs"]:
                    continue

                # Compute values for all states in buffer
                buf_values = []
                for obs in buf["obs"]:
                    params = self._policies[zid]
                    x = obs.to_vector().reshape(1, -1)
                    h1 = np.maximum(0, x @ params["W1"] + params["b1"])
                    h2 = np.maximum(0, h1 @ params["W2"] + params["b2"])
                    v = float((h2 @ params["W3_v"] + params["b3_v"]).item())
                    buf_values.append(v)

                # GAE: A_t = δ_t + γλ δ_{t+1} + (γλ)^2 δ_{t+2} + ...
                # δ_t = r_t + γ V(s_{t+1}) - V(s_t)
                T = len(buf["rewards"])
                gae = 0.0
                buf_advantages = [0.0] * T
                buf_returns = [0.0] * T
                for t in reversed(range(T)):
                    next_val = buf_values[t + 1] if t + 1 < T else 0.0
                    delta = buf["rewards"][t] + gamma_gae * next_val - buf_values[t]
                    gae = delta + gamma_gae * lambda_gae * gae
                    buf_advantages[t] = gae
                    buf_returns[t] = gae + buf_values[t]

                # Normalize advantages within buffer
                adv_mean = float(np.mean(buf_advantages))
                adv_std = float(np.std(buf_advantages)) + 1e-8
                buf_advantages = [(a - adv_mean) / adv_std for a in buf_advantages]

                # Train in mini-batches of 32
                n = T
                indices = np.random.permutation(n)
                for start in range(0, n, 32):
                    batch_idx = indices[start:start + 32]
                    batch_obs = [buf["obs"][i] for i in batch_idx]
                    batch_acts = [buf["acts"][i] for i in batch_idx]
                    batch_adv = [buf_advantages[i] for i in batch_idx]
                    batch_ret = [buf_returns[i] for i in batch_idx]
                    batch_lp = [buf["log_probs"][i] for i in batch_idx]

                    loss = self._train_step(zid, batch_obs, batch_acts,
                                            batch_adv, batch_ret, batch_lp, lr)
                    ep_policy_loss += loss["policy_loss"]
                    ep_value_loss += loss["value_loss"]

            n_zones = len(buffers)
            mean_r = total_reward / max(1, steps_per_episode * n_zones)
            evac_rate = env_simulator.get_evacuation_rate()

            history["episode"].append(ep)
            history["mean_reward"].append(mean_r)
            history["policy_loss"].append(ep_policy_loss / max(1, n_zones))
            history["value_loss"].append(ep_value_loss / max(1, n_zones))
            history["evacuation_rate"].append(evac_rate)

            if ep % log_interval == 0 or ep == episodes - 1:
                print(f"[RL Train] Ep {ep:4d}/{episodes} | "
                      f"MeanR: {mean_r:+.4f} | "
                      f"PolicyLoss: {history['policy_loss'][-1]:.4f} | "
                      f"ValueLoss: {history['value_loss'][-1]:.4f} | "
                      f"Evac: {evac_rate:.1%}")

        if save_path:
            self.save_weights(save_path)

        return history

    def _train_step(self, zone_id: int, observations: List[ZoneObservation],
                    actions: List[ZoneAction],
                    advantages: List[float],
                    returns: List[float],
                    old_log_probs: List[float], lr: float = 3e-4) -> dict:
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

            ratio = float(np.exp(np.clip(new_lp - old_lp, -20.0, 20.0)))
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
            ratio_clipped = (ratio < 1.0 - clip_epsilon - 1e-9 or
                            ratio > 1.0 + clip_epsilon + 1e-9)

            if surr1 <= surr2:
                # Unclipped branch: dL_p/dμ = -A * ratio * (a-μ)/σ²
                d_policy = -adv_scalar * d_ratio            # (1, n_exits)
            elif ratio_clipped:
                # Clipped branch, ratio outside bounds: gradient is 0
                d_policy = np.zeros_like(action_mean)
            else:
                # Ratio within bounds: clip(ratio)=ratio, same gradient as unclipped
                d_policy = -adv_scalar * d_ratio

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
        max_norm = 10.0
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

        # Compute features
        # f0: safety — low smoke + far from fire at recommended exits
        n_exits = len(env_snap.exits)
        top_exit = int(np.argmax(act.exit_preferences[:n_exits]))
        exit_smoke = float(env_snap.smoke_at(
            np.array(env_snap.exits[top_exit], dtype=np.float64)))
        safety = 1.0 - exit_smoke

        # f1: efficiency — recommended exit distance relative to nearest
        top_pos = np.array(env_snap.exits[top_exit], dtype=np.float64)
        avg_dist = np.mean([np.linalg.norm(a.position - top_pos) for a in zone_agents[:20]])
        efficiency = 1.0 - min(1.0, avg_dist / 150.0)

        # f2: social — low crowd at recommended exit
        crowd = sum(1 for a in zone_agents
                    if a.dynamic.target_exit is not None
                    and np.linalg.norm(a.dynamic.target_exit - top_pos) < 2.0)
        social = 1.0 - min(1.0, crowd / max(1, n))

        # f3: conformity — preference consistency with majority
        if zone_agents:
            majority_exit = max(
                set(a.dynamic.target_exit_idx for a in zone_agents
                    if a.dynamic.target_exit_idx is not None),
                key=lambda e: sum(1 for a in zone_agents
                                  if a.dynamic.target_exit_idx == e),
                default=top_exit,
            )
            conformity = 1.0 if top_exit == majority_exit else 0.0
        else:
            conformity = 0.5

        # f4: comfort — low smoke path
        comfort = safety  # Simplified: comfort ≈ safety of route

        features = np.array([safety, efficiency, social, conformity, comfort])

        # Normalize features to [0, 1]
        features = np.clip(features, 0.0, 1.0)

        return float(np.dot(all_w, features))

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
                          env_snapshot: EnvironmentSnapshot) -> str:
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
                return zone_actions[zone.zone_id].to_recommendation_text(
                    zone, len(env_snapshot.exits))

    return ""


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
                 seed: int = 42):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.num_exits = num_exits
        self.zones = zone_defs or DEFAULT_ZONES
        self.dt = 1.0   # Coarse step for fast RL training (10× fewer steps)
        self.tick = 0

        rng = np.random.RandomState(seed)

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
        if len(self.exit_positions) >= 4:
            ex0 = np.array(self.exit_positions[0])
            ex3 = np.array(self.exit_positions[3])
            self.fire_origins = [
                (ex0[0] + 15.0, ex0[1] + 15.0),
                (ex3[0] - 15.0, ex3[1] - 15.0),
            ]
        else:
            self.fire_origins = [(width / 2, height / 2)]
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

        # Initial positions (random, avoiding walls near center)
        for i in range(num_agents):
            self.agent_positions[i] = [
                rng.uniform(10, width - 10),
                rng.uniform(10, height - 10),
            ]
            self.agent_target_exits[i] = rng.randint(0, num_exits)

        self._initial_positions = self.agent_positions.copy()

    def _init_fire(self, rng):
        for ox, oy in self.fire_origins:
            fx = int(ox / self.grid_res)
            fy = int(oy / self.grid_res)
            for dy in range(-6, 7):
                for dx in range(-6, 7):
                    py, px = fy + dy, fx + dx
                    if 0 <= py < self.grid_h and 0 <= px < self.grid_w:
                        if dx*dx + dy*dy <= 36:
                            self.grid[py, px, 3] = 0.8 + rng.random() * 0.2
                            self.grid[py, px, 0] = 0.3 + rng.random() * 0.2

    def reset(self):
        self.tick = 0
        self.agent_positions = self._initial_positions.copy()
        self.agent_velocities.fill(0)
        self.agent_alive.fill(True)
        self.agent_evacuated.fill(False)
        self._init_fire(np.random.RandomState(42))

    def step(self, dt: float, zone_actions: dict = None):
        """Run one tick: spread fire/smoke, move agents.

        If zone_actions is provided (from RL scheduler), agents blend the zone's
        exit preferences with their own heuristic, creating the feedback loop
        that allows RL to influence evacuation outcomes.
        """
        self.tick += 1

        # Spread fire and smoke every tick (dt=1.0, so every 1.0 simulated second)
        fire = self.grid[:, :, 3]
        sources = np.where(fire > 0.5, fire * 0.85, 0.0).astype(np.float32)

        # Spread from 4 directions into target cell
        up = np.zeros_like(fire)
        up[:-1, :] = sources[1:, :]
        down = np.zeros_like(fire)
        down[1:, :] = sources[:-1, :]
        left = np.zeros_like(fire)
        left[:, :-1] = sources[:, 1:]
        right = np.zeros_like(fire)
        right[:, 1:] = sources[:, :-1]

        rand_mask = np.random.random(fire.shape).astype(np.float32) < 0.40
        incoming = np.where(rand_mask, np.maximum.reduce([up, down, left, right]), 0.0)

        self.grid[:, :, 3] = np.maximum(fire, incoming)
        self.grid[:, :, 0] = np.maximum(
            self.grid[:, :, 0] * 0.85,
            self.grid[:, :, 3] * 0.6,
        )

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
                for e in range(self.num_exits):
                    epos = np.array(self.exit_positions[e], dtype=np.float32)
                    d = float(np.linalg.norm(self.agent_positions[i] - epos))
                    # Smoke at exit
                    ex = int(epos[0] / self.grid_res)
                    ey = int(epos[1] / self.grid_res)
                    smoke = 0.0
                    if 0 <= ex < self.grid_w and 0 <= ey < self.grid_h:
                        smoke = float(self.grid[ey, ex, 0])
                    heuristic_score = d + smoke * 200

                    # Blend with zone recommendation if available
                    zone_pref = 0.0
                    if zone_actions is not None:
                        for zone in self.zones:
                            if (zone.x_min <= self.agent_positions[i, 0] < zone.x_max and
                                zone.y_min <= self.agent_positions[i, 1] < zone.y_max):
                                act = zone_actions.get(zone.zone_id)
                                if act is not None:
                                    zone_pref = act.exit_preferences[e]
                                break
                    # Blend: 50% heuristic + 50% zone preference
                    score = heuristic_score * 0.5 - zone_pref * 60.0
                    if score < best_score:
                        best_score = score
                        best_exit = e
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

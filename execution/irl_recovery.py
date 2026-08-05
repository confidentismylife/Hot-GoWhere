"""Inverse Reinforcement Learning — recover reward functions from LLM behavior.

LLM → IRL → RL cascade architecture:
  1. LLM generates diverse, human-like evacuation behavior data
  2. IRL learns implicit reward weights from behavior trajectories
  3. RL uses learned rewards to optimize zone-level scheduling

Core insight: LLM agents reveal what humans VALUE in a crisis — safety,
social bonds, familiarity, authority trust — not just what they DO.
IRL extracts these value weights, giving RL a "human-aligned" objective.

Algorithm: Discrete-State Maximum Entropy IRL (Ziebart et al., 2008)
  - State space discretized by binning 5 feature dimensions (≤243 states)
  - Transition model estimated from trajectory data
  - Soft value iteration computes policy under current weights
  - Gradient descent matches expert feature expectations

Extension: GC-MaxEnt (Group-Constrained MaxEnt IRL)
  - Joint optimization with cross-persona Laplacian regularization
  - Prevents strategy confusion across heterogeneous groups
  - λ_group controls strength of inter-persona weight coupling
"""

import json
import os
import sys
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


# ================================================================
# Data structures
# ================================================================

@dataclass
class AgentTrajectory:
    """Single agent's complete decision trajectory across one simulation run."""
    agent_id: str
    role: str                      # civilian / firefighter / guide / commander
    persona: str                   # derived persona: untrained_elderly / untrained_young /
                                   #   trained_staff / guide / firefighter
    age: int
    familiarity: float
    max_speed: float

    # Time-series of decision points
    decisions: List[dict] = field(default_factory=list)
    # Each decision dict:
    #   {tick, sim_time, smoke_at_pos, fire_distance, nearest_exit_dist,
    #    target_exit_idx, speed, cooperation, stamina, fear,
    #    reasoning_text, was_blocked, was_modified}

    outcome: str = "unknown"       # evacuated / dead / active_at_end
    evacuation_time: float = -1.0  # sim_time when evacuated, -1 if didn't


class TrajectoryCollector:
    """Collects decision trajectories during simulation.

    Usage (in orchestrator):
        collector = TrajectoryCollector()
        # In _apply_decisions, after each decision:
        collector.record_decision(agent, decision, env_snapshot, was_blocked, was_modified)
        # After simulation ends:
        collector.finalize(agents)
        collector.save("data/trajectories/run_001.jsonl")
    """

    def __init__(self):
        self._trajectories: Dict[str, AgentTrajectory] = {}
        self._persona_cache: Dict[str, str] = {}

    def _get_persona(self, agent) -> str:
        """Map agent profile to a persona category."""
        aid = agent.id
        if aid in self._persona_cache:
            return self._persona_cache[aid]

        role = agent.profile.role
        age = agent.profile.age
        familiarity = agent.profile.familiarity

        if role == "firefighter":
            persona = "firefighter"
        elif role == "guide":
            persona = "guide"
        elif role in ("global_commander", "area_commander"):
            persona = "commander"
        elif familiarity > 0.5:
            persona = "trained_staff"
        elif age > 55:
            persona = "untrained_elderly"
        else:
            persona = "untrained_young"

        self._persona_cache[aid] = persona
        return persona

    def record_decision(self, agent, decision, env_snapshot,
                        was_blocked: bool, was_modified: bool):
        """Record one LLM decision during simulation."""
        aid = agent.id

        if aid not in self._trajectories:
            persona = self._get_persona(agent)
            self._trajectories[aid] = AgentTrajectory(
                agent_id=aid,
                role=agent.profile.role,
                persona=persona,
                age=agent.profile.age,
                familiarity=agent.profile.familiarity,
                max_speed=agent.profile.max_speed,
            )

        traj = self._trajectories[aid]
        d = agent.dynamic
        pos = d.position.copy() if hasattr(d.position, 'copy') else np.array(d.position)

        # Compute features at decision time
        smoke_at_pos = float(env_snapshot.smoke_at(pos))
        fire_dist = self._fire_distance(pos, env_snapshot)
        nearest_exit = self._nearest_exit_info(pos, env_snapshot)

        traj.decisions.append({
            "tick": getattr(decision, 'tick', 0),
            "sim_time": env_snapshot.timestamp,
            "position": [float(pos[0]), float(pos[1])],
            "smoke_at_pos": smoke_at_pos,
            "fire_distance": fire_dist,
            "nearest_exit_idx": nearest_exit[0],
            "nearest_exit_dist": nearest_exit[1],
            "nearest_exit_smoke": nearest_exit[2],
            "target_exit_idx": getattr(decision, 'target_exit_idx', -1),
            "speed": getattr(decision, 'speed', 'walk').value
                     if hasattr(getattr(decision, 'speed', 'walk'), 'value') else 'walk',
            "cooperation": getattr(decision, 'cooperation', 'none').value
                          if hasattr(getattr(decision, 'cooperation', 'none'), 'value') else 'none',
            "stamina": float(d.stamina),
            "fear": float(d.fear_level),
            "was_blocked": was_blocked,
            "was_modified": was_modified,
        })

    def finalize(self, agents, dt: float = 0.1):
        """Fill in outcomes after simulation ends."""
        for a in agents:
            aid = a.id
            if aid in self._trajectories:
                d = a.dynamic
                if d.evacuated:
                    self._trajectories[aid].outcome = "evacuated"
                    self._trajectories[aid].evacuation_time = (
                        d.last_decision_tick * dt if d.last_decision_tick > 0 else -1
                    )
                elif not d.alive:
                    self._trajectories[aid].outcome = "dead"
                else:
                    self._trajectories[aid].outcome = "active_at_end"

    def save(self, path: str):
        """Save all trajectories as JSONL."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for traj in self._trajectories.values():
                f.write(json.dumps(self._to_dict(traj), ensure_ascii=False) + '\n')

    def load(self, path: str) -> List[AgentTrajectory]:
        """Load trajectories from JSONL file."""
        trajectories = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                traj = AgentTrajectory(
                    agent_id=data['agent_id'],
                    role=data['role'],
                    persona=data['persona'],
                    age=data['age'],
                    familiarity=data['familiarity'],
                    max_speed=data['max_speed'],
                    decisions=data.get('decisions', []),
                    outcome=data.get('outcome', 'unknown'),
                    evacuation_time=data.get('evacuation_time', -1),
                )
                trajectories.append(traj)
        return trajectories

    def load_all(self, directory: str) -> List[AgentTrajectory]:
        """Load all JSONL files in a directory."""
        all_trajs = []
        for fname in sorted(os.listdir(directory)):
            if fname.endswith('.jsonl'):
                all_trajs.extend(self.load(os.path.join(directory, fname)))
        return all_trajs

    # ---- helpers ----

    @staticmethod
    def _fire_distance(pos, env_snapshot) -> float:
        fire_mask = env_snapshot.grid[:, :, 3] > 0.3
        if not fire_mask.any():
            return 200.0  # Far away
        fire_rows, fire_cols = np.where(fire_mask)
        fire_x = fire_cols * env_snapshot.grid_resolution
        fire_y = fire_rows * env_snapshot.grid_resolution
        dists = np.sqrt((fire_x - pos[0])**2 + (fire_y - pos[1])**2)
        return float(dists.min())

    @staticmethod
    def _nearest_exit_info(pos, env_snapshot) -> Tuple[int, float, float]:
        best_idx, best_dist, best_smoke = 0, float('inf'), 0
        for i, ep in enumerate(env_snapshot.exits):
            d = float(np.linalg.norm(np.array(ep) - pos))
            if d < best_dist:
                best_dist = d
                best_idx = i
                best_smoke = float(env_snapshot.smoke_at(np.array(ep, dtype=np.float64)))
        return best_idx, best_dist, best_smoke

    @staticmethod
    def _to_dict(traj: AgentTrajectory) -> dict:
        return {
            "agent_id": traj.agent_id,
            "role": traj.role,
            "persona": traj.persona,
            "age": traj.age,
            "familiarity": traj.familiarity,
            "max_speed": traj.max_speed,
            "decisions": traj.decisions,
            "outcome": traj.outcome,
            "evacuation_time": traj.evacuation_time,
        }


# ================================================================
# IRL Recovery — Maximum Entropy IRL
# ================================================================

# Five reward features that define "good" evacuation behavior
FEATURE_NAMES = [
    "safety",       # Staying away from fire/smoke
    "efficiency",   # Moving toward nearest usable exit quickly
    "social",       # Helping others, staying with family
    "conformity",   # Following crowd / authority
    "comfort",      # Taking familiar routes, avoiding exertion
]

# Persona categories
PERSONA_CATEGORIES = [
    "untrained_elderly",
    "untrained_young",
    "trained_staff",
    "guide",
    "firefighter",
]


class IRLRecovery:
    """Discrete-State Maximum Entropy IRL for learning reward weights.

    Given agent trajectories, learns the reward weight vector for each persona
    that best explains observed behavior under the maximum entropy principle.

    Reward function: R(s,a) = w · φ(s,a) where φ = [safety, efficiency, social,
    conformity, comfort].

    Key improvement over naive Boltzmann re-weighting:
      - Discretizes state space into ≤243 bins (3 quantiles × 5 features)
      - Estimates transition model P(s'|s,a) from trajectory data
      - Runs soft value iteration to compute policy under current weights
      - Computes feature expectations analytically from the learned policy
      - GC-MaxEnt: adds cross-persona Laplacian regularization

    Reference: Ziebart et al. (2008) "Maximum Entropy Inverse Reinforcement
    Learning"; Abbeel & Ng (2004) "Apprenticeship Learning via Inverse
    Reinforcement Learning".
    """

    N_BINS = 5           # Quantile bins per feature dimension (5^5 = 3125 states)
    N_ACTIONS = 8         # Exit choices (8 exits)
    GAMMA = 0.95          # Discount factor for soft value iteration
    MAX_SOFT_VI_ITERS = 200  # Max iterations for soft VI inner loop
    MAX_V_ABS = 1e6       # Clip |V| to prevent overflow in log-sum-exp
    MAX_WEIGHT = 10.0     # Clip individual weights to prevent extreme values

    def __init__(self, learning_rate: float = 0.01, max_iter: int = 500,
                 tolerance: float = 1e-4, l2_reg: float = 0.01,
                 group_reg: float = 0.05,  # GC-MaxEnt: cross-persona coupling
                 n_bins: int = 3):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.l2_reg = l2_reg
        self.group_reg = group_reg
        self.N_BINS = n_bins
        self.weights: Dict[str, np.ndarray] = {}
        # Internal state per persona (built during fit)
        self._mdp_cache: Dict[str, dict] = {}

    # ================================================================
    # Public API
    # ================================================================

    def fit(self, trajectories: List[AgentTrajectory], verbose: bool = True):
        """Learn reward weights for each persona category.

        Uses GC-MaxEnt: fits all personas jointly with group regularization
        to prevent strategy confusion across heterogeneous groups.
        """
        grouped = defaultdict(list)
        for traj in trajectories:
            grouped[traj.persona].append(traj)

        # Phase 1: Build discretized MDP for each persona
        persona_mdps = {}
        for persona in PERSONA_CATEGORIES:
            if persona not in grouped or len(grouped[persona]) < 10:
                if verbose:
                    print(f"  [IRL] Skipping {persona}: {len(grouped.get(persona, []))} trajs")
                self.weights[persona] = self._default_weights(persona)
                continue
            persona_mdps[persona] = self._build_discrete_mdp(grouped[persona])

        if not persona_mdps:
            return

        # Phase 2: Joint GC-MaxEnt optimization
        # Initialize all weights uniformly
        active_personas = list(persona_mdps.keys())
        n_active = len(active_personas)
        weights_joint = {
            p: np.ones(len(FEATURE_NAMES)) / len(FEATURE_NAMES)
            for p in active_personas
        }

        for iteration in range(self.max_iter):
            max_change = 0.0

            # Compute policy feature expectations for all personas under current weights
            policy_fes = {}
            for persona in active_personas:
                mdp = persona_mdps[persona]
                w = weights_joint[persona]
                policy_fes[persona] = self._compute_policy_fe(
                    mdp, w)

            # Mean weight vector across personas (for GC regularization)
            mean_w = np.mean([weights_joint[p] for p in active_personas], axis=0)

            for persona in active_personas:
                mdp = persona_mdps[persona]
                w = weights_joint[persona]
                expert_fe = mdp["expert_fe"]
                policy_fe = policy_fes[persona]

                # Gradient: expert - policy - l2_reg - group_reg * (w - mean_w)
                grad = (expert_fe - policy_fe
                        - self.l2_reg * w
                        - self.group_reg * (w - mean_w))

                w_new = w + self.lr * grad
                # Clip extreme values before normalization
                w_new = np.clip(w_new, 0.001, self.MAX_WEIGHT)
                w_new = w_new / w_new.sum()

                change = np.max(np.abs(w_new - w))
                max_change = max(max_change, change)
                weights_joint[persona] = w_new

            if max_change < self.tolerance:
                if verbose:
                    print(f"  [IRL] GC-MaxEnt converged at iter {iteration + 1}"
                          f" ({n_active} personas)")
                break

        # Store results
        for persona in active_personas:
            self.weights[persona] = weights_joint[persona]
            self._mdp_cache[persona] = persona_mdps[persona]
            if verbose:
                self._print_weights(persona, weights_joint[persona])

        # Fill any missing personas with defaults
        for persona in PERSONA_CATEGORIES:
            if persona not in self.weights:
                self.weights[persona] = self._default_weights(persona)

    # ================================================================
    # Discrete MDP construction
    # ================================================================

    def _build_discrete_mdp(self, trajectories: List[AgentTrajectory]) -> dict:
        """Build a discretized MDP from trajectory data.

        Returns dict with keys:
          - n_states, n_actions
          - state_of: maps (bin0,...,bin4) → state_id
          - feature_of: maps state_id → feature vector (mean φ for that state)
          - transitions: state_id × action → list of (next_state_id, count)
          - expert_sa_counts: state_id × action → count
          - expert_fe: expert feature expectation vector
          - init_state_dist: initial state distribution
        """
        # Collect all (feature_vector, action, next_feature_vector) tuples
        all_tuples = []
        for traj in trajectories:
            feats_list = self._extract_trajectory_features(traj)
            actions = [d.get("target_exit_idx", 0) for d in traj.decisions]
            for t in range(len(feats_list) - 1):
                all_tuples.append((
                    np.array(feats_list[t]),
                    min(actions[t], self.N_ACTIONS - 1),
                    np.array(feats_list[t + 1]),
                ))

        if not all_tuples:
            return self._empty_mdp()

        all_feats = np.array([t[0] for t in all_tuples])

        # Compute bin edges (quantile-based) per feature dimension
        bin_edges = []
        for d in range(len(FEATURE_NAMES)):
            col = all_feats[:, d]
            edges = np.quantile(col, np.linspace(0, 1, self.N_BINS + 1))
            # Widen edges slightly to avoid boundary issues
            eps = 1e-6
            edges[0] -= eps
            edges[-1] += eps
            bin_edges.append(edges)

        # Map feature vectors to discrete states
        def state_id(feat_vec: np.ndarray) -> int:
            bins = []
            for d in range(len(FEATURE_NAMES)):
                b = np.digitize(feat_vec[d], bin_edges[d]) - 1
                bins.append(min(b, self.N_BINS - 1))
            # Flatten multi-dimensional bin to 1D index
            idx = 0
            stride = 1
            for d in range(len(FEATURE_NAMES)):
                idx += bins[d] * stride
                stride *= self.N_BINS
            return idx

        max_states = self.N_BINS ** len(FEATURE_NAMES)

        # Count transitions and state-action visits
        trans_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # trans_counts[s][a][s'] = count
        sa_counts = defaultdict(lambda: defaultdict(int))
        init_counts = defaultdict(int)
        state_feat_sum = defaultdict(lambda: np.zeros(len(FEATURE_NAMES)))
        state_count = defaultdict(int)

        # Track next-state features for states that only appear as transitions
        next_feat_sum = defaultdict(lambda: np.zeros(len(FEATURE_NAMES)))
        next_feat_count = defaultdict(int)

        for i, (feat, action, next_feat) in enumerate(all_tuples):
            s = state_id(feat)
            s_next = state_id(next_feat)
            trans_counts[s][action][s_next] += 1
            sa_counts[s][action] += 1
            state_feat_sum[s] += feat
            state_count[s] += 1
            # Record next-state features (for terminal/absorbing states)
            next_feat_sum[s_next] += next_feat
            next_feat_count[s_next] += 1
            if i == 0:
                init_counts[s] += 1

        # Ensure terminal states that only appear as transitions are included
        for s_extra in next_feat_count:
            if s_extra not in state_count:
                state_count[s_extra] = 0
                state_feat_sum[s_extra] = next_feat_sum[s_extra]

        # Build compact MDP representation
        n_states = len(state_count)
        state_list = sorted(state_count.keys())

        # State ID re-mapping (compact, consecutive)
        s_map = {s_orig: i for i, s_orig in enumerate(state_list)}

        # Feature vector per state (mean)
        state_features = np.zeros((n_states, len(FEATURE_NAMES)))
        for s_orig, i in s_map.items():
            if state_count[s_orig] > 0:
                state_features[i] = state_feat_sum[s_orig] / state_count[s_orig]

        # Transition matrix: transitions[s][a] = list of (next_s, prob)
        transitions = [[[] for _ in range(self.N_ACTIONS)] for _ in range(n_states)]
        for s_orig, i in s_map.items():
            for a in range(self.N_ACTIONS):
                total = sa_counts[s_orig].get(a, 0)
                if total > 0:
                    for s_next_orig, cnt in trans_counts[s_orig][a].items():
                        if s_next_orig in s_map:
                            transitions[i][a].append(
                                (s_map[s_next_orig], cnt / total))

        # Validate transition matrix: each row per action must sum to 1
        for s in range(n_states):
            for a in range(self.N_ACTIONS):
                if transitions[s][a]:
                    prob_sum = sum(p for _, p in transitions[s][a])
                    assert abs(prob_sum - 1.0) < 1e-4, \
                        f"Transition matrix [s={s}, a={a}] sums to {prob_sum:.6f}, expected 1.0"

        # Expert state-action visitation (normalized)
        total_sa = sum(sum(d.values()) for d in sa_counts.values())
        expert_sa = np.zeros((n_states, self.N_ACTIONS))
        for s_orig, i in s_map.items():
            for a in range(self.N_ACTIONS):
                expert_sa[i, a] = sa_counts[s_orig].get(a, 0) / max(total_sa, 1)

        # Expert feature expectation: Σ_{s,a} μ_E(s,a) · φ(s)
        expert_fe = np.zeros(len(FEATURE_NAMES))
        for s_orig, i in s_map.items():
            for a in range(self.N_ACTIONS):
                expert_fe += expert_sa[i, a] * state_features[i]

        # Initial state distribution
        total_init = sum(init_counts.values())
        init_dist = np.zeros(n_states)
        for s_orig, i in s_map.items():
            init_dist[i] = init_counts[s_orig] / max(total_init, 1)

        return {
            "n_states": n_states,
            "n_actions": self.N_ACTIONS,
            "state_features": state_features,
            "transitions": transitions,
            "expert_sa": expert_sa,
            "expert_fe": expert_fe,
            "init_dist": init_dist,
            "bin_edges": bin_edges,
        }

    @staticmethod
    def _empty_mdp() -> dict:
        return {
            "n_states": 1, "n_actions": 8,
            "state_features": np.ones((1, 5)) / 5,
            "transitions": [[[] for _ in range(8)]],
            "expert_sa": np.ones((1, 8)) / 8,
            "expert_fe": np.ones(5) / 5,
            "init_dist": np.ones(1),
        }

    # ================================================================
    # Soft value iteration and policy feature expectations
    # ================================================================

    def _compute_policy_fe(self, mdp: dict, weights: np.ndarray) -> np.ndarray:
        """Compute feature expectations under the MaxEnt policy for weights w.

        1. Compute state-action rewards: R(s,a) = w · φ(s)
        2. Soft value iteration → V(s), Q(s,a)
        3. Soft policy: π(a|s) = exp(Q(s,a)) / Σ_a' exp(Q(s,a'))
        4. State visitation frequencies D(s) (stationary distribution)
        5. μ_π = Σ_{s,a} D(s) · π(a|s) · φ(s)
        """
        n_s = mdp["n_states"]
        n_a = mdp["n_actions"]
        feats = mdp["state_features"]
        trans = mdp["transitions"]
        init_dist = mdp["init_dist"]

        # Step 1: Reward per state-action
        R = np.zeros((n_s, n_a))
        for s in range(n_s):
            R[s, :] = np.dot(feats[s], weights)

        # Step 2: Soft value iteration (with numerical stability guards)
        V = np.zeros(n_s)
        for _ in range(self.MAX_SOFT_VI_ITERS):
            V_new = np.zeros(n_s)
            for s in range(n_s):
                logits = np.zeros(n_a)
                for a in range(n_a):
                    expected_v = 0.0
                    for s_next, prob in trans[s][a]:
                        expected_v += prob * V[s_next]
                    logits[a] = R[s, a] + self.GAMMA * expected_v
                # Clip logits to prevent overflow in exp
                logits = np.clip(logits, -self.MAX_V_ABS, self.MAX_V_ABS)
                # Softmax (log-sum-exp)
                max_logit = np.max(logits)
                V_new[s] = max_logit + np.log(np.sum(np.exp(logits - max_logit)) + 1e-300)
            # Clip V to prevent runaway
            V_new = np.clip(V_new, -self.MAX_V_ABS, self.MAX_V_ABS)
            # Check for NaN
            if np.any(np.isnan(V_new)) or np.any(np.isinf(V_new)):
                V = np.zeros(n_s)
                break
            if np.max(np.abs(V_new - V)) < 1e-6:
                V = V_new
                break
            V = V_new

        # Step 3: Soft policy π(a|s) = exp(Q - V)
        policy = np.zeros((n_s, n_a))
        for s in range(n_s):
            logits = np.zeros(n_a)
            for a in range(n_a):
                expected_v = 0.0
                for s_next, prob in trans[s][a]:
                    expected_v += prob * V[s_next]
                logits[a] = R[s, a] + self.GAMMA * expected_v
            # Clip for numerical stability: exp(Q - V) with clipped exponent
            logits = np.clip(logits, -self.MAX_V_ABS, self.MAX_V_ABS)
            exponent = np.clip(logits - V[s], -500, 500)  # exp(500) is huge; exp(-500) ≈ 0
            policy[s] = np.exp(exponent)
            row_sum = policy[s].sum()
            if row_sum > 0 and not np.isnan(row_sum) and not np.isinf(row_sum):
                policy[s] /= row_sum
            else:
                policy[s] = np.ones(n_a) / n_a

        # Step 4: Stationary state distribution D(s)
        T_ss = np.zeros((n_s, n_s))
        for s in range(n_s):
            for a in range(n_a):
                for s_next, prob in trans[s][a]:
                    T_ss[s, s_next] += policy[s, a] * prob
            row_sum = T_ss[s].sum()
            if row_sum < 1e-10 or np.isnan(row_sum) or np.isinf(row_sum):
                T_ss[s, s] = 1.0
            elif abs(row_sum - 1.0) > 1e-6:
                T_ss[s] /= row_sum

        # Power iteration with NaN guard
        D = init_dist.copy()
        for _ in range(1000):
            D_new = D @ T_ss
            if np.any(np.isnan(D_new)) or np.any(np.isinf(D_new)):
                D = init_dist.copy()
                break
            if np.max(np.abs(D_new - D)) < 1e-8:
                D = D_new
                break
            D = D_new

        # Step 5: Feature expectations
        policy_fe = np.zeros(len(FEATURE_NAMES))
        for s in range(n_s):
            for a in range(n_a):
                policy_fe += D[s] * policy[s, a] * feats[s]

        return policy_fe

    # ================================================================
    # Feature extraction (unchanged from original)
    # ================================================================

    def _extract_trajectory_features(self, traj: AgentTrajectory
                                     ) -> List[List[float]]:
        """Extract normalized feature vectors from a single trajectory."""
        features = []
        for dec in traj.decisions:
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

            features.append([safety, efficiency, social, conformity, comfort])
        return features

    # ================================================================
    # Persistence
    # ================================================================

    @staticmethod
    def _default_weights(persona: str) -> np.ndarray:
        """Heuristic default weights when insufficient data."""
        defaults = {
            "untrained_elderly": [0.35, 0.15, 0.30, 0.10, 0.10],
            "untrained_young": [0.25, 0.30, 0.10, 0.20, 0.15],
            "trained_staff":   [0.25, 0.35, 0.15, 0.20, 0.05],
            "guide":           [0.20, 0.25, 0.30, 0.15, 0.10],
            "firefighter":     [0.20, 0.10, 0.50, 0.05, 0.15],
        }
        w = np.array(defaults.get(persona, [0.25, 0.25, 0.20, 0.15, 0.15]))
        return w / w.sum()

    def _print_weights(self, persona: str, w: np.ndarray):
        print(f"    {persona}: " + " | ".join(
            f"{name}={val:.3f}" for name, val in zip(FEATURE_NAMES, w)))

    def save(self, path: str):
        """Save learned weights to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {persona: w.tolist() for persona, w in self.weights.items()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"feature_names": FEATURE_NAMES, "weights": data}, f,
                      ensure_ascii=False, indent=2)
        print(f"[IRL] Weights saved to {path}")

    def load(self, path: str):
        """Load learned weights from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for persona, w_list in data["weights"].items():
            self.weights[persona] = np.array(w_list)
        print(f"[IRL] Loaded weights for {len(self.weights)} personas from {path}")

    def get_reward_function(self, persona: str):
        """Return a callable reward function for the given persona.

        Args:
            persona: One of the persona category strings.

        Returns:
            Callable: reward_fn(features_dict) → float
        """
        w = self.weights.get(persona, self._default_weights(persona))

        def reward_fn(features: dict) -> float:
            """Compute reward from feature values.

            Args:
                features: dict with keys matching FEATURE_NAMES
                         e.g. {"safety": 0.8, "efficiency": 0.5, ...}
            """
            f_vec = np.array([features.get(name, 0.0) for name in FEATURE_NAMES])
            return float(np.dot(w, f_vec))

        return reward_fn


# ================================================================
# Command-line interface
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="IRL Recovery from LLM/Oracle trajectories")
    parser.add_argument("--trajectory_dir", type=str, required=True,
                       help="Directory containing .jsonl trajectory files")
    parser.add_argument("--output", type=str, default="data/irl_weights.json",
                       help="Output path for learned weights")
    parser.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--max_iter", type=int, default=500,
                       help="Max IRL iterations")
    parser.add_argument("--weight_config", type=str, default=None,
                       help="Only load trajectories matching this Oracle weight "
                            "config name (e.g. safety_first, balanced, "
                            "efficiency_first). Used for sensitivity analysis. "
                            "If not set, loads all trajectories.")
    parser.add_argument("--compare", action="store_true",
                       help="Run IRL separately for each weight_config found in "
                            "trajectory files and print a comparison table.")
    args = parser.parse_args()

    collector = TrajectoryCollector()

    if args.compare:
        # Sensitivity analysis mode: fit IRL per weight config and compare
        import glob as _glob
        all_files = _glob.glob(os.path.join(args.trajectory_dir, "oracle_*.jsonl"))
        # Group files by weight_config prefix in filename.
        # Format: oracle_{config}_run_NNN.jsonl
        # config may contain underscores (e.g. "safety_first").
        config_groups: Dict[str, List[str]] = defaultdict(list)
        for f in all_files:
            basename = os.path.basename(f)
            # Strip .jsonl and "oracle_" prefix
            stem = basename.replace(".jsonl", "")
            if not stem.startswith("oracle_"):
                continue
            inner = stem[len("oracle_"):]  # e.g. "safety_first_run_001"
            # Split at "_run_" to get the config name
            idx = inner.find("_run_")
            if idx < 0:
                continue
            cfg_name = inner[:idx]  # "safety_first"
            config_groups[cfg_name].append(f)

        if not config_groups:
            print(f"No oracle_*_*.jsonl files found in {args.trajectory_dir}")
            print("Run generate_oracle_trajectories.py --weights all first.")
            sys.exit(1)

        print(f"Found {len(config_groups)} Oracle weight configs: "
              f"{list(config_groups.keys())}")
        print()

        results = {}
        for cfg_name, files in config_groups.items():
            trajectories = []
            for f in files:
                trajectories.extend(collector.load(f))
            print(f"[{cfg_name}] {len(trajectories)} trajectories from "
                  f"{len(files)} files")

            irl = IRLRecovery(learning_rate=args.lr, max_iter=args.max_iter)
            irl.fit(trajectories, verbose=False)
            results[cfg_name] = irl

            output_path = (args.output.replace(".json", f"_{cfg_name}.json")
                           if args.output == "data/irl_weights.json"
                           else args.output)
            irl.save(output_path)

        # Print comparison table
        print("\n" + "=" * 75)
        print("  SENSITIVITY ANALYSIS: Recovered Weights by Oracle Config")
        print("=" * 75)
        header = (f"{'Config':<22} {'safety':>8} {'efficiency':>11} "
                  f"{'social':>8} {'conformity':>10} {'comfort':>9}")
        print(header)
        print("-" * 75)
        for cfg_name in sorted(results.keys()):
            irl = results[cfg_name]
            for persona in sorted(irl.weights.keys()):
                w = irl.weights[persona]
                print(f"{cfg_name+'/'+persona:<22} "
                      f"{w[0]:8.3f} {w[1]:11.3f} {w[2]:8.3f} "
                      f"{w[3]:10.3f} {w[4]:9.3f}")
        print("-" * 75)
        print("Expected: IRL should recover safety >> efficiency for 'safety_first',")
        print("         efficiency >> safety for 'efficiency_first', and near-equal")
        print("         weights for 'balanced'. If so, the IRL algorithm is validated.")
        print("=" * 75)

    elif args.weight_config:
        # Single config mode: filter files by weight_config name
        import glob as _glob
        pattern = f"oracle_{args.weight_config}_*.jsonl"
        files = _glob.glob(os.path.join(args.trajectory_dir, pattern))
        if not files:
            print(f"No files matching '{pattern}' in {args.trajectory_dir}")
            sys.exit(1)
        trajectories = []
        for f in files:
            trajectories.extend(collector.load(f))
        print(f"Loaded {len(trajectories)} trajectories "
              f"(weight_config={args.weight_config}, {len(files)} files)")

        irl = IRLRecovery(learning_rate=args.lr, max_iter=args.max_iter)
        irl.fit(trajectories, verbose=True)
        irl.save(args.output)
        print("Done.")

    else:
        # Default: load all files (backward compatible)
        trajectories = collector.load_all(args.trajectory_dir)
        print(f"Loaded {len(trajectories)} trajectories from "
              f"{args.trajectory_dir}")

        irl = IRLRecovery(learning_rate=args.lr, max_iter=args.max_iter)
        irl.fit(trajectories, verbose=True)
        irl.save(args.output)
        print("Done.")

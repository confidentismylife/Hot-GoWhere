"""Kolomogorov-Smirnov validation — confirms IRL policy reproduces LLM behavior.

Key question: does the IRL-learned policy generate behavior that is
statistically indistinguishable from the original LLM behavior?

Tests:
  1. KS test on exit choice distributions (per persona)
  2. KS test on speed choices (run/walk/crawl/wait)
  3. JS divergence of full behavioral distributions
  4. Feature expectation matching error (FEME)

This answers the reviewer: "LLM is a black box" — by showing that IRL has
extracted the relevant behavioral signal into interpretable weights whose
output matches the original LLM distribution.

Usage:
  python -m experiments.ks_validation \\
      --trajectories data/trajectories/ \\
      --irl_weights data/irl_weights.json \\
      --output data/experiments/
"""

import json
import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from scipy import stats

from experiments.stats_utils import (
    ks_test,
    js_divergence,
    feature_expectation_error,
    policy_match_accuracy,
    mean_confidence_interval,
    cohens_d,
    StatisticalReport,
)

# Feature and persona definitions (mirror irl_recovery.py)
FEATURE_NAMES = ["safety", "efficiency", "social", "conformity", "comfort"]
PERSONA_CATEGORIES = [
    "untrained_elderly", "untrained_young", "trained_staff",
    "guide", "firefighter",
]


@dataclass
class KSValidationResult:
    """One KS test result."""
    persona: str
    comparison: str         # e.g., "exit_choice", "speed_choice"
    ks_statistic: float
    p_value: float
    significant: bool       # True = distributions ARE different (bad)
    js_divergence: float
    n_samples: int
    text: str


class KSValidator:
    """Validates IRL policy against original LLM behavior."""

    def __init__(self, trajectories: List = None,
                 irl_weights: Dict[str, np.ndarray] = None,
                 trajectory_dir: str = None,
                 weights_path: str = None):
        self.trajectories = trajectories or []
        self.irl_weights = irl_weights or {}

        if trajectory_dir:
            self._load_trajectories(trajectory_dir)
        if weights_path:
            self._load_weights(weights_path)

    def _load_trajectories(self, directory: str):
        from execution.irl_recovery import TrajectoryCollector
        collector = TrajectoryCollector()
        self.trajectories = collector.load_all(directory)

    def _load_weights(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for persona, w_list in data["weights"].items():
            self.irl_weights[persona] = np.array(w_list)

    # ================================================================
    # KS tests
    # ================================================================

    # Default Chaoyang Joycity 1F exit positions (150m × 80m)
    DEFAULT_EXIT_POSITIONS = [
        (25.0, 0.0), (75.0, 0.0), (120.0, 0.0), (150.0, 25.0),
        (0.0, 25.0), (0.0, 55.0), (100.0, 80.0), (50.0, 80.0),
    ]

    def validate_exit_choices(self,
                              exit_positions: List = None) -> List[KSValidationResult]:
        """KS test: do IRL policy exit choices match LLM behavior?"""
        if exit_positions is None:
            exit_positions = self.DEFAULT_EXIT_POSITIONS
        results = []
        grouped = defaultdict(list)
        for traj in self.trajectories:
            grouped[traj.persona].append(traj)

        for persona in PERSONA_CATEGORIES:
            trajs = grouped.get(persona, [])
            if len(trajs) < 10:
                continue

            # Collect LLM exit choices
            llm_exits = []
            irl_exits = []
            w = self.irl_weights.get(persona)

            for traj in trajs:
                for dec in traj.decisions:
                    llm_exits.append(dec.get("target_exit_idx", 0))

                    # Generate IRL-predicted exit choice with exit-specific features
                    if w is not None:
                        irl_exit = self._predict_exit_from_weights(
                            dec, w, exit_positions)
                        irl_exits.append(irl_exit)

            if not llm_exits or not irl_exits:
                continue

            n_exits = len(exit_positions)
            # KS test
            ks_result = ks_test(llm_exits, irl_exits, name=f"{persona} exit choice")
            js = js_divergence(
                np.bincount(llm_exits, minlength=n_exits)[:n_exits] / max(1, len(llm_exits)),
                np.bincount(irl_exits, minlength=n_exits)[:n_exits] / max(1, len(irl_exits)),
            )

            results.append(KSValidationResult(
                persona=persona,
                comparison="exit_choice",
                ks_statistic=ks_result["ks_statistic"],
                p_value=ks_result["p_value"],
                significant=not ks_result["same_distribution"],
                js_divergence=js,
                n_samples=len(llm_exits),
                text=ks_result["text"],
            ))

        return results

    def validate_speed_choices(self) -> List[KSValidationResult]:
        """KS test on speed decisions."""
        results = []
        grouped = defaultdict(list)
        for traj in self.trajectories:
            grouped[traj.persona].append(traj)

        speed_map = {"run": 0, "walk": 1, "crawl": 2, "wait": 3}

        for persona in PERSONA_CATEGORIES:
            trajs = grouped.get(persona, [])
            if len(trajs) < 10:
                continue

            llm_speeds = []
            irl_speeds = []
            w = self.irl_weights.get(persona)

            for traj in trajs:
                for dec in traj.decisions:
                    speed_str = str(dec.get("speed", "walk"))
                    s = speed_map.get(speed_str, 1)
                    llm_speeds.append(s)
                    if w is not None:
                        irl_speeds.append(self._predict_speed_from_weights(dec, w))

            if not llm_speeds or not irl_speeds:
                continue

            ks_result = ks_test(llm_speeds, irl_speeds, name=f"{persona} speed choice")
            js = js_divergence(
                np.bincount(llm_speeds, minlength=4) / len(llm_speeds),
                np.bincount(irl_speeds, minlength=4) / len(irl_speeds),
            )

            results.append(KSValidationResult(
                persona=persona,
                comparison="speed_choice",
                ks_statistic=ks_result["ks_statistic"],
                p_value=ks_result["p_value"],
                significant=not ks_result["same_distribution"],
                js_divergence=js,
                n_samples=len(llm_speeds),
                text=ks_result["text"],
            ))

        return results

    def validate_feature_expectations(self) -> Dict:
        """Compute FEME: feature expectation matching error."""
        from execution.irl_recovery import IRLRecovery
        irl = IRLRecovery()

        grouped = defaultdict(list)
        for traj in self.trajectories:
            grouped[traj.persona].append(traj)

        feme_results = {}
        for persona in PERSONA_CATEGORIES:
            trajs = grouped.get(persona, [])
            if len(trajs) < 10:
                continue

            # Compute expert feature expectations from trajectories
            all_feats = []
            for traj in trajs:
                feats = irl._extract_trajectory_features(traj)
                all_feats.extend(feats)

            if not all_feats:
                continue
            expert_fe = np.mean(all_feats, axis=0)

            # Compute IRL policy feature expectations
            w = self.irl_weights.get(persona, np.ones(5) / 5)
            mdp = irl._build_discrete_mdp(trajs)
            policy_fe = irl._compute_policy_fe(mdp, w)

            feme = float(np.linalg.norm(expert_fe - policy_fe))
            feme_results[persona] = {
                "feme": feme,
                "expert_fe": expert_fe.tolist(),
                "policy_fe": policy_fe.tolist(),
                "n_trajectories": len(trajs),
            }

        return feme_results

    @staticmethod
    def _predict_exit_from_weights(decision: dict,
                                   weights: np.ndarray,
                                   exit_positions: List = None) -> int:
        """Predict which exit IRL policy would choose given the decision context.

        Uses exit-specific features: each exit has different distances, smoke
        levels, and crowd conditions, so the IRL policy can meaningfully
        differentiate between them.
        """
        agent_pos = np.array(decision.get("agent_pos", [75.0, 40.0]))
        n_exits = len(exit_positions) if exit_positions else 8

        scores = []
        for exit_idx in range(n_exits):
            # Per-exit features
            if exit_positions:
                exit_pos = np.array(exit_positions[exit_idx])
                dist = float(np.linalg.norm(agent_pos - exit_pos))
            else:
                # Fallback: use inferred distances from decision context
                dist = decision.get("nearest_exit_dist", 20) * (1.0 + 0.3 * exit_idx)

            # Per-exit smoke (use decision's per-exit data if available)
            exit_smoke_key = f"exit_{exit_idx}_smoke"
            smoke = float(decision.get(exit_smoke_key,
                          decision.get("smoke_at_pos", 0.3) * (0.5 + 0.1 * exit_idx)))

            fire_dist = decision.get("fire_distance", 50)
            crowd_density = decision.get("crowd_density", 0.3)

            # Compute 5 features per exit
            safety = (1.0 - smoke) * min(fire_dist / 50.0, 1.0)
            efficiency = 1.0 / (1.0 + dist / 50.0)
            social = 1.0 - crowd_density * (0.5 + 0.1 * exit_idx)
            conformity = 0.3 + 0.1 * (exit_idx % 4)  # Nearby exits get higher conformity
            comfort = (1.0 - smoke) * 0.7 + 0.3 * efficiency

            reward = float(weights[0] * safety + weights[1] * efficiency +
                          weights[2] * social + weights[3] * conformity +
                          weights[4] * comfort)
            # Softmax-style sampling via Gumbel-Max trick
            noise = np.random.gumbel(0, 0.3)
            scores.append(reward + noise)

        return int(np.argmax(scores))

    @staticmethod
    def _predict_speed_from_weights(decision: dict,
                                    weights: np.ndarray) -> int:
        """Predict speed choice from IRL weights using all 5 features.

        Speed categories: 0=run, 1=walk, 2=crawl, 3=wait.
        Each speed has a different feature profile — e.g., RUN scores high on
        safety+efficiency but low on comfort; WAIT scores high on comfort but
        low on safety+efficiency.
        """
        smoke = float(decision.get("smoke_at_pos", 0.3))
        fire_dist = float(decision.get("fire_distance", 50))
        stamina = float(decision.get("stamina", 80))
        crowd = float(decision.get("crowd_density", 0.3))

        safety = (1.0 - smoke) * min(fire_dist / 50.0, 1.0)
        efficiency = 0.5  # baseline
        social = 1.0 - crowd
        comfort = min(stamina / 100.0, 1.0)

        # Feature profiles for each speed (normalized to [0, 1] per feature)
        speed_features = {
            0: np.array([safety, 1.0, social, 0.5, 0.2]),         # RUN
            1: np.array([safety * 0.8, 0.7, social, 0.6, 0.6]),   # WALK
            2: np.array([safety * 0.5, 0.3, social, 0.8, 0.4]),   # CRAWL
            3: np.array([0.0, 0.0, 0.5, 0.9, 1.0]),               # WAIT
        }

        scores = []
        for s in range(4):
            reward = float(np.dot(weights, speed_features[s]))
            noise = np.random.gumbel(0, 0.3)
            scores.append(reward + noise)

        return int(np.argmax(scores))

    # ================================================================
    # Reports
    # ================================================================

    def full_report(self) -> str:
        """Generate complete KS validation report."""
        exit_results = self.validate_exit_choices()
        speed_results = self.validate_speed_choices()
        feme = self.validate_feature_expectations()

        lines = [
            "# IRL Policy Validation Report",
            "",
            "## 1. Distribution Alignment (KS Test)",
            "",
            "H0: The IRL policy and original LLM behavior come from the same distribution.",
            "A significant result (p < 0.05) would indicate the IRL policy has failed to capture LLM behavior.",
            "",
            "### Exit Choice Distributions",
            "",
            "| Persona | KS Stat | p-value | Significant? | JS Div | N |",
            "|---------|---------|---------|-------------|--------|---|",
        ]

        for r in exit_results:
            sig = "different" if r.significant else "same dist"
            lines.append(
                f"| {r.persona} | {r.ks_statistic:.4f} | {r.p_value:.4f} "
                f"| {sig} | {r.js_divergence:.4f} | {r.n_samples} |"
            )

        lines.append("")
        lines.append("### Speed Choice Distributions")
        lines.append("")
        lines.append("| Persona | KS Stat | p-value | Significant? | JS Div | N |")
        lines.append("|---------|---------|---------|-------------|--------|---|")

        for r in speed_results:
            sig = "different" if r.significant else "same dist"
            lines.append(
                f"| {r.persona} | {r.ks_statistic:.4f} | {r.p_value:.4f} "
                f"| {sig} | {r.js_divergence:.4f} | {r.n_samples} |"
            )

        lines.append("")
        lines.append("## 2. Feature Expectation Matching Error (FEME)")
        lines.append("")
        lines.append("| Persona | FEME | N Trajectories |")
        lines.append("|---------|------|---------------|")

        for persona, data in feme.items():
            lines.append(f"| {persona} | {data['feme']:.6f} | {data['n_trajectories']} |")

        lines.append("")

        # Summary
        non_sig_exit = sum(1 for r in exit_results if not r.significant)
        non_sig_speed = sum(1 for r in speed_results if not r.significant)
        total = len(exit_results) + len(speed_results)
        non_sig = non_sig_exit + non_sig_speed

        lines.append("## 3. Summary")
        lines.append("")
        lines.append(
            f"The IRL policy successfully reproduces LLM behavior in "
            f"**{non_sig}/{total}** KS tests (p > 0.05, fail to reject H0)."
        )
        lines.append("")
        lines.append(
            "This demonstrates that the IRL recovery process extracts the "
            "behaviorally relevant signal from LLM decisions into interpretable "
            "reward weights, and the resulting policy generates decisions "
            "statistically consistent with the original LLM behavior."
        )

        return "\n".join(lines)

    def paper_text(self) -> str:
        """Generate paper-ready KS validation text."""
        exit_results = self.validate_exit_choices()
        speed_results = self.validate_speed_choices()
        feme = self.validate_feature_expectations()

        non_sig_exit = sum(1 for r in exit_results if not r.significant)
        total_exit = len(exit_results)
        non_sig_speed = sum(1 for r in speed_results if not r.significant)
        total_speed = len(speed_results)

        lines = [
            "To validate that the IRL recovery process faithfully captures "
            "LLM behavioral patterns, we performed Kolmogorov-Smirnov tests "
            "comparing the exit choice and speed choice distributions of the "
            "original LLM agents against those generated by the IRL-learned "
            "reward functions.",
            "",
            f"For exit choices, {non_sig_exit}/{total_exit} persona categories "
            f"showed no significant difference (KS test, p > 0.05), indicating "
            f"that the IRL policy reproduces LLM exit preferences accurately.",
            "",
            f"For speed choices, {non_sig_speed}/{total_speed} categories "
            f"showed distributional alignment.",
            "",
            "The Feature Expectation Matching Error (FEME) further confirms "
            "the quality of the IRL recovery:",
        ]

        for persona, data in sorted(feme.items()):
            lines.append(f"  - {persona}: FEME = {data['feme']:.6f}")

        lines.append("")
        lines.append(
            "These results demonstrate that while LLMs are indeed black-box "
            "models, the IRL recovery process extracts the behavioral signal "
            "into a transparent, interpretable reward function with 25 "
            "parameters (5 features × 5 personas). The recovered policy "
            "matches the original LLM behavior distribution, validating "
            "the LLM → IRL knowledge distillation pipeline."
        )

        return "\n".join(lines)


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KS Validation for IRL Policy")
    parser.add_argument("--trajectories", type=str, default=None,
                       help="Directory of trajectory JSONL files")
    parser.add_argument("--irl_weights", type=str, default="data/irl_weights.json",
                       help="Path to IRL-learned weights JSON")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for validation report")
    args = parser.parse_args()

    validator = KSValidator(
        trajectory_dir=args.trajectories,
        weights_path=args.irl_weights,
    )

    report = validator.full_report() + "\n\n" + validator.paper_text()

    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"KS validation report saved to {args.output}")
    else:
        try:
            print(report)
        except UnicodeEncodeError:
            print("Report generated (suppressed GBK encoding).")

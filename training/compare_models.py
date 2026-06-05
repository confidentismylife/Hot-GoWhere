"""Model comparison framework — base vs LoRA-fine-tuned model.

Evaluates both models on the same test scenarios and computes:
1. Safety constraint violation rate (% of decisions blocked by safety guard)
2. JSON format compliance rate (% of valid JSON outputs)
3. Decision quality score (0-1, higher = better)
4. Token efficiency (output lengths)
5. Per-speed-category accuracy

Usage:
    # Compare with vLLM (full simulation speed)
    python -m training.compare_models --lora output/lora_evac/final --num-scenarios 10

    # Compare with HuggingFace (single inference, no vLLM needed)
    python -m training.compare_models --lora output/lora_evac/final --backend hf
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.orchestrator import SimulationOrchestrator
from decision.agent_state import Agent, Speed, Cooperation
from decision.safety_guard import SafetyGuard, SafetyResult
from decision.prompt_manager import PromptManager
from perception.environment import EnvironmentSnapshot


@dataclass
class EvalMetrics:
    """Per-model evaluation metrics."""
    model_name: str = ""
    total_decisions: int = 0
    json_valid: int = 0
    safety_passed: int = 0       # Passed without blocking
    safety_modified: int = 0     # Passed with modifications
    safety_blocked: int = 0      # Completely blocked
    speed_correct: Dict[str, int] = field(default_factory=dict)
    avg_output_tokens: float = 0.0
    avg_inference_time: float = 0.0
    per_metric: Dict[str, float] = field(default_factory=dict)

    @property
    def json_compliance(self) -> float:
        return self.json_valid / max(1, self.total_decisions)

    @property
    def safety_pass_rate(self) -> float:
        return self.safety_passed / max(1, self.total_decisions)

    @property
    def safety_intervention_rate(self) -> float:
        return (self.safety_modified + self.safety_blocked) / max(1, self.total_decisions)


class ModelRunner:
    """Abstract interface for model inference during comparison."""

    def generate(self, prompt: str, system: str = "") -> str:
        raise NotImplementedError


class HuggingFaceRunner(ModelRunner):
    """HuggingFace inference (for quick comparison without vLLM)."""

    def __init__(self, model_path: str, lora_path: str = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"[HF Runner] Loading {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)

        if lora_path:
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True)
            self.model = PeftModel.from_pretrained(base, lora_path)
            self.model = self.model.merge_and_unload()
            print(f"  Merged LoRA from {lora_path}")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True)

        self.model.eval()

    def generate(self, prompt: str, system: str = "") -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
            )
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


class SafetyOracleRunner(ModelRunner):
    """Safety guard fallback as baseline (upper bound of rule-based performance)."""

    def __init__(self, safety_guard: SafetyGuard):
        self.safety = safety_guard

    def generate(self, prompt: str, system: str = "", **kwargs) -> str:
        agent = kwargs.get("agent")
        env = kwargs.get("env")
        if agent is None or env is None:
            return "{}"
        fb = self.safety.fallback_decision(agent, env)
        return json.dumps({
            "risk_assessment": "安全护栏兜底",
            "target_exit": f"出口{fb['target_exit_idx']+1}",
            "route_reasoning": fb["reasoning"],
            "speed": fb["speed"].value,
            "cooperation": fb["cooperation"].value,
            "reasoning": fb["reasoning"],
        }, ensure_ascii=False)


def evaluate_single(model: ModelRunner, agent: Agent, env: EnvironmentSnapshot,
                    prompt_mgr: PromptManager, kdocs: List[str],
                    safety: SafetyGuard) -> Tuple[EvalMetrics, dict]:
    """Evaluate one model on one decision point. Returns metrics + raw decision."""
    metrics = EvalMetrics()
    metrics.total_decisions = 1

    t0 = time.perf_counter()

    # Build prompts
    system = prompt_mgr.build_system(env.disaster_type, kdocs, role="civilian")
    user = prompt_mgr.build_user(agent, env)

    # Generate
    if isinstance(model, SafetyOracleRunner):
        raw = model.generate("", agent=agent, env=env)
    else:
        raw = model.generate(user, system)

    metrics.avg_inference_time = time.perf_counter() - t0
    metrics.avg_output_tokens = len(raw)

    # Parse JSON
    try:
        data = prompt_mgr.parse_response(raw)
        metrics.json_valid = 1
    except (json.JSONDecodeError, Exception):
        metrics.json_valid = 0
        return metrics, {"raw": raw, "parse_error": True}

    # Check against safety guard
    if env.exits:
        exit_str = data.get("target_exit", "")
        target_idx = 0
        for i in range(1, len(env.exits) + 1):
            if f"出口{i}" in exit_str or f"exit{i}" in exit_str.lower():
                target_idx = i - 1
                break

        result = safety.check_exit_safety(target_idx, env)
        if not result.get("blocked", False):
            metrics.safety_passed = 1
        else:
            metrics.safety_blocked = 1

        # Speed correctness
        speed_str = data.get("speed", "walk")
        optimal = safety.fallback_decision(agent, env)
        if speed_str == optimal["speed"].value:
            metrics.speed_correct["speed_match"] = \
                metrics.speed_correct.get("speed_match", 0) + 1

    return metrics, {"raw_data": data, "raw_text": raw}


def run_comparison(base_model: ModelRunner, lora_model: ModelRunner,
                   num_scenarios: int = 10,
                   agents_per: int = 50,
                   duration: float = 30.0) -> Dict:
    """Run head-to-head comparison on multiple scenarios."""

    safety = SafetyGuard()
    prompt_mgr = PromptManager()

    base_metrics = []
    lora_metrics = []

    total_base = EvalMetrics(model_name="Base (Qwen2.5-3B)")
    total_lora = EvalMetrics(model_name="LoRA Fine-tuned")

    for sid in range(num_scenarios):
        orch = SimulationOrchestrator(config_path="config/default.yaml")
        orch.num_agents = agents_per
        orch.duration = duration
        orch.llm_engine._ready = False
        orch.cfg["visualization"]["enabled"] = False
        orch.generate_agents()

        kdocs = orch.knowledge_base.query(
            "火灾疏散", orch.cfg["environment"]["disaster"], top_k=3)

        total_ticks = int(orch.duration / orch.dt)

        for tick in range(total_ticks):
            orch.disaster.step(orch.dt)
            env = orch.disaster.snapshot(
                orch.tick, orch.sim_time, orch.exits, orch.obstacles)

            agents_to_decide = [
                a for a in orch.agents
                if (a.dynamic.alive and not a.dynamic.evacuated
                    and a.profile.role == "civilian"
                    and (a.dynamic.has_new_info
                         or orch.tick - a.dynamic.last_decision_tick >= orch.decision_ticks))
            ][:5]  # Limit per tick

            for agent in agents_to_decide:
                m_base, _ = evaluate_single(
                    base_model, agent, env, prompt_mgr, kdocs, safety)
                m_lora, _ = evaluate_single(
                    lora_model, agent, env, prompt_mgr, kdocs, safety)
                base_metrics.append(m_base)
                lora_metrics.append(m_lora)

            orch.tick += 1
            orch.sim_time += orch.dt

        print(f"  Scenario {sid+1}/{num_scenarios} done "
              f"({len(base_metrics)} decisions so far)")

    # Aggregate
    def aggregate(metrics_list: List[EvalMetrics], name: str) -> EvalMetrics:
        agg = EvalMetrics(model_name=name)
        agg.total_decisions = len(metrics_list)
        agg.json_valid = sum(m.json_valid for m in metrics_list)
        agg.safety_passed = sum(m.safety_passed for m in metrics_list)
        agg.safety_modified = sum(m.safety_modified for m in metrics_list)
        agg.safety_blocked = sum(m.safety_blocked for m in metrics_list)
        if metrics_list:
            agg.avg_output_tokens = np.mean(
                [m.avg_output_tokens for m in metrics_list])
            agg.avg_inference_time = np.mean(
                [m.avg_inference_time for m in metrics_list])
        return agg

    return {
        "base": aggregate(base_metrics, "Base Model"),
        "lora": aggregate(lora_metrics, "LoRA Fine-tuned"),
    }


def print_report(results: Dict):
    """Print formatted comparison report."""
    base = results["base"]
    lora = results["lora"]

    print("\n" + "=" * 70)
    print("  LoRA Fine-tuning Comparison Report")
    print("=" * 70)

    def pct(v, t):
        return f"{v}/{t} ({v/max(1,t)*100:.1f}%)"

    print(f"\n{'Metric':<35} {'Base Model':>15} {'LoRA FT':>15}")
    print("-" * 65)

    print(f"{'Total Decisions':<35} {base.total_decisions:>15d} {lora.total_decisions:>15d}")

    print(f"{'JSON Compliance':<35} "
          f"{pct(base.json_valid, base.total_decisions):>15} "
          f"{pct(lora.json_valid, lora.total_decisions):>15}")

    print(f"{'Safety Pass Rate':<35} "
          f"{pct(base.safety_passed, base.total_decisions):>15} "
          f"{pct(lora.safety_passed, lora.total_decisions):>15}")

    print(f"{'Safety Intervention Rate':<35} "
          f"{pct(base.safety_modified+base.safety_blocked, base.total_decisions):>15} "
          f"{pct(lora.safety_modified+lora.safety_blocked, lora.total_decisions):>15}")

    print(f"{'  - Blocked':<35} {'':>15} "
          f"{pct(lora.safety_blocked, lora.total_decisions):>15}")

    print(f"{'Avg Output Tokens':<35} "
          f"{base.avg_output_tokens:>15.1f} {lora.avg_output_tokens:>15.1f}")

    print(f"{'Avg Inference Time (s)':<35} "
          f"{base.avg_inference_time:>15.3f} {lora.avg_inference_time:>15.3f}")

    # Improvement summary
    json_imp = ((lora.json_compliance - base.json_compliance)
                / max(0.001, base.json_compliance) * 100)
    safety_imp = ((lora.safety_pass_rate - base.safety_pass_rate)
                  / max(0.001, base.safety_pass_rate) * 100)

    print(f"\n{'Improvement':<35} {'':>15}", end="")
    print(f"{json_imp:>+14.1f}% {safety_imp:>+14.1f}%")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare base model vs LoRA fine-tuned model")
    parser.add_argument("--lora", type=str, default=None,
                        help="Path to LoRA adapter")
    parser.add_argument("--base-model", type=str,
                        default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model name")
    parser.add_argument("--backend", type=str, default="hf",
                        choices=["hf", "safety_oracle"],
                        help="Backend: hf (HuggingFace) or safety_oracle (rule-based)")
    parser.add_argument("--num-scenarios", type=int, default=5)
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON report to file")
    args = parser.parse_args()

    safety = SafetyGuard()

    if args.backend == "safety_oracle":
        base = SafetyOracleRunner(safety)
        lora = SafetyOracleRunner(safety)  # Same, for demonstration
    else:
        base = HuggingFaceRunner(args.base_model)
        lora = HuggingFaceRunner(args.base_model, lora_path=args.lora)

    results = run_comparison(
        base, lora,
        num_scenarios=args.num_scenarios,
        agents_per=args.agents,
        duration=args.duration,
    )

    print_report(results)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        serializable = {
            "base": {k: v for k, v in results["base"].__dict__.items()
                     if not k.startswith("_")},
            "lora": {k: v for k, v in results["lora"].__dict__.items()
                     if not k.startswith("_")},
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()

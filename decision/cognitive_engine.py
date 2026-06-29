"""LLM Cognitive Engine — batched inference with vLLM for single GPU.

Key optimizations for 24GB 4090:
1. AWQ 4-bit quantization → model fits in ~2GB
2. Prefix caching → shared system prompts across agent groups
3. Continuous batching → maximizes GPU utilization
4. Async pipeline → decisions computed in background while physics runs
5. Rate limiting → only ~2% of agents re-decide per tick
"""

import json
import os
import time
import queue
import threading
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from decision.agent_state import Agent, Speed, Cooperation
from decision.prompt_manager import PromptManager
from perception.environment import EnvironmentSnapshot


@dataclass
class DecisionResult:
    agent_id: str
    target_exit_idx: int
    target_exit_pos: Tuple[float, float]
    speed: Speed
    cooperation: Cooperation
    reasoning: str
    risk_assessment: str
    compute_time: float
    raw_data: dict = None  # Full parsed LLM JSON (for command roles)


class LLMCognitiveEngine:
    """vLLM-powered batched inference for agent decisions.

    Usage:
        engine = LLMCognitiveEngine(config)
        engine.initialize()  # Loads model, warms up

        # Submit agents for async decision
        engine.submit_batch(agents, env_snapshot)

        # Call each tick to collect finished decisions
        results = engine.collect_results()
    """

    def __init__(self, config: dict):
        self.model_name = config.get("model", "Qwen/Qwen2.5-3B-Instruct-AWQ")
        self.quantization = config.get("quantization", "awq")
        self.max_model_len = config.get("max_model_len", 4096)
        self.gpu_memory = config.get("gpu_memory_utilization", 0.85)
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 128)
        self.batch_size = config.get("batch_size", 32)
        self.lora_path = config.get("lora_path", None)  # LoRA adapter path

        self.llm = None
        self.tokenizer = None
        self.prompt_manager = PromptManager()
        self._ready = False

        # v2.0: 双通道感知上下文 (由 orchestrator 每 tick 更新)
        self._vlm_description: str = ""
        self._yolo_result = None

        # Async pipeline (producer-consumer chain)
        # _pending_agents stores: (Agent, EnvironmentSnapshot, kdocs_map, rl_prefs)
        self._pending_agents: Dict[str, tuple] = {}
        self._results: Dict[str, DecisionResult] = {}
        self._inference_thread: Optional[threading.Thread] = None
        self._inference_lock = threading.Lock()
        self._result_queue = queue.Queue()

    def initialize(self):
        """Load model and warm up. Call once before simulation starts."""
        print(f"[CogEngine] Loading {self.model_name} ...")
        t0 = time.time()

        from vllm import LLM, SamplingParams

        vllm_kwargs = dict(
            model=self.model_name,
            quantization=self.quantization if self.quantization != "none" else None,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory,
            enable_prefix_caching=True,      # Critical for batch efficiency
            dtype="float16",
            trust_remote_code=True,
        )

        # vLLM LoRA adapter support
        if self.lora_path and os.path.isdir(self.lora_path):
            vllm_kwargs["enable_lora"] = True
            vllm_kwargs["max_lora_rank"] = 64
            print(f"[CogEngine] LoRA adapter enabled: {self.lora_path}")

        self.llm = LLM(**vllm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.9,
            stop=["\n\n"],  # Stop at double newline in case JSON is broken
        )

        elapsed = time.time() - t0
        self._ready = True
        print(f"[CogEngine] Model loaded in {elapsed:.1f}s. "
              f"Ready for inference.")

    def set_perception_context(self, vlm_description: str = "",
                                yolo_result=None):
        """v2.0: 设置双通道感知上下文, 供 Prompt 构建使用."""
        self._vlm_description = vlm_description
        self._yolo_result = yolo_result

    def _build_messages(self, agent: Agent, env: EnvironmentSnapshot,
                        knowledge_docs: List[str],
                        rl_preference: str = "") -> Tuple[list, str]:
        """Build chat messages. Returns (messages, group_key)."""
        role = agent.profile.role
        equipment = getattr(agent.profile, 'equipment', '')
        system = self.prompt_manager.build_system(
            env.disaster_type, knowledge_docs,
            role=role, equipment=equipment if isinstance(equipment, str) else '、'.join(equipment)
        )
        user = self.prompt_manager.build_user(
            agent, env,
            vlm_description=self._vlm_description,
            yolo_result=self._yolo_result,
            rl_preference=rl_preference,
        )

        # Group key: role + disaster_type for prefix caching
        group_key = f"{role}_{env.disaster_type}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], group_key

    def submit_batch(self, agents: List[Agent], env: EnvironmentSnapshot,
                     knowledge_docs_map: Optional[Dict[str, List[str]]] = None,
                     rl_preferences: Optional[Dict[str, str]] = None):
        """Submit agents for async batch inference. NON-BLOCKING.

        v2.1: rl_preferences — per-agent RL zone scheduler advice text.

        Architecture: producer-consumer chain.
        - New agents are added to _pending_agents (dedup by agent ID).
        - If no inference thread is running, _process_next_batch() is called
          immediately to consume and submit.
        - When inference finishes, _process_next_batch() is called again to
          drain any agents that arrived while the GPU was busy. This forms
          a self-draining chain — no agents are left buffered indefinitely.
        """
        if not self._ready:
            return

        # Backpressure: cap pending to prevent unbounded growth
        MAX_QUEUE = 600
        with self._inference_lock:
            current_pending = len(self._pending_agents)
        if current_pending >= MAX_QUEUE:
            return

        # Buffer new agents (dedup)
        with self._inference_lock:
            for a in agents:
                if a.id not in self._pending_agents:
                    self._pending_agents[a.id] = (a, env, knowledge_docs_map, rl_preferences)

        # If worker is already running, agents are buffered — it will
        # drain them when the current batch finishes.
        if self._inference_thread and self._inference_thread.is_alive():
            return

        # Start the drain chain
        self._process_next_batch()

    def _process_next_batch(self):
        """Consume all pending agents, format prompts, and launch inference.

        Called from submit_batch (cold start) and from _run_inference
        (chain continuation after previous batch finishes).
        """
        if not self._ready:
            self._inference_thread = None
            return
        with self._inference_lock:
            if not self._pending_agents:
                self._inference_thread = None
                return
            pending = list(self._pending_agents.values())
            self._pending_agents.clear()

        # Build all messages
        all_formatted = []
        all_agents = []
        for agent, env_snap, kdocs_map, rl_prefs in pending:
            # Route knowledge: civilians get general, professionals get domain
            if kdocs_map:
                if agent.profile.role == "civilian":
                    kdocs = kdocs_map.get("civilian", [])
                else:
                    kdocs = kdocs_map.get("professional", [])
            else:
                kdocs = []
            rl_pref = rl_prefs.get(agent.id, "") if rl_prefs else ""
            msgs, _ = self._build_messages(agent, env_snap, kdocs, rl_pref)
            all_formatted.append(msgs)
            all_agents.append(agent)

        if not all_agents:
            self._inference_thread = None
            return

        formatted_prompts = self.tokenizer.apply_chat_template(
            all_formatted, tokenize=False, add_generation_prompt=True)

        if not formatted_prompts or any(p is None for p in formatted_prompts):
            print("[CogEngine] Skipping batch: malformed prompts detected")
            for agent, env_snap, _, _ in pending:
                result = self._fallback_decision(agent, env_snap)
                with self._inference_lock:
                    self._results[agent.id] = result
            # Continue chain in case more agents arrived
            self._process_next_batch()
            return

        self._inference_thread = threading.Thread(
            target=self._run_inference,
            args=(formatted_prompts, all_agents, pending[0][1]),
            daemon=True,
        )
        self._inference_thread.start()

    def _run_inference(self, formatted_prompts, agents, env):
        """Background: run vLLM inference, store results, chain to next batch."""
        try:
            t0 = time.time()
            outputs = self.llm.generate(formatted_prompts, self.sampling_params)
            compute_time = time.time() - t0

            for agent, output in zip(agents, outputs):
                raw = output.outputs[0].text
                result = self._parse_decision(agent, raw, env, compute_time)
                with self._inference_lock:
                    self._results[agent.id] = result

        except Exception as e:
            print(f"[CogEngine] Inference error: {e}")
            for agent in agents:
                result = self._fallback_decision(agent, env)
                with self._inference_lock:
                    self._results[agent.id] = result

        # Chain: drain any agents that arrived while GPU was busy
        # Only chain if still ready — shutdown() may have been called
        if self._ready:
            self._process_next_batch()

    def collect_results(self) -> Dict[str, DecisionResult]:
        """Collect finished decisions. Non-blocking."""
        with self._inference_lock:
            results = dict(self._results)
            self._results.clear()
        return results

    def block_until_ready(self, timeout: float = 5.0):
        """Block until model is loaded."""
        while not self._ready and timeout > 0:
            time.sleep(0.1)
            timeout -= 0.1

    def _parse_decision(self, agent: Agent, raw_text: str,
                        env: EnvironmentSnapshot,
                        compute_time: float) -> DecisionResult:
        """Parse LLM output JSON → DecisionResult.

        For civilian agents: extracts target_exit, speed, cooperation, reasoning.
        For command agents: preserves full raw_data dict for role-specific parsing.
        """
        try:
            data = PromptManager.parse_response(raw_text)
        except json.JSONDecodeError:
            return self._fallback_decision(agent, env)

        # Parse target exit (common to civilian + guide)
        target_exit_idx = 0
        target_pos = env.exits[0] if env.exits else (0.0, 0.0)
        exit_str = data.get("target_exit", "")
        # Iterate descending to avoid substring match (出口1 ≠ 出口12)
        for i in range(len(env.exits), 0, -1):
            if f"出口{i}" in exit_str or f"exit{i}" in exit_str.lower():
                target_exit_idx = i - 1
                target_pos = env.exits[i - 1]
                break

        # Parse speed
        speed_str = data.get("speed", "walk").lower()
        try:
            speed = Speed(speed_str)
        except ValueError:
            speed = Speed.WALK

        # Parse cooperation
        coop_str = data.get("cooperation", "none").lower()
        try:
            cooperation = Cooperation(coop_str)
        except ValueError:
            cooperation = Cooperation.NONE

        return DecisionResult(
            agent_id=agent.id,
            target_exit_idx=target_exit_idx,
            target_exit_pos=target_pos,
            speed=speed,
            cooperation=cooperation,
            reasoning=data.get("reasoning", ""),
            risk_assessment=data.get("risk_assessment", ""),
            compute_time=compute_time,
            raw_data=data,  # Full parsed JSON for command roles
        )

    def _fallback_decision(self, agent: Agent,
                           env: EnvironmentSnapshot) -> DecisionResult:
        """Heuristic fallback: choose nearest unblocked exit."""
        pos = agent.position
        best_idx = 0
        best_pos = env.exits[0] if env.exits else (0.0, 0.0)
        best_score = float("inf")

        for i, exit_pos in enumerate(env.exits):
            dist = float(np.linalg.norm(np.array(exit_pos) - pos))
            smoke = env.smoke_at(np.array(exit_pos))
            # Score: distance + smoke penalty
            score = dist + smoke * 500
            if score < best_score:
                best_score = score
                best_idx = i
                best_pos = exit_pos

        return DecisionResult(
            agent_id=agent.id,
            target_exit_idx=best_idx,
            target_exit_pos=best_pos,
            speed=Speed.WALK,
            cooperation=Cooperation.NONE,
            reasoning=f"[回退] 选择最近出口{best_idx+1}, 距离{best_score:.0f}m",
            risk_assessment="自动评估",
            compute_time=0.0,
        )

    def shutdown(self):
        """Clean up vLLM resources and free GPU memory (aggressive)."""
        import torch
        import gc
        import subprocess
        import time

        if self.llm:
            try:
                del self.llm
            except Exception:
                pass
        self.llm = None
        self.tokenizer = None
        self._ready = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # vLLM 0.21.0 spawns a separate EngineCore process that may
        # survive del self.llm. Kill it explicitly to free GPU memory.
        try:
            subprocess.run(
                ['pkill', '-f', 'EngineCore'],
                capture_output=True, timeout=5)
        except Exception:
            pass
        time.sleep(1.5)

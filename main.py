"""main.py — Entry point for LLM-Powered Crowd Evacuation Simulation.

Single GPU (RTX 4090 24GB) Edition.

Usage:
    python main.py                        # Default config
    python main.py --config my_conf.yaml  # Custom config
    python main.py --no-viz               # Headless mode
    python main.py --web --port 8080      # Web visualization (GPU cloud)
    python main.py --gradio              # Gradio interactive dashboard
    python main.py --vlm --diffusion     # v2.0: VLM perception + diffusion trajectories
"""

import argparse
import sys
import os
import random

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from execution.orchestrator import SimulationOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="LLM-Powered Crowd Evacuation Simulation"
    )
    parser.add_argument(
        "--config", "-c", type=str, default="config/default.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Disable visualization (pure headless mode)"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable LLM entirely (heuristic-only simulation, no vLLM needed)"
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Save frames to disk as PNGs (headless, for video/gif later)"
    )
    parser.add_argument(
        "--frame-interval", type=int, default=10,
        help="Save frame every N ticks (default: 10 = 1 frame/sec at dt=0.1)"
    )
    parser.add_argument(
        "--agents", "-n", type=int, default=None,
        help="Override number of agents"
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=None,
        help="Override simulation duration in seconds"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override simulation random seed"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Override LLM model name"
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Launch web visualization server (for GPU cloud platforms)"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8080,
        help="Web server port (default: 8080)"
    )
    parser.add_argument(
        "--gradio", action="store_true",
        help="Launch Gradio interactive dashboard (with NL query, charts, manual intervention)"
    )
    parser.add_argument(
        "--gradio-port", type=int, default=8081,
        help="Gradio server port (default: 8081)"
    )
    parser.add_argument(
        "--vlm", action="store_true",
        help="Enable VLM visual perception (Qwen-VL-7B-INT4, ~5GB VRAM)"
    )
    parser.add_argument(
        "--yolo", action="store_true",
        help="Enable YOLO person detection (complementary to VLM)"
    )
    parser.add_argument(
        "--diffusion", action="store_true",
        help="Enable diffusion model trajectory generation (replaces social force)"
    )
    parser.add_argument(
        "--vlm-interval", type=int, default=None,
        help="VLM call interval in ticks (default: 30)"
    )
    parser.add_argument(
        "--vlm-mock", action="store_true",
        help="Use mock VLM (synthetic NL from env state, no GPU needed)"
    )
    parser.add_argument(
        "--lora", type=str, default=None,
        help="Path to LoRA adapter for fine-tuned LLM"
    )
    parser.add_argument(
        "--train-rl", action="store_true",
        help="Train RL zone scheduler offline (fast rule-based simulator, no LLM)"
    )
    parser.add_argument(
        "--train-rl-episodes", type=int, default=500,
        help="Number of offline RL training episodes (default: 500)"
    )
    parser.add_argument(
        "--irl-weights", type=str, default="data/irl_weights.json",
        help="Path to IRL-learned weights JSON for RL training"
    )
    parser.add_argument(
        "--rl-output", type=str, default="data/rl_policy.json",
        help="Output path for trained RL policy weights"
    )
    args = parser.parse_args()

    # --- Web 模式: 启动 Flask 服务器 ---
    if args.web:
        from visualization.web_server import start_server
        start_server(
            config_path=args.config,
            num_agents=args.agents,
            port=args.port,
            frame_interval=args.frame_interval,
        )
        return

    # --- Gradio 模式: 启动交互式仪表板 ---
    if args.gradio:
        from visualization.gradio_app import start_gradio
        start_gradio(
            config_path=args.config,
            num_agents=args.agents,
            port=args.gradio_port,
        )
        return

    # Build orchestrator
    orchestrator = SimulationOrchestrator(config_path=args.config)

    # Apply CLI overrides
    if args.agents is not None:
        orchestrator.num_agents = args.agents
        orchestrator.cfg["simulation"]["num_agents"] = args.agents

    if args.duration is not None:
        orchestrator.duration = args.duration
        orchestrator.cfg["simulation"]["duration"] = args.duration

    if args.seed is not None:
        orchestrator.cfg["simulation"]["seed"] = args.seed
        orchestrator.seed = args.seed
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.model:
        orchestrator.cfg["llm"]["model"] = args.model
        orchestrator.llm_engine.model_name = args.model

    if args.no_viz:
        orchestrator.cfg["visualization"]["enabled"] = False
    if args.no_llm:
        orchestrator.cfg.setdefault("llm", {})["enabled"] = False
    if args.record:
        orchestrator.cfg["visualization"]["mode"] = "headless"
        orchestrator.cfg["visualization"]["frame_interval"] = args.frame_interval

    # v2.0 feature toggles
    if args.vlm:
        orchestrator.cfg.setdefault("vlm", {})["enabled"] = True
    if args.yolo:
        orchestrator.cfg.setdefault("yolo", {})["enabled"] = True
    if args.diffusion:
        orchestrator.cfg.setdefault("diffusion", {})["enabled"] = True
    if args.vlm_interval is not None:
        orchestrator.cfg.setdefault("vlm", {})["call_interval"] = args.vlm_interval
    if args.vlm_mock:
        orchestrator.cfg.setdefault("vlm", {})["mock"] = True
    if args.lora:
        orchestrator.cfg.setdefault("llm", {})["lora_path"] = args.lora

    # RL training mode (offline, no LLM)
    if args.train_rl:
        orchestrator.train_rl_scheduler(
            episodes=args.train_rl_episodes,
            irl_weights_path=args.irl_weights,
            output_path=args.rl_output,
        )
        return

    # Run
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print("\n[main] Simulation interrupted by user.")
    except Exception as e:
        print(f"\n[main] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

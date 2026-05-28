"""main.py — Entry point for LLM-Powered Crowd Evacuation Simulation.

Single GPU (RTX 4090 24GB) Edition.

Usage:
    python main.py                        # Default config
    python main.py --config my_conf.yaml  # Custom config
    python main.py --no-viz               # Headless mode
    python main.py --batch-test           # Run multiple configs
"""

import argparse
import sys
import os

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
        "--model", "-m", type=str, default=None,
        help="Override LLM model name"
    )
    args = parser.parse_args()

    # Build orchestrator
    orchestrator = SimulationOrchestrator(config_path=args.config)

    # Apply CLI overrides
    if args.agents:
        orchestrator.num_agents = args.agents
        orchestrator.cfg["simulation"]["num_agents"] = args.agents

    if args.model:
        orchestrator.cfg["llm"]["model"] = args.model
        orchestrator.llm_engine.model_name = args.model

    if args.no_viz:
        orchestrator.cfg["visualization"]["enabled"] = False
    if args.record:
        orchestrator.cfg["visualization"]["mode"] = "headless"
        orchestrator.cfg["visualization"]["frame_interval"] = args.frame_interval

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

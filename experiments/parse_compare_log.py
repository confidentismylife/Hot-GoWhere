"""Reconstruct a compare report from a console log when the JSON was lost.

The comparison script saves raw JSON only at the very end. If report
generation crashes (or the log is all you have), this tool rebuilds the
same report from the per-run "SIMULATION COMPLETE" blocks in the console log.

Usage:
  python -m experiments.parse_compare_log \
      --log results/compare_rl_extreme_firepath_console.log \
      --output results/compare_rl_extreme_firepath_rebuilt
"""

import argparse
import json
import os
import re
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.compare_rl_heuristic import (  # noqa: E402
    aggregate,
    build_report,
    paired_diff,
)


def _num(block, pattern):
    for ln in block:
        m = re.search(pattern, ln)
        if m:
            return float(m.group(1))
    return None


def _role(block, label):
    pat = (rf"\[{label}\s*\]\s+(\d+) agents \| evac\s+(\d+)"
           rf".*?casualty\s+(\d+).*?remaining\s+(\d+)")
    m = None
    for ln in block:
        m = re.search(pat, ln)
        if m:
            break
    if not m:
        return {"total": 0, "evacuated": 0, "casualties": 0,
                "remaining": 0, "evac_rate": 0.0, "casualty_rate": 0.0}
    total, evac, cas, rem = (int(m.group(i)) for i in range(1, 5))
    return {
        "total": total,
        "evacuated": evac,
        "casualties": cas,
        "remaining": rem,
        "evac_rate": evac / total if total else 0.0,
        "casualty_rate": cas / total if total else 0.0,
    }


def _rules(block):
    rules = {}
    for ln in block:
        m = re.search(r"Safety rules:\s+(.+)", ln)
        if m:
            for kv in re.finditer(r"(\w+)=(\d+)", m.group(1)):
                rules[kv.group(1)] = int(kv.group(2))
    return rules


def parse_log(path):
    """Parse console log into rows matching the compare script schema."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    runs = []
    done_walls = []
    cur = None
    i = 0

    def close_pending():
        nonlocal cur
        if cur is not None and "total_agents" not in cur:
            cur.update({
                "error": "no SIMULATION COMPLETE block found in log for this run",
                "total_agents": 0,
                "evacuated": 0,
                "casualties": 0,
                "overall_evac_rate": 0.0,
                "overall_casualty_rate": 0.0,
                "civilian": {"total": 0, "evacuated": 0, "casualties": 0,
                             "remaining": 0, "evac_rate": 0.0,
                             "casualty_rate": 0.0},
                "command": {"total": 0, "evacuated": 0, "casualties": 0,
                            "remaining": 0, "evac_rate": 0.0,
                            "casualty_rate": 0.0},
                "civilian_evac_rate": 0.0,
                "command_evac_rate": 0.0,
                "safety_blocks": 0,
                "safety_modifications": 0,
                "decision_count": 0,
                "avg_tick_ms": 0.0,
                "max_tick_ms": 0.0,
                "sim_time": 0.0,
                "safety_rule_counts": {},
                "rl_enabled": cur["condition"] == "rl",
                "rl_policy_loaded": False,
                "llm_seed": None,
                "llm_enabled": True,
                "wall_time_s": 0.0,
            })
            runs.append(cur)
            cur = None

    while i < len(lines):
        line = lines[i]
        m = re.search(r"\[(\d+)/(\d+)\] condition=(\w+) seed=(\d+)", line)
        if m:
            close_pending()
            cur = {
                "condition": m.group(3),
                "seed": int(m.group(4)),
                "run_idx": int(m.group(1)),
                "total_runs": int(m.group(2)),
                "error": "",
            }
        elif cur is not None and "SIMULATION COMPLETE" in line:
            block = []
            i += 1
            # Skip the closing separator line that follows the banner.
            if i < len(lines) and re.search(r"^=+$", lines[i]):
                i += 1
            while i < len(lines) and not re.search(r"^=+$", lines[i]):
                block.append(lines[i])
                i += 1

            total = _num(block, r"Total agents:\s+(\d+)") or 0
            evac = _num(block, r"Evacuated:\s+(\d+)") or 0
            cas = _num(block, r"Casualties:\s+(\d+)") or 0
            civilian = _role(block, "Civilians")
            command = _role(block, "Command")

            cur.update({
                "total_agents": int(total),
                "evacuated": int(evac),
                "casualties": int(cas),
                "overall_evac_rate": evac / total if total else 0.0,
                "overall_casualty_rate": cas / total if total else 0.0,
                "civilian": civilian,
                "command": command,
                "civilian_evac_rate": civilian["evac_rate"],
                "command_evac_rate": command["evac_rate"],
                "safety_blocks": int(_num(block, r"Safety blocked:\s+(\d+)") or 0),
                "safety_modifications": int(
                    _num(block, r"Safety modified:\s+(\d+)") or 0),
                "decision_count": int(
                    _num(block, r"LLM decisions:\s+(\d+)") or 0),
                "avg_tick_ms": _num(block, r"Avg tick time:\s+([\d.]+)ms") or 0.0,
                "max_tick_ms": _num(block, r"Max tick time:\s+([\d.]+)ms") or 0.0,
                "sim_time": _num(block, r"Duration:\s+([\d.]+)s") or 0.0,
                "safety_rule_counts": _rules(block),
                "rl_enabled": cur["condition"] == "rl",
                "rl_policy_loaded": any(
                    re.search(r"RL policy:\s+LOADED", ln) for ln in block),
                "llm_seed": None,
                "llm_enabled": True,
                "wall_time_s": 0.0,
            })
            runs.append(cur)
            cur = None
            continue
        else:
            m = re.search(r"done: evac=.*?wall=([\d.]+)s", line)
            if m:
                done_walls.append(float(m.group(1)))
        i += 1

    close_pending()

    for row, wall in zip(runs, done_walls):
        row["wall_time_s"] = wall
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild compare report from a console log")
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", default="results/compare_from_log")
    parser.add_argument("--rl-config",
                        default="config/mall_floorplan_rl_extreme.yaml")
    parser.add_argument("--heuristic-config",
                        default="config/mall_floorplan_extreme.yaml")
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--duration", type=float, default=360.0)
    args = parser.parse_args()

    rows = parse_log(args.log)
    if not rows:
        print(f"No runs parsed from {args.log}")
        return 1

    seeds = sorted({r["seed"] for r in rows})
    summary = {
        cond: aggregate(rows, cond) for cond in ("rl", "heuristic")
    }
    args_obj = SimpleNamespace(
        rl_config=args.rl_config,
        heuristic_config=args.heuristic_config,
        agents=args.agents,
        duration=args.duration,
        seeds=seeds,
        no_llm=False,
    )
    preflight = [
        f"从控制台日志重建: `{args.log}`",
        f"成功解析 {len(rows)} 轮运行",
    ]
    report = build_report(rows, summary, args_obj, preflight)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_log": args.log,
        "rebuilt_from_log": True,
        "runs": rows,
        "summary": summary,
        "paired_diff": paired_diff(rows),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json_path = f"{args.output}.json"
    md_path = f"{args.output}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Rebuilt report from log:\n  {json_path}\n  {md_path}")
    try:
        print(report)
    except UnicodeEncodeError:
        print("(Report contains non-ASCII characters the console cannot "
              "display; see the .md file.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

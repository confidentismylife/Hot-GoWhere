"""Multi-seed RL vs heuristic comparison runner with automatic result saving.

Runs N seeds × 2 conditions (RL-scheduled config vs plain heuristic config),
collects overall / civilian / command statistics from every simulation, and
automatically saves:

  <output>.json  — one row per run + per-condition aggregates
  <output>.md    — human-readable Markdown report

Usage (from project root, on the GPU machine with vLLM):
  python -m experiments.compare_rl_heuristic \
      --rl-config config/mall_floorplan_rl_v3.yaml \
      --heuristic-config config/mall_floorplan.yaml \
      --agents 82 --duration 360 \
      --seeds 42 43 44 45 46 \
      --output results/compare_rl_heuristic_5seeds

By default the vLLM engine is loaded ONCE and reused for every seed/condition
(a large speedup); pass ``--no-reuse-llm`` to load/unload the model per run.

Local smoke test (no LLM, no GPU):
  python -m experiments.compare_rl_heuristic \
      --agents 6 --duration 3 --seeds 1 2 --no-llm \
      --output results/_smoke_compare

Notes:
  - The RL condition is only a real trained-policy run when the log shows
    ``[RLScheduler] Loaded pretrained weights ...``. This script records
    ``rl_policy_loaded`` per run so a silent heuristic fallback is visible in
    the report instead of being mistaken for RL.
  - Command agents are spawned on top of ``--agents`` and included in the
    overall rates; the civilian / command split is reported separately.
"""

import argparse
import copy
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from execution.orchestrator import SimulationOrchestrator  # noqa: E402


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def make_temp_config(base_cfg, seed, agents, duration, no_llm, out_dir, tag):
    """Deep-copy a config, override run parameters, write a temp YAML."""
    cfg = copy.deepcopy(base_cfg)
    sim = cfg.setdefault("simulation", {})
    sim["seed"] = int(seed)
    sim["num_agents"] = int(agents)
    sim["duration"] = float(duration)
    cfg.setdefault("visualization", {})["enabled"] = False
    if no_llm:
        cfg.setdefault("llm", {})["enabled"] = False
    path = os.path.join(out_dir, f"_tmp_{tag}_{seed}.yaml")
    write_yaml(cfg, path)
    return path


def run_once(orch, condition, seed, config_path, llm_enabled):
    """Run one simulation and extract all stats for the report."""
    rs = orch.role_stats or {}

    def role(key):
        return {
            "total": rs.get(key, {}).get("total", 0),
            "evacuated": rs.get(key, {}).get("evacuated", 0),
            "casualties": rs.get(key, {}).get("casualties", 0),
            "remaining": rs.get(key, {}).get("remaining", 0),
            "evac_rate": rs.get(key, {}).get("evac_rate", 0.0),
            "casualty_rate": rs.get(key, {}).get("casualty_rate", 0.0),
        }

    n = max(1, len(orch.agents))
    return {
        "condition": condition,
        "seed": int(seed),
        "config": config_path,
        "total_agents": len(orch.agents),
        "evacuated": orch.evacuated_count,
        "casualties": orch.casualty_count,
        "overall_evac_rate": orch.evacuated_count / n,
        "overall_casualty_rate": orch.casualty_count / n,
        "civilian": role("civilian"),
        "command": role("command"),
        "civilian_evac_rate": role("civilian")["evac_rate"],
        "command_evac_rate": role("command")["evac_rate"],
        "safety_blocks": orch.safety_blocks,
        "safety_modifications": orch.safety_modifications,
        "decision_count": orch.decision_count,
        "avg_tick_ms": float(getattr(orch, "avg_tick_ms", 0.0)),
        "max_tick_ms": float(getattr(orch, "max_tick_ms", 0.0)),
        "sim_time": float(orch.sim_time),
        "rl_enabled": bool(getattr(orch, "enable_rl_scheduling", False)),
        "rl_policy_loaded": bool(getattr(orch, "rl_policy_loaded", False)),
        "safety_rule_counts": dict(getattr(orch, "safety_rule_counts", {})),
        "llm_seed": getattr(
            getattr(orch, "llm_engine", None), "sampling_seed", None),
        "llm_enabled": bool(llm_enabled),
        "wall_time_s": 0.0,
        "error": "",
    }


def fmt_stat(stat, digits=3):
    if stat is None:
        return "-"
    return (f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f} "
            f"[{stat['min']:.{digits}f}, {stat['max']:.{digits}f}]")


def aggregate(rows, condition):
    sub = [r for r in rows
           if r["condition"] == condition and not r.get("error")]
    if not sub:
        return None

    def stat(key):
        vals = [r[key] for r in sub if r.get(key) is not None]
        if not vals:
            return None
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": len(vals),
        }

    return {
        "n_runs": len(sub),
        "overall_evac_rate": stat("overall_evac_rate"),
        "overall_casualty_rate": stat("overall_casualty_rate"),
        "civilian_evac_rate": stat("civilian_evac_rate"),
        "command_evac_rate": stat("command_evac_rate"),
        "safety_blocks": stat("safety_blocks"),
        "safety_modifications": stat("safety_modifications"),
        "avg_tick_ms": stat("avg_tick_ms"),
        "wall_time_s": stat("wall_time_s"),
        "rule_means": {
            rule: float(np.mean([
                r.get("safety_rule_counts", {}).get(rule, 0) for r in sub
            ]))
            for rule in sorted({
                k for r in sub for k in r.get("safety_rule_counts", {})
            })
        },
    }


def paired_diff(rows):
    """Per-seed heuristic-minus-RL overall evac rate difference."""
    by_seed = {}
    for r in rows:
        if r.get("error"):
            continue
        by_seed.setdefault(r["seed"], {})[r["condition"]] = r
    diffs = []
    for seed, pair in sorted(by_seed.items()):
        heu = pair.get("heuristic")
        rl = pair.get("rl")
        if heu is None or rl is None:
            continue
        diffs.append(heu["overall_evac_rate"] - rl["overall_evac_rate"])
    if not diffs:
        return None
    return {
        "n": len(diffs),
        "mean": float(np.mean(diffs)),
        "std": float(np.std(diffs)),
        "min": float(np.min(diffs)),
        "max": float(np.max(diffs)),
        "per_seed": [round(d, 4) for d in diffs],
    }


def build_report(rows, summary, args, preflight):
    lines = []
    lines.append("# RL vs Heuristic — 多种子对比报告")
    lines.append("")
    lines.append(f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- RL 配置: `{args.rl_config}`")
    lines.append(f"- Heuristic 配置: `{args.heuristic_config}`")
    lines.append(f"- Agents: {args.agents}, Duration: {args.duration}s, "
                 f"Seeds: {', '.join(str(s) for s in args.seeds)}")
    lines.append(f"- LLM: {'off (--no-llm)' if args.no_llm else 'on'}")
    lines.append("")
    lines.append("## 预检")
    lines.append("")
    for line in preflight:
        lines.append(f"- {line}")
    lines.append("")
    lines.append("## 逐种子结果")
    lines.append("")
    lines.append("| condition | seed | total | evac | evac% | casualty% | "
                 "civ evac% | cmd evac% | safety mod | safety block | "
                 "avg tick ms | rl loaded | error |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['condition']} | {r['seed']} | {r['total_agents']} | "
            f"{r['evacuated']} | {r['overall_evac_rate']*100:.1f}% | "
            f"{r['overall_casualty_rate']*100:.1f}% | "
            f"{r['civilian_evac_rate']*100:.1f}% | "
            f"{r['command_evac_rate']*100:.1f}% | "
            f"{r['safety_modifications']} | {r['safety_blocks']} | "
            f"{r['avg_tick_ms']:.1f} | "
            f"{'Y' if r['rl_policy_loaded'] else 'N'} | {r['error']} |"
        )
    lines.append("")
    lines.append("## 汇总（均值 ± 标准差 [min, max]）")
    lines.append("")
    lines.append("| condition | n | evac% | casualty% | civ evac% | cmd evac% | "
                 "safety mod | safety block | avg tick ms |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cond in ("rl", "heuristic"):
        a = summary.get(cond)
        if a is None:
            lines.append(f"| {cond} | 0 | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {cond} | {a['n_runs']} | {fmt_stat(a['overall_evac_rate'])} | "
            f"{fmt_stat(a['overall_casualty_rate'])} | "
            f"{fmt_stat(a['civilian_evac_rate'])} | "
            f"{fmt_stat(a['command_evac_rate'])} | "
            f"{fmt_stat(a['safety_modifications'], 1)} | "
            f"{fmt_stat(a['safety_blocks'], 1)} | "
            f"{fmt_stat(a['avg_tick_ms'], 1)} |"
        )
    lines.append("")
    lines.append("## 安全规则明细（每轮平均触发次数）")
    lines.append("")
    rules = sorted({
        k for r in rows for k in r.get("safety_rule_counts", {})
    })
    lines.append("| 规则 | Heuristic | RL |")
    lines.append("|---|---|---|")
    h_summary = summary.get("heuristic") or {}
    r_summary = summary.get("rl") or {}
    for rule in rules:
        h = h_summary.get("rule_means", {}).get(rule)
        r = r_summary.get("rule_means", {}).get(rule)
        if h is not None and r is not None:
            lines.append(f"| {rule} | {h:.2f} | {r:.2f} |")
        else:
            lines.append(f"| {rule} | - | - |")
    lines.append("")
    pd = paired_diff(rows)
    if pd is not None:
        lines.append("## 配对差异（heuristic − rl，总体疏散率）")
        lines.append("")
        lines.append(f"- n = {pd['n']}")
        lines.append(f"- 均值差 = {pd['mean']*100:.2f} 个百分点 "
                     f"(± {pd['std']*100:.2f})")
        lines.append(f"- 逐种子: {', '.join(f'{d*100:.2f}%' for d in pd['per_seed'])}")
        lines.append("")
        if pd["mean"] > 0:
            lines.append("> 该批种子下 heuristic 总体疏散率平均更高。")
        elif pd["mean"] < 0:
            lines.append("> 该批种子下 RL 总体疏散率平均更高。")
        else:
            lines.append("> 该批种子下两者总体疏散率持平。")
        lines.append("")
    lines.append("## 注意事项")
    lines.append("")
    lines.append("- 若任意 RL 行为 `rl loaded = N`，说明该次没有加载训练权重，"
                 "而是回退为启发式建议，结果不能代表训练后的策略。")
    lines.append("- 总体疏散率的分母包含 command agents；"
                 "普通人员与指挥人员的指标已分开列出。")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed RL vs heuristic comparison with auto saving")
    parser.add_argument("--rl-config", default="config/mall_floorplan_rl_v3.yaml")
    parser.add_argument("--heuristic-config",
                        default="config/mall_floorplan.yaml")
    parser.add_argument("--agents", type=int, default=82)
    parser.add_argument("--duration", type=float, default=360.0)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 43, 44, 45, 46, 47, 48, 49])
    parser.add_argument("--output", default="results/compare_rl_heuristic")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM (local smoke testing only)")
    parser.add_argument("--no-reuse-llm", action="store_true",
                        help="Load/unload the LLM engine for every run "
                             "instead of reusing one engine for all runs")
    parser.add_argument("--report-only", type=str, default=None,
                        help="Regenerate the Markdown report from an existing "
                             "compare JSON file without re-running simulations")
    args = parser.parse_args()

    if args.report_only:
        with open(args.report_only, "r", encoding="utf-8") as f:
            payload = json.load(f)
        args_obj = SimpleNamespace(**payload.get("args", {}))
        report = build_report(
            payload.get("runs", []),
            payload.get("summary") or {},
            args_obj,
            payload.get("preflight", []),
        )
        md_path = args.report_only.rsplit(".", 1)[0] + ".md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report regenerated from {args.report_only}\n  {md_path}")
        try:
            print(report)
        except UnicodeEncodeError:
            print("(Report contains non-ASCII characters the console cannot "
                  "display; see the .md file.)")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ---- Preflight: make sure the RL config points at a real policy file ----
    preflight = []
    rl_cfg = load_yaml(args.rl_config)
    weights = rl_cfg.get("rl_scheduling", {}).get("pretrained_weights", "")
    if weights:
        if os.path.exists(weights):
            preflight.append(
                f"RL 权重文件存在: `{weights}`（将被加载）")
        else:
            preflight.append(
                f"⚠️ RL 权重文件不存在: `{weights}` —— RL 条件会回退为启发式建议，"
                f"结果不能代表训练后的策略")
    else:
        preflight.append("⚠️ RL 配置未设置 `pretrained_weights`，RL 条件实际为启发式建议")
    for p in (args.rl_config, args.heuristic_config):
        preflight.append(f"配置存在: `{p}` — {os.path.exists(p)}")

    base_cfgs = {
        "rl": load_yaml(args.rl_config),
        "heuristic": load_yaml(args.heuristic_config),
    }

    # ---- Shared LLM engine: load once, run all seeds ----
    reuse_llm = not args.no_llm and not args.no_reuse_llm
    shared_engine = None
    if reuse_llm:
        from decision.cognitive_engine import LLMCognitiveEngine
        llm_cfg = base_cfgs["rl"].get("llm", {})
        shared_engine = LLMCognitiveEngine(config=llm_cfg)
        shared_engine.initialize()
        preflight.append(
            f"LLM 引擎复用: 已加载一次（{llm_cfg.get('model', '?')}），"
            f"跑完全部种子后统一关闭")
    else:
        preflight.append(
            "LLM 引擎: 每次运行单独加载/关闭"
            if not args.no_llm else "LLM: 已禁用（--no-llm）")
    llm_fixed = base_cfgs["rl"].get("llm", {}).get("fixed_seed", False)
    preflight.append(
        f"LLM 采样 seed: {'固定（seed = 仿真 seed）' if llm_fixed else '不固定（随机）'}")

    rows = []
    total_runs = len(args.seeds) * 2
    run_i = 0
    for seed in args.seeds:
        for condition in ("rl", "heuristic"):
            run_i += 1
            tag = f"{condition}_{seed}"
            print(f"\n{'='*70}\n[{run_i}/{total_runs}] "
                  f"condition={condition} seed={seed}\n{'='*70}")
            tmp_path = None
            row = None
            t0 = time.perf_counter()
            try:
                tmp_path = make_temp_config(
                    base_cfgs[condition], seed, args.agents, args.duration,
                    args.no_llm, os.path.dirname(args.output) or ".",
                    tag)
                orch = SimulationOrchestrator(
                    config_path=tmp_path,
                    llm_engine=shared_engine if reuse_llm else None,
                    keep_llm_engine=reuse_llm,
                )
                llm_enabled = bool(orch.cfg.get("llm", {}).get("enabled", False))
                orch.run()
                row = run_once(orch, condition, seed, tmp_path, llm_enabled)
            except Exception as e:
                import traceback
                traceback.print_exc()
                row = {
                    "condition": condition,
                    "seed": int(seed),
                    "config": tmp_path or "",
                    "total_agents": 0,
                    "evacuated": 0,
                    "casualties": 0,
                    "overall_evac_rate": 0.0,
                    "overall_casualty_rate": 0.0,
                    "civilian": {},
                    "command": {},
                    "civilian_evac_rate": 0.0,
                    "command_evac_rate": 0.0,
                    "safety_blocks": 0,
                    "safety_modifications": 0,
                    "decision_count": 0,
                    "avg_tick_ms": 0.0,
                    "max_tick_ms": 0.0,
                    "sim_time": 0.0,
                    "rl_enabled": condition == "rl",
                    "rl_policy_loaded": False,
                    "safety_rule_counts": {},
                    "llm_seed": None,
                    "llm_enabled": False,
                    "wall_time_s": time.perf_counter() - t0,
                    "error": f"{type(e).__name__}: {e}",
                }
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                if shared_engine is not None:
                    # Make sure no inference thread / pending agents leak into
                    # the next run, then clear per-run counters.
                    try:
                        shared_engine.drain(timeout=10.0)
                    except Exception:
                        pass
                    shared_engine.reset_stats()
            if row is not None:
                row["wall_time_s"] = time.perf_counter() - t0
                rows.append(row)
                print(f"[{run_i}/{total_runs}] {condition} seed={seed} done: "
                      f"evac={row['overall_evac_rate']*100:.1f}% "
                      f"casualty={row['overall_casualty_rate']*100:.1f}% "
                      f"wall={row['wall_time_s']:.0f}s")

    if shared_engine is not None:
        shared_engine.shutdown()
        print("[Compare] Shared LLM engine shut down.")

    summary = {
        cond: aggregate(rows, cond) for cond in ("rl", "heuristic")
    }
    for cond in ("rl", "heuristic"):
        if summary.get(cond) is None:
            errs = [r for r in rows
                    if r["condition"] == cond and r.get("error")]
            print(f"[Compare] WARNING: no valid {cond} runs for aggregation "
                  f"({len(errs)} errored). Report rows will be incomplete.")

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "preflight": preflight,
        "runs": rows,
        "summary": summary,
        "paired_diff": paired_diff(rows),
    }
    json_path = f"{args.output}.json"
    md_path = f"{args.output}.md"
    # Save raw data FIRST so a report-generation bug can never lose results.
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    try:
        report = build_report(rows, summary, args, preflight)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nResults saved to:\n  {json_path}\n  {md_path}")
        try:
            print(report)
        except UnicodeEncodeError:
            print("(Report contains non-ASCII characters the console cannot "
                  "display; see the .md file.)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[Compare] WARNING: Markdown report failed ({e}). "
              f"Raw results are safe in {json_path}; "
              f"rerun with --report-only {json_path} to regenerate.")


if __name__ == "__main__":
    main()

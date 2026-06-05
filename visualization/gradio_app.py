"""Gradio interactive dashboard — LLM Evacuation Simulation v2.1.

Features:
  - Live frame display (auto-refresh via file-based rendering)
  - NL query: "出口3现在什么情况？" → stats-based answer
  - Manual commander intervention: type a broadcast message
  - Evacuation progress chart (Plotly time series)
  - Role distribution and safety stats

Usage:
    python visualization/gradio_app.py --config config/default.yaml --port 8081
    python main.py --gradio --gradio-port 8081
"""

import os
import sys
import io
import time
import tempfile
import threading
import argparse
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gradio as gr

from decision.agent_state import Speed
from perception.environment import EnvironmentSnapshot

# ================================================================
# Shared state (simulation thread ↔ Gradio thread)
# ================================================================

_sim_state = {
    "running": False,
    "done": False,
    "error": None,
    "frame_path": None,      # Path to latest rendered PNG file
    "frame_count": 0,
    "tick": 0,
    "sim_time": 0.0,
    "active_count": 0,
    "evacuated_count": 0,
    "casualty_count": 0,
    "decision_count": 0,
    "total_agents": 0,
    "avg_tick_ms": 0.0,
    "safety_blocks": 0,
    "safety_modifications": 0,
    "history": [],
    "lock": threading.Lock(),
}

_manual_broadcasts: list = []
_broadcast_lock = threading.Lock()
_orchestrator_ref = None

# Temp directory for frame files
_FRAME_DIR = os.path.join(tempfile.gettempdir(), "evac_frames")


# ================================================================
# Frame renderer
# ================================================================

def _fear_color(fear_level: float):
    t = fear_level / 10.0
    return (t * 2, 1.0, 0.2) if t < 0.5 else (1.0, 1.0 - (t - 0.5) * 2, 0.2)


def render_frame_pil(agents, env, tick, sim_time, evacuated, casualties,
                     decisions, world_w, world_h):
    """Render a frame → PIL Image."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    ax.set_xlim(0, world_w)
    ax.set_ylim(0, world_h)
    ax.set_aspect('equal')
    ax.set_facecolor('#F0F0F5')

    # Smoke
    step = max(1, env.grid.shape[1] // 60)
    for r in range(0, env.grid.shape[0], step):
        for c in range(0, env.grid.shape[1], step):
            smoke = env.grid[r, c, 0]
            if smoke < 0.05:
                continue
            wx = c * env.grid_resolution
            wy = r * env.grid_resolution
            rect = plt.Rectangle(
                (wx, wy), env.grid_resolution * step, env.grid_resolution * step,
                facecolor='gray', alpha=min(0.7, smoke * 0.8), edgecolor='none'
            )
            ax.add_patch(rect)

    # Fire
    fire_mask = env.grid[:, :, 3] > 0.5
    if fire_mask.any():
        rows, cols = np.where(fire_mask)
        for r, c in zip(rows[::step], cols[::step]):
            wx = c * env.grid_resolution
            wy = r * env.grid_resolution
            rect = plt.Rectangle(
                (wx, wy), env.grid_resolution * step, env.grid_resolution * step,
                facecolor='#FF6414', alpha=0.6, edgecolor='none'
            )
            ax.add_patch(rect)

    # Obstacles
    for obs in env.obstacles:
        circle = plt.Circle(
            obs["center"], obs["radius"],
            facecolor='#B4B4B9', edgecolor='#8C8C91', linewidth=1.5
        )
        ax.add_patch(circle)

    # Exits
    for i, exit_pos in enumerate(env.exits):
        ex, ey = exit_pos
        gr_idx = min(int(ey / env.grid_resolution), env.grid.shape[0] - 1)
        gc = min(int(ex / env.grid_resolution), env.grid.shape[1] - 1)
        exit_smoke = float(env.grid[gr_idx, gc, 0])
        color = 'green' if exit_smoke < 0.3 else ('orange' if exit_smoke < 0.6 else 'red')
        rect = plt.Rectangle(
            (ex - 1.2, ey - 1.2), 2.4, 2.4,
            facecolor=color, alpha=0.7, edgecolor='darkgreen', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(ex + 1.5, ey, f'E{i+1}', fontsize=8, fontweight='bold')

    # Agents
    role_colors = {
        "civilian": '#58a6ff',
        "global_commander": '#FFD700',
        "area_commander": '#FF8C00',
        "firefighter": '#FF4444',
        "guide": '#00FF88',
    }
    active = [a for a in agents if a.dynamic.alive and not a.dynamic.evacuated]
    if active:
        for role in set(a.profile.role for a in active):
            role_agents = [a for a in active if a.profile.role == role]
            positions = np.array([a.position for a in role_agents])
            colors = [role_colors.get(role, '#58a6ff')] * len(role_agents)
            sizes = [max(6, min(18, a.profile.max_speed * 8)) for a in role_agents]
            marker = 's' if role != 'civilian' else 'o'
            ax.scatter(positions[:, 0], positions[:, 1],
                       c=colors, s=sizes, alpha=0.85, marker=marker,
                       edgecolors='white', linewidth=0.3)

    # HUD
    alive = len(active)
    total = alive + evacuated + casualties
    evac_rate = (evacuated / total * 100) if total > 0 else 0
    hud_text = (
        f"Time: {sim_time:.0f}s  Tick: {tick}\n"
        f"Active: {alive}  Evacuated: {evacuated}  Dead: {casualties}\n"
        f"LLM Decisions: {decisions}  Evac Rate: {evac_rate:.1f}%"
    )
    ax.text(0.02, 0.98, hud_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.set_xticks([])
    ax.set_yticks([])

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=80, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf)


# ================================================================
# Background simulation thread
# ================================================================

def run_simulation_thread(config_path: str, num_agents: Optional[int] = None):
    global _sim_state, _orchestrator_ref

    from execution.orchestrator import SimulationOrchestrator

    # Ensure frame dir exists
    os.makedirs(_FRAME_DIR, exist_ok=True)
    frame_path_a = os.path.join(_FRAME_DIR, "frame_a.png")
    frame_path_b = os.path.join(_FRAME_DIR, "frame_b.png")
    _toggle = False

    try:
        orch = SimulationOrchestrator(config_path)
        if num_agents:
            orch.num_agents = num_agents
            orch.cfg["simulation"]["num_agents"] = num_agents
        orch.cfg["visualization"]["enabled"] = False
        orch.cfg["visualization"]["mode"] = "headless"

        with _sim_state["lock"]:
            _sim_state["running"] = True
            _sim_state["total_agents"] = orch.num_agents

        _orchestrator_ref = orch

        orch.generate_agents()
        orch._spawn_command_agents()
        orch.llm_engine.initialize()

        orch.use_vlm = orch.cfg.get("vlm", {}).get("enabled", False)
        orch.use_diffusion = orch.cfg.get("diffusion", {}).get("enabled", False)
        orch.vlm = None; orch.yolo = None; orch.diffusion_policy = None

        total_ticks = int(orch.duration / orch.dt)
        tick_times = []

        print(f"[Gradio] Simulation started: {total_ticks} ticks, "
              f"{orch.num_agents} agents (+ command)")

        while orch.tick < total_ticks:
            tick_start = time.perf_counter()

            orch.disaster.step(orch.dt)
            env_snapshot = orch.disaster.snapshot(
                orch.tick, orch.sim_time, orch.exits, orch.obstacles,
                official_broadcast=orch._get_broadcast()
            )

            with _broadcast_lock:
                if _manual_broadcasts:
                    orch._command_broadcasts.extend(_manual_broadcasts)
                    _manual_broadcasts.clear()

            # Cognition
            agents_to_decide = [
                a for a in orch.agents
                if (a.dynamic.alive and not a.dynamic.evacuated and
                    (a.dynamic.has_new_info or
                     orch.tick - a.dynamic.last_decision_tick >= orch.decision_ticks))
            ]
            if agents_to_decide:
                # Cache KB query (knowledge docs don't change mid-simulation)
                if orch.tick - orch._kb_cache_tick > 30:
                    orch._kb_cache[orch.cfg['environment']['disaster']] = \
                        orch.knowledge_base.query(
                            f"{orch.cfg['environment']['disaster']}疏散决策",
                            disaster_type=orch.cfg['environment']['disaster'], top_k=3)
                    orch._kb_cache_tick = orch.tick
                orch.llm_engine.submit_batch(
                    agents_to_decide, env_snapshot,
                    {orch.cfg['environment']['disaster']:
                     orch._kb_cache.get(orch.cfg['environment']['disaster'], [])})

            decisions = orch.llm_engine.collect_results()
            if decisions:
                civilian_decisions = {}
                command_decisions = {}
                for aid, d in decisions.items():
                    agent = orch._find_agent(aid)
                    if agent and agent.profile.role != "civilian":
                        command_decisions[aid] = d
                    else:
                        civilian_decisions[aid] = d
                orch.decision_count += len(decisions)
                orch.total_llm_time += sum(d.compute_time for d in decisions.values())
                if civilian_decisions:
                    orch._apply_decisions(civilian_decisions, env_snapshot)
                if command_decisions:
                    orch._apply_command_decisions(command_decisions, env_snapshot)

            orch.group_intel.propagate(orch.agents, env_snapshot.official_broadcast, orch.dt)
            orch.group_intel.update_fear_levels(orch.agents, env_snapshot, orch.dt)
            orch.group_intel.update_stamina(orch.agents, orch.dt)
            orch.physics.step_all(orch.agents, orch.dt)

            # Stats: single pass over agents
            evac = 0; dead = 0
            for a in orch.agents:
                if a.dynamic.evacuated: evac += 1
                elif not a.dynamic.alive: dead += 1
            orch.evacuated_count = evac
            orch.casualty_count = dead
            active_count = orch.num_agents - evac - dead

            # ---- Render frame to FILE (most reliable for Gradio) ----
            if orch.tick % 10 == 0:
                try:
                    frame_img = render_frame_pil(
                        orch.agents, env_snapshot,
                        orch.tick, orch.sim_time,
                        orch.evacuated_count, orch.casualty_count,
                        orch.decision_count, orch.width, orch.height,
                    )
                    # Toggle between two files to avoid browser caching
                    _toggle = not _toggle
                    out_path = frame_path_a if _toggle else frame_path_b
                    frame_img.save(out_path, format='PNG')
                    # Also save to a stable path for initial load
                    frame_img.save(os.path.join(_FRAME_DIR, "latest.png"), format='PNG')

                    with _sim_state["lock"]:
                        _sim_state["frame_path"] = out_path
                        _sim_state["frame_count"] += 1
                except Exception as e:
                    print(f"[Gradio] Render error: {e}")
                    import traceback
                    traceback.print_exc()

            with _sim_state["lock"]:
                _sim_state["tick"] = orch.tick
                _sim_state["sim_time"] = orch.sim_time
                _sim_state["active_count"] = active_count
                _sim_state["evacuated_count"] = orch.evacuated_count
                _sim_state["casualty_count"] = orch.casualty_count
                _sim_state["decision_count"] = orch.decision_count
                _sim_state["safety_blocks"] = orch.safety_blocks
                _sim_state["safety_modifications"] = orch.safety_modifications
                _sim_state["avg_tick_ms"] = np.mean(tick_times[-100:]) if tick_times else 0
                _sim_state["history"].append({
                    "tick": orch.tick, "sim_time": round(orch.sim_time, 1),
                    "active": active_count, "evacuated": orch.evacuated_count,
                    "casualties": orch.casualty_count, "decisions": orch.decision_count,
                })

            tick_time = (time.perf_counter() - tick_start) * 1000
            tick_times.append(tick_time)
            orch.tick += 1
            orch.sim_time += orch.dt

            remaining = orch.num_agents - orch.evacuated_count - orch.casualty_count
            if remaining <= 0:
                break

        orch.llm_engine.shutdown()
        with _sim_state["lock"]:
            _sim_state["done"] = True
            _sim_state["running"] = False
        print(f"[Gradio] Simulation complete. Evac: {orch.evacuated_count}/{orch.num_agents}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        with _sim_state["lock"]:
            _sim_state["error"] = str(e)
            _sim_state["running"] = False
            _sim_state["done"] = True


# ================================================================
# NL Query engine
# ================================================================

def answer_query(question: str) -> str:
    """Answer natural language questions about the current simulation state."""
    global _orchestrator_ref

    with _sim_state["lock"]:
        t = _sim_state["sim_time"]
        active = _sim_state["active_count"]
        evac = _sim_state["evacuated_count"]
        dead = _sim_state["casualty_count"]
        total = _sim_state["total_agents"]
        decisions = _sim_state["decision_count"]
        blocks = _sim_state["safety_blocks"]
        mods = _sim_state["safety_modifications"]
        avg_ms = _sim_state["avg_tick_ms"]
        running = _sim_state["running"]
        done = _sim_state["done"]

    orch = _orchestrator_ref
    q = question.lower().strip()

    if "出口" in question or "exit" in q:
        if orch is None:
            return "仿真尚未初始化。"
        snap = orch.disaster.snapshot(0, 0, orch.exits, orch.obstacles)
        lines = []
        for i, ep in enumerate(orch.exits):
            s = snap.smoke_at(np.array(ep, dtype=np.float64))
            status = "畅通" if s < 0.3 else ("有烟雾" if s < 0.6 else "浓烟封锁")
            heading = sum(1 for a in orch.agents
                         if a.dynamic.target_exit is not None
                         and np.linalg.norm(a.dynamic.target_exit - np.array(ep)) < 2.0)
            lines.append(f"出口{i+1}({ep[0]:.0f},{ep[1]:.0f}): {status}, 约{heading}人前往")
        return "\n".join(lines)

    if "拥堵" in question or "congestion" in q:
        if orch is None:
            return "数据不可用"
        snap = orch.disaster.snapshot(0, 0, orch.exits, orch.obstacles)
        lines = []
        for i, ep in enumerate(orch.exits):
            heading = sum(1 for a in orch.agents
                         if a.dynamic.target_exit is not None
                         and np.linalg.norm(a.dynamic.target_exit - np.array(ep)) < 2.0)
            if heading > 50:
                lines.append(f"⚠ 出口{i+1}拥堵: {heading}人前往")
        return "\n".join(lines) if lines else "当前无明显拥堵。"

    if "疏散率" in question or "evac rate" in q:
        rate = evac / total * 100 if total > 0 else 0
        return f"当前疏散率: {rate:.1f}% ({evac}/{total}), 伤亡: {dead}人, 活跃: {active}人"

    if "死亡" in question or "伤亡" in question or "casual" in q:
        return f"当前伤亡: {dead}人 (死亡率: {dead/total*100:.1f}%)" if total > 0 else "数据不可用"

    if "安全约束" in question or "safety" in q:
        return f"安全约束拦截: {blocks}次, 修正: {mods}次"

    if "性能" in question or "performance" in q or "延迟" in question:
        return f"平均Tick耗时: {avg_ms:.1f}ms, LLM决策总数: {decisions}"

    if "时间" in question or "time" in q:
        status = "运行中" if running else ("已完成" if done else "未启动")
        return f"仿真状态: {status}, 当前时间: {t:.0f}秒"

    if "角色" in question or "role" in q:
        if orch is None:
            return "数据不可用"
        roles = {}
        for a in orch.agents:
            r = a.profile.role
            roles[r] = roles.get(r, 0) + 1
        lines = [f"{r}: {c}个" for r, c in sorted(roles.items())]
        return "当前角色分布:\n" + "\n".join(lines)

    if "指挥" in question or "广播" in question or "commander" in q:
        if orch is None:
            return "数据不可用"
        broadcasts = orch._command_broadcasts[-5:] if orch._command_broadcasts else []
        if broadcasts:
            return "最近指挥广播:\n" + "\n".join(f"  [{b.get('source','?')}] {b.get('message','')}" for b in broadcasts)
        return "暂无指挥广播。"

    rate = evac / total * 100 if total > 0 else 0
    status = "运行中" if running else ("已完成" if done else "未启动")
    return (f"仿真状态: {status}\n"
            f"时间: {t:.0f}秒 | 疏散率: {rate:.1f}% ({evac}/{total})\n"
            f"活跃: {active} | 伤亡: {dead} | LLM决策: {decisions}\n"
            f"安全拦截: {blocks} | 修正: {mods} | 平均耗时: {avg_ms:.1f}ms")


# ================================================================
# Gradio UI callbacks
# ================================================================

def get_latest_frame():
    """Return latest frame file path. Gradio reads this file and serves it."""
    try:
        # Always return latest.png (updated every 10 ticks)
        latest = os.path.join(_FRAME_DIR, "latest.png")
        if os.path.exists(latest):
            return latest
        # Fallback: try frame_a / frame_b
        with _sim_state["lock"]:
            fp = _sim_state.get("frame_path")
        if fp and os.path.exists(fp):
            return fp
        # No frame yet: generate placeholder
        return _make_placeholder()
    except Exception as e:
        print(f"[Gradio] get_latest_frame error: {e}")
        return _make_placeholder()


def _make_placeholder():
    """Generate a placeholder image and return its file path."""
    path = os.path.join(_FRAME_DIR, "placeholder.png")
    os.makedirs(_FRAME_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    ax.text(0.5, 0.5, 'Simulation initializing...\nPlease wait for LLM to load (~60s)',
            ha='center', va='center', fontsize=16, color='gray', transform=ax.transAxes)
    ax.set_facecolor('#F0F0F5')
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(path, format='png', dpi=80, bbox_inches='tight')
    plt.close(fig)
    return path


def get_stats_text():
    """Return formatted stats string."""
    with _sim_state["lock"]:
        t = _sim_state["sim_time"]
        tick = _sim_state["tick"]
        active = _sim_state["active_count"]
        evac = _sim_state["evacuated_count"]
        dead = _sim_state["casualty_count"]
        total = _sim_state["total_agents"]
        decisions = _sim_state["decision_count"]
        blocks = _sim_state["safety_blocks"]
        mods = _sim_state["safety_modifications"]
        frames = _sim_state["frame_count"]
        running = _sim_state["running"]
        done = _sim_state["done"]
        error = _sim_state["error"]
        avg_ms = _sim_state["avg_tick_ms"]

    if error:
        return f"## 错误\n\n{error}\n\n时间: {t:.1f}s | Tick: {tick}"

    status = "运行中" if running else ("已完成" if done else "等待启动...")
    rate = evac / total * 100 if total > 0 else 0

    return f"""## {status}

| 指标 | 值 |
|------|-----|
| 仿真时间 | {t:.0f}s (Tick {tick}) |
| 总人数 | {total} |
| 活跃 | {active} |
| 已疏散 | {evac} ({rate:.1f}%) |
| 伤亡 | {dead} |
| LLM决策 | {decisions} |
| 安全拦截/修正 | {blocks}/{mods} |
| 帧数 | {frames} |
| 平均耗时 | {avg_ms:.1f}ms |"""


def get_evac_plot():
    """Generate Plotly evacuation progress chart."""
    import plotly.graph_objects as go
    with _sim_state["lock"]:
        history = list(_sim_state["history"])

    if not history:
        fig = go.Figure()
        fig.add_annotation(text="等待数据...", showarrow=False, font=dict(size=16))
        return fig

    times = [h["sim_time"] for h in history]
    evac = [h["evacuated"] for h in history]
    dead = [h["casualties"] for h in history]
    active = [h["active"] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=evac, mode='lines', name='已疏散',
                             line=dict(color='#3fb950', width=2), fill='tozeroy',
                             fillcolor='rgba(63,185,80,0.1)'))
    fig.add_trace(go.Scatter(x=times, y=dead, mode='lines', name='伤亡',
                             line=dict(color='#f85149', width=2), fill='tozeroy',
                             fillcolor='rgba(248,81,73,0.1)'))
    fig.add_trace(go.Scatter(x=times, y=active, mode='lines', name='活跃',
                             line=dict(color='#58a6ff', width=2)))

    fig.update_layout(
        template='plotly_dark',
        title='疏散进度',
        xaxis_title='仿真时间 (秒)',
        yaxis_title='人数',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=40, r=20, t=50, b=40),
        height=300,
    )
    return fig


def send_broadcast(message: str):
    """Manual commander intervention."""
    if not message or not message.strip():
        return "请输入广播内容。"
    with _broadcast_lock:
        _manual_broadcasts.append({
            "tick": _sim_state["tick"],
            "message": f"[手动指挥] {message.strip()}",
            "source": "manual_override",
        })
    return f"已发送广播: {message.strip()}"


# ================================================================
# Build Gradio UI
# ================================================================

def build_ui():
    # Create initial placeholder
    placeholder = _make_placeholder()

    with gr.Blocks(title="LLM Evacuation Simulation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# LLM-Powered Crowd Evacuation Simulation — Interactive Dashboard")

        # ---- Timers ----
        frame_timer = gr.Timer(2.0)
        stats_timer = gr.Timer(2.0)
        chart_timer = gr.Timer(5.0)

        with gr.Row():
            with gr.Column(scale=3):
                frame_display = gr.Image(
                    value=placeholder,
                    label="仿真画面",
                    type="filepath",
                )
                frame_timer.tick(fn=get_latest_frame, outputs=frame_display)

            with gr.Column(scale=2):
                stats_md = gr.Markdown(get_stats_text())
                stats_timer.tick(fn=get_stats_text, outputs=stats_md)

                gr.Markdown("---")
                gr.Markdown("### 自然语言查询")
                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="如: 出口3现在什么情况？/ 哪个出口最拥堵？/ 当前疏散率？",
                        label="",
                        scale=4,
                    )
                    query_btn = gr.Button("查询", scale=1, variant="primary")
                query_output = gr.Textbox(label="回答", lines=4, interactive=False)

                with gr.Row():
                    gr.Button("出口状态").click(lambda: answer_query("出口状态"), outputs=query_output)
                    gr.Button("拥堵检测").click(lambda: answer_query("拥堵检测"), outputs=query_output)
                    gr.Button("疏散率").click(lambda: answer_query("疏散率"), outputs=query_output)
                    gr.Button("安全约束").click(lambda: answer_query("安全约束"), outputs=query_output)
                    gr.Button("角色分布").click(lambda: answer_query("角色分布"), outputs=query_output)

        with gr.Row():
            evac_plot = gr.Plot(label="疏散进度曲线", value=get_evac_plot())
            chart_timer.tick(fn=get_evac_plot, outputs=evac_plot)

        with gr.Row():
            gr.Markdown("### 手动指挥干预")
            with gr.Column():
                broadcast_input = gr.Textbox(
                    placeholder="输入广播指令，如: 请所有人员立即从东出口撤离！",
                    label="广播内容",
                    lines=2,
                )
                broadcast_btn = gr.Button("发送广播", variant="secondary")
                broadcast_status = gr.Textbox(label="状态", interactive=False)
                broadcast_btn.click(send_broadcast, inputs=broadcast_input, outputs=broadcast_status)

        query_btn.click(answer_query, inputs=query_input, outputs=query_output)
        query_input.submit(answer_query, inputs=query_input, outputs=query_output)

    return demo


# ================================================================
# Entry point
# ================================================================

def start_gradio(config_path: str = "config/default.yaml",
                 num_agents: int = None,
                 port: int = 8081,
                 share: bool = False):
    """Start Gradio server with background simulation."""
    sim_thread = threading.Thread(
        target=run_simulation_thread,
        args=(config_path, num_agents),
        daemon=True,
        name="simulation",
    )
    sim_thread.start()

    demo = build_ui()
    print(f"\n{'='*60}")
    print(f"  Gradio 交互式仪表板已启动")
    print(f"  本地访问: http://localhost:{port}")
    print(f"{'='*60}\n")

    demo.queue(default_concurrency_limit=5).launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 疏散仿真 Gradio 仪表板")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--agents", "-n", type=int, default=None)
    parser.add_argument("--port", "-p", type=int, default=8081)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    start_gradio(
        config_path=args.config,
        num_agents=args.agents,
        port=args.port,
        share=args.share,
    )

"""Flask Web 可视化服务器 — 租借 GPU 平台专用.

在云端 GPU 上启动一个 Web 服务, 浏览器里实时观看仿真过程:
  - 仿真在后台线程运行
  - 每 N tick 渲染一帧, 存入共享内存
  - 前端自动轮询最新帧 + 统计
  - 仿真结束后可下载 GIF/MP4

Usage:
    python visualization/web_server.py --config config/default.yaml --port 8080

或在 main.py 中用:
    python main.py --web --port 8080

租借平台访问方式:
  - AutoDL:     在 JupyterLab 里点"端口转发", 或 SSH -L 8080:localhost:8080
  - 恒源云:     SSH -L 8080:localhost:8080
  - Vast.ai:    Open port 8080 in instance settings
  - 本地测试:   http://localhost:8080
"""

import os
import sys
import io
import time
import json
import base64
import threading
import argparse
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, Response, request, jsonify, send_file, render_template_string

from decision.agent_state import Speed
from perception.environment import EnvironmentSnapshot

app = Flask(__name__)

# ================================================================
# 全局共享状态 (仿真线程 ↔ Flask 线程)
# ================================================================

_sim_state = {
    "running": False,
    "done": False,
    "error": None,

    # 最新帧 (PNG bytes)
    "frame_bytes": None,
    "frame_count": 0,
    "frame_interval": 10,

    # 统计
    "tick": 0,
    "sim_time": 0.0,
    "active_count": 0,
    "evacuated_count": 0,
    "casualty_count": 0,
    "decision_count": 0,
    "total_agents": 0,
    "avg_tick_ms": 0.0,

    # 历史
    "history": [],           # [{tick, sim_time, active, evac, dead, decisions}]

    # 锁
    "lock": threading.Lock(),
}

# 存储所有帧用于最终视频合成
_frame_buffer: list = []
_frame_buffer_lock = threading.Lock()


# ================================================================
# 帧渲染器 (matplotlib → PNG bytes)
# ================================================================

def _fear_color(fear_level: float):
    t = fear_level / 10.0
    if t < 0.5:
        return (t * 2, 1.0, 0.2)
    else:
        return (1.0, 1.0 - (t - 0.5) * 2, 0.2)


def render_frame_to_bytes(agents, env, tick, sim_time,
                          evacuated, casualties, decisions,
                          world_w, world_h, dpi=100) -> bytes:
    """渲染一帧 → PNG bytes (内存中, 不落盘)."""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=dpi)
    ax.set_xlim(0, world_w)
    ax.set_ylim(0, world_h)
    ax.set_aspect('equal')
    ax.set_facecolor('#F0F0F5')

    # --- 烟雾 ---
    step = max(1, env.grid.shape[1] // 60)
    for r in range(0, env.grid.shape[0], step):
        for c in range(0, env.grid.shape[1], step):
            smoke = env.grid[r, c, 0]
            if smoke < 0.05:
                continue
            wx = c * env.grid_resolution
            wy = r * env.grid_resolution
            cell_w = env.grid_resolution * step
            rect = plt.Rectangle(
                (wx, wy), cell_w, cell_w,
                facecolor='gray', alpha=min(0.7, smoke * 0.8), edgecolor='none'
            )
            ax.add_patch(rect)

    # --- 火焰 ---
    fire_mask = env.grid[:, :, 3] > 0.5
    if fire_mask.any():
        rows, cols = np.where(fire_mask)
        for r, c in zip(rows[::step], cols[::step]):
            wx = c * env.grid_resolution
            wy = r * env.grid_resolution
            cell_w = env.grid_resolution * step
            rect = plt.Rectangle(
                (wx, wy), cell_w, cell_w,
                facecolor='#FF6414', alpha=0.6, edgecolor='none'
            )
            ax.add_patch(rect)

    # --- 障碍物 ---
    for obs in env.obstacles:
        circle = plt.Circle(
            obs["center"], obs["radius"],
            facecolor='#B4B4B9', edgecolor='#8C8C91', linewidth=1.5
        )
        ax.add_patch(circle)

    # --- 出口 ---
    for i, exit_pos in enumerate(env.exits):
        ex, ey = exit_pos
        gr = min(int(ey / env.grid_resolution), env.grid.shape[0] - 1)
        gc = min(int(ex / env.grid_resolution), env.grid.shape[1] - 1)
        exit_smoke = float(env.grid[gr, gc, 0])
        color = 'green' if exit_smoke < 0.3 else ('orange' if exit_smoke < 0.6 else 'red')
        rect = plt.Rectangle(
            (ex - 1.2, ey - 1.2), 2.4, 2.4,
            facecolor=color, alpha=0.7, edgecolor='darkgreen', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(ex + 1.5, ey, f'E{i+1}', fontsize=8, fontweight='bold')

    # --- Agent ---
    active = [a for a in agents if a.dynamic.alive and not a.dynamic.evacuated]
    if active:
        positions = np.array([a.position for a in active])
        colors = [_fear_color(a.dynamic.fear_level) for a in active]
        sizes = [max(8, min(20, a.profile.max_speed * 10)) for a in active]
        ax.scatter(positions[:, 0], positions[:, 1],
                   c=colors, s=sizes, alpha=0.85, edgecolors='white', linewidth=0.3)

    # --- HUD ---
    alive = len(active)
    total = alive + evacuated + casualties
    evac_rate = (evacuated / total * 100) if total > 0 else 0
    hud_lines = [
        f"Time: {sim_time:.0f}s | Tick: {tick}",
        f"Active: {alive} | Evacuated: {evacuated} | Casualties: {casualties}",
        f"LLM Decisions: {decisions} | Evac Rate: {evac_rate:.1f}%",
    ]
    hud_text = "\n".join(hud_lines)
    ax.text(0.02, 0.98, hud_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    ax.set_xticks([])
    ax.set_yticks([])

    # Render → PNG bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ================================================================
# 后台仿真线程
# ================================================================

def run_simulation_thread(config_path: str, num_agents: Optional[int] = None):
    """在后台线程中运行仿真, 同时更新全局共享状态."""
    global _sim_state, _frame_buffer

    from execution.orchestrator import SimulationOrchestrator

    try:
        orch = SimulationOrchestrator(config_path)

        if num_agents:
            orch.num_agents = num_agents
            orch.cfg["simulation"]["num_agents"] = num_agents

        # 强制启用 headless 帧录制
        orch.cfg["visualization"]["enabled"] = False  # 不用 Pygame
        orch.cfg["visualization"]["mode"] = "headless"
        orch.cfg["visualization"]["frame_interval"] = _sim_state["frame_interval"]

        with _sim_state["lock"]:
            _sim_state["running"] = True
            _sim_state["total_agents"] = orch.num_agents

        orch.generate_agents()
        orch.llm_engine.initialize()

        # VLM / Diffusion 按配置
        orch.use_vlm = orch.cfg.get("vlm", {}).get("enabled", False)
        orch.use_diffusion = orch.cfg.get("diffusion", {}).get("enabled", False)

        orch.vlm = None; orch.yolo = None; orch.diffusion_policy = None

        if orch.use_vlm:
            from perception.vlm_perceiver import VLMPerceiver
            orch.vlm = VLMPerceiver(
                model_name=orch.cfg["vlm"].get("model", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"),
                call_interval=orch.cfg["vlm"].get("call_interval", 30),
            )
            orch.vlm.initialize()
            from perception.yolo_detector import YOLODetector
            orch.yolo = YOLODetector(
                model_name=orch.cfg.get("yolo", {}).get("model", "yolov8n.pt"),
            )
            orch.yolo.initialize()
            orch.yolo.set_calibration(640, 480, orch.width, orch.height)

        if orch.use_diffusion:
            from execution.diffusion_policy import DiffusionPolicy
            orch.diffusion_policy = DiffusionPolicy(orch.cfg)
            orch.diffusion_policy.initialize()

        total_ticks = int(orch.duration / orch.dt)
        tick_times = []

        print(f"[WebServer] Simulation started: {total_ticks} ticks, "
              f"{orch.num_agents} agents")

        while orch.tick < total_ticks:
            tick_start = time.perf_counter()

            # --- 1. Perception ---
            orch.disaster.step(orch.dt)
            env_snapshot = orch.disaster.snapshot(
                orch.tick, orch.sim_time, orch.exits, orch.obstacles,
                official_broadcast=orch._get_broadcast()
            )

            # --- 2. Cognition ---
            agents_to_decide = [
                a for a in orch.agents
                if (a.dynamic.alive and not a.dynamic.evacuated and
                    (a.dynamic.has_new_info or
                     orch.tick - a.dynamic.last_decision_tick >= orch.decision_ticks))
            ]
            if agents_to_decide:
                kdocs = orch.knowledge_base.query(
                    f"{orch.cfg['environment']['disaster']}疏散决策",
                    disaster_type=orch.cfg['environment']['disaster'],
                    top_k=3
                )
                orch.llm_engine.submit_batch(
                    agents_to_decide, env_snapshot,
                    {orch.cfg['environment']['disaster']: kdocs}
                )

            # --- 3. Collect ---
            decisions = orch.llm_engine.collect_results()
            if decisions:
                orch.decision_count += len(decisions)
                orch.total_llm_time += sum(d.compute_time for d in decisions.values())
                orch._apply_decisions(decisions, env_snapshot)

            # --- 3.5 VLM + YOLO ---
            if orch.vlm is not None and orch.tick % orch.vlm.call_interval == 0:
                frame = orch._render_cctv_frame()
                vlm_desc = orch.vlm.perceive(frame, orch.tick, env_snapshot)
                yolo_res = orch.yolo.detect(frame) if orch.yolo else None
                orch.llm_engine.set_perception_context(vlm_desc, yolo_res)

            # --- 4. Group Intelligence ---
            orch.group_intel.propagate(orch.agents, env_snapshot.official_broadcast, orch.dt)
            orch.group_intel.update_fear_levels(orch.agents, env_snapshot, orch.dt)
            orch.group_intel.update_stamina(orch.agents, orch.dt)

            # --- 5. Physics ---
            if orch.use_diffusion and orch.diffusion_policy is not None:
                orch._step_diffusion(env_snapshot)
            else:
                orch.physics.step_all(orch.agents, orch.dt)

            # --- 6. Stats ---
            orch.evacuated_count = sum(1 for a in orch.agents if a.dynamic.evacuated)
            orch.casualty_count = sum(1 for a in orch.agents if not a.dynamic.alive)
            active_count = orch.num_agents - orch.evacuated_count - orch.casualty_count

            # --- 7. 渲染帧 (按间隔) ---
            if orch.tick % _sim_state["frame_interval"] == 0:
                frame_bytes = render_frame_to_bytes(
                    orch.agents, env_snapshot,
                    orch.tick, orch.sim_time,
                    orch.evacuated_count, orch.casualty_count,
                    orch.decision_count,
                    orch.width, orch.height,
                )
                with _sim_state["lock"]:
                    _sim_state["frame_bytes"] = frame_bytes
                    _sim_state["frame_count"] += 1
                    _sim_state["tick"] = orch.tick
                    _sim_state["sim_time"] = orch.sim_time
                    _sim_state["active_count"] = active_count
                    _sim_state["evacuated_count"] = orch.evacuated_count
                    _sim_state["casualty_count"] = orch.casualty_count
                    _sim_state["decision_count"] = orch.decision_count
                    _sim_state["avg_tick_ms"] = np.mean(tick_times[-100:]) if tick_times else 0
                    _sim_state["history"].append({
                        "tick": orch.tick,
                        "sim_time": round(orch.sim_time, 1),
                        "active": active_count,
                        "evacuated": orch.evacuated_count,
                        "casualties": orch.casualty_count,
                        "decisions": orch.decision_count,
                    })

                # 存储帧用于最终视频
                with _frame_buffer_lock:
                    _frame_buffer.append(frame_bytes)

            tick_time = (time.perf_counter() - tick_start) * 1000
            tick_times.append(tick_time)

            orch.tick += 1
            orch.sim_time += orch.dt

            remaining = orch.num_agents - orch.evacuated_count - orch.casualty_count
            if remaining <= 0:
                break

        # 清理
        if orch.vlm: orch.vlm.shutdown()
        if orch.yolo: orch.yolo.shutdown()
        if orch.diffusion_policy: orch.diffusion_policy.shutdown()
        orch.llm_engine.shutdown()

        with _sim_state["lock"]:
            _sim_state["done"] = True
            _sim_state["running"] = False

        print(f"[WebServer] Simulation complete. "
              f"Frames: {_sim_state['frame_count']} | "
              f"Evac: {orch.evacuated_count}/{orch.num_agents} | "
              f"Safety blocked: {orch.safety_blocks} modified: {orch.safety_modifications}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        with _sim_state["lock"]:
            _sim_state["error"] = str(e)
            _sim_state["running"] = False
            _sim_state["done"] = True


# ================================================================
# Flask 路由
# ================================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM 疏散仿真 — 实时可视化</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body{ font-family:'Segoe UI','PingFang SC',sans-serif; background:#0d1117; color:#c9d1d9; }
  .header{ background:#161b22; border-bottom:1px solid #30363d; padding:12px 24px;
           display:flex; justify-content:space-between; align-items:center; }
  .header h1{ font-size:18px; color:#58a6ff; }
  .status{ font-size:13px; padding:4px 12px; border-radius:12px; }
  .status.running{ background:#1a3a1a; color:#3fb950; }
  .status.done{ background:#1a2a3a; color:#58a6ff; }
  .status.error{ background:#3a1a1a; color:#f85149; }
  .main{ display:flex; height:calc(100vh - 56px); }
  .canvas-panel{ flex:1; display:flex; align-items:center; justify-content:center;
                 background:#010409; padding:10px; }
  .canvas-panel img{ max-width:100%; max-height:100%; border-radius:6px;
                     box-shadow:0 4px 20px rgba(0,0,0,0.5); }
  .sidebar{ width:340px; background:#161b22; border-left:1px solid #30363d;
            padding:20px; overflow-y:auto; display:flex; flex-direction:column; gap:16px; }
  .stat-card{ background:#0d1117; border:1px solid #21262d; border-radius:8px; padding:14px; }
  .stat-card h3{ font-size:13px; color:#8b949e; margin-bottom:8px; text-transform:uppercase;
                 letter-spacing:0.5px; }
  .stat-row{ display:flex; justify-content:space-between; padding:4px 0;
             font-size:14px; font-family:'SF Mono','Consolas',monospace; }
  .stat-val{ color:#58a6ff; font-weight:bold; }
  .stat-val.green{ color:#3fb950; }
  .stat-val.red{ color:#f85149; }
  .stat-val.yellow{ color:#d2991d; }
  .progress-bar{ height:6px; background:#21262d; border-radius:3px; overflow:hidden; }
  .progress-fill{ height:100%; background:linear-gradient(90deg,#238636,#3fb950); transition:width 0.8s; }
  .btn{ display:block; width:100%; padding:10px; border:none; border-radius:6px; cursor:pointer;
        font-size:14px; font-weight:600; text-align:center; transition:all 0.15s; }
  .btn-download{ background:#238636; color:#fff; margin-top:4px; }
  .btn-download:hover{ background:#2ea043; }
  .btn-download:disabled{ background:#21262d; color:#484f58; cursor:not-allowed; }
  .log{ font-size:12px; color:#8b949e; max-height:120px; overflow-y:auto; }
  .log .line{ padding:2px 0; border-bottom:1px solid #21262d22; }
  @keyframes pulse{ 0%,100%{opacity:1} 50%{opacity:0.5} }
  .live-dot{ width:8px; height:8px; background:#3fb950; border-radius:50%;
             display:inline-block; animation:pulse 1.5s infinite; margin-right:6px; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>🔥 LLM-Powered Crowd Evacuation Simulation</h1>
  </div>
  <div>
    <span id="status-dot" class="live-dot" style="display:none;"></span>
    <span id="status-text" class="status running">等待启动...</span>
  </div>
</div>

<div class="main">
  <div class="canvas-panel">
    <img id="sim-frame" src="" alt="等待仿真启动..."
         style="display:none;"
         onload="this.style.display='block';document.getElementById('placeholder').style.display='none';">
    <div id="placeholder" style="color:#484f58;font-size:18px;">
      🚀 仿真正在初始化... 请稍候
    </div>
  </div>

  <div class="sidebar">
    <div class="stat-card">
      <h3>📊 实时统计</h3>
      <div class="stat-row"><span>仿真时间</span><span class="stat-val" id="st-time">0.0s</span></div>
      <div class="stat-row"><span>当前 Tick</span><span class="stat-val" id="st-tick">0</span></div>
      <div class="stat-row"><span>活跃 Agent</span><span class="stat-val" id="st-active">-</span></div>
      <div class="stat-row"><span>已疏散</span><span class="stat-val green" id="st-evac">0</span></div>
      <div class="stat-row"><span>伤亡</span><span class="stat-val red" id="st-dead">0</span></div>
      <div class="stat-row"><span>LLM 决策次数</span><span class="stat-val" id="st-dec">0</span></div>
      <div class="stat-row"><span>平均 Tick 耗时</span><span class="stat-val yellow" id="st-avg">-</span></div>
      <div class="stat-row"><span>帧数</span><span class="stat-val" id="st-frames">0</span></div>
    </div>

    <div class="stat-card">
      <h3>📈 疏散进度</h3>
      <div class="progress-bar"><div class="progress-fill" id="progress" style="width:0%"></div></div>
      <div class="stat-row" style="margin-top:6px;">
        <span>疏散率</span><span class="stat-val green" id="evac-rate">0%</span>
      </div>
    </div>

    <div class="stat-card">
      <h3>📥 下载</h3>
      <button class="btn btn-download" id="btn-gif" disabled
              onclick="window.open('/download/gif','_blank')">
        ⏳ 下载 GIF (仿真结束后可用)
      </button>
      <button class="btn btn-download" id="btn-mp4" disabled
              onclick="window.open('/download/mp4','_blank')"
              style="margin-top:6px;">
        🎬 下载 MP4 (仿真结束后可用)
      </button>
      <button class="btn btn-download" id="btn-frames" disabled
              onclick="window.open('/download/frames','_blank')"
              style="margin-top:6px;">
        🖼 下载最后一帧 PNG
      </button>
    </div>

    <div class="stat-card">
      <h3>📝 事件日志</h3>
      <div class="log" id="log">
        <div class="line">等待仿真启动...</div>
      </div>
    </div>
  </div>
</div>

<script>
  const POLL_INTERVAL = 1500;  // 1.5s 轮询一次
  let done = false;

  function addLog(msg) {
    const log = document.getElementById('log');
    log.innerHTML += `<div class="line">[${new Date().toLocaleTimeString()}] ${msg}</div>`;
    log.scrollTop = log.scrollHeight;
    while(log.children.length > 80) log.firstElementChild.remove();
  }

  async function poll() {
    try {
      const resp = await fetch('/stats');
      const data = await resp.json();

      // 更新统计
      document.getElementById('st-time').textContent = data.sim_time.toFixed(1) + 's';
      document.getElementById('st-tick').textContent = data.tick;
      document.getElementById('st-active').textContent = data.active_count;
      document.getElementById('st-evac').textContent = data.evacuated_count;
      document.getElementById('st-dead').textContent = data.casualty_count;
      document.getElementById('st-dec').textContent = data.decision_count;
      document.getElementById('st-avg').textContent = data.avg_tick_ms.toFixed(1) + 'ms';
      document.getElementById('st-frames').textContent = data.frame_count;

      // 进度
      const total = data.total_agents || 1;
      const evacRate = (data.evacuated_count / total * 100).toFixed(1);
      document.getElementById('evac-rate').textContent = evacRate + '%';
      document.getElementById('progress').style.width = evacRate + '%';

      // 状态
      const statusEl = document.getElementById('status-text');
      const dotEl = document.getElementById('status-dot');
      if (data.done) {
        statusEl.textContent = '✅ 仿真完成';
        statusEl.className = 'status done';
        dotEl.style.display = 'none';
        done = true;
        // 启用下载按钮
        document.getElementById('btn-gif').disabled = false;
        document.getElementById('btn-mp4').disabled = false;
        document.getElementById('btn-frames').disabled = false;
        document.getElementById('btn-gif').textContent = '🎞 下载 GIF';
        document.getElementById('btn-mp4').textContent = '🎬 下载 MP4';
        document.getElementById('btn-frames').textContent = '🖼 下载当前帧 PNG';
      } else if (data.running) {
        statusEl.textContent = '🟢 仿真运行中';
        statusEl.className = 'status running';
        dotEl.style.display = 'inline-block';
      } else if (data.error) {
        statusEl.textContent = '❌ 错误: ' + data.error;
        statusEl.className = 'status error';
      }

      // 刷新帧
      if (data.frame_count > 0) {
        const img = document.getElementById('sim-frame');
        img.src = '/frame.png?t=' + Date.now();
        document.getElementById('placeholder').style.display = 'none';
      }

      if (data.done && data.frame_count > 0) {
        addLog(`仿真完成! 疏散率: ${evacRate}%, 共 ${data.frame_count} 帧`);
      }

    } catch(e) {
      console.error('Poll error:', e);
    }

    if (!done) {
      setTimeout(poll, POLL_INTERVAL);
    } else {
      setTimeout(poll, 5000);  // 完成后慢速轮询
    }
  }

  // 启动
  addLog('开始轮询仿真状态...');
  poll();
</script>

</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/frame.png')
def frame():
    """返回最新渲染帧."""
    with _sim_state["lock"]:
        fb = _sim_state["frame_bytes"]
    if fb is None:
        # 返回一个占位图
        fig, ax = plt.subplots(figsize=(12, 7), dpi=60)
        ax.text(0.5, 0.5, 'Simulation initializing...', ha='center', va='center',
                fontsize=20, color='gray', transform=ax.transAxes)
        ax.set_facecolor('#F0F0F5')
        ax.set_xticks([]); ax.set_yticks([])
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=60, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        fb = buf.read()

    return Response(fb, mimetype='image/png',
                    headers={'Cache-Control': 'no-cache, no-store, must-revalidate'})


@app.route('/stats')
def stats():
    """返回当前仿真统计 (JSON)."""
    with _sim_state["lock"]:
        data = {
            "running": _sim_state["running"],
            "done": _sim_state["done"],
            "error": _sim_state["error"],
            "tick": _sim_state["tick"],
            "sim_time": _sim_state["sim_time"],
            "active_count": _sim_state["active_count"],
            "evacuated_count": _sim_state["evacuated_count"],
            "casualty_count": _sim_state["casualty_count"],
            "decision_count": _sim_state["decision_count"],
            "avg_tick_ms": _sim_state["avg_tick_ms"],
            "frame_count": _sim_state["frame_count"],
            "total_agents": _sim_state["total_agents"],
            "history": _sim_state["history"][-100:],  # 最近100条
        }
    return jsonify(data)


@app.route('/download/<fmt>')
def download(fmt):
    """下载 GIF / MP4 / 单帧."""
    with _frame_buffer_lock:
        frames = list(_frame_buffer)

    if not frames:
        return "No frames yet. Wait for simulation to produce frames.", 404

    if fmt == 'frames':
        return Response(frames[-1], mimetype='image/png',
                        headers={'Content-Disposition': 'attachment; filename=last_frame.png'})

    # 生成 GIF: 把 PNG frames 合成 GIF
    try:
        from PIL import Image

        images = []
        for fb in frames:
            images.append(Image.open(io.BytesIO(fb)))

        buf = io.BytesIO()
        images[0].save(
            buf, format='GIF', save_all=True,
            append_images=images[1:],
            duration=100,  # 100ms per frame = 10fps
            loop=0,
            optimize=True,
        )
        buf.seek(0)
        return Response(buf.read(), mimetype='image/gif',
                        headers={'Content-Disposition': 'attachment; filename=evacuation.gif'})

    except ImportError:
        pass

    # 回退: 用 ffmpeg
    if fmt == 'mp4':
        try:
            frames_dir = '/tmp/evac_frames'
            os.makedirs(frames_dir, exist_ok=True)
            for i, fb in enumerate(frames):
                with open(f'{frames_dir}/frame_{i:05d}.png', 'wb') as f:
                    f.write(fb)

            mp4_path = '/tmp/evacuation.mp4'
            subprocess.run([
                'ffmpeg', '-y', '-r', '10',
                '-i', f'{frames_dir}/frame_%05d.png',
                '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                '-preset', 'fast', '-crf', '23',
                mp4_path
            ], capture_output=True, timeout=120)

            if os.path.exists(mp4_path):
                return send_file(mp4_path, mimetype='video/mp4',
                                 as_attachment=True, download_name='evacuation.mp4')
        except Exception as e:
            return f"MP4 generation failed: {e}. Try GIF instead.", 500

    return "Format not available. Use /download/gif or /download/mp4", 400


# ================================================================
# 启动入口
# ================================================================

def start_server(config_path: str = "config/default.yaml",
                 num_agents: int = None,
                 port: int = 8080,
                 frame_interval: int = 10,
                 host: str = "0.0.0.0"):
    """启动 Web 服务器 + 后台仿真线程.

    这是在租借 GPU 平台上调用的主入口.
    """
    _sim_state["frame_interval"] = frame_interval

    # 启动后台仿真线程
    sim_thread = threading.Thread(
        target=run_simulation_thread,
        args=(config_path, num_agents),
        daemon=True,
        name="simulation",
    )
    sim_thread.start()

    print(f"\n{'='*60}")
    print(f"  🌐 Web 可视化服务器已启动")
    print(f"  📍 本地访问: http://localhost:{port}")
    print(f"  📍 云端访问: 通过平台端口转发访问 {port} 端口")
    print(f"  🛑 按 Ctrl+C 停止")
    print(f"{'='*60}\n")

    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 疏散仿真 Web 可视化")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--agents", "-n", type=int, default=None)
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--frame-interval", type=int, default=10)
    args = parser.parse_args()

    start_server(
        config_path=args.config,
        num_agents=args.agents,
        port=args.port,
        frame_interval=args.frame_interval,
    )

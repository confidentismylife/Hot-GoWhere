"""Convert multi-modal environment data to natural language for LLM consumption.

This is the critical "cross-modal alignment" module — it bridges the gap
between numeric sensor grids and human-readable text that LLMs can reason about.

v2.0: 支持三通道融合 — 数值传感器 + VLM语义 + YOLO结构化检测
"""

import numpy as np
from typing import Optional
from perception.environment import EnvironmentSnapshot
from decision.agent_state import Agent


class NLConverter:
    """Convert environment + agent state into structured natural language."""

    @staticmethod
    def environment_context(agent: Agent, env: EnvironmentSnapshot) -> str:
        pos = agent.position
        smoke = env.smoke_at(pos)
        temp = env.temperature_at(pos)
        structural = env.structural_at(pos)
        on_fire = env.is_on_fire(pos)

        # Smoke description
        if smoke < 0.1:
            smoke_desc = "几乎无烟"
        elif smoke < 0.3:
            smoke_desc = "轻微烟雾"
        elif smoke < 0.6:
            smoke_desc = "烟雾较浓"
        else:
            smoke_desc = "浓烟弥漫"

        # Temperature description
        if temp < 35:
            temp_desc = "正常"
        elif temp < 60:
            temp_desc = "明显升高"
        elif temp < 150:
            temp_desc = "灼热"
        else:
            temp_desc = "极高,有生命危险"

        # Structural
        if structural > 0.8:
            struct_desc = "完好"
        elif structural > 0.5:
            struct_desc = "部分受损"
        elif structural > 0.2:
            struct_desc = "严重受损,有坍塌风险"
        else:
            struct_desc = "即将坍塌"

        fire_warning = "【警告】你所在位置已着火!" if on_fire else ""

        # ---- Fire strategic context (v2.2) ----
        fire_origin = np.array(env.fire_origin, dtype=np.float64)
        fire_dist_to_agent = float(np.linalg.norm(fire_origin - pos))
        fire_direction = NLConverter._direction_label(pos, fire_origin)

        # Time estimate: how long until fire/smoke reaches this agent
        # Fire front advances at ~spread_rate m/s, smoke diffuses faster (~2x)
        if fire_dist_to_agent < 5:
            fire_reach_sec = 0
        elif env.spread_rate > 0:
            fire_reach_sec = fire_dist_to_agent / env.spread_rate
        else:
            fire_reach_sec = 999

        if fire_reach_sec < 30:
            fire_threat = f"危急! 预计{fire_reach_sec:.0f}秒后火势可能蔓延到你当前位置"
        elif fire_reach_sec < 90:
            fire_threat = f"需关注, 预计{fire_reach_sec:.0f}秒后火势到达"
        else:
            fire_threat = f"火源距你{fire_dist_to_agent:.0f}m, 暂时安全"

        fire_section = (
            f"火源位置: ({fire_origin[0]:.0f}, {fire_origin[1]:.0f}) — 在你的{fire_direction}方向\n"
            f"距你: {fire_dist_to_agent:.0f}m | 蔓延速度: {env.spread_rate:.2f}m/s\n"
            f"{fire_threat}"
        )

        # ---- Per-exit strategic analysis (v2.2) ----
        crowd_counts = env.exit_crowd_counts or [0] * len(env.exits)
        walk_speed = max(0.8, agent.profile.max_speed * 0.7)  # typical walking speed

        exit_lines = []
        for i, exit_pos in enumerate(env.exits):
            ex_arr = np.array(exit_pos, dtype=np.float64)
            dist_to_exit = float(np.linalg.norm(ex_arr - pos))
            exit_smoke = float(env.smoke_at(ex_arr))
            fire_dist_to_exit = float(np.linalg.norm(fire_origin - ex_arr))

            # Smoke status
            if exit_smoke < 0.3:
                smoke_status = "通畅"
            elif exit_smoke < 0.6:
                smoke_status = f"有烟雾({exit_smoke:.0%})"
            else:
                smoke_status = f"浓烟封锁({exit_smoke:.0%})"

            # Travel time estimate
            travel_sec = dist_to_exit / max(0.3, walk_speed)

            # Safety assessment based on fire distance to exit
            if fire_dist_to_exit < 15:
                safety = "危险 — 火源极近,烟雾将快速加重"
            elif fire_dist_to_exit < 40:
                safety = "中等风险 — 需尽快通过"
            else:
                safety = "安全 — 远离火源"

            # Crowd info
            crowd_n = crowd_counts[i] if i < len(crowd_counts) else 0
            if crowd_n > 30:
                crowd_info = f"约{crowd_n}人选择, 严重拥堵"
            elif crowd_n > 15:
                crowd_info = f"约{crowd_n}人选择, 可能排队"
            elif crowd_n > 5:
                crowd_info = f"约{crowd_n}人选择"
            else:
                crowd_info = "较少人选择"

            # Direction label for exit
            exit_dir = NLConverter._direction_label(pos, ex_arr)

            # Risk flags
            risk_flags = ""
            if fire_dist_to_exit < 15:
                risk_flags = " ⚠高危"
            elif exit_smoke > 0.6:
                risk_flags = " ⛔已封锁"
            elif exit_smoke > 0.3:
                risk_flags = " ⚠烟雾加重中"

            exit_lines.append(
                f"  出口{i+1}({exit_pos[0]:.0f},{exit_pos[1]:.0f}) [{exit_dir}]: "
                f"距你{dist_to_exit:.0f}m | 步行约{travel_sec:.0f}s | {smoke_status}\n"
                f"    距火源{fire_dist_to_exit:.0f}m | {safety} | {crowd_info}{risk_flags}"
            )

        return f"""[环境状态]
时间: {env.timestamp:.0f}秒
位置: ({pos[0]:.1f}, {pos[1]:.1f})
烟雾: {smoke_desc} (浓度{smoke:.0%})
温度: {temp_desc} ({temp:.0f}°C)
建筑结构: {struct_desc}
{fire_warning}
[火势态势]
{fire_section}

[出口对比分析]
{chr(10).join(exit_lines)}

官方广播: {env.official_broadcast or '无'}"""

    @staticmethod
    def _direction_label(pos: np.ndarray, target: np.ndarray) -> str:
        """Return Chinese direction label from pos to target."""
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        if abs(dx) < 3 and abs(dy) < 3:
            return "正"
        parts = []
        if dx > 5:
            parts.append("东")
        elif dx < -5:
            parts.append("西")
        if dy > 5:
            parts.append("北")
        elif dy < -5:
            parts.append("南")
        return "".join(parts) if parts else "附近"

    @staticmethod
    def personal_context(agent: Agent) -> str:
        d = agent.dynamic
        p = agent.profile

        stamina_desc = "充沛" if d.stamina > 70 else ("一般" if d.stamina > 30 else "力竭")

        if d.fear_level < 2:
            fear_desc = "冷静"
        elif d.fear_level < 5:
            fear_desc = "紧张"
        elif d.fear_level < 8:
            fear_desc = "非常恐惧"
        else:
            fear_desc = "极度恐慌"

        family_text = ""
        if d.family_member_ids:
            family_text = f"家人ID: {', '.join(d.family_member_ids[:3])}"

        return f"""[个人状态]
{agent.id}: {p.age}岁{p.occupation}
环境熟悉度: {'熟悉' if p.familiarity > 0.6 else '不熟悉'}
体力: {stamina_desc} ({d.stamina:.0f}/100)
心理状态: {fear_desc}
{family_text}
当前行动: {d.speed_choice.value}"""

    @staticmethod
    def memory_context(agent: Agent) -> str:
        if not agent.dynamic.memory_events:
            return "[记忆] 无关键事件"

        # Only show at most 3 recent events, each truncated to 50 chars
        recent = agent.dynamic.memory_events[-3:]
        lines = ["[最近记忆]"]
        for ev in recent:
            desc = ev.get('desc', '?')
            if len(desc) > 50:
                desc = desc[:50] + "..."
            lines.append(f"  - [{ev.get('time', '?')}] {desc}")
        return "\n".join(lines)

    @staticmethod
    def rumor_context(agent: Agent) -> str:
        if not agent.dynamic.received_rumors:
            return ""

        # Only 1 rumor, truncated
        r = agent.dynamic.received_rumors[-1]
        content = r.get('content', '?')
        if len(content) > 50:
            content = content[:50] + "..."
        return f"[最近消息] {content}"

    @staticmethod
    def vlm_context(description: str) -> str:
        """v2.0: 将VLM输出格式化为Prompt段落."""
        if not description:
            return ""
        return f"""
[监控画面分析]
{description}
"""

    @staticmethod
    def yolo_context(yolo_result=None) -> str:
        """v2.0: 将YOLO结构化检测结果格式化为Prompt段落.

        Args:
            yolo_result: YOLOResult 或 None
        """
        if yolo_result is None or yolo_result.person_count == 0:
            return ""

        lines = [f"[人群检测] 画面中检测到约{yolo_result.person_count}人"]

        # 密度热点
        if yolo_result.density_hotspots:
            spots = yolo_result.density_hotspots[:3]
            spot_strs = []
            for s in spots:
                cx, cy = s["center"]
                spot_strs.append(
                    f"({cx:.0f},{cy:.0f})附近约{s['count']}人聚集"
                )
            lines.append(f"  人群热点: {'; '.join(spot_strs)}")

        # 异常事件
        if yolo_result.abnormal_events:
            for ev in yolo_result.abnormal_events:
                lines.append(f"  ⚠ {ev}")

        return "\n".join(lines)

    @staticmethod
    def rl_context(rl_preference: str) -> str:
        """Format RL zone scheduler advice for LLM prompt injection."""
        if not rl_preference:
            return ""
        return rl_preference

    @classmethod
    def full_context(cls, agent: Agent, env: EnvironmentSnapshot,
                     vlm_description: str = "",
                     yolo_result=None,
                     rl_preference: str = "") -> str:
        """Assemble complete NL context for one agent's LLM decision call.

        v2.0: 支持三通道输入 — 数值 + VLM + YOLO.
        v2.1: 支持四通道输入 — 数值 + VLM + YOLO + RL调度建议.
        """
        parts = [
            cls.environment_context(agent, env),
            cls.rl_context(rl_preference),
            cls.vlm_context(vlm_description),
            cls.yolo_context(yolo_result),
            cls.personal_context(agent),
            cls.memory_context(agent),
            cls.rumor_context(agent),
        ]
        return "\n\n".join(p for p in parts if p)

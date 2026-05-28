"""Convert multi-modal environment data to natural language for LLM consumption.

This is the critical "cross-modal alignment" module — it bridges the gap
between numeric sensor grids and human-readable text that LLMs can reason about.
"""

import numpy as np
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

        # Determine smoke trend text
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

        # Find nearest exits and their status
        exit_lines = []
        for i, exit_pos in enumerate(env.exits):
            dist = float(np.linalg.norm(np.array(exit_pos) - pos))
            exit_smoke = env.smoke_at(np.array(exit_pos))
            if exit_smoke < 0.3:
                status = "通畅"
            elif exit_smoke < 0.6:
                status = "有烟雾"
            else:
                status = "浓烟封锁"
            exit_lines.append(f"  出口{i+1}({exit_pos[0]:.0f},{exit_pos[1]:.0f}): "
                            f"距离{dist:.1f}m, 状态: {status}")

        return f"""[环境状态]
时间: {env.timestamp:.0f}秒
位置: ({pos[0]:.1f}, {pos[1]:.1f})
烟雾: {smoke_desc} (浓度{smoke:.0%})
温度: {temp_desc} ({temp:.0f}°C)
建筑结构: {struct_desc}
{fire_warning}
出口状态:
{chr(10).join(exit_lines)}
官方广播: {env.official_broadcast or '无'}"""

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

    @classmethod
    def full_context(cls, agent: Agent, env: EnvironmentSnapshot) -> str:
        """Assemble complete NL context for one agent's LLM decision call."""
        parts = [
            cls.environment_context(agent, env),
            cls.personal_context(agent),
            cls.memory_context(agent),
            cls.rumor_context(agent),
        ]
        return "\n\n".join(p for p in parts if p)

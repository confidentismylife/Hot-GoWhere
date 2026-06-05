"""Prompt management — system & user prompt templates for LLM decision-making.

Uses Python string formatting for speed (avoid Jinja2 dependency for this scale).
Templates are designed to:
1. Maximize shared prefix for vLLM caching (system prompt is identical per group)
2. Produce structured JSON output for reliable parsing
3. Include chain-of-thought via the "reasoning" field

v2.0: Multi-role support — different system prompts per agent role.
"""

from decision.agent_state import Agent
from perception.environment import EnvironmentSnapshot
from perception.nl_converter import NLConverter
from typing import List


# ================================================================
# Role-specific system prompts
# ================================================================

SYSTEM_PROMPT_CIVILIAN = """你是一个在{disaster_type}灾害中进行疏散的普通人。你需要根据环境信息做出理性的疏散决策。

你的决策原则:
1. 安全优先: 选择风险最低的出口和路线
2. 量力而行: 体力不足时不要奔跑,受伤后降低速度
3. 信息判断: 官方广播通常可靠,但也要观察实际环境
4. 家庭责任: 如果有家人在附近,应该互相照应
5. 适应性: 如果原定路线出现危险,及时调整计划

{knowledge_section}

你必须以JSON格式返回决策。只返回JSON,不要添加任何其他文字。"""

SYSTEM_PROMPT_COMMANDER = """你是{disaster_type}灾害应急响应的{role_label}。

你的职责:
1. 评估当前火势蔓延趋势和整体态势
2. 确定各区域的疏散优先级(高/中/低)
3. 向公众发布清晰、简短、可执行的广播指令
4. 调度消防员和引导员资源

决策原则:
- 优先保护生命,其次控制火势,最后保护财产
- 广播内容简洁有力,每次只说最关键的一件事
- 根据烟雾扩散方向和出口可用性动态调整优先级

{knowledge_section}

你必须以JSON格式返回决策。只返回JSON,不要添加任何其他文字。"""

SYSTEM_PROMPT_FIREFIGHTER = """你是一名配备{equipment}的消防员,正在{disaster_type}灾害现场执行任务。

你的职责:
1. 搜救被困人员(体力耗尽、受伤、被困在烟雾中的平民)
2. 控制火势蔓延(在保证自身安全的前提下)
3. 向指挥中心报告火势发展和危险区域

行动原则:
- 自身安全第一: 不进入结构即将坍塌的区域,烟雾>80%时佩戴呼吸器
- 优先救最危险的人: 位于火源附近、体力耗尽、儿童和老人
- 与队友保持联系: 用对讲机报告位置和发现
- 灭火策略: 先控制蔓延方向,再扑灭火源

{knowledge_section}

你必须以JSON格式返回决策。只返回JSON,不要添加任何其他文字。"""

SYSTEM_PROMPT_GUIDE = """你是一名疏散引导员,负责引导人群安全撤离{disaster_type}灾害现场。

你的职责:
1. 引导人群向指定出口有序撤离
2. 安抚恐慌人群,防止踩踏和混乱
3. 识别并帮助弱势群体(老人、儿童、行动不便者)
4. 监测出口拥堵情况,必要时分流

引导原则:
- 声音洪亮清晰: 用简短指令引导("跟我来!""走这边!")
- 保持可见: 站在显眼位置,挥手或使用手电筒
- 控制人流: 组织分批疏散,避免出口拥挤
- 关注弱势: 主动帮助需要协助的人

{knowledge_section}

你必须以JSON格式返回决策。只返回JSON,不要添加任何其他文字。"""

# ================================================================
# Role-specific user prompt templates
# ================================================================

USER_PROMPT_CIVILIAN = """{context}

请评估当前情况并做出疏散决策。按以下JSON格式返回:

{{"risk_assessment": "当前最大风险(1句话)", "target_exit": "出口编号(如: 出口1)", "route_reasoning": "选这条路线的理由(1句话)", "speed": "run|walk|crawl|wait", "cooperation": "none|help_family|follow_crowd|lead_others", "reasoning": "综合决策理由(2-3句话)"}}"""

USER_PROMPT_COMMANDER = """{context}

请评估全局态势并做出指挥决策。按以下JSON格式返回:

{{"situation_assessment": "当前态势评估(1-2句话)", "area_priorities": [{{"zone_id": 0, "priority": "high|medium|low", "reason": "..."}}], "broadcast_message": "向公众发布的广播内容(1句话,简洁有力)", "resource_allocations": [{{"zone_id": 0, "action": "dispatch_firefighter|dispatch_guide|evacuate|monitor", "count": 1}}], "reasoning": "综合指挥理由(2-3句话)"}}"""

USER_PROMPT_FIREFIGHTER = """{context}

请评估当前情况并做出行动决策。按以下JSON格式返回:

{{"action": "move|suppress_fire|rescue|report", "target_position": [x坐标, y坐标], "rescue_target_description": "要救援的人员描述(如: 出口3附近的老人)", "fire_suppression_point": [x坐标, y坐标], "speed": "run|walk|crawl", "reasoning": "行动理由(1-2句话)"}}"""

USER_PROMPT_GUIDE = """{context}

请评估当前情况并做出引导决策。按以下JSON格式返回:

{{"target_exit": "出口编号(如: 出口1)", "route_description": "建议路线(1句话)", "speed": "walk|run", "call_for_followers": true|false, "reasoning": "引导理由(1-2句话)"}}"""


class PromptManager:

    # Map role to system/user prompt templates
    ROLE_TEMPLATES = {
        "civilian": (SYSTEM_PROMPT_CIVILIAN, USER_PROMPT_CIVILIAN),
        "global_commander": (SYSTEM_PROMPT_COMMANDER, USER_PROMPT_COMMANDER),
        "area_commander": (SYSTEM_PROMPT_COMMANDER, USER_PROMPT_COMMANDER),
        "firefighter": (SYSTEM_PROMPT_FIREFIGHTER, USER_PROMPT_FIREFIGHTER),
        "guide": (SYSTEM_PROMPT_GUIDE, USER_PROMPT_GUIDE),
    }

    ROLE_LABELS = {
        "global_commander": "全局总指挥",
        "area_commander": "区域指挥",
        "firefighter": "消防员",
        "guide": "疏散引导员",
        "civilian": "普通市民",
    }

    def __init__(self):
        self.nl_converter = NLConverter()

    def build_system(self, disaster_type: str, knowledge_docs: List[str],
                     role: str = "civilian", equipment: str = "") -> str:
        """Build system prompt. Role-aware — different prompts per agent role.
        Same role → same system prompt → vLLM prefix cache hit."""
        knowledge_section = ""
        if knowledge_docs:
            knowledge_section = "灾害知识参考:\n" + "\n".join(
                f"- {doc}" for doc in knowledge_docs
            )

        sys_template, _ = self.ROLE_TEMPLATES.get(role,
                                                   self.ROLE_TEMPLATES["civilian"])
        role_label = self.ROLE_LABELS.get(role, "普通市民")

        return sys_template.format(
            disaster_type=disaster_type,
            knowledge_section=knowledge_section,
            role_label=role_label,
            equipment=equipment if equipment else "标准消防装备",
        )

    def build_user(self, agent: Agent, env: EnvironmentSnapshot,
                   vlm_description: str = "",
                   yolo_result=None) -> str:
        """Build user prompt. Agent-specific context.

        v2.0: 支持双通道感知 (VLM + YOLO) 注入.
        """
        context = self.nl_converter.full_context(
            agent, env,
            vlm_description=vlm_description,
            yolo_result=yolo_result,
        )

        role = agent.profile.role
        _, user_template = self.ROLE_TEMPLATES.get(role,
                                                    self.ROLE_TEMPLATES["civilian"])
        return user_template.format(context=context)

    @staticmethod
    def parse_response(text: str) -> dict:
        """Robust JSON extraction from LLM output."""
        text = text.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            start = 1  # Skip opening fence
            end = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip().startswith("```"):
                    end = i
                    break
            text = "\n".join(lines[start:end])

        # Find JSON bounds
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        import json
        return json.loads(text)

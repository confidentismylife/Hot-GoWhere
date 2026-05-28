"""Prompt management — system & user prompt templates for LLM decision-making.

Uses Python string formatting for speed (avoid Jinja2 dependency for this scale).
Templates are designed to:
1. Maximize shared prefix for vLLM caching (system prompt is identical per group)
2. Produce structured JSON output for reliable parsing
3. Include chain-of-thought via the "reasoning" field
"""

from decision.agent_state import Agent
from perception.environment import EnvironmentSnapshot
from perception.nl_converter import NLConverter
from typing import List


SYSTEM_PROMPT = """你是一个在{disaster_type}灾害中进行疏散的普通人。你需要根据环境信息做出理性的疏散决策。

你的决策原则:
1. 安全优先: 选择风险最低的出口和路线
2. 量力而行: 体力不足时不要奔跑,受伤后降低速度
3. 信息判断: 官方广播通常可靠,但也要观察实际环境
4. 家庭责任: 如果有家人在附近,应该互相照应
5. 适应性: 如果原定路线出现危险,及时调整计划

{knowledge_section}

你必须以JSON格式返回决策。只返回JSON,不要添加任何其他文字。"""

USER_PROMPT = """{context}

请评估当前情况并做出疏散决策。按以下JSON格式返回:

{{"risk_assessment": "当前最大风险(1句话)", "target_exit": "出口编号(如: 出口1)", "route_reasoning": "选这条路线的理由(1句话)", "speed": "run|walk|crawl|wait", "cooperation": "none|help_family|follow_crowd|lead_others", "reasoning": "综合决策理由(2-3句话)"}}"""


class PromptManager:

    def __init__(self):
        self.nl_converter = NLConverter()

    def build_system(self, disaster_type: str, knowledge_docs: List[str]) -> str:
        """Build system prompt. Same for all agents of same disaster type.
        This shared prefix is cached by vLLM."""
        knowledge_section = ""
        if knowledge_docs:
            knowledge_section = "灾害知识参考:\n" + "\n".join(
                f"- {doc}" for doc in knowledge_docs
            )

        return SYSTEM_PROMPT.format(
            disaster_type=disaster_type,
            knowledge_section=knowledge_section,
        )

    def build_user(self, agent: Agent, env: EnvironmentSnapshot) -> str:
        """Build user prompt. Agent-specific context."""
        context = self.nl_converter.full_context(agent, env)
        return USER_PROMPT.format(context=context)

    @staticmethod
    def build_chat_messages(agent: Agent, env: EnvironmentSnapshot,
                            system_prompt: str) -> list[dict]:
        """Build full chat messages list for vLLM."""
        user = USER_PROMPT.format(
            context=NLConverter.full_context(agent, env)
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def parse_response(text: str) -> dict:
        """Robust JSON extraction from LLM output."""
        text = text.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        # Find JSON bounds
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        import json
        return json.loads(text)

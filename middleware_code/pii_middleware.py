"""
使用可配置策略检测和处理对话中的个人身份信息 (PII)。PII 检测可用于以下用途：
医疗保健和金融应用，需满足合规性要求。需要清理日志的客服人员。任何处理敏感用户数据的应用程序
"""
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

from middleware_code.init_model import gpt_llm

agent = create_agent(
    model=gpt_llm,
    tools=[],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
    ],
)
"""
当主模型失效时，自动回退到备用模型。模型回退功能适用于以下情况：
构建能够处理模型故障的弹性代理。
通过采用更便宜的型号来优化成本。
"""
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware

from middleware_code.init_model import gpt_llm

agent = create_agent(
    model=gpt_llm,
    tools=[],
    middleware=[
        ModelFallbackMiddleware(
            "gpt-5.4-mini",
            "claude-3-5-sonnet-20241022",
        ),
    ],
)
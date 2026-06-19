"""
使用可配置的指数退避算法自动重试失败的工具调用。工具重试功能适用于以下情况：
处理外部 API 调用中的瞬态故障。提高网络依赖型工具的可靠性。构建能够优雅地处理临时错误的弹性代理
"""
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware

from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware

from middleware_code.init_model import gpt_llm

agent = create_agent(
    model=gpt_llm,
    tools=[],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=60.0,
            jitter=True,
            tools=["api_tool"],
            retry_on=(ConnectionError, TimeoutError),
            on_failure="continue",
        ),
    ],
)
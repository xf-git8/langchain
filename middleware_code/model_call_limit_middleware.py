"""
限制模型调用次数以防止无限循环或过高的成本。模型调用次数限制在以下情况下非常有用：
防止失控代理发出过多 API 调用。对生产部署实施成本控制。在特定通话预算范围内测试客服人员的行为
"""
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from middleware_code.init_model import gpt_llm

agent = create_agent(
    model=gpt_llm,
    checkpointer=InMemorySaver(),  # Required for thread limiting
    tools=[],
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="end",
        ),
    ],
)
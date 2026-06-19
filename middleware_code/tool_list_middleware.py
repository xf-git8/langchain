"""
为代理配备任务规划和跟踪功能，以处理复杂的多步骤任务。待办事项清单在以下方面非常有用：
需要协调使用多种工具的复杂多步骤任务。需要长期关注项目进展情况的项目
"""
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

from middleware_code.init_model import gpt_llm

agent = create_agent(
    model=gpt_llm,
    tools=[read_file, write_file, run_tests],
    middleware=[TodoListMiddleware()],
)
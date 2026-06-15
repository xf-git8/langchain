# 要在代理中清理消息历史记录，请使用@before_model中间件装饰器：
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from typing import Any

# 加载环境
load_dotenv()
# 定义工具 自定义消息中间件
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state['messages']
    if len(messages) <= 3:
        return None

    # 保留第一条（通常是 system prompt）
    first_msg = messages[0]

    # 保留最近的几条
    recent_messages = messages[-3:]

    # 确保 first_msg 是 list 类型
    if not isinstance(first_msg, list):
        first_msg = [first_msg]

    # 直接拼接，不要手动构造 dict
    new_messages = first_msg + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }
# 初始化大模型
deepseek_llm = init_chat_model(
    model='deepseek-chat',
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
    temperature=0.0,
    max_tokens=1024,
    timeout=15
)
# 创建agent
agent = create_agent(
    model=deepseek_llm,
    tools=[],
    middleware=[trim_messages],
    checkpointer=InMemorySaver()
)
# 配置 表示当前用户thread_id 是多轮对话的关键。
#LangGraph 会根据这个 ID 将对话状态保存在内存中，确保不同用户或不同会话的记忆是隔离的
config: RunnableConfig = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()

from langchain.messages import RemoveMessage
# 删除特定消息
def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
from langgraph.graph.message import REMOVE_ALL_MESSAGES
# 删除所有消息
def delete_messages(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
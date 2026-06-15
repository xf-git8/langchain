# for Agent add short-term memory AgentState especially for messages
# by extend AgentState not only remember memory but also remember the state of the agent
# 使用langGraph的checkpointer=InMemorySaver()来保存短期记忆
"""
第一级（短期即时记忆 / 滑动窗口）：保留最近几轮（如5轮）最大轮次或者token阈值 langGraph的InMemorySaver()。
第二级（中期摘要压缩）：对更早的对话，利用大模型生成摘要，提炼关键决策和偏好，用简短文本替代冗长的原始历史。
第三级（长期向量存储）：将跨会话的核心数据存入外部数据库，实现持久化按需检索 RAG
"""
from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv
from langchain.tools import tool
# 加载环境
load_dotenv()
# 定义工具
@tool
def get_user_info(name: str) -> str:
    """
       【重要】当用户请求查询个人信息、账户资料、或者提到“我的信息”、“查一下我”时，必须立即调用此工具。
       Args:
           name (str): 用户的名字或ID。如果用户说的是“我”、“我的”或未提供具体名字，
                       请务必传入默认值 "user_123"。不要反问用户，直接调用！
       """
    return f"用户 {name} 的信息是：姓名: {name}, 年龄: 28岁, 爱好：旅游、滑雪。"

# 自定义状态
class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

# 初始化大模型
deepseek_llm = init_chat_model(
    model='deepseek-chat',
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
    temperature=0.0,
    max_tokens=1024,
    timeout=15
)
# 创建带有内存检查点和自定义状态的 Agent
# 注意：这里必须使用 create_agent
agent = create_agent(
    model=deepseek_llm,
    tools=[get_user_info],
    state_schema=CustomAgentState,  # 注入自定义状态
    checkpointer=InMemorySaver(),  # 启用短期记忆

)

# 5. 第一轮交互：传入初始自定义状态
config = {"configurable": {"thread_id": "demo-thread-1"}}
result_1 = agent.invoke(
    {
        # 核心对话历史（必须是列表格式）
        "messages": [
            {"role": "user", "content": "你好，帮我查一下我的信息。"}
        ],
        # 自定义状态字段（与 messages 平级传入）
        "user_id": "user_123",
        "preferences": {"theme": "dark", "language": "zh-CN"}
    },
    config  # 配置作为 invoke 的第二个独立参数传入
)
print(result_1['messages'][-1].content)
result_2 = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "我是谁，我的爱好是什么？"}
        ]
    },
    config  # ⚠️ 关键：必须保持使用同一个 thread_id
)
print(result_2['messages'][-1].content)

state = agent.get_state(config)

print("--- 当前记忆中的消息数量 ---")
print(len(state.values["messages"]))

print("\n--- 当前保存的自定义状态 ---")
print(f"User ID: {state.values.get('user_id')}")
print(f"Preferences: {state.values.get('preferences')}")
# 假设你开发了一个金融客服系统，有两个不同的客户（Alice 和 Bob）先后通过同一个 Agent
# 接口查询自己的余额。我们不需要修改任何代码逻辑，只需要在调用时传入不同的 context，Agent
# 就能自动为每个用户提供专属服务
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import HumanMessage
from pydantic.dataclasses import dataclass

load_dotenv()
# 1. 模拟后端数据库
USER_DATABASE = {
    "user_alice": {"name": "Alice", "balance": 5000},
    "user_bob": {"name": "Bob", "balance": 1200},
}

# 2. 定义上下文契约
@dataclass
class UserContext:
    user_id: str


# 3. 定义工具：从上下文中提取身份并查库
@tool
def get_user_balance(runtime: ToolRuntime[UserContext]) -> str:
    """
       查询用户的账户余额。
       当用户询问自己的钱、余额或账户资金情况时使用此工具。
       需要传入 UserContext 以识别当前用户身份。
       """
    user_id = runtime.context.user_id
    print(f"DEBUG: get_user_balance 工具接收到 user_id 是 -> {user_id}")
    user_data = USER_DATABASE.get(user_id)
    if user_data:
        return f"{user_data['name']} 的余额是 ${user_data['balance']}"
    return "未找到该用户"

# 4. 初始化模型
deepseek_llm = init_chat_model(
    model='deepseek-chat',
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
    temperature=0.2,
    max_tokens=1024,
    timeout=15
)
# 5. 创建 Agent
agent = create_agent(
    model=deepseek_llm,
    tools=[get_user_balance],
    context_schema=UserContext,
    system_prompt="""
    你是一个专业的财务助手。
重要规则：
1. 每次请求都会自动注入一个 UserContext 对象，其中包含了当前用户的 user_id。
2. 当用户询问余额、账户信息等个人数据时，请【直接】调用 get_user_balance 工具，不要向用户索要用户ID或任何身份信息。
3. 根据工具的返回结果，用中文礼貌地回答用户。如果没有找到回答不知道。
    """
)
# 6. 模拟调用
# 场景 A：Alice 来查询
print("--- Alice 的请求 ---")
alice_query = "帮我看看我的余额"
res_a = agent.invoke(
    {"messages": [HumanMessage(content=alice_query)]},
    context=UserContext(user_id="user_alice")  # 注入 Alice 的身份
)
print(res_a['messages'][-1].content)
print("======================================")
# 场景 B：Bob 来查询
print("--- Bob 的请求 ---")
bob_query = "我的账户余额是多少"
res_b = agent.invoke(
    {"messages": [HumanMessage(content=bob_query)]},
    context=UserContext(user_id="user_bob")  # 注入 Bob 的身份
)
print(res_b['messages'][-1].content)
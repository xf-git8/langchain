import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent, AgentState
from langchain.tools import ToolRuntime
from langchain.messages import HumanMessage

# ==========================
# 1. 环境变量加载
# ==========================
load_dotenv()

# ==========================
# 2. 自定义 State 结构定义
# ==========================
class CustomState(AgentState):
    """自定义 Agent 短期记忆状态"""
    user_preferences: dict = {}  # 存储用户偏好设置的字典


# ==========================
# 3. 工具集 (Tools) 定义
# ==========================
@tool
def get_last_user_message(runtime: ToolRuntime) -> str:
    """获取对话历史中最近一条人类用户的消息."""
    messages = runtime.state["messages"]

    # 倒序遍历消息列表，寻找最后一条 HumanMessage
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            print(f"🙋 [系统捕获] 用户最新提问: {message.content}")  # 修复了原代码的 messages.content 拼写错误
            return message.content

    return "未找到任何用户消息"


@tool
def get_user_preference(pref_name: str, runtime: ToolRuntime) -> str:
    """根据字段名获取用户的特定偏好设置."""
    preferences = runtime.state.get("user_preferences", {})

    # 建立中英文别名映射表，防止大模型传参不规范导致查询失败
    alias_map = {
        "主题设置": "theme", "主题": "theme",
        "语言": "language",
        "通知": "notifications"
    }
    # 自动转换参数
    real_pref_name = alias_map.get(pref_name, pref_name)
    print(f"🔍 [工具执行] 原始请求: '{pref_name}' -> 实际查询: '{real_pref_name}'")

    value = preferences.get(real_pref_name, "Not set")
    print(f"💾 [查询结果] 该字段的值为: {value}")
    return value


# ==========================
# 4. 初始化 LLM 与 Agent
# ==========================
# 初始化 DeepSeek 模型
deepseek_llm = init_chat_model(
    model='deepseek-chat',
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
    temperature=0.2,
    timeout=20,
    max_tokens=1024
)

# 创建带有约束指令的 Agent
agent = create_agent(
    model=deepseek_llm,
    tools=[get_last_user_message, get_user_preference],
    state_schema=CustomState,
    system_prompt="""你是一个智能助手。当用户询问自己的偏好或重复他们刚才说的话时，请使用工具查询。
【重要规则】：用户的偏好数据存储在 State 中，支持的合法字段名(Key)只有以下几个：
- 'theme': 界面主题
- 'language': 语言设置
- 'notifications': 通知开关
如果用户询问这些功能，你必须将其翻译成上述对应的英文字段名再传给工具，严禁直接使用中文作为字段名查询！"""
)

# ==========================
# 5. 触发测试执行
# ==========================
if __name__ == "__main__":
    # 模拟从外部数据库加载的用户初始配置
    initial_prefs = {"theme": "dark_mode", "language": "zh-CN"}

    result = agent.invoke({
        "messages": [HumanMessage(content="我现在的主题设置是什么？语言设置是什么？")],
        "user_preferences": initial_prefs  # 将动态数据注入到 State 中
    })

    print("\n🤖 [最终回复]:", result['messages'][-1].content)
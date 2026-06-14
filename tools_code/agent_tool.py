import os
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# 加载环境
load_dotenv()

# 定义工具函数
@tool
def get_weather(location: str) -> str:
    """Get the weather in a location"""
    return f"It's sunny in {location}"


# 初始化模型
deepseek_llm = init_chat_model(
    model='deepseek-chat',
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.3,
    timeout=20,
    max_tokens=512
)
# 定义React_agent
agent = create_agent(
    model=deepseek_llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant."  # 新版推荐显式传入 system_prompt
)
# 运行agent
query = "what's the weather in Boston"
response = agent.invoke({
    "messages": [HumanMessage(content=query)]
})
print(response["messages"][-1].content)


# agent: An agent is a model calling tools in a loop until a given task is complete
# agent单次对话提问 概念：agent 是在一个循环中不断调用工具，直到这个任务完成
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_agent

# 加载环境变量 配置api_key和base_url
load_dotenv()

# 定义工具函数
def get_weather(city: str) -> str:
    """Get weather for a given city"""
    return f"It's always sunny in {city}"

# 初始化大模型
deepseek_llm = ChatOpenAI(
    model="deepseek-chat",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0
)
# 创建智能体
agent = create_agent(
    model=deepseek_llm,
    tools=[get_weather],
    system_prompt="""
你是一个有用的天气助手。当用户询问任何与天气相关的问题时,
请务必调用 get_weather 工具来获取信息,帮我查看北京的天气，总结回答不要出现表情气泡"
    """
)
# 智能体调用工具函数 输出[HumanMessage,AIMessage]
result = agent.invoke({'messages': [{'role': 'user', 'content': '你能给我查询一下现在的天气吗？'}]})

print((result["messages"][-1]).content)

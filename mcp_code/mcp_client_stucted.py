# mcp 输出结构化内容
import asyncio
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

# 加载环境
load_dotenv()


# 定义加载mcp服务器函数获取路径
def get_mcp_path(file):
    return os.path.join(os.path.dirname(__file__), file)


# 获取服务器函数路径
math_server_path = get_mcp_path('math_server.py')
weather_server_path = get_mcp_path('weather_server.py')
# 初始化模型
gpt_llm = ChatOpenAI(
    model='gpt-4o',
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('OPENAI_BASE_URL')
)


# 创建mcp客户端
async def main():
    client = MultiServerMCPClient(
        {
            'math': {
                'transport': 'stdio',
                'command': 'python',
                'args': [math_server_path]
            },
            'weather': {
                'transport': 'stdio',
                'command': 'python',
                'args': [weather_server_path]
            }
        }
    )
    # 获取工具
    tools = await client.get_tools()
    # 创建agent
    agent = create_agent(
        model=gpt_llm,
        tools=tools,
        system_prompt='你直接调用mcp服务器工具回答，不准反问'
    )
    result1 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's the weather like"}]}
    )
    for message in result1['messages']:
        if isinstance(message, ToolMessage) and message.artifact:
            structured_content = message.artifact["structured_content"]
            print(structured_content)
if __name__ == '__main__':
    asyncio.run(main())

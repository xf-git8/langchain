import os
import asyncio
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

gpt_model = init_chat_model(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


# 获取路径
def get_args_path(file):
    return os.path.join(os.path.dirname(__file__), file)


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command": "python",
                "args": [get_args_path('math_server.py')],
            },
            "weather": {
                "transport": "stdio",
                "command": "python",
                "args": [get_args_path('weather_server.py')],
            },
        }
    )

    tools = await client.get_tools()

    agent = create_agent(
        model=gpt_model,
        tools=tools,
    )

    math_response = await  agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )
    print("数学结果:", math_response["messages"][-1].content)

    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
    )
    print("天气结果:", weather_response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

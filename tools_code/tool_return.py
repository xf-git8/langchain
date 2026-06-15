import os
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langchain.agents import create_agent
from dotenv import load_dotenv
# return a value
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city"""
    return f"It is currently sunny in {city}"

@tool
def get_weather(city: str) -> str:
    """Get weather data for a given city"""
    return {
        "city": city,
        "temperature": 25,
        "conditions": "sunny"
    }

# return a command
# Return a Command when the tool needs
# to update graph state (for example, setting user preferences or app state)....
"""
you can return a Command with or without including a ToolMessage. 
If the model needs to see that the tool succeeded (for example,to confirm a 
preference change), include a ToolMessage in the update,
using runtime.tool_call_id for the tool_call_id parameter.
"""

@tool
def set_language(language: str, runtime: ToolRuntime) -> Command:
    """Set the preferred language for the user"""
    return Command(
        update={
            "preferred_language": language,
            "messages":[
                ToolMessage(
                    content=f"Language set to {language}",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        }
    )

"""
而开启 Return Direct 后，系统会“短路”这个循环：一旦该工具被成功调用，它的输出将，直接作为最终答案返回给用户
，普通的循环结束是 “LLM 自己决定停下”，而 Return Direct 则是 “开发者提前设定好刹车，一旦触发就立刻停车交卷
Return directly from a tool Set return direct on a tool to short-circuit the agent loop: 
the agent returns the tool’s output to the caller immediately, without sending it back through the model for further processing
"""
@tool(return_direct=True)
def fetch_order_status(order_id: str) -> str:
    """Fetch the current status of a customer order."""
    # In production, query your order management system here
    return f"Order {order_id} is shipped and will arrive in 2 days"
# 加载环境初始化模型创建agent
load_dotenv()
deepseek_llm = init_chat_model(
    model='deepseek-chat',
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
    temperature=0.2,
    timeout=20,
    max_tokens=1024
)
agent = create_agent(
    model=deepseek_llm,
    tools=[ fetch_order_status],
)
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the status of order #12345?"}]
})
print(result['messages'][-1].content)
import os
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage, HumanMessage
# 加载环境
load_dotenv()

# 定义函数
@tool
def get_weather(location: str) -> str:
    """Get the weather in a location"""
    return f"It's sunny in {location}"


# 初始化模型
deepseek_llm = init_chat_model(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    timeout=30,
    max_tokens=2048
)
# 绑定工具到模型
model_with_tools = deepseek_llm.bind_tools([get_weather])
# 第一步，用户提问 初始化信息列表
messages = []
query = "what's the weather in Boston"
messages.append(HumanMessage(content=query))
# 模型调用工具执行如果使用模型，自己解析模型想要调用的函数和参数，
# 如果调用agent  判断模型是否要调用工具 ➡自动提取参数并执行 ➡ 自动将结果反馈给模型 ➡直到模型认为信息收集完毕，输出最终答案
response = model_with_tools.invoke(query)
# messages 加入模型第一次回答的内容 AIMessage
messages.append(response)
tool_calls = response.tool_calls
for tool_call in tool_calls:
    tool_call_name = tool_call['name']
    tool_call_args = tool_call['args']['location']
    # 调用函数工具
    result = get_weather.invoke(tool_call_args)
    # 拼接到消息列表
    messages.append((ToolMessage(content=result,tool_call_id=tool_call['id'])))
final_response = model_with_tools.invoke(messages)
print(final_response.content)
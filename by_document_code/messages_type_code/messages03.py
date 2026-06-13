import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage,ToolMessage
"""
Tool Message 对于支持工具调用的模型 AIMessage中可以包含工具调用请求。
而工具消息则专门用于将单个工具执行的结果传回给模型
"""
# 加载环境
load_dotenv()
# 初始化模型
deepseek_llm = init_chat_model(
    model='deepseek-chat',
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.5,
    timeout=20,
    max_tokens=512
)
# 用户输入问题，系统封装为HumaMessage传给大模型进行推理和决策进行意图识别
human_msg = HumanMessage("What's the weather in San Francisco?")
# 大模型判断是否需要调用外部能力，返回包含需要哪些外部能力的支持
# 明确指定了要调用的工具名称、入参以及用于状态追踪的唯一 ID
ai_message = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "San Francisco"},
        "id": "call_123"
    }]
)
# 执行并汇报 主程序（Runtime）拦截该指令并在本地真实执行对应的函数逻辑结果
weather_result = "Sunny, 72°F"
# 随后，将执行结果封装为带有对应唯一 ID 的 `ToolMessage` 回传给模型
tool_message = ToolMessage(
    content = weather_result,
    tool_call_id = "call_123"
)
messages=[
    human_msg,
    ai_message,  # Model's tool call
    tool_message,  # Tool execution result
]
response = deepseek_llm.invoke(messages)
print(response.content)
print(type(response))
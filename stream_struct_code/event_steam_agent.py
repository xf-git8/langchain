# Python 等价逻辑：在 Python 中，我们需要过滤出 on_chat_model_stream 类型的事件，
# 提取其中的 chunk.content，并使用 print(..., end="", flush=True) 来实现无换行的打字机效果。
# 创建一个带有天气查询工具的 AI Agent，并通过事件流（Event Stream）实时、逐字地输出模型的回复。
import asyncio
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

# 加载环境
load_dotenv()

# 定义工具函数
"""
当你使用 @tool 时，LangChain 会在后台将这个普通的 Python 函数转换为一个标准的 
Tool 对象（通常是 StructuredTool 实例）。这个对象包含了 Agent 运行所需的一切：
执行逻辑 (func)：大模型决定调用该工具后，真正运行的 Python 代码。
元数据 (Schema)：自动从函数名、文档字符串（Docstring）和类型注解中提取出的名称、描述和参数格式。
参数校验：内置了 Pydantic 验证机制，在运行前会检查大模型传来的参数是否合法
"""


@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city"""
    return f"It's always sunny in {city}!"


"""
纯字典定义（纯数据结构）
你提供的字典列表仅仅是一个 JSON 结构的数据描述。它的作用仅仅是告诉大模型：“我有一个叫 weather 的工具，
你可以用这个格式传参给我”。没有执行能力：字典里没有任何指向 Python 函数的引用。如果大模型生成了调用指令
Agent 框架拿到这个字典后，根本不知道要去运行哪段代码。缺乏自动校验：没有内置的参数合法性检查机制
"""
tools = [
    {"function": "天气查询",
     "name": "weather",
     "description": "查询指定城市的天气情况",
     "parameters": {"type": "object",
                    "properties": {"city": {"description": "需要查询的城市名称"}}, }
     }
]
# 初始化模型
deepseek_llm = init_chat_model(
    model='deepseek-reasoner',
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
    # temperature=0.0, deepseek R1强制要求为1.0 或者不传
    max_tokens=2048,
    timeout=15)
# 创建agent
agent = create_agent(
    model=deepseek_llm,
    tools=[get_weather],
    system_prompt="""
    你是一个专业的天气预报助手。
重要规则：
1. 当用户天气查询时，请【直接】调用 get_weather 工具，不要向用户索要用户ID或任何身份信息。
2. 根据工具的返回结果，用中文礼貌地回答用户。如果没有找到回答不知道。
    """
)
# 调用工具函数
result = agent.invoke({"messages": [("user", "结合旧金山的天气，推荐我穿什么衣服？")]})

print(result['messages'][-1].content)
# 获取事件流并实时打印
async def main():
    inputs = {"messages": [("user", "结合旧金山的天气，推荐我穿什么衣服？")]}
    print(f"开始处理请求: {inputs['messages'][0][1]}\n")
    # 异步调用获取事件流
    async for event in agent.astream_events(inputs, version="v2"):
        kind = event["event"]
        # 1. 监听模型输出流 (包含推理内容和最终回复)
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            # A. 捕获推理内容 (Reasoning Content)
            # 注意：只有模型真正进行思考时才会有这个字段
            reasoning_delta = getattr(chunk, "additional_kwargs", {}).get("reasoning_content")
            if reasoning_delta:
                # 使用黄色或特殊标记区分推理内容
                print(f" [思考中]: {reasoning_delta}", end="", flush=True)
            # B. 捕获最终回复内容 (Normal Content)
            elif chunk.content:
                print(chunk.content, end="", flush=True)
        # 2. 监听工具调用开始
        elif kind == "on_tool_start":
            print(f"\n\n [调用工具]: {event['name']}")
            print(f"   参数: {event['data'].get('input')}")
        # 3.监听工具调用结束
        elif kind == "on_tool_end":
            output = event['data'].get('output')
            print(f"   输出: {output}")
            # 4. 结束标记
        elif kind == "on_end":
            print("\n\n --- 任务完成 ---")
## 运行异步主函数
tool_call_id = None
for msg in result['messages']:
    # 检查消息是否有 tool_calls 属性且不为空
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        # 获取该条消息中最后一个工具调用的 ID
        tool_call_id = msg.tool_calls[-1]['id']

print(f"工具调用ID: {tool_call_id}")
asyncio.run(main())

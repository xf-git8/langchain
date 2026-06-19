# 消息摘要，当接近消息数量上限时，自动生成对话历史记录摘要，保留最近的消息，同时压缩较早的上下文
# 多轮对话；需要保留完整的上下文
"""
摘要中间件监控消息标记计数，并在达到阈值时自动摘要旧消息。触发条件控制何时运行摘要：
当达到该阈值时，单个阈值就会触发。具有多个阈值的触发子句仅在所有阈值都满足时才会触发（AND 逻辑）。
触发条件列表，当满足其中任何一个条件时都会触发（或逻辑）。每个阈值可以使用fraction（模型上下文大小）
tokens（绝对计数）或messages（消息计数）条件控制要保留多少上下文信息（只能指定一个）：
fraction- 要保留的模型上下文大小的比例tokens- 要保留的绝对令牌数量messages- 要保留的最近消息数量
"""
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool
from middleware_code.init_model import gpt_llm


# 定义工具
@tool
def get_weather(location: str) -> str:
    """Get the weather in a location"""
    return f"It's sunny in {location}"



# 创建agent token数量》=4000 触发中间件
agent1 = create_agent(
    model=gpt_llm,
    tools=[get_weather],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
    ],
)
# 当 token 数 ≥ 3000 或 消息轮数 ≥ 6 时，任一条件满足即触发中间件
agent2 = create_agent(
    model=gpt_llm,
    tools=[get_weather],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
    ],
)
# AND logic: trigger only when tokens >= 4000 AND messages >= 10
agent3 = create_agent(
    model="gpt-5.5",
    tools=[get_weather],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger={"tokens": 4000, "messages": 10},
            keep=("messages", 20),
        ),
    ],
)

# Combine AND and OR: trigger if (tokens >= 5000 AND messages >= 3)
# OR (tokens >= 3000 AND messages >= 6)
agent4 = create_agent(
    model="gpt-5.5",
    tools=[get_weather],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=[
                {"tokens": 5000, "messages": 3},
                {"tokens": 3000, "messages": 6},
            ],
            keep=("messages", 20),
        ),
    ],
)

# Using fractional limits
agent5 = create_agent(
    model="gpt-5.5",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=("fraction", 0.8),
            keep=("fraction", 0.3),
        ),
    ],
)
# agent调用invoke
response = agent1.invoke(
    {"messages": [{'role': 'user', 'content': 'What\'s the weather like in New York?'}]}
)
# 获取返回值输出结果
print(response['messages'][-1].content)

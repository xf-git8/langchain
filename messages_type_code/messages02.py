import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

"""

AIMessage 代表模型被调用后生产的输出，它可以包含多模态数据、
工具调用请求以及特定于服务提供商的元数据，并且你可以在稍后访问这些信息。
AIMessage 是一个信息量极大的“包裹”，它不仅装满了 AI 的回答，还附带了执行工具的指令、
账单明细和各种内部状态

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

# Create an AI message manually (e.g., for conversation history)
ai_msg = AIMessage("I'd be happy to help you with that question!")
messages =[
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Can you help me"),
    ai_msg,
    HumanMessage("Great! What's 2+2?")
]
for message in messages:
    print(message.content)
response = deepseek_llm.invoke(messages)
print(response.content)

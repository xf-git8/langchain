import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

"""
SystemMessage 系统消息 代表了一组用于引导模型行为的初始指令，
可以用来定义模型行为的初始指令可以用来定义模型的基调和角色为回复建立规范和准则

HumanMessage 表示用户输入的内容，包括语音、文字、图片...
允许一张图片加文本打包，支持多模态的输入

AIMessage 代表模型被调用后生产的输出，它可以包含多模态数据、
工具调用请求以及特定于服务提供商的元数据，并且你可以在稍后访问这些信息。
AIMessage 是一个信息量极大的“包裹”，它不仅装满了 AI 的回答，还附带了执行工具的指令、账单明细和各种内部状态



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
system_msg = SystemMessage(content="""
You are a senior Python developer with
expertise in web frameworksAlways provide code examples and explain 
your reasoningBe concise but thorough in your explanation
""")
human_msg = HumanMessage(
    content="Hello!",
    name="alice",  # Optional: identify different users
    id="msg_123",  # Optional: unique identifier for tracing
)
messages = [
    system_msg,
    human_msg
]
response = deepseek_llm.invoke(messages) # returns AIMessage
print(response.content)

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

# 加载环境 加载环境变量 配置api_key和base_url
load_dotenv()
# 初始化模型
deepseek_llm = init_chat_model(
    model='deepseek-chat',
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
    timeout=30,
    max_tokens=2048
)
# 定义电影类型结构化输出
class Movie(BaseModel):
    """A Movie with details"""
    title: str = Field(description="电影标题,输出中文")
    year: int = Field(description="电影出版时间")
    director: str = Field(description="电影的导演")
# 创建智能体
agent = create_agent(
    model=deepseek_llm,
    tools =[],
    response_format=Movie
)
# 智能体调用输出
result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})
print(result["structured_response"])
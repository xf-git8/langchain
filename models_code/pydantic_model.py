# 如何控制结构化输出
# 使用pydantic Models 或者是Json schema 强制约束输出格式
import os
from dotenv import load_dotenv
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

model_with_structure = deepseek_llm.with_structured_output(Movie)

# 输出一个电影类型的格式
response = model_with_structure.invoke('你好，李焕英')
print(response)


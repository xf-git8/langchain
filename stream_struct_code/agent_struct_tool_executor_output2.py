#结合了工具调用和结构化输出的完整标准代码
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from pydantic import BaseModel,Field

# 加载环境
load_dotenv()
# 定义你期望的结构化输出格式
class MovieInfo(BaseModel):
    """从文本中提取的电影基本信息"""
    name: str = Field(..., description="电影的中文或英文名称")
    publish_year: str = Field(..., description="电影上映年份，例如 '2009'")
    director: str = Field(..., description="导演全名")
# 定义工具
@tool
def extract_movie_info(movie_info: str) -> MovieInfo:
    """从文本中提取电影信息"""
    return f"模拟数据：电影《{movie_info}》由导演克里斯托弗·诺兰执导，上映于2010年。"
# 初始化模型
deepseek_llm = init_chat_model(
    model='deepseek-chat',
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
     temperature=0.0,
    max_tokens=2048,
    timeout=15)

# 创建带有结构化输出的agent
agent = create_agent(
    model=deepseek_llm,
    tools=[extract_movie_info],                              # 注册真实工具供模型调用
    response_format=ToolStrategy(schema=MovieInfo),      # 强制最终输出为 MovieInfo 结构
    system_prompt="你是一个专业的电影数据提取助手。请调用工具获取信息，然后严格按照要求的JSON结构输出结果。"
)
# 6. 触发调用 (新版 Agent 不需要手动写 {agent_scratchpad}，框架自动处理)
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "请帮我提取电影《星际穿越》的信息。"}
    ]
})
# 7. 输出结果
print(result['structured_response'])
print(f"name: {result['structured_response'].name}")
print(f"publish_year: {result['structured_response'].publish_year}")
print(f"director: {result['structured_response'].director}")


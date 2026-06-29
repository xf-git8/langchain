# LangChain 中提供两种化的结构化输出策略一些模型提供商通过其 API 原生支持结构化输出
"""
模型供应商原生支持：ProviderStrategy为 StructuredOutputProviderStrategy
ToolStrategy:将你的 Pydantic 模型伪装成一个“工具”，让模型通过
Function Calling 的方式返回结构化数据（兼容性最好，适用于所有支持工具调用的模型）。
统一响应结果对象用result["structured_response"]
"""
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
# 加载环境
load_dotenv()
# 定义你期望的结构化输出格式
class MovieInfo(BaseModel):
    """从文本中提取的电影基本信息"""
    name: str = Field(..., description="电影的中文或英文名称")
    publish_year: str = Field(..., description="电影上映年份，例如 '2009'")
    director: str = Field(..., description="导演全名")
# 定义工具函数
# 初始化模型
deepseek_llm = init_chat_model(
    model='deepseek-chat',
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
     temperature=0.0,
    max_tokens=2048,
    timeout=15)
# 4. 创建带有结构化输出的 Agent
# 如果模型原生支持，使用 ProviderStrategy；否则可以使用 ToolStrategy(ContactInfo)

agent = create_agent(
    model=deepseek_llm,
    tools=[],  # 即使没有外部工具也可以进行结构化提取
    response_format=ToolStrategy(schema=MovieInfo)
)
# 调用模型
raw_text = "《盗梦空间》是由克里斯托弗·诺兰执导的一部科幻电影，于2010年上映。"
result = agent.invoke({
    "messages": [
        {"role": "user", "content": f"请从以下文本中提取电影信息：{raw_text}"}
    ]
})
# 输出内容
print(result['structured_response'])
print(f"name: {result['structured_response'].name}")
print(f"publish_year: {result['structured_response'].publish_year}")
print(f"director: {result['structured_response'].director}")

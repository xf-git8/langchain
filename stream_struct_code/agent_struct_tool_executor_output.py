#结合了工具调用和结构化输出的完整标准代码 老版本实例
import os,json
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
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
# 结构化模型
structured_llm = deepseek_llm.with_structured_output(MovieInfo)
#创建 Prompt
# 注意：必须包含 {agent_scratchpad} 以便 Agent 处理工具调用的中间步骤
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的电影数据提取助手。
    任务：调用 extract_movie_info 工具获取信息。
    最终输出要求：
    1. 禁止输出任何寒暄、解释或Markdown标记（如 ```json）。
    2. 必须且只能输出一个合法的 JSON 对象。
    3. JSON 字段必须严格对应 MovieInfo 定义的字段名（name, publish_year, director）。"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
# 创建带有结构化输出的agent 这里使用 create_tool_calling_agent，它会自动处理工具调用
agent = create_tool_calling_agent(
    deepseek_llm,
    tools=[extract_movie_info],  # 添加工具
    prompt=prompt,  # 指定 Prompt
)
# 创建agent执行器 创建 Executor (这才是真正执行 Agent 并处理流式/结构化输出的关键)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[extract_movie_info],
    verbose=True,
    handle_parsing_errors=True  # 建议开启：当模型输出格式错误时自动重试
)

# 获取结果 (注意：必须传入字典 {"input": ...})
query = "请帮我提取电影《星际穿越》的信息。"
result = agent_executor.invoke({"input": query})
# 输出结果
output_str = result.get('output')
print(output_str)
# 输出结构化结果
output = json.loads(output_str)
print(f"name: {output['name']}")
print(f"publish_year: {output['publish_year']}")
print(f"director: {output['director']}")

# 模型 非Agent的情况下结构化输出
"""
需要明确你期望模型输出什么样的数据格式。在 LangChain 中，最优雅的方式是使用 Pydantic 来定义数据模型。
你需要指定字段的名称、类型，并尽量使用 Field 添加清晰的描述LangChain 提供的 .with_structured_output()
方法，将你在第一步定义的 Pydantic 模型“绑定”到模型上。需要注意的是，这个方法并不会立即执行请求，
它只是返回了一个带有约束规则的新可执行对象Runnable对象调用 .invoke() 方法，并传入包含待处理文本的 Prompt。
模型接收到指令后，会在底层自动将其转换为工具调用来强制约束输出格式，最终直接返回一个干净的 Pydantic 对象.
你可以像操作普通 Python 类一样直接访问它的属性。
"""
import os
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# 定义你期望的结构化输出格式
class MovieInfo(BaseModel):
    """从文本中提取的电影基本信息"""
    name: str = Field(..., description="电影的中文或英文名称")
    publish_year: str = Field(..., description="电影上映年份，例如 '2009'")
    director: str = Field(..., description="导演全名")
# 加载环境
load_dotenv()
# 定义工具函数
# 初始化模型
deepseek_llm = init_chat_model(
    model='deepseek-chat',
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
     temperature=0.0,
    max_tokens=2048,
    timeout=15)
# 模型调用内置函数 生成结构化输出
raw_text= "《盗梦空间》是由克里斯托弗·诺兰执导的一部科幻电影，于2010年上映。"
result = deepseek_llm.with_structured_output(MovieInfo).invoke(raw_text)
# 输出内容
print(f"name:{result.name}")
print(f"publish_year:{result.publish_year}")
print(f"director:{result.director}")
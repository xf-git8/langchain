"""
Tools
工具（Tools）扩展了智能体（Agents）的能力——使它们能够获取实时数据、
执行代码、查询外部数据库，并在现实世界中采取行动
模型的职责：它只根据当前的对话上下文进行推理，判断是否需要调用工具。
如果需要，它会生成一个包含“函数名称”和“所需参数”的结构化 JSON 请求。
程序的职责：你的后端宿主程序接收到这个 JSON 后，再去真实的环境中运行相应的代码，并将结果返回给模型。
"""
from langchain.tools import tool


# 基本工具定义
@tool
def search_database(query: str, limit: int = 10) -> str:
    """
        Search the customer database for records matching the query.
        Args: query:Search terms to look for
              limit:Maximum number of result to return
    """
    return f"Found {limit} result for '{query}'"


# 自定义工具属性
# 名称
@tool("web_search")  # Custom name
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"
print(search.name)
#描述
@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))
print(calc.description)
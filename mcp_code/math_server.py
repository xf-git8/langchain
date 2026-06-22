# 数学服务器
from fastmcp import FastMCP
from langchain_mcp_adapters import tools
# 创建实例
mcp = FastMCP('Math')
#注册工具
@mcp.tool()
def add(a, b):
    """Add two numbers"""
    return a + b
@mcp.tool()
def multiply(a, b):
    """Multiply two numbers"""
    return a * b
# 启动服务
if __name__ == '__main__':
    mcp.run()
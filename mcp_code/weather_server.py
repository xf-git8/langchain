from fastmcp import FastMCP
from langchain_mcp_adapters import tools
# 创建实例
mcp = FastMCP("Weather")
# 注册工具
@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return "It's always sunny in New York"

# 启动服务
if __name__ == "__main__":
    mcp.run()
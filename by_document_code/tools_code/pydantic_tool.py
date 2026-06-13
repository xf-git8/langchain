# 使用pydantic 模型定义工具
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    """Input for weather queries"""
    # 必填的字符串类型参数
    location: str = Field(description="city name or coordinates")
    # 带有固定选项和默认值的参数  摄氏度和华氏度
    untis: Literal["ssd", "hsd"] = Field(
        default="ssd",
        description="Temperature unit preference"
    )
    # 定义bool类型的开关参数 默认 false 返回当天  显示true 预报5天内信息
    include_forcast: bool


@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "ssd", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    # temp = 22 if units == 'ssd' else 72
    if units == 'ssd':
        temp = 22
    else:
        temp = 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "Next 5 days:sunny"
    return result

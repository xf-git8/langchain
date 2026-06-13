# 使用json schema 定义工具
from langchain_core.tools import tool

weather_schema = {
    'type': object,
    'properties': {
        'location': {'type': 'string'},
        'units': {'type': 'string'},
        'include_forecast': {'type': 'boolean'}
    },
    'required': ['location', 'units', 'include_forcast']
}


@tool(args_schema=weather_schema)
def get_weather(location: str, units: str = 'ssd', include_forecast: bool = False) -> str:
    """Get current weather and optional forecast"""
    if units == 'ssd':
        temp = 22
    else:
        temp = 72
    result = f"Current weather in {location}:{temp} degree {units[0].upper()}"
    if include_forecast:
        result += "Next 5 days:sunny"
    return result

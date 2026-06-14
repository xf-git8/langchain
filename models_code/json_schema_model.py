import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 加载环境 加载环境变量 配置api_key和base_url
load_dotenv()
# json_schame 格式
json_schema = {
    "title": "Movie",
    "description": "A movie with details",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "The title of the movie"
        },
        "year": {
            "type": "integer",
            "description": "The year the movie was released"
        },
        "director": {
            "type": "string",
            "description": "The director of the movie"
        },
        "rating": {
            "type": "number",
            "description": "The movie's rating out of 10"
        }
    },
    "required": ["title", "year", "director", "rating"]
}
# 初始化模型
deepseek_llm = init_chat_model(
    model='deepseek-chat',
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
    timeout=30,
    max_tokens=2048
)
# 绑定json_schema
model_with_structure = deepseek_llm.with_structured_output(
    json_schema
)
response = model_with_structure.invoke("你好，李焕英")
print(response)

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 加载环境
load_dotenv()
# 初始化模型
gpt_llm = init_chat_model(
    model="gpt-5.4-mini",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=1.0,
    timeout=15
)
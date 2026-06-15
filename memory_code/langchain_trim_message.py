# langchain provide trim_message function can trim the message history
#  according to the max_token limit,remain recent messages and system prompt
# 实现滑动窗口：保留最近的几轮对话，按对话轮次删除
        # 1. 保留最近的几轮对话，按对话轮次删除
        # 2. token阈值，token数超过阈值，从最早的消息删除对话

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import trim_messages
# 加载环境初始化模型
load_dotenv()
deepseek_llm = init_chat_model(
    model='deepseek-chat',
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
    temperature=0.0,
    timeout=15,
    max_tokens=4096
)
messages = 'dfasdfdss';
trimmer = trim_messages(
    max_tokens=4000,  # 设定最大保留的 token 数
    strategy="last",  # 从最新的消息开始往前算（保留最近的对话）
    token_counter=deepseek_llm,  # 传入你的 LLM 实例，让它自动计算 token
    include_system=True,  # 始终保留第一条 SystemMessage
    allow_partial=False  # 不允许把一条消息切一半，要么全留要么删掉
)
# 在调用前应用裁剪
messages = trimmer(messages)
# 使用python切片保留
recent_messages = messages[-5:]

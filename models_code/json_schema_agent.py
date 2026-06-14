import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# 加载环境与要求json格式
load_dotenv()
json_schema = {
    "title": "extract_contact_info",
    "description": "Contact information for a person.",
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "The name of the person"},
        "email": {"type": "string", "description": "The email address of the person"},
        "phone": {"type": "string", "description": "The phone number of the person"}
    },
    "required": ["name", "email", "phone"]
}
# 初始化模型
deepseek_llm = init_chat_model(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.2,
    timeout=30,
    max_tokens=2048,
)
# 创建智能体
agent = create_agent(
    model=deepseek_llm,
    tools=[],
    response_format=json_schema
)
model_with_structure = deepseek_llm.with_structured_output(
    json_schema
)
# 4. 绑定结构化输出约束
structured_llm = deepseek_llm.with_structured_output(json_schema)
result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})
print(result["structured_response"])

# 5. 构建 Prompt 并执行 (无需 Agent，直接用链式调用)
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个专业的信息提取助手。请严格按照给定的JSON Schema从用户输入的文本中提取联系人信息。如果缺失某项信息，请填入 null 或空字符串。不要输出任何额外的解释文本。"),
    ("human", "Extract contact info from: {text}")
])

# 组合 Prompt 和 结构化模型
chain = prompt | structured_llm

# 6. 执行并获取结果
result = chain.invoke({"text": "John Doe, john@example.com, (555) 123-4567"})
print(result)

# 多轮对话
# 引入while 循环使用messages列表保存历史对话（上下文）
import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_agent

# 加载环境变量
load_dotenv()


# 定义工具函数
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    # 模拟真实场景：如果是北京，返回具体天气；其他城市返回通用回复
    if "北京" in city:
        return "北京今天晴朗，气温 25°C，微风。"
    return f"{city} 今天也是阳光明媚的好天气！"


# 初始化大模型
deepseek_llm = ChatOpenAI(
    model="deepseek-chat",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0
)
# 创建智能体
agent = create_agent(
    model=deepseek_llm,
    tools=[get_weather],
    system_prompt="""你是一个有用的天气助手。
        当用户询问任何与天气相关的问题时，请务必调用 get_weather 工具来获取信息。
        如果用户没有提供城市名称，请追问用户。
        请用中文回答。"""
)
# 开启交互式对话循环
print("=== 天气助手已启动 (输入 'exit' 或 'quit' 退出) ===")
# 初始化消息列表
messages = []
while True:
    try:
        user_input = input("please input your question:").strip()
        # 退出机制
        if user_input.lower() in ['quit', 'exit']:
            print('bye')
            break
        if not user_input:
            continue
        # 将用户的新问题加入messages
        messages.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": messages})
        # 获取AI的最新回复
        response = result["messages"][-1]
        print(response.content)
        # 将回复也加入到消息列表中
        messages.append(response)

    except KeyboardInterrupt:
        print("\n程序被强制中断，再见！")
        break
    except Exception as e:
        print(f"\n发生错误: {e}")

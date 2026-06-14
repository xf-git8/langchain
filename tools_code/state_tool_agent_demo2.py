# 设你有一个 Agent，它的任务是：先调用搜索工具，找到一篇关于某个主题的文章链接（URL）。
# 再调用阅读工具，去读取这个 URL 的内容并生成摘要。痛点在于：第二个工具（阅读工具）怎么知道该去读哪个 URL？
# 它需要拿到第一步生成的结果。 如果把这个 URL 塞进大模型的 Prompt 里让它传给工具，既浪费 Token 又容易出错。
# 解决方案：把中间产生的 URL 存入 State，让下一个工具直接从 runtime.state 中读取！
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage
from dotenv import load_dotenv

# 自定义state 存放获取的url中间值
class WebState(AgentState):
    fetch_url: str = ''

# 加载环境
load_dotenv()

# 定义工具
@tool
def search_article(topic: str, runtime: ToolRuntime) -> str:
    """根据主题搜索文章."""
    # 模拟返回搜到链接
    fake_url = f"https://news.example.com/{topic.replace(' ', '-')}"
    # 状态存储链接
    runtime.state['fetch_url'] = fake_url
    return f"已找到关于 '{topic}' 的文章，链接为: {fake_url}"
@tool
def read_and_summary(runtime:ToolRuntime):
    """读取之前搜到的文章并总结"""
    url = runtime.state.get('fetch_url','')
    if not url:
        return "没有找到可以读取的文章，请先进行搜索"
    return f"成功读取了{url}的内容，这篇文章主要讲述了该领域的最新发展趋势，内容详实且结构清晰"
# 初始化模型
deepseek_llm = init_chat_model(
    model='deepseek-chat',
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL'),
    temperature=0.2,
    timeout=20,
    max_tokens=1024
)
# 创建agent
agent = create_agent(
    model=deepseek_llm,
    tools=[search_article, read_and_summary],
    state_schema=WebState,
    system_prompt="""你是一个新闻助手。请严格遵守以下工作流：
    1. 当用户要求了解某话题时，必须先调用 search_article 搜索文章。
    2. 拿到搜索结果后，必须紧接着调用 read_and_summary 进行阅读和总结。
    3. 只有当两个工具都执行完毕后，才能向用户输出最终的总结内容
    4. 如果没有，就说：'不知道，请见谅！'"""
)
# agent 执行工具
result = agent.invoke({
    'messages':[HumanMessage(content='帮我总结下最新的AI发展动态')]
})
print(result['messages'][-1].content)
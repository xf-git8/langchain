"""
在工具调用执行前，暂停代理程序的执行，以便人工审批、编辑或拒绝这些调用。人机协作机制在以下情况下非常有用：
需要人工批准的高风险操作（例如数据库写入、金融交易）。
需要人工监督的合规工作流程。
长时间的对话，其中人类的反馈会指导智能体
"""
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

from middleware_code.init_model import gpt_llm

def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

agent = create_agent(
    model=gpt_llm,
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "read_email_tool": False,
            }
        ),
    ],
)
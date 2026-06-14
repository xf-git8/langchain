# 工具可以通过一下方式访问当前对话状态 runtime.state
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage

# 访问状态 访问和读取短期记忆 . 获取最近的用户消息 (get_last_user_message)
@tool
def get_last_user_message(runtime: ToolRuntime) -> str:
    """Get the most recent message from the user."""
    messages = runtime.state['messages']
    # find the last human message
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return 'No user message found'


# Access custom state fields . 访问自定义状态字段 (get_user_preference)
@tool
def get_user_preference(
        pref_name: str,
        runtime: ToolRuntime
) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")
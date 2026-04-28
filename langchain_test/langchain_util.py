
from langchain_deepseek import ChatDeepSeek
import os
# apikey = "sk-10f6076d16da41e1b7dd0377d407be82"
os.environ["DEEPSEEK_API_KEY"] = "sk-0157cfa583d24e3f9ea99f3790066cb0"

def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"

agent = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None
)

messages = [
    ("system", "你是一个中英互译助手。"),
    ("human", "旧金山的天气怎么样")
]
resp = agent.invoke(messages)
print(resp.content)

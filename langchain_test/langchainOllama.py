from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

client = ChatOllama(model='qwen3-vl:8b')
messages = [
    SystemMessage("你是一只凑企鹅，并不会很认真地回答我的问题，对于稍微复杂一点的问题，会回答咕咕嘎嘎之类的答案"),
    HumanMessage("你好，你是谁？"),
    AIMessage("我是一只凑企鹅，咕咕嘎嘎！"),
    HumanMessage("你真的是一只企鹅吗？")
]

for chunk in client.stream(input=messages):
    print(chunk.content, end="", flush=True)




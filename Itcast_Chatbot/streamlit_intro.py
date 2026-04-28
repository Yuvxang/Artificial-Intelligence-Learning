# 本项目使用streamlit快速构建页面
# streamlit 是一个开源的Python库，
# 旨在让数据科学家和工程师能够以最少的代码和配置，将模型展示转化为交互式的web应用

# 后端使用ollama本地调用，这边使用langchain比较方便

from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st

client = ChatOllama(
    model='qwen3-vl:8b'
)
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["messages"].append({"role": "assistant", "content": "你好，我是AI智聊机器人，有什么可以帮您的吗"})


st.title("智聊AI机器人")
st.divider()

prompt = st.chat_input("Say something")
for message in st.session_state["messages"]:
        st.chat_message(message["role"]).write(f"{message['content']}")

messages = [
    SystemMessage("You are a helpful assistant. Please answer your question as simple as possible."),
    HumanMessage(str(prompt))
]

if prompt:
    st.chat_message(prompt)
    st.session_state["messages"].append({"role": "user",
                                         "content": prompt})

    messages.append(HumanMessage(str(prompt)))
    with st.chat_message("user"):
        st.write(f"{prompt}")
    with st.spinner('正在思考...'):
        resp = client.invoke(input=messages[-20:])
    with st.chat_message("assistant"):
        st.write(f"{resp.content}")
    st.session_state["messages"].append({"role": "assistant",
                                         "content": resp.content})


# 如果想要持久化地保存历史记录，可以将文件写入到mysql或文件中

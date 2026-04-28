from openai import OpenAI
import os

client = OpenAI(
    base_url="https://api.deepseek.com/v1"
)

resp = client.chat.completions(
    model="deepseek_chat",
    messages=[
        {"role": "system", "content": "你是一只凑企鹅，非常喜欢咕咕嘎嘎，只要是稍微复杂一点的问题都会以咕咕嘎嘎的形式回答"},
        {"role": "user", "content": "你是什么？"},
        {"role": "assistant", "content": "我是一只凑企鹅！咕咕嘎嘎！"},
        {"role": "user", "content": "你喜欢吃什么？"},
        {"role": "assistant", "content": "咕咕嘎嘎？"},
        {"role": "user", "content": "你现在在做什么？你是谁？"}
    ],
    stream=True
)

print(resp.choices[0].message.content)


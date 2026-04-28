import streamlit as st

st.title("传智教育用户注册平台")
st.divider()
index = 1

st.write("请输入用户名:")
user_name = st.text_input("用户名")
password = st.text_input('密码', type="password")
age = st.number_input('年龄', value=18, min_value=0, max_value=100)
sex = st.radio('性别:', options=['男', '女', '保密'], horizontal=True)
birthday = st.date_input('出生日期')
user_height = st.slider('身高', min_value=0, max_value=250, value=100)
if st.button('确认'):
    st.write("信息录入成功")

with open(f'D:/Coding/RAG/Itcast_Chatbot/user{index}.txt', 'w', encoding='utf-8') as fu:
    fu.write(f'用户名: {str(user_name)}\n')
    fu.write(f'密码: {str(password)}\n')
    fu.write(f'年龄: {str(age)}\n')







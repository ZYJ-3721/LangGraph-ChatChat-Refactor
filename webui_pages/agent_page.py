import streamlit as st
from streamlit_extras.bottom_container import bottom

from .utils import model_settings_init, model_settings_dialog, get_chatllm


AGENT_PAGE_INTRODUCTION = "你好，我是你的AI小助手`（Agent对话模式）`，我可以基于你左侧选择的工具回答问题，请问有什么可以帮助您？"

def get_chat_response(platform, model, api_url, api_key, max_tokens, temperature, input):
    try:
        llm = get_chatllm(platform, model, api_url, api_key, max_tokens, temperature)
        for chunk in llm.stream(input):
            yield chunk.content
    except Exception as e:
        st.error(e)

def display_chat_history():
    for message in st.session_state["agent_chat_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def clear_chat_history():
    st.session_state["agent_chat_history"] = [
        {"role": "assistant", "content": AGENT_PAGE_INTRODUCTION}]

def agent_page():
    model_settings_init()
    if "agent_chat_history" not in st.session_state:
        st.session_state["agent_chat_history"] = [
            {"role": "assistant", "content": AGENT_PAGE_INTRODUCTION}]
    display_chat_history()

    with bottom():
        cols = st.columns([1, 10, 1])
        if cols[0].button(":gear:", help="模型配置"):
            model_settings_dialog()
        if cols[2].button(":wastebasket:", help="清空对话"):
            clear_chat_history()
        input = cols[1].chat_input("请输入您的问题")
    
    tools = ["fake1","fake2", "fake3"]
    agent_workflows = ["基础Agent工作流"]
    with st.sidebar:
        selected_agent_workflow = st.selectbox("选择工作流", options=agent_workflows)
        selected_tools = st.multiselect("选择工具", options=tools)
    
    if input:
        with st.chat_message("user"):
            st.write(input)
        st.session_state["agent_chat_history"] += [{'role': 'user', 'content': input}]
        stream_response = get_chat_response(
            st.session_state["platform"],
            st.session_state["model"],
            st.session_state["api_url"],
            st.session_state["api_key"],
            st.session_state["max_tokens"],
            st.session_state["temperature"],
            st.session_state["agent_chat_history"][-st.session_state["history_len"]:])
        with st.chat_message("assistant"):
            response = st.write_stream(stream_response)
        st.session_state["agent_chat_history"] += [{'role': 'assistant', 'content': response}]

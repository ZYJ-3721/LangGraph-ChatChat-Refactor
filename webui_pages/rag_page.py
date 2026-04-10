import streamlit as st
from streamlit_extras.bottom_container import bottom

from webui_pages.utils import model_settings_init, model_settings_dialog, get_chatllm
from rag.graphs.base_rag import create_rag_graph


RAG_PAGE_INTRODUCTION = "你好，我是你的AI小助手`（RAG对话模式）`，我可以基于你左侧选择的知识库回答问题，请问有什么可以帮助您？"

def get_rag_response(platform, model, api_url, api_key, max_tokens, temperature, selected_rag_graph, selected_kb_names, input):
    try:
        llm = get_chatllm(platform, model, api_url, api_key, max_tokens, temperature)
        graph = create_rag_graph(llm)
        for event in graph.stream(input={"messages": input},
            config={"configurable": {"thread_id": 21}},
            stream_mode="messages"):
            yield event[0].content
    except Exception as e:
        st.error(e)

def display_chat_history():
    for message in st.session_state["rag_chat_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def clear_chat_history():
    st.session_state["rag_chat_history"] = [
        {"role": "assistant", "content": RAG_PAGE_INTRODUCTION}]

def rag_page():
    model_settings_init()
    if "rag_chat_history" not in st.session_state:
        st.session_state["rag_chat_history"] = [
            {"role": "assistant", "content": RAG_PAGE_INTRODUCTION}]
    display_chat_history()

    with bottom():
        cols = st.columns([1, 10, 1])
        if cols[0].button(":gear:", help="模型配置"):
            model_settings_dialog()
        if cols[2].button(":wastebasket:", help="清空对话"):
            clear_chat_history()
        input = cols[1].chat_input("请输入您的问题")
    
    from kbm.kbs_table import get_kb_names
    kb_names=get_kb_names()
    rag_graphs = ["基础RAG"]
    with st.sidebar:
        selected_rag_graph = st.selectbox("选择工作流", options=rag_graphs, key="selected_rag_graph")
        selected_kb_names = st.multiselect("选择知识库", options=kb_names, key="selected_kb_names")
    
    if input:
        with st.chat_message("user"):
            st.write(input)
        st.session_state["rag_chat_history"] += [{'role': 'user', 'content': input}]
        stream_response = get_rag_response(
            st.session_state["platform"],
            st.session_state["model"],
            st.session_state["api_url"],
            st.session_state["api_key"],
            st.session_state["max_tokens"],
            st.session_state["temperature"],
            st.session_state["selected_rag_graph"],
            st.session_state["selected_kb_names"],
            st.session_state["rag_chat_history"][-st.session_state["history_len"]:])
        with st.chat_message("assistant"):
            response = st.write_stream(stream_response)
        st.session_state["rag_chat_history"] += [{'role': 'assistant', 'content': response}]

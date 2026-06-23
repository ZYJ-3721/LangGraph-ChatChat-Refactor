import streamlit as st

from langchain_core.messages import AIMessageChunk
from webui_pages.utils import init_model_settings, model_settings_dialog, get_chatllm
from agent.graphs import AGENT_GRAPHS, create_agent_graph
from agent.tools import AGETN_TOOLS


AGENT_PAGE_INTRODUCTION = "你好，我是你的AI助手`（Agent对话模式）`，我可以基于你左侧选择的工具回答问题，请问有什么可以帮助您？"

def get_agent_response(platform, model, api_url, api_key, thinking, max_tokens, temperature, selected_rag_graph, selected_tool_names, input):
    llm = get_chatllm(platform, model, api_url, api_key, thinking, max_tokens, temperature)
    graph = create_agent_graph(llm, selected_rag_graph, selected_tool_names)
    tools_status = {}
    main_state = "complete"
    reasoning_state = "complete"
    for event in graph.stream(
        input={"messages": input},
        config={"configurable": {"thread_id": 21}},
        stream_mode=["updates", "tools", "messages"],
    ):
        # st.write(event)
        if event[0] == "updates":
            if (node_data := event[1].get("call_llm")) and (tool_calls := node_data["messages"][0].tool_calls):
                tool_name_list_str = " ".join([f"`{tool_call['name']}`" for tool_call in tool_calls])
                batch_status = main_status.status(f"正在调用 {tool_name_list_str} ...", expanded=True)
                for tool_call in tool_calls:
                    tools_status[tool_call["id"]] = {
                        "name": tool_call["name"],
                        "status": batch_status.status(f"正在执行 `{tool_call['name']}` ...", expanded=True)
                    }
                    tools_status[tool_call["id"]]["status"].write(f"工具输入：`{tool_call['args']}`")
            if (node_data := event[1].get("tools")):
                batch_status.update(label=f"{tool_name_list_str} 已完成", expanded=False, state="complete")
        
        if event[0] == "tools":
            if event[1]["event"] == "tool-finished":
                tools_status[event[1]["tool_call_id"]]["status"].write(f"工具输出：{event[1]['output'].content}")
                tools_status[event[1]["tool_call_id"]]["status"].update(
                    label=f"`{event[1]['output'].name}` 已完成", expanded=False, state="complete")
            if event[1]["event"] == "tool-error":
                tools_status[event[1]["tool_call_id"]]["status"].error(f"错误消息：{event[1]['message']}")
                tools_status[event[1]["tool_call_id"]]["status"].update(
                    label=f"`{tools_status[event[1]['tool_call_id']]['name']}` 执行失败", expanded=True, state="error")
                batch_status.update(label=f"`{tool_name_list_str}` 调用失败", expanded=True, state="error")
                main_status.update(label="Agent 工作失败", expanded=True, state="error")
        
        if event[0] == "messages":
            if type(event[1][0]) == AIMessageChunk:
                reasoning_content = event[1][0].additional_kwargs.get("reasoning_content")

                if (reasoning_content or event[1][0].tool_calls) and main_state == "complete":
                    main_status = st.status("Agent 工作中 ...", expanded=True)
                    main_state = "running"
                
                if reasoning_content and reasoning_state == "complete":
                    reasoning_status = main_status.status("正在思考 ...", expanded=True)
                    reasoning_placeholder = reasoning_status.empty()
                    reasoning_state = "running"
                    reasoning_c = ""
                
                if reasoning_content and reasoning_state == "running":
                    reasoning_c += reasoning_content
                    reasoning_placeholder._markdown(reasoning_c, unterminated_parsing=True)
                
                if not reasoning_content and reasoning_state == "running":
                    reasoning_status.update(label=f"思考已完成", expanded=False, state="complete")
                    reasoning_state = "complete"
                
                if not event[1][0].tool_calls and event[1][0].content.strip():
                    if main_state == "running":
                        main_status.update(label="Agent 工作已完成", expanded=False, state="complete")
                        main_state = "complete"
                    yield event[1][0].content

def display_chat_history():
    with st.chat_message("assistant"):
        st.write(AGENT_PAGE_INTRODUCTION)
    for message in st.session_state["agent_chat_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def init_chat_history():
    if "agent_chat_history" not in st.session_state:
        st.session_state["agent_chat_history"] = []

def clear_chat_history():
    st.session_state["agent_chat_history"] = []

def agent_page():
    init_model_settings()
    init_chat_history()
    display_chat_history()

    with st.bottom:
        cols = st.columns([1, 11, 1], vertical_alignment="center")
        if cols[0].button(":gear:", help="模型配置"):
            model_settings_dialog()
        if cols[2].button(":wastebasket:", help="清空对话"):
            clear_chat_history()
        input = cols[1].chat_input("请输入您的问题")
    
    with st.sidebar:
        selected_agent_workflow = st.selectbox("选择工作流", options=AGENT_GRAPHS.keys(), key="selected_agent_graph")
        selected_tool_names = st.multiselect("选择工具", options=AGETN_TOOLS.keys(), key="selected_tool_names")
    
    if input:
        with st.chat_message("user"):
            st.write(input)
        st.session_state["agent_chat_history"].append({'role': 'user', 'content': input})

        with st.chat_message("assistant"):
            try:
                stream_response = get_agent_response(
                    st.session_state["platform"],
                    st.session_state["model"],
                    st.session_state["api_url"],
                    st.session_state["api_key"],
                    st.session_state["thinking"],
                    st.session_state["max_tokens"],
                    st.session_state["temperature"],
                    st.session_state["selected_agent_graph"],
                    st.session_state["selected_tool_names"],
                    st.session_state["agent_chat_history"][-st.session_state["history_len"]:])
            except Exception as e:
                st.session_state["agent_chat_history"].pop()
                st.error(e)
            response = st.write_stream(stream_response)
        st.session_state["agent_chat_history"].append({'role': 'assistant', 'content': response})

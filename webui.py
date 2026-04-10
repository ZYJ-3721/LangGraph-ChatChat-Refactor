import streamlit as st
import streamlit_antd_components as sac

from webui_pages import rag_page, agent_page, kbm_page


if __name__ == "__main__":

    st.set_page_config(
        page_title="AI Chat WebUI",
        # page_icon="img/icon.png",
        # layout="wide",
        menu_items={
            "About": "**欢迎使用 AI Chat WebUI !**",
            "Get help": "https://github.com/xxx/xxx",
            "Report a bug": "https://github.com/xxx/xxx",
        }
    )
    
    with st.sidebar:
        # st.logo(
        #     size="large",
        #     image="img/long_icon.png",
        #     icon_image="img/short_icon.png")
        selected_page = sac.menu(
            items=[
                sac.MenuItem(label="RAG 对 话", icon="chat-dots"),
                sac.MenuItem(label="Agent 对 话", icon="robot"),
                sac.MenuItem(label="知 识 库 管 理", icon="database-add"),
            ]
        )
        sac.divider() # 水平分割线
    
    if selected_page == "RAG 对 话":
        rag_page()
    elif selected_page == "Agent 对 话":
        agent_page()
    elif selected_page == "知 识 库 管 理":
        kbm_page()
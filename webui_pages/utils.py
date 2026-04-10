import streamlit as st
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


PLATFORMS_API_URL = {
    "Xinference": "http://192.168.12.19:9997/v1",
    "Ollama": "http://127.0.0.1:11434/v1",
    "OpenAI": "https://api.openai.com/v1",
    "ZhipuAI": "https://open.bigmodel.cn/api/paas/v4", # 99b83f53f2c84fcd885939e3742b3cd2.zgywhYo4dLTwxViS
}
PLATFORMS = list(PLATFORMS_API_URL.keys())

def model_settings_init():
    if "platform" not in st.session_state:
        st.session_state["platform"] = PLATFORMS[0]
        st.session_state["api_url"] = ""
        st.session_state["api_key"] = ""
        st.session_state["model"] = next(iter(get_llm_models(st.session_state["platform"])), "")
        st.session_state["max_tokens"] = 1024
        st.session_state["history_len"] = 25
        st.session_state["temperature"] = .7

@st.dialog("模型配置", width="large")
def model_settings_dialog():
    cols1 = st.columns(2) # 第一行两列
    cols2 = st.columns(2) # 第二行两列
    platform = cols1[0].selectbox(
        "选择平台", options=PLATFORMS, index=PLATFORMS.index(st.session_state.platform))
    api_url = cols2[0].text_input(
        "API URL", st.session_state.api_url, placeholder=PLATFORMS_API_URL[platform])
    api_key = cols2[1].text_input(
        "API KEY", st.session_state.api_key, placeholder="EMPTY")
    if platform in ["Xinference", "Ollama"]:
        model = cols1[1].selectbox("选择模型", options=get_llm_models(platform, api_url, api_key))
    else:
        model = cols1[1].text_input("选择模型", st.session_state.model, placeholder="（必填）")
    max_tokens = st.slider("Max Tokens", 0, 4096, st.session_state.max_tokens)
    history_len = st.slider("History Len", 1, 50, st.session_state.history_len)
    temperature = st.slider("Temperature", 0., 1., st.session_state.temperature)
    if st.button("确认", use_container_width=True):
        st.session_state.platform = platform
        st.session_state.api_url = api_url
        st.session_state.api_key = api_key
        st.session_state.model = model
        st.session_state.max_tokens = max_tokens
        st.session_state.history_len = history_len
        st.session_state.temperature = temperature
        st.rerun()

def get_base_url(api_url):
    if not api_url: return
    parsed_url = urlparse(api_url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"

def get_llm_models(platform: str, api_url: str=None, api_key: str=None):
    if platform == "Xinference":
        try:
            from xinference_client import RESTfulClient as Client
            client = Client(get_base_url(api_url or PLATFORMS_API_URL[platform]))
            return [k for k, v in client.list_models().items() if "LLM" in v.get("model_type")]
        except Exception as e:
            st.error(e)
            return []
    elif platform == "Ollama":
        try:
            from ollama import Client
            client = Client(get_base_url(api_url or PLATFORMS_API_URL[platform]))
            return [m["model"] for m in client.list().models if "llama" in m.details.get("family")]
        except Exception as e:
            st.error(e)
            return []
    elif platform == "OpenAI":
        return []

def get_chatllm(
        platform: str, model: str, 
        api_url: str=None, api_key: str=None,
        max_tokens: int=None, temperature: float=None):
    api_url = api_url or PLATFORMS_API_URL[platform]
    api_key = api_key or "EMPTY"
    return ChatOpenAI(
        model=model, base_url=api_url, api_key=api_key,
        max_tokens=max_tokens, temperature=temperature, streaming=True)

def get_embedding_models(platform: str, api_url: str=None, api_key: str=None):
    if platform == "Xinference":
        try:
            from xinference_client import RESTfulClient as Client
            client = Client(get_base_url(api_url or PLATFORMS_API_URL[platform]))
            return [k for k, v in client.list_models().items() if "embedding" in v.get("model_type")]
        except Exception as e:
            st.error(e)
            return []
    elif platform == "Ollama":
        try:
            from ollama import Client
            client = Client(get_base_url(api_url or PLATFORMS_API_URL[platform]))
            return [m["model"] for m in client.list().models if "bert" in m.details.get("family")]
        except Exception as e:
            st.error(e)
            return []
    elif platform == "OpenAI":
        return []

def get_embedding(
        platform: str, model: str,
        api_url: str=None, api_key: str=None):
    api_url = api_url or PLATFORMS_API_URL[platform]
    api_key = api_key or "EMPTY"
    return OpenAIEmbeddings(model=model, base_url=api_url, api_key=api_key)


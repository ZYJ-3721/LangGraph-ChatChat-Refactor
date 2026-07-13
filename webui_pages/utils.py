import streamlit as st
from urllib.parse import urlparse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import settings


PLATFORMS_API_URL = {
    "GPUStack": "http://192.168.12.242:8780/v1",
    "Xinference": "http://192.168.12.19:9997/v1",
    "Ollama": "http://127.0.0.1:11434/v1",
    "OpenAI": "https://api.openai.com/v1",
    "ZhipuAI": "https://open.bigmodel.cn/api/paas/v4",
}
PLATFORMS = list(PLATFORMS_API_URL.keys())

def init_model_settings():
    if "platform" not in st.session_state:
        st.session_state["platform"] = settings.platform or PLATFORMS[0]
        st.session_state["api_url"] = settings.api_url
        st.session_state["api_key"] = settings.api_key
        st.session_state["model"] = next(iter(get_llm_models(st.session_state["platform"])), settings.model)
        st.session_state["thinking"] = 0
        st.session_state["max_tokens"] = 65536
        st.session_state["history_len"] = 50
        st.session_state["temperature"] = .7

@st.dialog("模型配置", width="medium")
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
    thinking = st.slider("Thinking", 0, 8192, st.session_state.thinking)
    max_tokens = st.slider("Max Tokens", 0, 262144, st.session_state.max_tokens)
    history_len = st.slider("History Len", 1, 100, st.session_state.history_len)
    temperature = st.slider("Temperature", 0., 1., st.session_state.temperature)
    if st.button("确认", width="stretch"):
        st.session_state.platform = platform
        st.session_state.api_url = api_url
        st.session_state.api_key = api_key
        st.session_state.model = model
        st.session_state.thinking = thinking
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
            client = Client(get_base_url(api_url or PLATFORMS_API_URL[platform]), api_key)
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
    else:
        return []

def get_chatllm(
        platform: str, model: str, 
        api_url: str=None, api_key: str=None,
        thinking: int=0, max_tokens: int=0, temperature: float=.7):
    api_url = api_url or PLATFORMS_API_URL[platform]
    api_key = api_key or "EMPTY"
    extra_body={
        "chat_template_kwargs": {"enable_thinking": thinking > 0},
        "thinking_token_budget": thinking,
        "reasoning_budget": thinking,
        "thinking": {
            "type": "enabled" if thinking else "disabled",
            "budget_tokens": thinking
        }
    }
    return ChatOpenAIWithReasoning(
        model=model, base_url=api_url, api_key=api_key,
        max_tokens=max_tokens, temperature=temperature,
        extra_body=extra_body, streaming=True, stream_usage=True
    )

def get_embedding_models(platform: str, api_url: str=None, api_key: str=None):
    if platform == "Xinference":
        try:
            from xinference_client import RESTfulClient as Client
            client = Client(get_base_url(api_url or PLATFORMS_API_URL[platform]), api_key)
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
    else:
        return []

def get_embedding(
        platform: str, model: str,
        api_url: str=None, api_key: str=None):
    api_url = api_url or PLATFORMS_API_URL[platform]
    api_key = api_key or "EMPTY"
    return OpenAIEmbeddings(model=model, base_url=api_url, api_key=api_key)



import openai
from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatResult, ChatGenerationChunk

class ChatOpenAIWithReasoning(ChatOpenAI):
    """支持 reasoning_content 的 ChatOpenAI"""

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        """提取结果中的 reasoning_content 到 additional_kwargs"""
        chat_result = super()._create_chat_result(
            response, generation_info
        )
        if (isinstance(response, openai.BaseModel)
            and (choices := getattr(response, "choices", None))):
            reasoning_content = (
                getattr(choices[0].message, "reasoning", None) or
                getattr(choices[0].message, "reasoning_content", None)
            )
            if reasoning_content is not None:
                chat_result.generations[0].message.additional_kwargs["reasoning_content"] = reasoning_content
        return chat_result
    
    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """提取流式响应中的 reasoning_content 到 additional_kwargs"""
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if (generation_chunk
            and (choices := chunk.get("choices", [])) 
            and (delta := choices[0].get("delta", {}))):
            reasoning_content = (
                delta.get("reasoning") or
                delta.get("reasoning_content")
            )
            if reasoning_content is not None:
                generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning_content
        return generation_chunk



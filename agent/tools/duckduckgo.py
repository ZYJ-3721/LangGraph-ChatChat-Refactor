from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults


def duckduckgo_search():
    """DuckDuckGO 网页搜索"""
    api_wrapper = DuckDuckGoSearchAPIWrapper(
        region="cn",
        safesearch="strict",
    )
    return DuckDuckGoSearchResults(
        api_wrapper=api_wrapper,
        output_format="list",
        max_results=5,
    )

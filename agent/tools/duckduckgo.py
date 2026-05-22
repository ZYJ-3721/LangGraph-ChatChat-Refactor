from langchain_community.tools import DuckDuckGoSearchResults


def duckduckgo_search():
    """DuckDuckGO 网页搜索"""
    return DuckDuckGoSearchResults(output_format="list")

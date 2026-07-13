from langchain_community.utilities import SearxSearchWrapper
from langchain_community.tools import SearxSearchResults


def searxng_search():
    """SearXNG 网页搜索"""
    wrapper = SearxSearchWrapper(
        searx_host="http://localhost:8781",
        params={"language": "zh"},
    )
    return SearxSearchResults(
        wrapper=wrapper,
        kwargs={"format": "json"},
        max_results=5,
    )

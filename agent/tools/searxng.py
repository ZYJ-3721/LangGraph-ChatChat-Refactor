from langchain_community.utilities import SearxSearchWrapper
from langchain_community.tools import SearxSearchResults
from config import settings


def searxng_search():
    """SearXNG 网页搜索"""
    wrapper = SearxSearchWrapper(
        searx_host=settings.searxng_host,
        params={"language": "zh"},
    )
    return SearxSearchResults(
        wrapper=wrapper,
        kwargs={"format": "json"},
        max_results=10,
    )

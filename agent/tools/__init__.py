from .duckduckgo import duckduckgo_search
from .searxng import searxng_search
from .nowtime import get_nowtime
from .weather import get_weather


AGETN_TOOLS = {
    "Duckduckgo 搜索": duckduckgo_search(),
    "SearXNG 搜索": searxng_search(),
    "天气查询": get_weather,
    "时间获取": get_nowtime,
}

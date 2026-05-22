from langchain.tools import tool
from datetime import datetime
from zoneinfo import ZoneInfo


@tool
def get_nowtime():
    """获取当前时间"""
    tz = ZoneInfo("Asia/Shanghai")
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M %A")

from langchain.tools import tool
import requests


@tool
def get_weather(city: str):
    """获取指定城市的天气
    Args:
        city: 输入城市的英文或拼音
    """
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=zh"
    geo_response = requests.get(geo_url).json()
    if not geo_response.get("results"):
        return f"未找到{city}的城市坐标，无法查询天气"
    
    timezone = geo_response["results"][0]["timezone"]
    latitude = geo_response["results"][0]["latitude"]
    longitude = geo_response["results"][0]["longitude"]

    weather_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "timezone": timezone,
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": "true",
        "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m,winddirection_10m",
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,weathercode",
    }
    response = requests.get(weather_url, params=params)
    if response.status_code != 200:
        return f"请求错误，{city}天气查询失败"
    
    return response.json()

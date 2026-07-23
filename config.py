from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mcp_json_dir: str = ""
    skills_dir: str = ""
    platform: str = ""
    api_url: str = ""
    api_key: str = ""
    model: str = ""
    searxng_host: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

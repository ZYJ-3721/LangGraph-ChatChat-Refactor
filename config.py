from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    platform: str = ""
    api_url: str = ""
    api_key: str = ""
    model: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

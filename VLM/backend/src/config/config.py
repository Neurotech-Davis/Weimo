"""requirements.txt
pydantic_settings
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Defining 1 main API key:
    HF_KEY
    """

    HF_KEY: str
    model_config = SettingsConfigDict(env_file=".env")

@lru_cache
def get_settings():
    """
    Loads API keys and other settings, caching the result.
    """
    # returns settings object
    return Settings()  # type: ignore (making linter shut up)

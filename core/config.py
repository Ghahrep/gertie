# in core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Manages application settings and secrets by loading them from a .env file.
    """
    # --- Database Configuration ---
    DATABASE_URL: str

    # --- LLM API Keys ---
    # We need a field to load the key from the .env file.
    # We make it Optional so the app doesn't crash if it's not set.
    ANTHROPIC_API_KEY: Optional[str] = None 

    # --- JWT Authentication ---
    SECRET_KEY: str = "a_very_secret_key_that_you_should_change"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # This tells Pydantic to load the variables from a file named '.env'
    model_config = SettingsConfigDict(env_file=".env")

# Create a single, importable instance of the settings
settings = Settings()
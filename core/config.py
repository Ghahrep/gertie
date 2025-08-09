from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Manages application settings and secrets by loading them from a .env file.
    """
    # --- Database Configuration ---
    # This is the variable you will define in your .env file.
    # Pydantic will automatically read it.
    DATABASE_URL: str

    # --- JWT Authentication (for Task 2) ---
    # We can add these now to prepare for the next steps.
    SECRET_KEY: str = "a_very_secret_key_that_you_should_change"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # This tells Pydantic to load the variables from a file named '.env'
    model_config = SettingsConfigDict(env_file=".env")

# Create a single, importable instance of the settings
settings = Settings()
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """Application settings, loaded from environment variables or .env file."""
    
    # App Settings
    ENV: str = "development"
    DEBUG: bool = True
    
    # Logging Config
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: str = str(BASE_DIR / "logs" / "app.log")
    
    # Dataset Config
    DATASET_PATH: str = str(BASE_DIR / "data" / "ONAM EXCEL.csv")
    
    # AI Config
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY_1: Optional[str] = None
    OPENROUTER_API_KEY_2: Optional[str] = None
    OPENROUTER_API_KEY_3: Optional[str] = None
    OPENROUTER_API_KEY_4: Optional[str] = None
    
    # Configurations to read from environment or .env
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate settings
settings = Settings()

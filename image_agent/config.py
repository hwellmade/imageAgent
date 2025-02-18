from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Application settings."""
    # Base
    APP_NAME: str = "Image Agent"
    DEBUG: bool = True
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    
    # CORS
    CORS_ORIGINS: list[str] = ["*"]
    
    # API Keys (will be loaded from environment variables)
    GOOGLE_CLOUD_PROJECT: str | None = None
    GOOGLE_APPLICATION_CREDENTIALS: str | None = None
    OPENAI_API_KEY: str | None = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 
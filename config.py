# src/config.py
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "pratc"
    DEBUG: bool = True

    VECTOR_STORE: str = "chroma"
    CHROMA_DB_DIR: str = ".chroma"

    EMBEDDING_PROVIDER: str = "cohere"
    COHERE_API_KEY: str = Field(..., env="COHERE_API_KEY")
    EMBEDDING_MODEL: str = "embed-english-v3.0"

    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")
    GROQ_MODEL: str = "deepseek-r1-distill-llama-70b"

    # CHUNK_SIZE: int = 800
    # CHUNK_OVERLAP: int = 160
    # TOP_K: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

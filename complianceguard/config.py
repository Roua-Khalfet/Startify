import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Serper (Search)
    SERPER_API_KEY: str = Field(default="", env="SERPER_API_KEY")

    # Neo4j Configureation
    NEO4J_URI: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    NEO4J_USERNAME: str = Field(default="neo4j", env="NEO4J_USERNAME")
    NEO4J_PASSWORD: str = Field(default="password", env="NEO4J_PASSWORD")
    
    # Qdrant Configuration
    QDRANT_URL: str = Field(default="", env="QDRANT_URL")
    QDRANT_API_KEY: str = Field(default="", env="QDRANT_API_KEY")
    QDRANT_PATH: str = Field(default="", env="QDRANT_PATH")
    QDRANT_COLLECTION_NAME: str = Field(default="complianceguard_chunks", env="QDRANT_COLLECTION_NAME")
    
    # Ollama Embeddings
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    OLLAMA_EMBED_MODEL: str = Field(default="qwen3-embedding:0.6b", env="OLLAMA_EMBED_MODEL")
    OLLAMA_INFERENCE_MODEL: str = Field(default="llama3.1", env="OLLAMA_MODEL") # Fallback model
    
    # Azure OpenAI Configuration (Extraction/Agent)
    AZURE_API_KEY: str = Field(default="", env="AZURE_API_KEY")
    AZURE_API_BASE: str = Field(default="", env="AZURE_API_BASE")
    AZURE_API_VERSION: str = Field(default="2024-02-15-preview", env="AZURE_API_VERSION")
    AZURE_MODEL: str = Field(default="gpt-4o", env="AZURE_MODEL")
    
    # Processing specifics
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 300
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Instanciate global singleton
config = Settings()

# Azure helpers
def get_azure_llm_kwargs():
    """Returns kwargs for AzureChatOpenAI standard init."""
    return {
        "azure_endpoint": config.AZURE_API_BASE,
        "api_key": config.AZURE_API_KEY,
        "api_version": config.AZURE_API_VERSION,
        "azure_deployment": config.AZURE_MODEL,
        "temperature": 0.0,
    }

def get_ollama_embed_kwargs():
    """Returns kwargs for OllamaEmbeddings init."""
    return {
        "model": config.OLLAMA_EMBED_MODEL,
        "base_url": config.OLLAMA_BASE_URL
    }

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = "llama-3.1-8b-instant"  # Current working model

    # Alternative models if primary fails
    GROQ_FALLBACK_MODELS = [
        "llama-3.1-70b-versatile",
        "gemma2-9b-it"
    ]
    
    # Document Processing
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 50
    MAX_CHUNKS_FOR_CONTEXT = 5
    
    # Embedding Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = int(os.getenv("PORT", 8000))

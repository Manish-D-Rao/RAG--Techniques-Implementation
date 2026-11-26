import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    """Get an integer environment variable value, or return default if unset/invalid."""
    val = os.getenv(name)

    if not val or val.strip() == "":
        return default
    try:
        return int(val)
    except:
        print(f"Invalid integer value for {name}, using default {default}")
        return default


PROJECT_ROOT = Path(__file__).parent

# OpenAI (required)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEXUS_BASE_URL = os.getenv("NEXUS_BASE_URL")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Vector database
VECTOR_DB = os.getenv("VECTOR_DB", "faiss")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "advanced-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# Chunking
PARENT_CHUNK_SIZE = _get_int("PARENT_CHUNK_SIZE", 1600)
PARENT_CHUNK_OVERLAP = _get_int("PARENT_CHUNK_OVERLAP", 200)
CHILD_CHUNK_SIZE = _get_int("CHILD_CHUNK_SIZE", 400)
CHILD_CHUNK_OVERLAP = _get_int("CHILD_CHUNK_OVERLAP", 60)

# Retrieval
NUM_QUERY_VARIANTS = _get_int("NUM_QUERY_VARIANTS", 3)
VECTOR_TOP_K = _get_int("VECTOR_TOP_K", 6)
BM25_TOP_K = _get_int("BM25_TOP_K", 6)
HYBRID_TOP_K = _get_int("HYBRID_TOP_K", 10)
RERANK_TOP_K = _get_int("RERANK_TOP_K", 4)
MIN_RERANK_SCORE = os.getenv("MIN_RERANK_SCORE", "-4.0")
MAX_CONTEXT_CHARS = _get_int("MAX_CONTEXT_CHARS", 8000)
MEMORY_WINDOW = _get_int("MEMORY_WINDOW", 4)

# Reranker
RERANKER = os.getenv("RERANKER", "cross-encoder")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def check_openai_api_key() -> bool:
    """Validate that the OpenAI API key is set."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required but not set in .env")
    
    return True


def check_pinecone_api_key() -> bool:
    """Validate that the Pinecone API key is set when using Pinecone."""
    if VECTOR_DB == "pinecone" and not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is required when VECTOR_DB=pinecone")
    if VECTOR_DB == "pinecone" and not PINECONE_INDEX_NAME:
            raise RuntimeError("PINECONE_INDEX_NAME is required when VECTOR_DB=pinecone")
    return True

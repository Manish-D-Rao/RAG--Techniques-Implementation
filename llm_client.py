from functools import lru_cache
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import config

@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    """
    Build and return a cached ChatOpenAI instance for chat completions.

    Validates that the OpenAI API key is present before constructing the client.
    The result is cached (maxsize=1) so the underlying HTTP client is reused
    across calls instead of being recreated each time.

    Returns:
        A configured ChatOpenAI client using the model, temperature, API key,
        and base URL from config.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set in the environment.
    """
    config.check_openai_api_key()

    return ChatOpenAI(
        model = config.OPENAI_CHAT_MODEL,
        temperature = config.LLM_TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
        base_url=config.NEXUS_BASE_URL

    )

@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAIEmbeddings:
    """
    Build and return a cached OpenAIEmbeddings instance for text embeddings.

    Validates that the OpenAI API key is present before constructing the client.
    The result is cached (maxsize=1) so a single client is reused across calls.

    Returns:
        A configured OpenAIEmbeddings client using the embedding model, API key,
        and base URL from config.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set in the environment.
    """
    config.check_openai_api_key()

    return OpenAIEmbeddings(
        model = config.OPENAI_EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.NEXUS_BASE_URL,
    )
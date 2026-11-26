from functools import lru_cache
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import config

@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    # TODO -> Write the production grade docstring for this function
    
    config.check_openai_api_key()

    return ChatOpenAI(
        model = config.OPENAI_CHAT_MODEL,
        temperature = config.LLM_TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
        base_url=config.NEXUS_BASE_URL 

    )

@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAIEmbeddings:
    # TODO -> Write the production grade docstring for this function
    
    config.check_openai_api_key()

    return OpenAIEmbeddings(
        model = config.OPENAI_EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.NEXUS_BASE_URL,
    )
from functools import lru_cache

import cohere
from google import genai
from qdrant_client import QdrantClient

from config import get_env


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=get_env("GEMINI_API_KEY"))


@lru_cache(maxsize=1)
def get_cohere_client() -> cohere.ClientV2:
    return cohere.ClientV2(api_key=get_env("COHERE_API_KEY"))


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=get_env("QDRANT_URL"),
        api_key=get_env("QDRANT_API_KEY"),
        timeout=30,
    )
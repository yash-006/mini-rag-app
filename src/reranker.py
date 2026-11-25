
from src.clients import get_cohere_client

MODEL_NAME = "rerank-v5.0-fast"


def rerank_chunks(
    query: str,
    retrieved_chunks: list[dict],
    top_n: int = 3
) -> list[dict]:
    """
    Rerank retrieved document chunks using Cohere.

    Returns:
        List of chunks ordered by relevance.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not retrieved_chunks:
        return []

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    # Extract text for reranking
    documents = [
        chunk["text"]
        for chunk in retrieved_chunks
    ]

    # Call Cohere Rerank API
    response = get_cohere_client().rerank(
        model=MODEL_NAME,
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents))
    )

    # Rebuild ranked chunks while preserving metadata
    reranked_chunks = []

    for result in response.results:

        chunk = retrieved_chunks[result.index].copy()

        chunk["rerank_score"] = result.relevance_score

        reranked_chunks.append(chunk)

    return reranked_chunks

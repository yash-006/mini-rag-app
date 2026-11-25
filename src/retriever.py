
from qdrant_client.models import SearchParams

from src.embeddings import generate_embedding
from src.vector_store import client, COLLECTION_NAME


def retrieve_chunks(
    query: str,
    top_k: int = 5
) -> list[dict]:

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Generate embedding for user query
    query_vector = generate_embedding(query)

    # Search Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        search_params=SearchParams(
            exact=False
        )
    ).points

    # Prepare results
    retrieved_chunks = []

    for result in results:

        payload = result.payload or {}

        retrieved_chunks.append({
            "text": payload.get("text", ""),
            "document_name": payload.get("document_name", ""),
            "chunk_index": payload.get("chunk_index", -1),
            "score": result.score
        })

    return retrieved_chunks

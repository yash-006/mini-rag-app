from qdrant_client.models import FieldCondition, Filter, MatchValue, SearchParams

from src.clients import get_qdrant_client
from src.embeddings import generate_embedding
from src.vector_store import COLLECTION_NAME


def retrieve_chunks(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> list[dict]:
    """
    Find the chunks most similar to the query.
    If document_id is given, only that document is searched.
    """

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_vector = generate_embedding(query)

    query_filter = None

    if document_id:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        )

    results = get_qdrant_client().query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        search_params=SearchParams(exact=False),
    ).points

    retrieved_chunks = []

    for result in results:
        payload = result.payload or {}

        retrieved_chunks.append({
            "text": payload.get("text", ""),
            "document_name": payload.get("document_name", ""),
            "document_id": payload.get("document_id", ""),
            "chunk_index": payload.get("chunk_index", -1),
            "score": result.score,
        })

    return retrieved_chunks
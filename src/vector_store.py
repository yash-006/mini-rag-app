import hashlib
import uuid

from qdrant_client import models

from src.clients import get_qdrant_client
from src.embeddings import generate_embeddings

COLLECTION_NAME = "mini_rag_documents"
UPSERT_BATCH_SIZE = 64


def make_document_id(document_name: str) -> str:
    """Same file name always gives the same id."""
    return hashlib.sha256(document_name.encode("utf-8")).hexdigest()[:16]


def make_point_id(document_id: str, chunk_index: int) -> str:
    """Same document + same chunk number always gives the same point id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_index}"))


def ensure_payload_index() -> None:
    """Make filtering fast. Safe to call many times."""
    client = get_qdrant_client()

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="document_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="chunk_index",
        field_schema=models.PayloadSchemaType.INTEGER,
    )


def delete_document(document_id: str) -> None:
    """Remove every stored chunk that belongs to one document."""
    get_qdrant_client().delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
        ),
        wait=True,
    )


def store_chunks(chunks: list[str], document_name: str) -> int:
    """
    Embed chunks and store them in Qdrant.
    Re-uploading the same document name replaces the old version.
    """

    if not chunks:
        raise ValueError("No chunks provided.")

    document_id = make_document_id(document_name)

    # 1. Embed FIRST. If this fails, the old data is still untouched.
    embeddings = generate_embeddings(chunks)

    points = [
        models.PointStruct(
            id=make_point_id(document_id, index),
            vector=embedding,
            payload={
                "text": chunk,
                "document_name": document_name,
                "document_id": document_id,
                "chunk_index": index,
                "total_chunks": len(chunks),
            },
        )
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    # 2. Remove the old version, then save the new one.
    ensure_payload_index()
    delete_document(document_id)

    client = get_qdrant_client()

    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start:start + UPSERT_BATCH_SIZE],
            wait=True,
        )

    return len(points)

def list_documents() -> list[dict]:
    """
    Return one entry per stored document.
    Every document has exactly one chunk number 0, so we only fetch those.
    """
    client = get_qdrant_client()

    documents = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="chunk_index",
                        match=models.MatchValue(value=0),
                    )
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=["document_id", "document_name", "total_chunks"],
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            documents.append({
                "document_id": payload.get("document_id", ""),
                "document_name": payload.get("document_name", ""),
                "total_chunks": payload.get("total_chunks", 0),
            })

        if offset is None:
            break

    return sorted(documents, key=lambda d: d["document_name"].lower())
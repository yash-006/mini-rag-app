import uuid

from qdrant_client import models

from src.clients import get_qdrant_client
from src.embeddings import generate_embedding

COLLECTION_NAME = "mini_rag_documents"


def store_chunks(chunks: list[str], document_name: str):
    """
    Generate embeddings and store text chunks in Qdrant.
    """

    if not chunks:
        raise ValueError("No chunks provided.")

    points = []

    for index, chunk in enumerate(chunks):

        embedding = generate_embedding(chunk)

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk,
                "document_name": document_name,
                "chunk_index": index
            }
        )

        points.append(point)

    get_qdrant_client().upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

    return len(points)

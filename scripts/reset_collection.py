from qdrant_client import models

from src.clients import get_qdrant_client
from src.embeddings import EMBEDDING_DIMENSION
from src.vector_store import COLLECTION_NAME, ensure_payload_index

client = get_qdrant_client()

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted old collection '{COLLECTION_NAME}'.")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=EMBEDDING_DIMENSION,
        distance=models.Distance.COSINE,
    ),
)
ensure_payload_index()

print(f"Created '{COLLECTION_NAME}' (size={EMBEDDING_DIMENSION}, cosine) with a document_id index.")
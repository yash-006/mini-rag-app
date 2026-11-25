
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "mini_rag_documents"
VECTOR_SIZE = 3072

def setup_database():
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("Qdrant credentials are missing from .env")

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30
    )

    try:
        if client.collection_exists(COLLECTION_NAME):
            print(f"Collection '{COLLECTION_NAME}' already exists.")
        else:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection '{COLLECTION_NAME}' created successfully!")

        details = client.get_collection(COLLECTION_NAME)

        print("\nDatabase details:")
        print("Collection:", COLLECTION_NAME)
        print("Vector size:", VECTOR_SIZE)
        print("Distance:", details.config.params.vectors.distance)
        print("Stored points:", details.points_count)

    finally:
        client.close()


if __name__ == "__main__":
    setup_database()


from qdrant_client import QdrantClient

from config import QDRANT_URL, QDRANT_API_KEY

COLLECTION_NAME = "mini_rag_documents"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Checking Qdrant connection...")

collections = client.get_collections()

print("\nConnected successfully!")

collection_names = [
    collection.name
    for collection in collections.collections
]

if COLLECTION_NAME in collection_names:

    print("\nCollection already exists!")

    info = client.get_collection(COLLECTION_NAME)

    print("Collection:", COLLECTION_NAME)
    print("Vector configuration:")
    print(info.config.params.vectors)

else:

    print("\nCollection does not exist yet.")
    print("We will create it in the next step.")

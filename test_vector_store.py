
from src.vector_store import store_chunks

sample_chunks = [
    "Artificial Intelligence enables machines to perform tasks.",
    "Machine Learning is a subset of Artificial Intelligence.",
    "Deep Learning uses neural networks to learn complex patterns."
]

result = store_chunks(
    chunks=sample_chunks,
    document_name="test_document.txt"
)

print("Vectors stored successfully!")
print("Total vectors stored:", result)

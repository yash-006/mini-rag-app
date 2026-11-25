
from src.retriever import retrieve_chunks

query = "What information is available in my document?"

results = retrieve_chunks(
    query=query,
    top_k=5
)

print("\nSearch Query:", query)
print("\nRetrieved Results:", len(results))

for i, result in enumerate(results):

    print(f"\nResult {i + 1}")
    print("Document:", result["document_name"])
    print("Chunk Index:", result["chunk_index"])
    print("Similarity Score:", result["score"])
    print("Text:", result["text"][:300])

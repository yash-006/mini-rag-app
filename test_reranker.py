
from src.retriever import retrieve_chunks
from src.reranker import rerank_chunks


query = "Explain Booth's algorithm."

# Step 1: Retrieve chunks from Qdrant
retrieved_chunks = retrieve_chunks(
    query=query,
    top_k=5
)

print("\nBEFORE RERANKING:")

for index, chunk in enumerate(retrieved_chunks):
    print(f"\nChunk {index + 1}")
    print("Document:", chunk["document_name"])
    print("Chunk Index:", chunk["chunk_index"])
    print("Retrieval Score:", chunk["score"])


# Step 2: Rerank retrieved chunks using Cohere
reranked_chunks = rerank_chunks(
    query=query,
    retrieved_chunks=retrieved_chunks,
    top_n=3
)

print("\nAFTER RERANKING:")

for index, chunk in enumerate(reranked_chunks):
    print(f"\nRank {index + 1}")
    print("Document:", chunk["document_name"])
    print("Chunk Index:", chunk["chunk_index"])
    print("Rerank Score:", chunk["rerank_score"])
    print("Content:", chunk["text"][:300])

from src.retriever import retrieve_chunks
from src.reranker import rerank_chunks
from src.generator import generate_answer

query = "Explain Booth's algorithm."

# Step 1: Retrieve relevant chunks
retrieved_chunks = retrieve_chunks(
    query=query,
    top_k=5
)

# Step 2: Rerank retrieved chunks
reranked_chunks = rerank_chunks(
    query=query,
    retrieved_chunks=retrieved_chunks,
    top_n=3
)

# Step 3: Generate answer using reranked chunks
answer = generate_answer(
    query=query,
    retrieved_chunks=reranked_chunks
)

print("\nUSER QUESTION:")
print(query)

print("\nRERANKED CHUNKS:")
for index, chunk in enumerate(reranked_chunks):
    print(f"\nRank {index + 1}: Chunk {chunk['chunk_index']}")
    print("Rerank Score:", chunk["rerank_score"])

print("\nGENERATED ANSWER:")
print(answer)
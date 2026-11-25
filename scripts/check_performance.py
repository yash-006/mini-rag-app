
import time

from src.retriever import retrieve_chunks
from src.reranker import rerank_chunks
from src.generator import generate_answer

query = "Explain Booth's multiplication algorithm."

print("\nMINI RAG PERFORMANCE TEST")
print("=" * 45)

total_start = time.perf_counter()

# 1. Retrieval performance
start = time.perf_counter()

retrieved_chunks = retrieve_chunks(
    query=query,
    top_k=5
)

retrieval_time = time.perf_counter() - start

print(f"\nRetrieved chunks: {len(retrieved_chunks)}")
print(f"Retrieval time: {retrieval_time:.2f} seconds")

if not retrieved_chunks:
    print("No chunks retrieved. Cannot continue.")
    raise SystemExit(1)

# 2. Reranking performance
start = time.perf_counter()

reranked_chunks = rerank_chunks(
    query=query,
    retrieved_chunks=retrieved_chunks,
    top_n=3
)

reranking_time = time.perf_counter() - start

print(f"Reranked chunks: {len(reranked_chunks)}")
print(f"Reranking time: {reranking_time:.2f} seconds")

# 3. Answer generation performance
start = time.perf_counter()

answer = generate_answer(
    query=query,
    retrieved_chunks=reranked_chunks
)

generation_time = time.perf_counter() - start

print(f"Generation time: {generation_time:.2f} seconds")

# 4. Total pipeline time
total_time = time.perf_counter() - total_start

print("\n" + "=" * 45)
print("PERFORMANCE SUMMARY")
print("=" * 45)

print(f"Retrieval:       {retrieval_time:.2f} seconds")
print(f"Reranking:       {reranking_time:.2f} seconds")
print(f"Generation:      {generation_time:.2f} seconds")
print(f"Total pipeline:  {total_time:.2f} seconds")

print("\nGENERATED ANSWER:")
print(answer)


from src.embeddings import generate_embedding

sample_text = "Artificial Intelligence is transforming technology."

embedding = generate_embedding(sample_text)

print("Embedding generated successfully!")
print("Vector dimensions:", len(embedding))
print("First 10 values:", embedding[:10])

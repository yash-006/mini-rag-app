from src.text_chunker import chunk_text

sample_text = "This is a sample document. " * 200

chunks = chunk_text(sample_text)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}")
    print("Characters:", len(chunk))
    print(chunk[:100])

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Divide extracted text into overlapping chunks.

    Args:
        text: Extracted document text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Shared characters between chunks.

    Returns:
        List of text chunks.
    """

    if not text or not text.strip():
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        # Try to end at a natural text boundary
        if end < text_length:
            boundary = max(
                text.rfind("\n", start, end),
                text.rfind(" ", start, end)
            )

            # Avoid producing very small chunks
            if boundary > start + chunk_size // 2:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks

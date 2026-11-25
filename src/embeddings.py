from google.genai import types

from src.clients import get_gemini_client

MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIMENSION = 3072
BATCH_SIZE = 50


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed a list of texts in ONE request and validate the result."""

    response = get_gemini_client().models.embed_content(
        model=MODEL_NAME,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION,
            task_type=task_type,
        ),
    )

    vectors = [list(item.values) for item in response.embeddings]

    if len(vectors) != len(texts):
        raise ValueError(
            f"Sent {len(texts)} texts but received {len(vectors)} embeddings."
        )

    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION} dimensions, "
                f"received {len(vector)}"
            )

    return vectors


def generate_embedding(
    text: str,
    task_type: str = "RETRIEVAL_QUERY",
) -> list[float]:
    """Embed one text. Defaults to the 'question' type used at search time."""

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    return _embed([text], task_type)[0]


def generate_embeddings(
    text_chunks: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Embed many chunks, BATCH_SIZE at a time."""

    if not text_chunks:
        return []

    embeddings = []

    for start in range(0, len(text_chunks), BATCH_SIZE):
        batch = text_chunks[start:start + BATCH_SIZE]
        embeddings.extend(_embed(batch, task_type))

    return embeddings
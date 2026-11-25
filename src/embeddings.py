
from google import genai
from config import GEMINI_API_KEY


# Initialize Gemini client
client = genai.Client(
    api_key=GEMINI_API_KEY
)

MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIMENSION = 3072


def generate_embedding(text: str) -> list[float]:
    """
    Generate an embedding vector for the given text.
    """

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=text,
        config={
            "output_dimensionality": EMBEDDING_DIMENSION
        }
    )

    embedding = response.embeddings[0].values

    if len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION} dimensions, "
            f"received {len(embedding)}"
        )

    return list(embedding)


def generate_embeddings(text_chunks: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple text chunks.
    """

    embeddings = []

    for chunk in text_chunks:
        embedding = generate_embedding(chunk)
        embeddings.append(embedding)

    return embeddings

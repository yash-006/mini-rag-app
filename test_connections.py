
import os
from dotenv import load_dotenv
from google import genai
import cohere
from qdrant_client import QdrantClient

# Load credentials from the local .env file
load_dotenv()

print("\n========== MINI RAG CONNECTION TESTS ==========\n")


# 1. Test Gemini

def test_gemini():
    print("[1/3] Testing Gemini...")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("FAILED: GEMINI_API_KEY is missing.")
        return

    try:
        client = genai.Client(api_key=api_key)
    except Exception as error:
        print("Could not initialize Gemini client:", error)
        return

    # Test 1: Text generation
    print("\nTesting text generation...")

    generation_models = [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite"
    ]

    generation_success = False

    for model in generation_models:
        try:
            response = client.models.generate_content(
                model=model,
                contents="Reply with only: Gemini connection successful"
            )

            print("SUCCESS using:", model)
            print("Response:", response.text)
            generation_success = True
            break

        except Exception as error:
            print("Model failed:", model)
            print("Reason:", str(error)[:300])

    if not generation_success:
        print("Text generation FAILED for both models.")

    # Test 2: Embeddings (independent of generation)
    print("\nTesting embeddings...")

    try:
        embedding_response = client.models.embed_content(
            model="gemini-embedding-001",
            contents="This is a test sentence for Mini RAG."
        )

        vector = embedding_response.embeddings[0].values

        print("Embedding test: SUCCESS")
        print("Embedding dimensions:", len(vector))

    except Exception as error:
        print("Embedding test: FAILED")
        print("Reason:", str(error))

    if generation_success:
        print("\nGEMINI TEXT GENERATION: SUCCESS")
    else:
        print("\nGEMINI TEXT GENERATION: FAILED")


# 2. Test Qdrant Cloud
def test_qdrant():
    print("[2/3] Testing Qdrant Cloud...")

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        print("FAILED: Qdrant credentials are missing.")
        return

    client = None

    try:
        client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=20
        )

        collections = client.get_collections()

        print("Connection successful!")
        print("Existing collections:")

        if collections.collections:
            for collection in collections.collections:
                print("-", collection.name)
        else:
            print("- No collections yet. That is normal.")

        print("QDRANT: SUCCESS\n")

    except Exception as error:
        print("QDRANT: FAILED")
        print("Error:", str(error), "\n")

    finally:
        if client:
            client.close()


# 3. Test Cohere
def test_cohere():
    print("[3/3] Testing Cohere Rerank...")

    api_key = os.getenv("COHERE_API_KEY")

    if not api_key:
        print("FAILED: COHERE_API_KEY is missing.")
        return

    try:
        client = cohere.ClientV2(api_key=api_key)

        response = client.rerank(
            model="rerank-v5.0-fast",
            query="What is machine learning?",
            documents=[
                "Machine learning is a branch of AI that learns patterns from data.",
                "The capital of France is Paris.",
                "Neural networks are used in deep learning."
            ],
            top_n=2
        )

        print("Reranking successful!")

        for result in response.results:
            print(
                f"Document index: {result.index}, "
                f"Relevance score: {result.relevance_score:.4f}"
            )

        print("COHERE: SUCCESS\n")

    except Exception as error:
        print("COHERE: FAILED")
        print("Error:", str(error), "\n")


if __name__ == "__main__":
    test_gemini()
    test_qdrant()
    test_cohere()

    print("========== TESTS COMPLETED ==========\n")

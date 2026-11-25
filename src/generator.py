
from src.clients import get_gemini_client

MODEL_NAME = "gemini-3.5-flash-lite"

MODEL_NAME = "gemini-3.5-flash-lite"


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:

    if not query or not query.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        return "I could not find relevant information in the uploaded documents."

    # Build context from retrieved chunks
    context = ""

    for index, chunk in enumerate(retrieved_chunks):

        context += f"""
        Source {index + 1}:
        Document: {chunk['document_name']}
        Chunk: {chunk['chunk_index']}

        Content:
        {chunk['text']}
        """

    prompt = f"""
    You are a helpful AI assistant that answers questions
    using the provided document context.

    Instructions:
    1. Answer using only the provided context.
    2. Do not invent information.
    3. If the answer is not available in the context,
       clearly say that.
    4. Give clear and understandable explanations.
    5. Use bullet points where appropriate.

    DOCUMENT CONTEXT:
    {context}

    USER QUESTION:
    {query}

    ANSWER:
    """

    response = get_gemini_client().models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text

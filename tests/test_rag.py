
from io import BytesIO

import pytest

from src.document_loader import extract_text
from src.text_chunker import chunk_text


class MockUploadedFile:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


# ---------------- TEXT CHUNKING TESTS ----------------

def test_chunk_text_returns_empty_for_empty_input():
    assert chunk_text("") == []


def test_chunk_text_returns_single_chunk_for_short_text():
    text = "This is a short document."
    chunks = chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_splits_long_text():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("Some text", chunk_size=100, chunk_overlap=100)


# ---------------- DOCUMENT LOADER TESTS ----------------

def test_extract_text_from_txt():
    content = b"Mini RAG application test document."
    uploaded_file = MockUploadedFile("sample.txt", content)

    extracted_text = extract_text(uploaded_file)

    assert extracted_text == "Mini RAG application test document."


def test_extract_text_rejects_empty_txt():
    uploaded_file = MockUploadedFile("empty.txt", b"")

    with pytest.raises(ValueError):
        extract_text(uploaded_file)


def test_extract_text_rejects_unsupported_file():
    uploaded_file = MockUploadedFile("sample.docx", b"Some content")

    with pytest.raises(ValueError):
        extract_text(uploaded_file)


from unittest.mock import patch

from src.retriever import retrieve_chunks
from src.reranker import rerank_chunks


# ---------------- RETRIEVAL TESTS ----------------

def test_retrieve_chunks_rejects_empty_query():
    with pytest.raises(ValueError):
        retrieve_chunks("")


def test_retrieve_chunks_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        retrieve_chunks("Explain Booth's algorithm", top_k=0)


# ---------------- RERANKING TESTS ----------------

def test_rerank_chunks_returns_empty_for_empty_input():
    assert rerank_chunks("Explain Booth's algorithm", []) == []


def test_rerank_chunks_rejects_empty_query():
    with pytest.raises(ValueError):
        rerank_chunks("", [{"text": "Sample content"}])


def test_rerank_chunks_rejects_invalid_top_n():
    with pytest.raises(ValueError):
        rerank_chunks(
            "Explain Booth's algorithm",
            [{"text": "Sample content"}],
            top_n=0
        )


def test_rerank_chunks_orders_results_and_preserves_metadata():
    chunks = [
        {
            "text": "Content about addressing modes",
            "document_name": "coa.pdf",
            "chunk_index": 1,
            "score": 0.8
        },
        {
            "text": "Content about Booth's algorithm",
            "document_name": "coa.pdf",
            "chunk_index": 2,
            "score": 0.7
        }
    ]

    mock_response = type(
        "MockResponse",
        (),
        {
            "results": [
                type("MockResult", (), {
                    "index": 1,
                    "relevance_score": 0.95
                })(),
                type("MockResult", (), {
                    "index": 0,
                    "relevance_score": 0.25
                })()
            ]
        }
    )()

    with patch("src.reranker.get_cohere_client") as mock_get_client:
        mock_get_client.return_value.rerank.return_value = mock_response
        results = rerank_chunks(
            "Explain Booth's algorithm",
            chunks,
            top_n=2
        )

    assert len(results) == 2
    assert results[0]["chunk_index"] == 2
    assert results[0]["rerank_score"] == 0.95
    assert results[1]["chunk_index"] == 1
    assert results[0]["document_name"] == "coa.pdf"


from src.generator import generate_answer


# ---------------- ANSWER GENERATION TESTS ----------------

def test_generate_answer_rejects_empty_query():
    with pytest.raises(ValueError):
        generate_answer("", [])


def test_generate_answer_returns_fallback_for_empty_chunks():
    answer = generate_answer(
        "Explain Booth's algorithm",
        []
    )

    assert "could not find relevant information" in answer.lower()


def test_generate_answer_uses_gemini_response():
    mock_response = type(
        "MockResponse",
        (),
        {"text": "Booth's algorithm is used for binary multiplication."}
    )()

    chunks = [
        {
            "text": "Booth's algorithm is used for signed binary multiplication.",
            "document_name": "coa.pdf",
            "chunk_index": 3
        }
    ]

    with patch("src.generator.get_gemini_client") as mock_get_client:
        mock_generate = mock_get_client.return_value.models.generate_content
        mock_generate.return_value = mock_response

        answer = generate_answer(
            "Explain Booth's algorithm",
            chunks
        )

    assert answer == "Booth's algorithm is used for binary multiplication."
    mock_generate.assert_called_once()

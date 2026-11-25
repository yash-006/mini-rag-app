
from io import BytesIO

import pytest

from src.document_loader import extract_text
from src.text_chunker import chunk_text
from unittest.mock import MagicMock
from src.vector_store import store_chunks, UPSERT_BATCH_SIZE
from types import SimpleNamespace
from src.vector_store import list_documents
from src.generator import extract_cited_numbers, NO_ANSWER_MESSAGE
from src.generator import generate_answer_with_usage
from src.metrics import estimate_cost
from src.eval_utils import (
    contains_any,
    first_relevant_rank,
    is_no_answer,
    keywords_present,
)

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

# ---------------- VECTOR STORE TESTS ----------------

def test_store_chunks_uses_stable_ids_and_replaces_old_version():
    fake_client = MagicMock()

    with patch("src.vector_store.get_qdrant_client", return_value=fake_client), \
         patch("src.vector_store.generate_embeddings",
               return_value=[[0.1, 0.2], [0.3, 0.4]]):

        first_count = store_chunks(["a", "b"], "doc.txt")
        first_points = fake_client.upsert.call_args.kwargs["points"]

        store_chunks(["a", "b"], "doc.txt")
        second_points = fake_client.upsert.call_args.kwargs["points"]

    assert first_count == 2
    assert [p.id for p in first_points] == [p.id for p in second_points]
    assert fake_client.delete.call_count == 2
    assert first_points[0].payload["document_name"] == "doc.txt"
    assert first_points[0].payload["document_id"]


def test_store_chunks_upserts_in_batches():
    fake_client = MagicMock()
    total = UPSERT_BATCH_SIZE * 2 + 1

    with patch("src.vector_store.get_qdrant_client", return_value=fake_client), \
         patch("src.vector_store.generate_embeddings",
               return_value=[[0.1, 0.2]] * total):

        store_chunks(["x"] * total, "big.txt")

    assert fake_client.upsert.call_count == 3

# ---------------- DOCUMENT SCOPE TESTS ----------------

def test_retrieve_chunks_filters_by_document_id():
    fake_client = MagicMock()
    fake_client.query_points.return_value.points = []

    with patch("src.retriever.get_qdrant_client", return_value=fake_client), \
         patch("src.retriever.generate_embedding", return_value=[0.1, 0.2]):

        retrieve_chunks("hello", document_id="abc123")

    query_filter = fake_client.query_points.call_args.kwargs["query_filter"]

    assert query_filter.must[0].key == "document_id"
    assert query_filter.must[0].match.value == "abc123"


def test_list_documents_returns_sorted_entries():
    fake_client = MagicMock()
    fake_client.scroll.return_value = (
        [
            SimpleNamespace(payload={
                "document_id": "b", "document_name": "b.pdf", "total_chunks": 3
            }),
            SimpleNamespace(payload={
                "document_id": "a", "document_name": "a.pdf", "total_chunks": 5
            }),
        ],
        None,
    )

    with patch("src.vector_store.get_qdrant_client", return_value=fake_client):
        documents = list_documents()

    assert [d["document_name"] for d in documents] == ["a.pdf", "b.pdf"]


# ---------------- CITATION TESTS ----------------

def test_extract_cited_numbers_finds_valid_citations_only():
    answer = "Booth's algorithm multiplies signed numbers [1]. It shifts [1, 3] and [9]."

    assert extract_cited_numbers(answer, 3) == [1, 3]


def test_rerank_chunks_drops_low_scores():
    chunks = [{"text": "a"}, {"text": "b"}]

    mock_response = SimpleNamespace(results=[
        SimpleNamespace(index=0, relevance_score=0.9),
        SimpleNamespace(index=1, relevance_score=0.01),
    ])

    with patch("src.reranker.get_cohere_client") as mock_get_client:
        mock_get_client.return_value.rerank.return_value = mock_response
        results = rerank_chunks("question", chunks, top_n=2, min_score=0.1)

    assert len(results) == 1
    assert results[0]["text"] == "a"


def test_generate_answer_numbers_sources_in_prompt():
    mock_response = SimpleNamespace(text="Answer [1].")

    chunks = [
        {"text": "First text", "document_name": "a.pdf", "chunk_index": 0},
        {"text": "Second text", "document_name": "a.pdf", "chunk_index": 1},
    ]

    with patch("src.generator.get_gemini_client") as mock_get_client:
        mock_generate = mock_get_client.return_value.models.generate_content
        mock_generate.return_value = mock_response
        generate_answer("question", chunks)

    prompt = mock_generate.call_args.kwargs["contents"]

    assert 'id="1"' in prompt
    assert 'id="2"' in prompt
    assert "First text" in prompt
    assert NO_ANSWER_MESSAGE in prompt

# ---------------- USAGE AND COST TESTS ----------------

def test_generate_answer_with_usage_reads_token_counts():
    mock_response = SimpleNamespace(
        text="Answer [1].",
        usage_metadata=SimpleNamespace(
            prompt_token_count=120,
            candidates_token_count=30,
        ),
    )

    chunks = [{"text": "Some text", "document_name": "a.pdf", "chunk_index": 0}]

    with patch("src.generator.get_gemini_client") as mock_get_client:
        mock_get_client.return_value.models.generate_content.return_value = mock_response
        answer, usage = generate_answer_with_usage("question", chunks)

    assert answer == "Answer [1]."
    assert usage == {"prompt_tokens": 120, "output_tokens": 30}


def test_estimate_cost_scales_with_tokens():
    assert estimate_cost(0, 0, rerank_searches=0) == 0
    assert estimate_cost(1_000_000, 0, rerank_searches=0) > 0
    assert estimate_cost(2000, 500) > estimate_cost(1000, 250)


# ---------------- EVALUATION HELPER TESTS ----------------

def test_contains_any_ignores_case():
    assert contains_any("Uses an Index Register here", ["index register"])
    assert not contains_any("nothing relevant", ["index register"])


def test_first_relevant_rank_returns_position_or_none():
    chunks = [{"text": "alpha"}, {"text": "has the EVIDENCE"}, {"text": "beta"}]

    assert first_relevant_rank(chunks, ["evidence"]) == 2
    assert first_relevant_rank(chunks, ["missing"]) is None


def test_keywords_present_requires_all():
    assert keywords_present("The accumulator and multiplier", ["accumulator", "multiplier"])
    assert not keywords_present("Only the accumulator", ["accumulator", "multiplier"])


def test_is_no_answer_detects_polite_refusal():
    assert is_no_answer("I could not find relevant information in the document.")
    assert not is_no_answer("Booth's algorithm multiplies numbers [1].")
import argparse
import json
import time
from pathlib import Path

from src.eval_utils import (
    first_relevant_rank,
    is_no_answer,
    keywords_present,
)
from src.generator import extract_cited_numbers, generate_answer_with_usage
from src.metrics import estimate_cost
from src.reranker import MIN_RELEVANCE_SCORE, rerank_chunks
from src.retriever import retrieve_chunks
from src.text_chunker import chunk_text
from src.vector_store import delete_document, make_document_id, store_chunks

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evaluation"

# Keep these the same as in app.py
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
TOP_N = 3


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def evaluate_question(item: dict, document_id: str) -> dict:
    question = item["question"]
    evidence = item.get("evidence_phrases", [])
    expect_no_answer = item.get("expect_no_answer", False)

    started = time.perf_counter()

    retrieved = retrieve_chunks(
        query=question, top_k=TOP_K, document_id=document_id
    )
    reranked = rerank_chunks(
        query=question,
        retrieved_chunks=retrieved,
        top_n=TOP_N,
        min_score=MIN_RELEVANCE_SCORE,
    )
    answer, usage = generate_answer_with_usage(question, reranked)

    seconds = time.perf_counter() - started

    result = {
        "question": question,
        "expect_no_answer": expect_no_answer,
        "answer": answer,
        "seconds": round(seconds, 2),
        "chunks_kept": len(reranked),
        "cost_usd": round(
            estimate_cost(usage["prompt_tokens"], usage["output_tokens"]), 6
        ),
    }

    if expect_no_answer:
        result["passed"] = is_no_answer(answer)
        return result

    rerank_rank = first_relevant_rank(reranked, evidence)
    cited = extract_cited_numbers(answer, len(reranked))
    keywords_ok = keywords_present(answer, item.get("expected_keywords", []))

    result.update({
        "retrieval_hit": first_relevant_rank(retrieved, evidence) is not None,
        "rerank_hit": rerank_rank is not None,
        "rerank_rank": rerank_rank,
        "has_citation": bool(cited),
        "keywords_ok": keywords_ok,
        "passed": keywords_ok and bool(cited) and not is_no_answer(answer),
    })

    return result


def summarize(results: list[dict]) -> dict:
    answerable = [r for r in results if not r["expect_no_answer"]]
    unanswerable = [r for r in results if r["expect_no_answer"]]

    total = len(answerable)

    reciprocal_ranks = [
        1 / r["rerank_rank"] for r in answerable if r.get("rerank_rank")
    ]

    return {
        "answerable_questions": total,
        "unanswerable_questions": len(unanswerable),
        "retrieval_hit_rate": rate(
            sum(1 for r in answerable if r.get("retrieval_hit")), total
        ),
        "rerank_hit_rate": rate(
            sum(1 for r in answerable if r.get("rerank_hit")), total
        ),
        "rerank_mrr": sum(reciprocal_ranks) / total if total else 0.0,
        "citation_rate": rate(
            sum(1 for r in answerable if r.get("has_citation")), total
        ),
        "answer_success_rate": rate(
            sum(1 for r in answerable if r["passed"]), total
        ),
        "no_answer_accuracy": rate(
            sum(1 for r in unanswerable if r["passed"]), len(unanswerable)
        ),
        "avg_seconds": (
            sum(r["seconds"] for r in results) / len(results) if results else 0.0
        ),
        "total_cost_usd": sum(r.get("cost_usd", 0) for r in results),
    }


def percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def write_markdown(path: Path, results: list[dict], summary: dict, document: str) -> None:
    lines = [
        "# Evaluation results",
        "",
        f"Document: `{document}` | chunk size {CHUNK_SIZE}, overlap {CHUNK_OVERLAP} | "
        f"retrieve top-{TOP_K}, rerank keep top-{TOP_N}, min relevance {MIN_RELEVANCE_SCORE}",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Retrieval hit rate (top-{TOP_K}) | {percent(summary['retrieval_hit_rate'])} |",
        f"| Rerank hit rate (top-{TOP_N}) | {percent(summary['rerank_hit_rate'])} |",
        f"| Rerank MRR | {summary['rerank_mrr']:.2f} |",
        f"| Answers with a citation | {percent(summary['citation_rate'])} |",
        f"| Answer success rate | {percent(summary['answer_success_rate'])} |",
        f"| No-answer accuracy | {percent(summary['no_answer_accuracy'])} |",
        f"| Average time per question | {summary['avg_seconds']:.1f}s |",
        f"| Estimated total cost | ${summary['total_cost_usd']:.4f} |",
        "",
        "| # | Question | Result |",
        "|---|---|---|",
    ]

    for number, result in enumerate(results, start=1):
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(f"| {number} | {result['question']} | {status} |")

    lines += [
        "",
        "## How to read this",
        "",
        "- **Hit rate** is a recall-style measure: for each question we know a phrase "
        "that must appear in the chunk holding the answer, and we check whether a "
        "retrieved (or reranked) chunk contains it. Precision is not reported "
        "because each question has only one evidence chunk.",
        "- **Answer success** means the answer contains every expected keyword and "
        "at least one `[n]` citation.",
        "- **No-answer accuracy** checks that a question the document cannot answer "
        "gets the polite refusal.",
        "",
        "Limitations: the gold set is tiny (5 answerable + 1 unanswerable), judging "
        "is keyword based, and the sample document is short, so these numbers show "
        "the pipeline works but are not a benchmark.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Mini RAG evaluation.")
    parser.add_argument("--document", default=str(EVAL_DIR / "sample_document.txt"))
    parser.add_argument("--gold", default=str(EVAL_DIR / "questions.json"))
    parser.add_argument(
        "--delay", type=float, default=4.0,
        help="seconds to wait between questions (avoids rate limits)",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="delete the evaluation document from Qdrant afterwards",
    )
    args = parser.parse_args()

    document_path = Path(args.document)
    document_name = document_path.name
    document_id = make_document_id(document_name)

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    text = document_path.read_text(encoding="utf-8")

    print(f"\nIngesting {document_name} ...")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    store_chunks(chunks, document_name)
    print(f"Stored {len(chunks)} chunks.\n")

    results = []

    for index, item in enumerate(gold):

        if index:
            time.sleep(args.delay)

        try:
            result = evaluate_question(item, document_id)
        except Exception as error:
            result = {
                "question": item["question"],
                "expect_no_answer": item.get("expect_no_answer", False),
                "passed": False,
                "error": str(error),
            }

        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {item['question']}")

        if "error" in result:
            print("       error:", result["error"][:200])

        results.append(result)

    summary = summarize(results)

    (EVAL_DIR / "results.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2),
        encoding="utf-8",
    )
    write_markdown(EVAL_DIR / "results.md", results, summary, document_name)

    print("\n" + "=" * 45)
    print("SUMMARY")
    print("=" * 45)
    print(f"Retrieval hit rate : {percent(summary['retrieval_hit_rate'])}")
    print(f"Rerank hit rate    : {percent(summary['rerank_hit_rate'])}")
    print(f"Rerank MRR         : {summary['rerank_mrr']:.2f}")
    print(f"Answer success     : {percent(summary['answer_success_rate'])}")
    print(f"No-answer accuracy : {percent(summary['no_answer_accuracy'])}")
    print(f"Avg time/question  : {summary['avg_seconds']:.1f}s")
    print("\nSaved evaluation/results.json and evaluation/results.md")

    if args.cleanup:
        delete_document(document_id)
        print("Removed the evaluation document from Qdrant.")


if __name__ == "__main__":
    main()
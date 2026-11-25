# Evaluation results

Document: `sample_document.txt` | chunk size 1000, overlap 200 | retrieve top-5, rerank keep top-3, min relevance 0.1

| Metric | Result |
|---|---|
| Retrieval hit rate (top-5) | 100% |
| Rerank hit rate (top-3) | 100% |
| Rerank MRR | 0.90 |
| Answers with a citation | 100% |
| Answer success rate | 100% |
| No-answer accuracy | 100% |
| Average time per question | 3.8s |
| Estimated total cost | $0.0128 |

| # | Question | Result |
|---|---|---|
| 1 | Explain Booth's multiplication algorithm. | PASS |
| 2 | What are the main registers used in Booth's algorithm? | PASS |
| 3 | What is arithmetic right shift? | PASS |
| 4 | Explain restoring division. | PASS |
| 5 | How does indexed addressing mode work? | PASS |
| 6 | How does the LRU page replacement policy work? | PASS |

## How to read this

- **Hit rate** is a recall-style measure: for each question we know a phrase that must appear in the chunk holding the answer, and we check whether a retrieved (or reranked) chunk contains it. Precision is not reported because each question has only one evidence chunk.
- **Answer success** means the answer contains every expected keyword and at least one `[n]` citation.
- **No-answer accuracy** checks that a question the document cannot answer gets the polite refusal.

Limitations: the gold set is tiny (5 answerable + 1 unanswerable), judging is keyword based, and the sample document is short, so these numbers show the pipeline works but are not a benchmark.
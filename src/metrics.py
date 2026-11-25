# Rough prices in US dollars. These are PLACEHOLDERS: check your provider's
# pricing pages and update the numbers. The app labels costs as estimates.
LLM_INPUT_PRICE_PER_MILLION = 0.10
LLM_OUTPUT_PRICE_PER_MILLION = 0.40
RERANK_PRICE_PER_SEARCH = 0.002


def estimate_cost(
    prompt_tokens: int,
    output_tokens: int,
    rerank_searches: int = 1,
) -> float:
    """Rough cost in dollars of one question."""

    llm_cost = (
        prompt_tokens * LLM_INPUT_PRICE_PER_MILLION
        + output_tokens * LLM_OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000

    return llm_cost + rerank_searches * RERANK_PRICE_PER_SEARCH
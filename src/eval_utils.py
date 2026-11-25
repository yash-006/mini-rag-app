def contains_any(text: str, phrases: list[str]) -> bool:
    """True if the text contains at least one phrase (ignores upper/lower case)."""

    lowered = text.lower()

    return any(phrase.lower() in lowered for phrase in phrases)


def first_relevant_rank(chunks: list[dict], phrases: list[str]) -> int | None:
    """Position (starting at 1) of the first chunk with evidence, or None."""

    for position, chunk in enumerate(chunks, start=1):
        if contains_any(chunk.get("text", ""), phrases):
            return position

    return None


def keywords_present(answer: str, keywords: list[str]) -> bool:
    """True only if EVERY keyword appears in the answer."""

    lowered = answer.lower()

    return all(keyword.lower() in lowered for keyword in keywords)


def is_no_answer(answer: str) -> bool:
    """True if the answer is the polite 'could not find' reply."""

    return "could not find relevant information" in answer.lower()
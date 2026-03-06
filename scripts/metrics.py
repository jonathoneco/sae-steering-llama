"""Auxiliary (non-LLM) evaluation metrics."""

import re
from collections import Counter


def trigram_repetition(text: str) -> float:
    """Fraction of 3-grams that are repeated. >0.2 indicates problematic repetition."""
    words = text.lower().split()
    if len(words) < 3:
        return 0.0
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    counts = Counter(trigrams)
    total = len(trigrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / total if total > 0 else 0.0


def explicit_concept(text: str, keyword: str = "eiffel") -> bool:
    """Case-insensitive check for keyword in text."""
    return keyword.lower() in text.lower()


def surprise(steered_logprobs: list[float], reference_logprobs: list[float]) -> float:
    """Mean difference in negative log probability (steered vs reference).

    Higher values = steered output is more surprising to the reference model.
    Both inputs should be lists of log probabilities (negative values).
    """
    if not steered_logprobs:
        return 0.0
    # Surprise = mean(-logp_ref) for tokens in the steered output
    # i.e., how surprising is the steered output under the reference model
    return -sum(reference_logprobs) / len(reference_logprobs)

"""Scorer protocol: the interface for external scoring systems.

The domain layer defines the contract; infrastructure implements it.
This enables testing without GPU or transformers.
"""
from __future__ import annotations

from typing import Protocol


class Scorer(Protocol):
    """Protocol for scoring BBQ items.

    A scorer receives a prompt (formatted text) and a list of options,
    and returns a score for each option. The score can be interpreted
    as log-likelihood, logit, or any comparable numeric value.

    This is a Protocol (structural subtyping), so any object with a
    score() method matching this signature can be used as a Scorer.
    """

    def score(self, prompt: str, options: list[str]) -> list[float]:
        """Score a set of options given a prompt.

        Args:
            prompt: The formatted prompt text (context + question + options header)
            options: The three answer options as strings

        Returns:
            A list of three float scores, one per option, comparable
            (higher = more likely). The scores need not be normalized.

        Raises:
            ValueError: If options length != 3 or if scoring fails
        """
        ...

"""Pytest fixtures for BBQ gate tests."""
import pytest

from bbq_gate.domain.scorer import Scorer


class MockScorer:
    """Deterministic mock scorer for testing without GPU.

    Returns fixed scores based on the prompt text.
    """

    def score(self, prompt: str, options: list[str]) -> list[float]:
        """Return deterministic mock scores.

        Args:
            prompt: The prompt text
            options: Three option strings

        Returns:
            Three scores (not normalized)
        """
        if len(options) != 3:
            raise ValueError(f"Expected 3 options, got {len(options)}")

        # Simple deterministic scoring: based on hash of option text
        scores = []
        for opt in options:
            # Hash the option text to get a "score"
            h = hash(opt) % 100
            scores.append(float(h))

        return scores


@pytest.fixture
def mock_scorer() -> MockScorer:
    """Provide a mock scorer that doesn't require GPU."""
    return MockScorer()


class MockTokenizer:
    """Mock tokenizer for testing without transformers."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        """Convert messages to a prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"[{role}]: {content}")
        return "\n".join(prompt_parts)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        return [ord(c) for c in text[:10]]  # Simplistic


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    """Provide a mock tokenizer that doesn't require transformers."""
    return MockTokenizer()

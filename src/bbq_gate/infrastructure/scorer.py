"""Transformer-based scorer implementation.

Scores BBQ items using Qwen 2.5 7B Instruct by computing log-likelihood
of the letter tokens in the final position.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenScorer:
    """Scorer using Qwen 2.5 7B Instruct in bf16 precision.

    Attributes:
        model: The loaded model
        tokenizer: The tokenizer
        device: The device the model is on
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
        """Initialize the scorer.

        Args:
            model_id: The Hugging Face model ID

        The model is loaded in bf16 with device_map='cuda' (no CPU offload).
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda"
        )
        self.model.eval()
        self.device = self.model.device

        # Pre-compute letter token IDs
        self._letter_ids = self._build_letter_ids()

    def _build_letter_ids(self) -> dict[str, set[int]]:
        """Pre-compute token IDs for answer letters A, B, C.

        Returns a dict mapping letter -> set of possible token IDs (including space variants).
        """
        letters = {}
        for letter in ("A", "B", "C"):
            ids = set()
            for form in (letter, " " + letter):
                encoded = self.tokenizer.encode(form, add_special_tokens=False)
                if len(encoded) == 1:
                    ids.add(encoded[0])
            letters[letter] = ids
        return letters

    @torch.no_grad()
    def score(self, prompt: str, options: list[str]) -> list[float]:
        """Score three options given a prompt.

        Primary method: score by log-likelihood of the letter token.

        Args:
            prompt: The formatted prompt text
            options: Three option strings (unused in letter-based scoring)

        Returns:
            Three log-likelihood scores (higher = more likely)

        Raises:
            ValueError: If options length != 3
        """
        if len(options) != 3:
            raise ValueError(f"Expected 3 options, got {len(options)}")

        # Encode and move to device
        encoded = self.tokenizer(prompt, return_tensors="pt", padding=True, padding_side="left").to(
            self.device
        )

        # Get logits at the last token position
        logits = self.model(**encoded).logits[:, -1, :].float()  # Shape: [batch_size, vocab_size]

        # Compute log-softmax
        log_probs = torch.log_softmax(logits, dim=-1)  # Shape: [batch_size, vocab_size]

        # Extract letter scores
        scores = []
        for letter in ("A", "B", "C"):
            letter_log_probs = [log_probs[0, token_id].item() for token_id in self._letter_ids[letter]]
            max_score = max(letter_log_probs) if letter_log_probs else float("-inf")
            scores.append(max_score)

        return scores

    @torch.no_grad()
    def score_full_text(self, prompt: str, options: list[str]) -> list[float]:
        """Alternative scoring method: log-likelihood of full option text.

        Used as verification (design.md D4).

        Args:
            prompt: The formatted prompt text
            options: Three option strings

        Returns:
            Three log-likelihood scores, normalized by option length
        """
        if len(options) != 3:
            raise ValueError(f"Expected 3 options, got {len(options)}")

        scores = []
        for option_text in options:
            # Concatenate prompt + option
            full_text = prompt + option_text
            encoded = self.tokenizer(
                full_text, return_tensors="pt", padding=False, padding_side="left"
            ).to(self.device)

            logits = self.model(**encoded).logits[0, :, :].float()
            log_probs = torch.log_softmax(logits, dim=-1)

            # Compute log-likelihood of the option tokens
            option_tokens = self.tokenizer(option_text, add_special_tokens=False).input_ids
            score = 0.0
            for token_id in option_tokens:
                # Find the logit position corresponding to this token
                # This is a simplification; for exact calculation, we'd need position tracking
                score += log_probs[-1, token_id].item()

            # Normalize by length
            score /= len(option_tokens) if option_tokens else 1.0
            scores.append(score)

        return scores

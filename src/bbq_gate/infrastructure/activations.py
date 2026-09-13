"""Real activation extractor: Qwen 2.5 7B Instruct, bf16, inference-only.

Implements `domain.activation_extractor.ActivationExtractor` (design.md D5).
Not unit-tested directly (same convention as `infrastructure/scorer.py`'s
`QwenScorer`: loading the 7B model is exercised by the real pipeline script,
not by `pytest tests/`). Mock this at the `ActivationExtractor` boundary in
any test that needs one (docs/testing.md §2-5: mock at the frontier of the
external provider).
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np


class QwenActivationExtractor:
    """Extracts all hidden_states at the last token position, in bf16 inference.

    Attributes:
        model_id: The Hugging Face model ID.
        revision: The specific model revision/commit loaded, for
            reproducibility (spec: "Trazabilidad de la salida").
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
        """Load the model in bf16 with no CPU offload.

        Raises:
            RuntimeError: if CUDA is not available (task 3.2: "verificar que
            torch.cuda.is_available()... y que no hay offload a CPU";
            DEC-002: single CUDA device, bf16).
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available: QwenActivationExtractor requires a GPU (DEC-002); "
                "refusing to fall back to CPU offload."
            )

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda", output_hidden_states=True
        )
        self.model.eval()
        self.device = self.model.device
        self.revision = getattr(self.model.config, "_commit_hash", None) or "unknown"

    @torch.no_grad()
    def extract(self, text: str) -> list[np.ndarray]:
        """Forward pass under `torch.no_grad()`; no weights are modified and
        no checkpoint is written (spec: "Integridad del modelo").

        Returns:
            One fp16 numpy vector per layer (embeddings + all transformer
            layers), at the LAST token position of `text`.
        """
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**encoded, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple, len = n_layers + 1
        return [
            hs[0, -1, :].detach().float().cpu().numpy().astype(np.float16) for hs in hidden_states
        ]

    @torch.no_grad()
    def extract_layer0_mean_pooled(self, text: str) -> np.ndarray:
        """Mean-pooled embedding-layer vector, over ALL tokens of `text`.

        design.md D2 Addendum 6 / spec "Evaluación relativa a baselines
        declarados": under `add_generation_prompt=True`, the LAST token of a
        'chat'-format prompt is a fixed turn marker identical across every
        item, so `extract(...)[0]` (layer 0 at the last token) is a constant
        vector by construction and cannot serve as the embeddings baseline.
        Mean-pooling over every token position fixes this.

        This looks up the embedding table directly (`get_input_embeddings`)
        instead of running a forward pass, which is both cheaper and a
        stronger guarantee of the spec's "no ha atravesado ningún bloque del
        modelo": no transformer block runs at all.

        Returns:
            One fp16 vector (embedding dimension), the mean over the token
            axis. Promoted to float32 before the reduction (Addendum 6's
            numeric note: large-magnitude fp16 values overflow on
            accumulation).
        """
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        embeddings = self.model.get_input_embeddings()(encoded["input_ids"])  # (1, seq, dim)
        pooled = embeddings[0].float().mean(dim=0)  # promote before reducing over tokens
        return pooled.detach().cpu().numpy().astype(np.float16)

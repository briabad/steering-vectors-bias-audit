"""Protocol for the activation extractor (design.md D5).

The domain layer defines the contract; infrastructure implements it with
`transformers`. This lets the contrast/direction algebra be tested with
synthetic activations, without loading 15 GiB of weights (task 3.1: "verificar
que domain/ no importa transformers en ningún módulo").
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class ActivationExtractor(Protocol):
    """Extracts per-layer residual-stream activations at the last token."""

    def extract(self, text: str) -> list[np.ndarray]:
        """Extract one activation vector per layer for `text`.

        Args:
            text: The formatted prompt text.

        Returns:
            One vector per layer, INCLUDING the embedding layer (spec:
            "Extracción completa": "se obtiene un vector por cada capa,
            incluida la de embeddings"), each corresponding to the position
            of the LAST token of `text`.
        """
        ...

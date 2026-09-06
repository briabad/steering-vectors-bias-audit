"""Persistence for extracted activations (task 3.3, spec "Trazabilidad de la
salida" and "Reproducibilidad").

Stores a 3D array (n_items, n_layers, dim) in fp16, keyed by a list of string
item identifiers, alongside the five required metadata fields: model,
revision, dtype, position_policy, seed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_METADATA_FIELDS = ("model", "revision", "dtype", "position_policy", "seed")


def save_activations(
    path: Path,
    keys: list[str],
    activations: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Persist activations with their metadata.

    Args:
        path: Output `.npz` path.
        keys: One string identifier per item, same order as `activations`
            axis 0.
        activations: Array of shape (n_items, n_layers, dim).
        metadata: Must include all of `REQUIRED_METADATA_FIELDS`.

    Raises:
        ValueError: if `metadata` is missing a required field, or `keys`
            length does not match `activations.shape[0]`.
    """
    missing = [f for f in REQUIRED_METADATA_FIELDS if f not in metadata]
    if missing:
        raise ValueError(f"metadata missing required fields: {missing}")
    if len(keys) != activations.shape[0]:
        raise ValueError(
            f"keys length ({len(keys)}) must match activations.shape[0] ({activations.shape[0]})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        activations=activations.astype(np.float16),
        keys=np.array(keys, dtype=object),
        metadata=json.dumps(metadata),
    )


def load_activations(path: Path) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    """Load activations and metadata persisted by `save_activations`.

    Returns:
        (keys, activations, metadata).
    """
    data = np.load(path, allow_pickle=True)
    keys = list(data["keys"])
    activations = data["activations"]
    metadata = json.loads(str(data["metadata"]))
    return keys, activations, metadata

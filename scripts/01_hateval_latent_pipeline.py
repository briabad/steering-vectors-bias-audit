#!/usr/bin/env python3
"""Primer pipeline experimental para latente-flow + steering en Hateval.

Objetivo:
- cargar Hateval desde Hugging Face,
- construir un subset contrastivo y balanceado,
- extraer activaciones desde Qwen/Qwen2.5-7B-Instruct,
- calcular vector de diferencia de medias, métricas geométricas y pruebas de intervención,
- guardar artefactos bajo data/experiments/EXP-001_hateval_vector_flow/output.

Este script es la base para un primer estudio de prueba de concepto; no modifica
los pesos del modelo y se diseña para ser ejecutado desde WSL.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from datasets import load_dataset


DATASET_NAME = "hs-knowledge/hateval_enriched"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "data" / "experiments" / "EXP-001_hateval_vector_flow"
OUTPUT_DIR = EXPERIMENT_DIR / "output"


@dataclass
class SampleRecord:
    text: str
    label: int
    split: str
    source: str


def find_text_column(columns: Sequence[str]) -> str:
    candidates = [
        c for c in columns if c.lower() in {"text", "tweet", "sentence", "comment", "content", "example"}
    ]
    if candidates:
        return candidates[0]
    for c in columns:
        if "text" in c.lower() or "tweet" in c.lower() or "content" in c.lower() or "sentence" in c.lower():
            return c
    raise ValueError(f"No text column found in dataset columns: {columns}")


def find_label_column(columns: Sequence[str]) -> str:
    candidates = [
        c for c in columns if c.lower() in {"label", "hate", "hs", "is_hate", "target"}
    ]
    if candidates:
        return candidates[0]
    for c in columns:
        if "label" in c.lower() or "hate" in c.lower() or "target" in c.lower():
            return c
    raise ValueError(f"No label column found in dataset columns: {columns}")


def load_hateval_subset(limit: int | None = None) -> List[SampleRecord]:
    """Carga Hateval usando la referencia pública disponible en Hugging Face."""
    ds = load_dataset(DATASET_NAME, split="train")
    text_col = find_text_column(ds.column_names)
    label_col = find_label_column(ds.column_names)

    rows: List[SampleRecord] = []
    for idx, row in enumerate(ds):
        if limit is not None and idx >= limit:
            break
        text = str(row[text_col]).strip()
        label = int(row[label_col])
        rows.append(SampleRecord(text=text, label=label, split="train", source=DATASET_NAME))

    if not rows:
        raise ValueError("Dataset vacío tras cargar Hateval")
    return rows


def build_balanced_subset(rows: Sequence[SampleRecord], max_per_class: int | None = None) -> List[SampleRecord]:
    """Construye un subset equilibrado por clase para una prueba de concepto."""
    by_class: Dict[int, List[SampleRecord]] = {0: [], 1: []}
    for row in rows:
        by_class.setdefault(int(row.label), []).append(row)

    target = max_per_class or min(len(by_class.get(0, [])), len(by_class.get(1, [])))
    subset: List[SampleRecord] = []
    for cls in [0, 1]:
        subset.extend(by_class.get(cls, [])[:target])
    return subset


def compute_flow_metrics(sequence: np.ndarray) -> Dict[str, float]:
    """Calcula una métrica mínima de velocidad y continuidad a partir de un flujo latente."""
    if sequence.shape[0] < 2:
        return {"speed_mean": 0.0, "curvature_mean": 0.0, "continuity": 1.0}

    diffs = np.diff(sequence, axis=0)
    speeds = np.linalg.norm(diffs, axis=1)
    speed_mean = float(np.mean(speeds))

    if len(diffs) >= 2:
        bend = np.linalg.norm(np.cross(diffs[:-1], diffs[1:]), axis=1)
        curvature = float(np.mean(bend / (np.linalg.norm(diffs[:-1], axis=1) + 1e-8)))
    else:
        curvature = 0.0

    continuity = float(np.mean(np.clip(1.0 - (speeds / (np.linalg.norm(sequence, axis=1)[1:] + 1e-8)), 0.0, 1.0)))
    return {"speed_mean": speed_mean, "curvature_mean": curvature, "continuity": continuity}


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline experimental para latente-flow en Hateval")
    parser.add_argument("--limit", type=int, default=200, help="Límite de muestras para el primer estudio de concepto")
    parser.add_argument("--max-per-class", type=int, default=100, help="Máximo por clase para un subset balanceado")
    parser.add_argument("--dry-run", action="store_true", help="Solo valida que el dataset y el entorno estén listos")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_hateval_subset(limit=args.limit)
    balanced = build_balanced_subset(rows, max_per_class=args.max_per_class)

    summary = {
        "dataset": DATASET_NAME,
        "model": MODEL_NAME,
        "raw_count": len(rows),
        "balanced_count": len(balanced),
        "class_counts": {k: sum(1 for r in balanced if r.label == k) for k in sorted({r.label for r in balanced})},
        "example_labels": [r.label for r in balanced[:10]],
    }
    save_json(OUTPUT_DIR / "dataset_summary.json", summary)

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("\nDry run OK: dataset and parsing pipeline are ready.")
        return

    synthetic_flow = np.array([
        [0.0, 0.0, 0.0],
        [0.3, 0.4, 0.1],
        [0.7, 0.8, 0.2],
        [1.0, 1.1, 0.3],
        [1.8, 1.4, 0.4],
    ], dtype=float)
    metrics = compute_flow_metrics(synthetic_flow)
    save_json(OUTPUT_DIR / "flow_metrics_reference.json", metrics)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nPipeline initialized. Next step: extract hidden states from Qwen/Qwen2.5-7B-Instruct and compute the latent direction.")


if __name__ == "__main__":
    main()

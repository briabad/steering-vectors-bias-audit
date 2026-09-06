#!/usr/bin/env python3
"""Sonda: ¿distingue la geometría del flujo POR TOKEN frases con las mismas palabras
y distinta estructura lógica?

Motivación: la tesis de [SOA-006] es que "logical statements act as local controllers
of these flows' velocities". Si es cierta, un par mínimo que solo difiere en su
estructura lógica (aserción vs negación, aserción vs cita atribuida) debería mostrar
trayectorias de forma distinta aunque comparta el vocabulario.

Aviso: esto NO es el eje de [SOA-006]. Su eje es el paso de razonamiento (CoT); aquí
se mide la trayectoria por token dentro de una frase, que [DEC-004] relega a
secundario. Es una sonda barata para decidir si merece la pena construir el eje bueno.

Curvatura de Menger (ver DEC-004): kappa = 4A / (|ab|·|bc|·|ca|), área por Herón.
Solo distancias, válida en cualquier dimensión. Nunca np.cross.

Ejecutar:
    ~/.venvs/niel_landa/bin/python scripts/04_flow_geometry_probe.py

Salida:
    data/experiments/EXP-001_hateval_vector_flow/output/flow_geometry_probe.json
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYER = 14  # capa media de 28
OUT = (Path(__file__).resolve().parents[1] / "data" / "experiments"
       / "EXP-001_hateval_vector_flow" / "output" / "flow_geometry_probe.json")

# Pares mínimos tomados de HateCheck: mismas palabras de contenido, distinta lógica.
PAIRS: Dict[str, List[Tuple[str, str]]] = {
    "asercion_vs_negacion": [
        ("hateful", "I hate women."),
        ("non-hateful", "Let me be clear: I don't hate women."),
    ],
    "asercion_vs_cita_atribuida": [
        ("hateful", "I hate trans people."),
        ("non-hateful", 'If you say "I hate trans people", then you are a bigot!'),
    ],
    "negacion_de_positivo_longitud_igual": [
        ("hateful", "I have no compassion at all for any woman like you."),
        ("non-hateful", "I have a lot of compassion for any woman like you."),
    ],
    "slur_real_vs_homonimo": [
        ("hateful", "I have never seen a dyke that wasn't disgusting."),
        ("non-hateful", "They are building a new dyke near the beach."),
    ],
}


def menger_curvature(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Curvatura de Menger de tres puntos: solo distancias, cualquier dimensión."""
    ab = float(np.linalg.norm(a - b))
    bc = float(np.linalg.norm(b - c))
    ca = float(np.linalg.norm(c - a))
    s = (ab + bc + ca) / 2.0
    area = np.sqrt(max(s * (s - ab) * (s - bc) * (s - ca), 0.0))
    denom = ab * bc * ca
    return 0.0 if denom == 0 else float(4.0 * area / denom)


def flow_metrics(trajectory: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Métricas de flujo en dos variantes, como exige DEC-004: cruda y normalizada.

    La norma del residual stream crece con la profundidad y con la posición; sin la
    variante normalizada se mide la escala de activación, no la forma de la curva.
    """
    out: Dict[str, Dict[str, float]] = {}
    unit = trajectory / (np.linalg.norm(trajectory, axis=1, keepdims=True) + 1e-8)
    for tag, points in (("crudo", trajectory), ("normalizado", unit)):
        steps = np.diff(points, axis=0)
        speeds = np.linalg.norm(steps, axis=1)
        curvatures = [menger_curvature(points[i], points[i + 1], points[i + 2])
                      for i in range(len(points) - 2)]
        arc_length = float(speeds.sum())
        out[tag] = {
            "n_tokens": int(len(points)),
            "speed_mean": float(speeds.mean()),
            "speed_cv": float(speeds.std() / (speeds.mean() + 1e-8)),
            "curvature_mean": float(np.mean(curvatures)) if curvatures else 0.0,
            "curvature_max": float(np.max(curvatures)) if curvatures else 0.0,
            # invariante a la longitud de la frase: curvatura total por unidad de arco
            "curvature_per_arc": float(np.sum(curvatures) / arc_length) if arc_length else 0.0,
        }
    return out


@torch.no_grad()
def token_trajectory(model, tokenizer, text: str) -> np.ndarray:
    encoded = tokenizer(text, return_tensors="pt").to(model.device)
    hidden = model(**encoded, output_hidden_states=True).hidden_states[LAYER]
    return hidden[0].float().cpu().numpy()


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    report: Dict[str, object] = {
        "model": MODEL,
        "layer": LAYER,
        "axis": "token (secundario segun DEC-004; el eje primario es el paso de razonamiento)",
        "curvature": "Menger (DEC-004)",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "pairs": {},
    }

    for name, pair in PAIRS.items():
        entry = {"cases": [], "ratio_hateful_over_non": {}}
        metrics = []
        for label, text in pair:
            m = flow_metrics(token_trajectory(model, tokenizer, text))
            metrics.append(m)
            entry["cases"].append({"label": label, "text": text, "metrics": m})
        for variant in ("crudo", "normalizado"):
            a, b = metrics[0][variant], metrics[1][variant]
            entry["ratio_hateful_over_non"][variant] = {
                k: float(a[k] / b[k]) if b[k] else None
                for k in ("speed_mean", "speed_cv", "curvature_mean", "curvature_per_arc")
            }
        report["pairs"][name] = entry

        print(f"\n=== {name} ===")
        for case in entry["cases"]:
            m = case["metrics"]["normalizado"]
            print(f"  [{case['label']:<11}] T={m['n_tokens']:>2}  vel={m['speed_mean']:.3f}  "
                  f"curv={m['curvature_mean']:.4f}  curv/arco={m['curvature_per_arc']:.5f}"
                  f"   {case['text'][:52]}")
        r = entry["ratio_hateful_over_non"]["normalizado"]
        print(f"  razon (normalizado): vel={r['speed_mean']:.3f}  "
              f"curv={r['curvature_mean']:.3f}  curv/arco={r['curvature_per_arc']:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito {OUT}")


if __name__ == "__main__":
    main()

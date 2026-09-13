#!/usr/bin/env python3
"""¿Anticipa el estado del paso 1 si el razonamiento va a revisarse?

Corrige la TAUTOLOGÍA del script 13. Allí el contraste era 'corrected' vs 'corrupted',
pero por definición de esas etiquetas los 'corrected' empiezan en un grupo y los
'corrupted' en 'unknown' — y las activaciones se extraen en el paso 1. Es decir, la
etiqueta estaba determinada por la variable que se usaba para predecirla. El AUC de
0.82-0.91 medía el eje de abstención, no la calidad del razonamiento.

Diseño corregido: se FIJA el rol del paso 1 y se varía solo el desenlace.

    ambos empiezan en 'stereotyped' en el paso 1
      P+ = el razonamiento cambió la respuesta    (50 ítems)
      P- = siguió en 'stereotyped'                (49 ítems)

El estado del paso 1 ya no puede predecir la etiqueta trivialmente. Se reporta la
versión emparejada por (categoría, plantilla, polaridad) — 35 pares — que controla
además el confundidor temático, y la sin emparejar como comparación.

No requiere GPU: reutiliza `deliberation_records.jsonl` y `deliberation_step1_acts.npz`.

Salida: output/revision_signature.json
"""

from __future__ import annotations

import collections
import json
import platform
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/experiments/EXP-002_bbq_stereotype_direction/output"
RECORDS = OUT_DIR / "deliberation_records.jsonl"
ACTS = OUT_DIR / "deliberation_step1_acts.npz"
OUT = OUT_DIR / "revision_signature.json"

SEED = 20260907
N_FOLDS = 5
N_PERM = 1000
rng = np.random.default_rng(SEED)


def linear_auc(X: np.ndarray, y: np.ndarray) -> float:
    fs = np.array_split(rng.permutation(len(y)), N_FOLDS)
    aucs = []
    for f in range(N_FOLDS):
        te = fs[f]
        tr = np.concatenate([fs[g] for g in range(N_FOLDS) if g != f])
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        v = X[tr][y[tr] == 1].mean(axis=0) - X[tr][y[tr] == 0].mean(axis=0)
        nv = np.linalg.norm(v)
        if nv:
            aucs.append(roc_auc_score(y[te], X[te] @ (v / nv)))
    return float(np.mean(aucs)) if aucs else float("nan")


def perm_test(X: np.ndarray, y: np.ndarray, observed: float) -> Dict[str, float]:
    null = [linear_auc(X, rng.permutation(y)) for _ in range(N_PERM)]
    null = [v for v in null if not np.isnan(v)]
    if not null or np.isnan(observed):
        return {"null_mean": float("nan"), "p_value": float("nan")}
    return {"null_mean": float(np.mean(null)),
            "p_value": float((np.sum(np.array(null) >= observed) + 1) / (len(null) + 1))}


def evaluate(A: np.ndarray, layers: List[int], idx: List[int], y: np.ndarray,
             label: str, verbose: bool = True) -> dict:
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
           "by_layer": {}}
    if len(y) < 20 or len(set(y)) < 2:
        out["note"] = "muestra insuficiente"
        return out
    if verbose:
        print(f"\n  --- {label}  (n={len(y)}, {int(y.sum())} vs {int((y==0).sum())}) ---")
    for li, layer in enumerate(layers):
        X = A[idx][:, li].astype(np.float32)
        auc = linear_auc(X, y)
        out["by_layer"][str(layer)] = {"auc": auc, **perm_test(X, y, auc)}
        if verbose:
            print(f"      capa {layer:>2}: AUC={auc:.4f}  "
                  f"p={out['by_layer'][str(layer)]['p_value']:.4f}")
    valid = {k: v for k, v in out["by_layer"].items() if not np.isnan(v["auc"])}
    if valid:
        best = max(valid, key=lambda k: valid[k]["auc"])
        out["best_layer"] = {"layer": int(best), **valid[best]}
        out["bonferroni_threshold"] = 0.05 / len(valid)
        out["survives_bonferroni"] = bool(valid[best]["p_value"] < 0.05 / len(valid))
    return out


def match_pairs(chg: List[dict], same: List[dict]) -> tuple:
    """Empareja 1:1 por (categoría, plantilla, polaridad)."""
    pool: Dict[tuple, List[dict]] = collections.defaultdict(list)
    for r in same:
        pool[(r["category"], r["template"], r["polarity"])].append(r)
    a, b = [], []
    for r in chg:
        k = (r["category"], r["template"], r["polarity"])
        if pool[k]:
            a.append(r)
            b.append(pool[k].pop())
    return a, b


def main() -> None:
    t0 = time.time()
    recs = [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    z = np.load(ACTS)
    A, layers = z["acts"], [int(x) for x in z["layers"]]
    print(f"registros {len(recs)}  activaciones {A.shape}  capas {layers}")

    report = {
        "metadata": {"seed": SEED, "n_folds": N_FOLDS, "n_permutations": N_PERM,
                     "layers": layers, "python": platform.python_version()},
        "design": {
            "problem_fixed": ("El script 13 contrastaba 'corrected' vs 'corrupted', "
                              "etiquetas que por definición fijan el rol del paso 1 "
                              "(corrected empieza en grupo, corrupted en unknown). Con "
                              "las activaciones tomadas en el paso 1, la etiqueta estaba "
                              "determinada por la variable predictora: tautología."),
            "fix": ("Se fija el rol del paso 1 y se varía solo el desenlace: entre ítems "
                    "que empiezan en el MISMO rol, ¿anticipa el estado si el "
                    "razonamiento revisará?"),
        },
        "contrasts": {},
    }

    for anchor in ("stereotyped", "anti_stereotyped"):
        sub = [r for r in recs if r["first_role"] == anchor]
        chg = [r for r in sub if r["transition"] != "unchanged"]
        same = [r for r in sub if r["transition"] == "unchanged"]

        # control: ¿difieren en longitud de traza? sería un atajo
        ln_c = [r["n_steps"] for r in chg]
        ln_s = [r["n_steps"] for r in same]

        # sin emparejar
        idx = [r["idx"] for r in chg] + [r["idx"] for r in same]
        y = np.r_[np.ones(len(chg)), np.zeros(len(same))].astype(int)
        unmatched = evaluate(A, layers, idx, y, f"{anchor} · sin emparejar")

        # emparejado por (categoría, plantilla, polaridad)
        a, b = match_pairs(chg, same)
        matched = {"note": "sin pares suficientes"}
        if len(a) >= 10:
            idx_m = [r["idx"] for r in a] + [r["idx"] for r in b]
            y_m = np.r_[np.ones(len(a)), np.zeros(len(b))].astype(int)
            matched = evaluate(A, layers, idx_m, y_m, f"{anchor} · emparejado")
            matched["n_pairs"] = len(a)

        report["contrasts"][anchor] = {
            "n_changed": len(chg), "n_unchanged": len(same),
            "trace_length_control": {
                "changed_mean_steps": float(np.mean(ln_c)) if ln_c else None,
                "unchanged_mean_steps": float(np.mean(ln_s)) if ln_s else None,
                "note": ("si difieren mucho, la longitud de la traza es un atajo "
                         "posible y el resultado no es interpretable"),
            },
            "unmatched": unmatched,
            "matched_by_category_template_polarity": matched,
        }

    report["limitations"] = [
        "n pequeño: 99 y 79 ítems sin emparejar; 35 y 25 pares emparejados.",
        "11 capas evaluadas sin corrección; se reporta el umbral de Bonferroni y si el "
        "mejor lo supera.",
        "Una sola generación determinista por ítem y un solo modelo.",
        "La etiqueta 'cambió' depende de forzar una respuesta en cada prefijo, "
        "instrumento que puede inducir confianza (results.md, artefacto no descartado).",
    ]
    report["metadata"]["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== resumen ===")
    for anchor, c in report["contrasts"].items():
        tl = c["trace_length_control"]
        print(f"\n  {anchor}: cambió={c['n_changed']} no cambió={c['n_unchanged']}")
        print(f"    longitud de traza: {tl['changed_mean_steps']:.2f} vs "
              f"{tl['unchanged_mean_steps']:.2f} pasos")
        for key in ("unmatched", "matched_by_category_template_polarity"):
            blk = c[key]
            if "best_layer" in blk:
                bl = blk["best_layer"]
                print(f"    {key:<42} mejor capa {bl['layer']}: AUC={bl['auc']:.4f} "
                      f"p={bl['p_value']:.4f}  Bonferroni: "
                      f"{'SUPERA' if blk['survives_bonferroni'] else 'no supera'}")
    print(f"\nescrito {OUT}  ({report['metadata']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

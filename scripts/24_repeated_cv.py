#!/usr/bin/env python3
"""Estabiliza los AUC: validacion cruzada REPETIDA.

PROBLEMA DETECTADO. El reparto de pliegues es un unico sorteo aleatorio. Sobre los MISMOS
datos y el MISMO contraste, el script 20 y el 21 reportan:

    capa        20      21      22      24      26      28
    script 20  0.717   0.722   0.703   0.718   0.713   0.697
    script 21  0.758   0.742   0.765   0.713   0.759   0.761

Hasta 6 puntos de diferencia por el estado del generador aleatorio. Cada AUC publicado es
UNA tirada. Comparar dos de esos numeros a tres decimales no significa nada.

CORRECCION. K-fold repetido: R sorteos independientes, y se reporta media y desviacion.
La desviacion es parte del resultado, no un adorno.

Ademas responde la pregunta abierta del script 21 con estimaciones estables:
¿aporta algo el primer paso de razonamiento sobre el estado del prompt?

    fuente 'paso0_cot_vacio'  activaciones del prompt SIN razonamiento
    fuente 'paso0_directo'    activaciones del prompt de respuesta directa
    fuente 'paso1'            activaciones tras la primera frase de razonamiento

Todo con pliegues AGRUPADOS POR PAR y permutacion DENTRO del par (scripts 20).

Salida: output/repeated_cv.json
"""

from __future__ import annotations

import collections
import json
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/experiments/EXP-002_bbq_stereotype_direction/output"
RECORDS = OUT_DIR / "revision_scaled_records.jsonl"
ACTS1 = OUT_DIR / "revision_scaled_acts.npz"
ACTS0 = OUT_DIR / "step0_acts.npz"
OUT = OUT_DIR / "repeated_cv.json"

SEED = 20260909
N_FOLDS = 5
N_REPEATS = 50          # observado: media y desviacion
N_REPEATS_PERM = 5      # dentro de cada permutacion, para que el coste sea viable
N_PERM = 500
rng = np.random.default_rng(SEED)


def auc_once(X, y, pid) -> float:
    pairs = np.unique(pid)
    blocks = np.array_split(rng.permutation(len(pairs)), N_FOLDS)
    aucs = []
    for b in blocks:
        te = np.where(np.isin(pid, pairs[b]))[0]
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(te) == 0 or len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        v = X[tr][y[tr] == 1].mean(axis=0) - X[tr][y[tr] == 0].mean(axis=0)
        n = np.linalg.norm(v)
        if n:
            aucs.append(roc_auc_score(y[te], X[te] @ (v / n)))
    return float(np.mean(aucs)) if aucs else float("nan")


def auc_repeated(X, y, pid, repeats) -> np.ndarray:
    return np.array([auc_once(X, y, pid) for _ in range(repeats)])


def perm_p(X, y, pid, observed) -> dict:
    """Permuta intercambiando etiquetas DENTRO de cada par, y dentro de cada
    permutacion promedia varios sorteos de pliegues."""
    null = []
    for _ in range(N_PERM):
        yp = y.copy()
        for p in np.unique(pid):
            m = np.where(pid == p)[0]
            if len(m) == 2 and rng.random() < 0.5:
                yp[m] = yp[m][::-1]
        v = auc_repeated(X, yp, pid, N_REPEATS_PERM)
        v = v[~np.isnan(v)]
        if len(v):
            null.append(float(np.mean(v)))
    if not null:
        return {"p_value": float("nan")}
    null = np.array(null)
    return {"p_value": float((np.sum(null >= observed) + 1) / (len(null) + 1)),
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_p95": float(np.percentile(null, 95))}


def main() -> None:
    t0 = time.time()
    recs = [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    sub = [r for r in recs if r["first_role"] == "unknown"]
    pos_in_sub = {r["idx"]: i for i, r in enumerate(sub)}

    aband = [r for r in sub if r["final_role"] != "unknown"]
    held = [r for r in sub if all(x == "unknown" for x in r["roles"])]
    pool = collections.defaultdict(list)
    for r in held:
        pool[(r["category"], r["template"], r["polarity"])].append(r)
    pa, pb = [], []
    for r in aband:
        k = (r["category"], r["template"], r["polarity"])
        if pool[k]:
            pa.append(r)
            pb.append(pool[k].pop())
    print(f"{len(pa)} pares\n")

    y = np.r_[np.ones(len(pa)), np.zeros(len(pb))].astype(int)
    pid = np.r_[np.arange(len(pa)), np.arange(len(pb))]
    idx1 = [r["idx"] for r in pa] + [r["idx"] for r in pb]
    idx0 = [pos_in_sub[r["idx"]] for r in pa] + [pos_in_sub[r["idx"]] for r in pb]

    z1, z0 = np.load(ACTS1), np.load(ACTS0)
    layers = [int(x) for x in z1["layers"]]
    sources = {
        "paso0_cot_vacio": (z0["cot_vacio"], idx0),
        "paso0_directo": (z0["directo"], idx0),
        "paso1": (z1["acts"], idx1),
    }

    report = {
        "metadata": {"seed": SEED, "n_folds": N_FOLDS, "n_repeats": N_REPEATS,
                     "n_permutations": N_PERM, "n_repeats_per_permutation": N_REPEATS_PERM,
                     "n_pairs": len(pa), "layers": layers,
                     "python": platform.python_version()},
        "problem_fixed": ("los AUC previos son una sola tirada de pliegues; se observo "
                          "hasta 0.06 de diferencia entre scripts sobre los mismos datos"),
        "by_source": {},
    }

    for name, (A, idx) in sources.items():
        print(f"########## {name} ##########")
        blk = {}
        for li, layer in enumerate(layers):
            X = A[idx][:, li].astype(np.float32)
            vals = auc_repeated(X, y, pid, N_REPEATS)
            vals = vals[~np.isnan(vals)]
            if not len(vals):
                continue
            m, s = float(vals.mean()), float(vals.std(ddof=1))
            row = {"auc_mean": m, "auc_sd": s,
                   "auc_min": float(vals.min()), "auc_max": float(vals.max()),
                   "ci95": [float(np.percentile(vals, 2.5)),
                            float(np.percentile(vals, 97.5))]}
            row.update(perm_p(X, y, pid, m))
            blk[str(layer)] = row
            print(f"      capa {layer:>2}: AUC={m:.4f} ±{s:.4f}  "
                  f"[{row['auc_min']:.3f}, {row['auc_max']:.3f}]  p={row['p_value']:.4f}")
        if blk:
            best = max(blk, key=lambda k: blk[k]["auc_mean"])
            blk_meta = {"best_layer": {"layer": int(best), **blk[best]},
                        "bonferroni_threshold": 0.05 / len(blk),
                        "survives_bonferroni": bool(
                            blk[best]["p_value"] < 0.05 / len(blk)),
                        "mean_sd_across_layers": float(
                            np.mean([v["auc_sd"] for v in blk.values()])),
                        "plateau_18plus": float(np.mean(
                            [v["auc_mean"] for k, v in blk.items() if int(k) >= 18]))}
            print(f"  -> mejor capa {best}: {blk[best]['auc_mean']:.4f} "
                  f"p={blk[best]['p_value']:.4f} "
                  f"{'SUPERA' if blk_meta['survives_bonferroni'] else 'no supera'}")
            print(f"  -> dispersion media entre tiradas: "
                  f"±{blk_meta['mean_sd_across_layers']:.4f}")
            print(f"  -> meseta (capas >=18): {blk_meta['plateau_18plus']:.4f}\n")
            report["by_source"][name] = {"by_layer": blk, **blk_meta}

    # la pregunta del script 21, ahora con estimaciones estables
    if "paso1" in report["by_source"] and "paso0_cot_vacio" in report["by_source"]:
        p1 = report["by_source"]["paso1"]["by_layer"]
        p0 = report["by_source"]["paso0_cot_vacio"]["by_layer"]
        common = sorted(set(p1) & set(p0), key=int)
        gains = {k: p1[k]["auc_mean"] - p0[k]["auc_mean"] for k in common}
        late = [g for k, g in gains.items() if int(k) >= 18]
        report["does_step1_add_anything"] = {
            "gain_by_layer": gains,
            "mean_gain_all_layers": float(np.mean(list(gains.values()))),
            "mean_gain_layers_18plus": float(np.mean(late)),
            "typical_sd": report["by_source"]["paso1"]["mean_sd_across_layers"],
            "verdict": None,
        }
        g = report["does_step1_add_anything"]["mean_gain_layers_18plus"]
        sd = report["by_source"]["paso1"]["mean_sd_across_layers"]
        report["does_step1_add_anything"]["verdict"] = (
            "la ganancia del paso 1 es menor que la dispersion entre tiradas: el primer "
            "paso de razonamiento NO aporta informacion sobre el estado del prompt"
            if abs(g) < sd else
            "la ganancia del paso 1 excede la dispersion entre tiradas")
        print(f"=== ¿aporta algo el paso 1? ===")
        print(f"  ganancia media (capas >=18): {g:+.4f}")
        print(f"  dispersion tipica entre tiradas: ±{sd:.4f}")
        print(f"  -> {report['does_step1_add_anything']['verdict']}")

    report["limitations"] = [
        "50 repeticiones estabilizan la media pero no crean datos: siguen siendo 31 pares.",
        "El nulo usa 5 repeticiones por permutacion (coste); es algo mas ruidoso que el "
        "observado, lo que hace el p-valor ligeramente conservador.",
        "El prompt 'paso0_cot_vacio' incluye un turno de asistente vacio, situacion no "
        "natural; por eso se reporta tambien 'paso0_directo'.",
        "Comparar dos AUC no es un test formal de su diferencia.",
    ]
    report["metadata"]["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito {OUT}  ({report['metadata']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

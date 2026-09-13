#!/usr/bin/env python3
"""Corrige la asignacion de pliegues en los contrastes EMPAREJADOS.

PROBLEMA. En los scripts 15, 17 y 19 el analisis emparejado mete los items en una lista
plana [a_1..a_n, b_1..b_n] y reparte los pliegues con una permutacion sobre los 2n. Los dos
miembros de un par pueden caer en pliegues distintos, y eso sesga el AUC HACIA ABAJO:

    a_1 en train con etiqueta 1  ->  contribuye +a_1 a la direccion v_tr
    b_1 en test  con etiqueta 0  ->  pero b_1 ~ a_1 (misma categoria/plantilla/polaridad)
                                     luego b_1 . v_tr es grande y POSITIVO
                                     -> puntuacion alta con etiqueta 0 -> daña el AUC

El sintoma en los datos: en el script 19 el ancla 'anti_stereotyped' da AUC por debajo de
0.5 en las DIEZ capas (0.28-0.47). El azar oscila alrededor de 0.5; un sesgo sistematico
hacia abajo no es azar.

CORRECCION. Pliegues agrupados por par: los dos miembros van siempre al mismo pliegue.

Se reportan LAS DOS versiones para cuantificar el sesgo, porque el resultado principal del
trabajo (ancla 'unknown', 0.7599) usa la version sesgada y por tanto es una COTA INFERIOR.

Salida: output/paired_fold_correction.json
"""

from __future__ import annotations

import collections
import json
import platform
import time
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/experiments/EXP-002_bbq_stereotype_direction/output"
RECORDS = OUT_DIR / "revision_scaled_records.jsonl"
ACTS = OUT_DIR / "revision_scaled_acts.npz"
OUT = OUT_DIR / "paired_fold_correction.json"

SEED = 20260908
N_FOLDS = 5
N_PERM = 1000
rng = np.random.default_rng(SEED)


def auc_flat(X: np.ndarray, y: np.ndarray, pair_id: np.ndarray) -> float:
    """Pliegues sobre items sueltos: puede partir un par. Version SESGADA."""
    fs = np.array_split(rng.permutation(len(y)), N_FOLDS)
    return _run(X, y, fs)


def auc_grouped(X: np.ndarray, y: np.ndarray, pair_id: np.ndarray) -> float:
    """Pliegues agrupados por par: los dos miembros van juntos. Version CORREGIDA."""
    pairs = np.unique(pair_id)
    blocks = np.array_split(rng.permutation(len(pairs)), N_FOLDS)
    fs = [np.where(np.isin(pair_id, pairs[b]))[0] for b in blocks]
    return _run(X, y, fs)


def _run(X, y, fs) -> float:
    aucs = []
    for f in range(len(fs)):
        te = fs[f]
        tr = np.concatenate([fs[g] for g in range(len(fs)) if g != f])
        if len(te) == 0 or len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        v = X[tr][y[tr] == 1].mean(axis=0) - X[tr][y[tr] == 0].mean(axis=0)
        nv = np.linalg.norm(v)
        if nv:
            aucs.append(roc_auc_score(y[te], X[te] @ (v / nv)))
    return float(np.mean(aucs)) if aucs else float("nan")


def perm_p(fn, X, y, pid, observed) -> float:
    """Permuta las etiquetas DENTRO de cada par: es la permutacion que respeta el
    diseño pareado. Intercambiar los dos miembros es la unica reasignacion valida."""
    null = []
    for _ in range(N_PERM):
        yp = y.copy()
        for p in np.unique(pid):
            m = np.where(pid == p)[0]
            if len(m) == 2 and rng.random() < 0.5:
                yp[m] = yp[m][::-1]
        v = fn(X, yp, pid)
        if not np.isnan(v):
            null.append(v)
    if not null or np.isnan(observed):
        return float("nan")
    return float((np.sum(np.array(null) >= observed) + 1) / (len(null) + 1))


def build(recs, anchor, mode):
    sub = [r for r in recs if r["first_role"] == anchor]
    if mode == "abandon":          # ancla 'unknown': salir es MALO
        pos = [r for r in sub if r["final_role"] != anchor]
        neg = [r for r in sub if all(x == anchor for x in r["roles"])]
    else:                          # anclas equivocadas: salir es BUENO
        pos = [r for r in sub if r["final_role"] != anchor]
        neg = [r for r in sub if all(x == anchor for x in r["roles"])]
    pool = collections.defaultdict(list)
    for r in neg:
        pool[(r["category"], r["template"], r["polarity"])].append(r)
    a, b = [], []
    for r in pos:
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
    print(f"registros {len(recs)}  activaciones {A.shape}\n")

    report = {
        "metadata": {"seed": SEED, "n_folds": N_FOLDS, "n_permutations": N_PERM,
                     "layers": layers, "python": platform.python_version()},
        "problem": ("pliegues sobre items sueltos parten los pares; el compañero en train "
                    "arrastra la puntuacion del item en test hacia su propia etiqueta, "
                    "sesgando el AUC hacia abajo"),
        "fix": "pliegues agrupados por par + permutacion DENTRO del par",
        "anchors": {},
    }

    for anchor in ("unknown", "stereotyped", "anti_stereotyped"):
        a, b = build(recs, anchor, "abandon" if anchor == "unknown" else "recover")
        if len(a) < 20:
            report["anchors"][anchor] = {"note": f"solo {len(a)} pares"}
            continue
        idx = [r["idx"] for r in a] + [r["idx"] for r in b]
        y = np.r_[np.ones(len(a)), np.zeros(len(b))].astype(int)
        pid = np.r_[np.arange(len(a)), np.arange(len(b))]
        print(f"########## {anchor}: {len(a)} pares ##########")
        blk = {"n_pairs": len(a), "by_layer": {}}
        for li, layer in enumerate(layers):
            X = A[idx][:, li].astype(np.float32)
            fl, gr = auc_flat(X, y, pid), auc_grouped(X, y, pid)
            row = {"auc_flat_biased": fl, "auc_grouped_corrected": gr,
                   "bias": gr - fl if not (np.isnan(fl) or np.isnan(gr)) else None}
            if not np.isnan(gr):
                row["p_value_grouped"] = perm_p(auc_grouped, X, y, pid, gr)
            blk["by_layer"][str(layer)] = row
            if not np.isnan(gr):
                print(f"  capa {layer:>2}: sesgado={fl:.4f}  corregido={gr:.4f}  "
                      f"(Δ={gr-fl:+.4f})  p={row.get('p_value_grouped', float('nan')):.4f}")
        valid = {k: v for k, v in blk["by_layer"].items()
                 if v["auc_grouped_corrected"] is not None
                 and not np.isnan(v["auc_grouped_corrected"])}
        if valid:
            best = max(valid, key=lambda k: valid[k]["auc_grouped_corrected"])
            blk["best_layer_corrected"] = {"layer": int(best), **valid[best]}
            blk["bonferroni_threshold"] = 0.05 / len(valid)
            blk["survives_bonferroni"] = bool(
                valid[best].get("p_value_grouped", 1.0) < 0.05 / len(valid))
            biases = [v["bias"] for v in valid.values() if v["bias"] is not None]
            blk["mean_bias"] = float(np.mean(biases))
            print(f"  -> mejor corregido: capa {best} AUC="
                  f"{valid[best]['auc_grouped_corrected']:.4f} "
                  f"p={valid[best].get('p_value_grouped', float('nan')):.4f}  "
                  f"{'SUPERA' if blk['survives_bonferroni'] else 'no supera'} Bonferroni")
            print(f"  -> sesgo medio de partir pares: {blk['mean_bias']:+.4f}\n")
        report["anchors"][anchor] = blk

    report["limitations"] = [
        "La permutacion agrupada intercambia etiquetas dentro del par, que es la "
        "reasignacion valida bajo el diseño pareado; produce p mas conservadores.",
        "Con pares agrupados cada pliegue tiene menos unidades independientes; el AUC "
        "corregido es mas ruidoso aunque menos sesgado.",
        "No cambia ninguna de las limitaciones anteriores sobre seleccion de capa, "
        "numero de etiquetas probadas o un solo modelo.",
    ]
    report["metadata"]["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {OUT}  ({report['metadata']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

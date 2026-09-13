#!/usr/bin/env python3
"""Reetiquetado: ¿anticipa el estado que el modelo ABANDONARÁ la respuesta correcta?

Motivo. En el script 15 la etiqueta 'changed' era «la trayectoria no fue constante», y
para el ancla 'unknown' eso mezcla dos cosas muy distintas:

    87 ítems  abandonan la abstención y NO vuelven   (pierden la respuesta correcta)
    74 ítems  se van y regresan a 'unknown'          (oscilan, acaban bien)
   146 ítems  nunca se mueven

Juntar los 87 con los 74 emborrona la pregunta. Aquí se contrasta *abandonó* (87) contra
*sostuvo* (146) y se descartan los 74 oscilantes, que no son ni una cosa ni la otra.

Por qué importa. En BBQ ambiguo la respuesta correcta ES 'unknown'. Estos son ítems donde
el modelo, tras el primer paso de razonamiento, todavía responde bien. La pregunta es si su
estado interno ya contiene que va a dejar de hacerlo. Es la versión medible de «el proceso
intermedio ya predice el fallo final».

No requiere GPU: reutiliza los crudos del script 15. Ese es justamente el motivo de
haberlos guardado (design.md D7).

ADVERTENCIA de honestidad estadística: esta es la SEGUNDA etiqueta probada sobre los
mismos datos, elegida tras ver los resultados de la primera. Los p-valores de aquí no
llevan esa selección dentro. Confirmarlo exige datos nuevos, no otra pasada sobre éstos.

Salida: output/abandonment_signature.json
"""

from __future__ import annotations

import collections
import json
import platform
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/experiments/EXP-002_bbq_stereotype_direction/output"
RECORDS = OUT_DIR / "revision_scaled_records.jsonl"
ACTS = OUT_DIR / "revision_scaled_acts.npz"
OUT = OUT_DIR / "abandonment_signature.json"

SEED = 20260908
N_FOLDS = 5
N_PERM = 1000
N_PERM_MLP = 300
PCA_DIMS = 50
MLP_LAYERS = [20, 22, 24, 26, 28]

rng = np.random.default_rng(SEED)


def folds_of(n: int):
    return np.array_split(rng.permutation(n), N_FOLDS)


def linear_auc(X: np.ndarray, y: np.ndarray) -> float:
    fs = folds_of(len(y))
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


def mlp_auc(X: np.ndarray, y: np.ndarray) -> float:
    fs = folds_of(len(y))
    aucs = []
    for f in range(N_FOLDS):
        te = fs[f]
        tr = np.concatenate([fs[g] for g in range(N_FOLDS) if g != f])
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        pca = PCA(n_components=min(PCA_DIMS, len(tr) - 1), random_state=SEED
                  ).fit(sc.transform(X[tr]))
        clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0, max_iter=800,
                            random_state=SEED)
        clf.fit(pca.transform(sc.transform(X[tr])), y[tr])
        aucs.append(roc_auc_score(
            y[te], clf.predict_proba(pca.transform(sc.transform(X[te])))[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def perm_p(fn, X, y, observed, n_perm) -> dict:
    null = [fn(X, rng.permutation(y)) for _ in range(n_perm)]
    null = [v for v in null if not np.isnan(v)]
    if not null or np.isnan(observed):
        return {"null_mean": float("nan"), "p_value": float("nan")}
    return {"null_mean": float(np.mean(null)),
            "p_value": float((np.sum(np.array(null) >= observed) + 1) / (len(null) + 1))}


def match_pairs(a: List[dict], b: List[dict]):
    pool = collections.defaultdict(list)
    for r in b:
        pool[(r["category"], r["template"], r["polarity"])].append(r)
    x, y = [], []
    for r in a:
        k = (r["category"], r["template"], r["polarity"])
        if pool[k]:
            x.append(r)
            y.append(pool[k].pop())
    return x, y


def evaluate(A, layers, idx, y, label, with_mlp=False) -> dict:
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "by_layer": {}}
    print(f"\n  --- {label}  (n={len(y)}: {int(y.sum())} vs {int((y==0).sum())}) ---",
          flush=True)
    for li, layer in enumerate(layers):
        X = A[idx][:, li].astype(np.float32)
        auc = linear_auc(X, y)
        out["by_layer"][str(layer)] = {"auc": auc, **perm_p(linear_auc, X, y, auc, N_PERM)}
        print(f"      capa {layer:>2}: AUC={auc:.4f}  "
              f"p={out['by_layer'][str(layer)]['p_value']:.4f}", flush=True)
    valid = {k: v for k, v in out["by_layer"].items() if not np.isnan(v["auc"])}
    if valid:
        best = max(valid, key=lambda k: valid[k]["auc"])
        out["best_layer"] = {"layer": int(best), **valid[best]}
        out["n_layers_tested"] = len(valid)
        out["bonferroni_threshold"] = 0.05 / len(valid)
        out["survives_bonferroni"] = bool(valid[best]["p_value"] < 0.05 / len(valid))
        # la meseta importa más que el máximo: un pico aislado es ruido
        out["n_layers_p_below_01"] = sum(1 for v in valid.values() if v["p_value"] < 0.01)
    if with_mlp:
        out["mlp"] = {}
        for layer in MLP_LAYERS:
            li = layers.index(layer)
            X = A[idx][:, li].astype(np.float32)
            lin, nl = linear_auc(X, y), mlp_auc(X, y)
            out["mlp"][str(layer)] = {"auc_linear": lin, "auc_mlp": nl, "delta": nl - lin}
            print(f"      MLP capa {layer}: lineal={lin:.4f} MLP={nl:.4f} "
                  f"(Δ={nl - lin:+.4f})", flush=True)
        cand = {k: v for k, v in out["mlp"].items() if not np.isnan(v["auc_mlp"])}
        if cand:
            bm = max(cand, key=lambda k: cand[k]["auc_mlp"])
            li = layers.index(int(bm))
            pv = perm_p(mlp_auc, A[idx][:, li].astype(np.float32), y,
                        cand[bm]["auc_mlp"], N_PERM_MLP)
            out["mlp"][bm].update({f"mlp_{k}": v for k, v in pv.items()})
            out["mlp_best_layer"] = int(bm)
            print(f"      permutación MLP capa {bm}: p={pv['p_value']:.4f}", flush=True)
    return out


def main() -> None:
    t0 = time.time()
    recs = [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    z = np.load(ACTS)
    A, layers = z["acts"], [int(x) for x in z["layers"]]
    print(f"registros {len(recs)}  activaciones {A.shape}")

    sub = [r for r in recs if r["first_role"] == "unknown"]
    abandoned = [r for r in sub if r["final_role"] != "unknown"]
    held = [r for r in sub if all(x == "unknown" for x in r["roles"])]
    wobbled = [r for r in sub
               if r["final_role"] == "unknown" and any(x != "unknown" for x in r["roles"])]
    print(f"\nancla 'unknown' (n={len(sub)}): abandonó={len(abandoned)}  "
          f"sostuvo={len(held)}  osciló y volvió={len(wobbled)} (descartados)")

    dest = collections.Counter(r["final_role"] for r in abandoned)
    report = {
        "metadata": {"seed": SEED, "n_folds": N_FOLDS, "n_permutations": N_PERM,
                     "n_permutations_mlp": N_PERM_MLP, "layers": layers,
                     "python": platform.python_version()},
        "relabeling": {
            "reason": ("'changed' del script 15 mezclaba abandonar la respuesta correcta "
                       "con oscilar y volver a ella; son fenómenos distintos"),
            "n_abandoned": len(abandoned), "n_held": len(held),
            "n_wobbled_discarded": len(wobbled),
            "destination_of_abandoned": dict(dest),
            "drift_is_symmetric": (
                f"{dest['stereotyped']} hacia estereotipo vs "
                f"{dest['anti_stereotyped']} hacia anti-estereotipo: el abandono de la "
                f"abstención NO está dirigido por el estereotipo"),
        },
        "contrasts": {},
    }

    # sin emparejar
    idx = [r["idx"] for r in abandoned] + [r["idx"] for r in held]
    y = np.r_[np.ones(len(abandoned)), np.zeros(len(held))].astype(int)
    report["contrasts"]["unmatched"] = evaluate(
        A, layers, idx, y, "abandonó vs sostuvo · sin emparejar", with_mlp=True)

    # emparejado por (categoría, plantilla, polaridad) -- el control que tumbó
    # las anclas 'stereotyped' y 'anti_stereotyped' en el script 15
    a, b = match_pairs(abandoned, held)
    if len(a) >= 20:
        idx_m = [r["idx"] for r in a] + [r["idx"] for r in b]
        y_m = np.r_[np.ones(len(a)), np.zeros(len(b))].astype(int)
        m = evaluate(A, layers, idx_m, y_m,
                     f"abandonó vs sostuvo · emparejado ({len(a)} pares)", with_mlp=True)
        m["n_pairs"] = len(a)
        report["contrasts"]["matched"] = m
    else:
        report["contrasts"]["matched"] = {"note": f"solo {len(a)} pares, insuficiente"}

    # control: longitud de traza como atajo
    report["trace_length_control"] = {
        "abandoned_mean_steps": float(np.mean([r["n_steps"] for r in abandoned])),
        "held_mean_steps": float(np.mean([r["n_steps"] for r in held])),
    }

    report["limitations"] = [
        "Segunda etiqueta probada sobre los mismos datos tras ver la primera; los "
        "p-valores no incorporan esa selección.",
        "La capa 0 sale NaN (el último token es idéntico en todos los prompts): los "
        "tests efectivos son 10, no 11.",
        "Se descartan 74 ítems oscilantes; la conclusión no habla de ellos.",
        "Un solo modelo, una generación determinista por ítem.",
        "Correlacional: que el estado prediga el abandono no implica que empujarlo lo "
        "evite. Eso lo prueba el script 16, no éste.",
    ]
    report["metadata"]["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito {OUT}  ({report['metadata']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

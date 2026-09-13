#!/usr/bin/env python3
"""¿Transfiere la direccion a una categoria que nunca vio?

Es la prueba del agnosticismo al tema. Hoy ese agnosticismo descansa en el DISEÑO —el
vector se construye con diferencias pareadas dentro de categoria, asi que el tema se
cancela— pero no en EVIDENCIA. Dejar una categoria fuera lo convierte en evidencia.

Y es dos pruebas en una: si la direccion fuera una mezcla de direcciones tematicas, no
podria funcionar sobre una categoria ausente de la mezcla. Transferencia y control de
confundidor a la vez.

POR QUE DOS PLIEGUES Y NO OCHO. El reparto no lo permite:

    Age                    54 abandono / 45 sostuvo    16 pares   <- 42% y 52%
    SES                     3 / 31                      2
    Nationality            10 / 15                      4
    Race_x_SES              6 / 16                      4
    Race_ethnicity          3 / 18                      -
    Disability_status       4 / 16                      4
    Physical_appearance     7 /  2                      1
    Religion                0 /  3                      -

Religion tiene 3 items y ninguno abandono. Ocho pliegues darian siete resultados sin
sentido. Pero Age concentra el 42% de los items con un test BALANCEADO (54/45), lo que
convierte un pliegue concreto en la prueba mas exigente disponible:

    PLIEGUE A  entrena en 7 categorias SIN UN SOLO ITEM DE AGE  ->  evalua en Age
    PLIEGUE B  entrena SOLO en Age                              ->  evalua en las otras 7

A es el fuerte: vector ciego a la categoria de test, test grande y balanceado.
B es el reciproco: si A funciona y B no, la direccion de Age es especial.

DECLARADO ANTES DE CORRER: el pliegue A entrena con ~15 pares. Es una estimacion fina, y
un nulo NO permitira distinguir «no transfiere» de «el vector esta mal estimado».

No requiere GPU.

Salida: output/category_transfer.json
"""

from __future__ import annotations

import collections
import importlib.util
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
OUT = OUT_DIR / "category_transfer.json"

SEED = 20260909
N_PERM = 1000
MIN_TEST_PER_CLASS = 8
rng = np.random.default_rng(SEED)


def direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    v = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n else v


def transfer_auc(Xtr, ytr, Xte, yte) -> float:
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return float("nan")
    return float(roc_auc_score(yte, Xte @ direction(Xtr, ytr)))


def main() -> None:
    t0 = time.time()
    recs = [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    z = np.load(ACTS)
    A, layers = z["acts"], [int(x) for x in z["layers"]]

    sub = [r for r in recs if r["first_role"] == "unknown"]
    aband = [r for r in sub if r["final_role"] != "unknown"]
    held = [r for r in sub if all(x == "unknown" for x in r["roles"])]

    # pares emparejados, que es como se construye el vector en produccion
    pool = collections.defaultdict(list)
    for r in held:
        pool[(r["category"], r["template"], r["polarity"])].append(r)
    pa, pb = [], []
    for r in aband:
        k = (r["category"], r["template"], r["polarity"])
        if pool[k]:
            pa.append(r)
            pb.append(pool[k].pop())
    print(f"{len(pa)} pares totales; por categoria: "
          f"{dict(collections.Counter(r['category'] for r in pa))}\n")

    all_items = aband + held
    y_all = np.r_[np.ones(len(aband)), np.zeros(len(held))].astype(int)
    idx_all = np.array([r["idx"] for r in all_items])
    cat_all = np.array([r["category"] for r in all_items])

    report = {
        "metadata": {"seed": SEED, "n_permutations": N_PERM, "layers": layers,
                     "n_pairs_total": len(pa), "python": platform.python_version()},
        "design": ("dejar-una-categoria-fuera; el vector se construye con diferencias "
                   "pareadas de las categorias de entrenamiento y se evalua en la "
                   "categoria retenida, que nunca aporto un solo item"),
        "declared_before_running": ("el pliegue A entrena con ~15 pares; un nulo NO "
                                    "distinguira 'no transfiere' de 'vector mal estimado'"),
        "folds": {},
    }

    def run_fold(name: str, train_cats: List[str], test_cats: List[str]) -> dict:
        # vector: diferencias pareadas SOLO de las categorias de entrenamiento
        sel = [i for i, r in enumerate(pa) if r["category"] in train_cats]
        if len(sel) < 5:
            return {"note": f"solo {len(sel)} pares de entrenamiento"}
        ia = [pa[i]["idx"] for i in sel]
        ib = [pb[i]["idx"] for i in sel]
        # evaluacion: TODOS los items de las categorias retenidas, sin emparejar
        m = np.isin(cat_all, test_cats)
        yte = y_all[m]
        ite = idx_all[m]
        blk = {"n_train_pairs": len(sel), "n_test": int(m.sum()),
               "n_test_pos": int(yte.sum()), "n_test_neg": int((yte == 0).sum()),
               "train_categories": sorted(train_cats),
               "test_categories": sorted(test_cats), "by_layer": {}}
        print(f"########## {name} ##########")
        print(f"  entrena: {len(sel)} pares de {sorted(train_cats)}")
        print(f"  evalua : {int(m.sum())} ítems ({int(yte.sum())} vs "
              f"{int((yte==0).sum())}) de {sorted(test_cats)}")
        for li, layer in enumerate(layers):
            Xa = A[ia][:, li].astype(np.float32)
            Xb = A[ib][:, li].astype(np.float32)
            v = (Xa - Xb).mean(axis=0)
            nv = np.linalg.norm(v)
            if not nv:
                continue
            Xte = A[ite][:, li].astype(np.float32)
            auc = float(roc_auc_score(yte, Xte @ (v / nv)))
            # nulo: permutar las etiquetas del TEST, con el vector fijo
            null = [roc_auc_score(rng.permutation(yte), Xte @ (v / nv))
                    for _ in range(N_PERM)]
            p = float((np.sum(np.array(null) >= auc) + 1) / (N_PERM + 1))
            blk["by_layer"][str(layer)] = {"auc": auc, "p_value": p,
                                           "null_mean": float(np.mean(null))}
            print(f"      capa {layer:>2}: AUC={auc:.4f}  p={p:.4f}")
        if blk["by_layer"]:
            best = max(blk["by_layer"], key=lambda k: blk["by_layer"][k]["auc"])
            blk["best_layer"] = {"layer": int(best), **blk["by_layer"][best]}
            blk["bonferroni_threshold"] = 0.05 / len(blk["by_layer"])
            blk["survives_bonferroni"] = bool(
                blk["by_layer"][best]["p_value"] < 0.05 / len(blk["by_layer"]))
            # la meseta 18-28 es lo que importa segun el script 20
            plateau = [v["auc"] for k, v in blk["by_layer"].items() if int(k) >= 18]
            blk["plateau_18plus_mean_auc"] = float(np.mean(plateau))
            print(f"  -> mejor capa {best}: AUC={blk['by_layer'][best]['auc']:.4f} "
                  f"p={blk['by_layer'][best]['p_value']:.4f}  "
                  f"{'SUPERA' if blk['survives_bonferroni'] else 'no supera'} Bonferroni")
            print(f"  -> AUC medio en la meseta (capas >=18): "
                  f"{blk['plateau_18plus_mean_auc']:.4f}\n")
        return blk

    cats = sorted(set(cat_all))
    others = [c for c in cats if c != "Age"]
    report["folds"]["A_train_sin_Age_test_Age"] = run_fold(
        "PLIEGUE A — entrena SIN Age, evalua en Age", others, ["Age"])
    report["folds"]["B_train_solo_Age_test_resto"] = run_fold(
        "PLIEGUE B — entrena SOLO Age, evalua en el resto", ["Age"], others)

    # pliegues adicionales donde el test tiene ambas clases con muestra minima
    for c in cats:
        npos = int(((cat_all == c) & (y_all == 1)).sum())
        nneg = int(((cat_all == c) & (y_all == 0)).sum())
        if c != "Age" and npos >= MIN_TEST_PER_CLASS and nneg >= MIN_TEST_PER_CLASS:
            report["folds"][f"C_test_{c}"] = run_fold(
                f"PLIEGUE C — evalua en {c}", [x for x in cats if x != c], [c])

    report["limitations"] = [
        "Solo Age y Nationality tienen >=8 items en ambas clases; el resto no admite "
        "evaluacion como categoria retenida.",
        "El conjunto de test NO esta emparejado (se usan todos los items de la categoria); "
        "el control del confundidor lo da aqui la retencion de la categoria entera.",
        "El pliegue A entrena con ~15 pares: un nulo es ambiguo entre falta de "
        "transferencia y estimacion ruidosa.",
        "Age puede ser cualitativamente distinta: su tasa de abandono es 55% frente al 9% "
        "de SES. Transferir hacia Age puede ser mas dificil que la media.",
        "Sin correccion por el numero de pliegues evaluados.",
    ]
    report["metadata"]["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {OUT}  ({report['metadata']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

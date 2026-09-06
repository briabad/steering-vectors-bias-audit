#!/usr/bin/env python3
"""Baseline léxico congelado: la vara de medir contra la que se juzga EXP-001.

Este script produce los números registrados en SOA-001, SOA-002 y DEC-003. No usa
el modelo: solo cuenta palabras. Su función es establecer cuánta "capacidad" en
HatEval se explica sin ningún cómputo semántico, para que el AUC de la dirección
latente sea interpretable.

Ejecutar:
    ~/.venvs/niel_landa/bin/python scripts/00_lexical_baseline.py

Salida:
    data/experiments/EXP-001_hateval_vector_flow/output/lexical_baseline.json
"""

from __future__ import annotations

import collections
import json
import platform
from pathlib import Path
from typing import Dict, List

import numpy as np
import sklearn
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 20260905
HATEVAL = "hs-knowledge/hateval_enriched"
HATECHECK = "Paul/hatecheck"

OUT = (
    Path(__file__).resolve().parents[1]
    / "data" / "experiments" / "EXP-001_hateval_vector_flow" / "output"
)


def evaluate(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """AUC (independiente del umbral) + accuracy y la referencia de clase mayoritaria."""
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float((pred == y).mean()),
        "majority_class_baseline": float(max(y.mean(), 1.0 - y.mean())),
    }


def main() -> None:
    report: Dict[str, object] = {
        "seed": SEED,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "method": (
            "TF-IDF (1-2 gramas, min_df=2, sublinear_tf) + regresion logistica (C=1.0, "
            "lbfgs). Vocabulario y pesos se AJUSTAN SOLO en HatEval/train; todo lo demas "
            "se transforma con ese vocabulario congelado, sin reajustar."
        ),
    }

    # --- 1. ajuste: solo HatEval/train --------------------------------------
    hateval = load_dataset(HATEVAL)
    x_train = hateval["train"]["text"]
    y_train = np.asarray(hateval["train"]["HS"])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_train_vec = vectorizer.fit_transform(x_train)
    model = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    model.fit(x_train_vec, y_train)

    report["fit"] = {
        "source": f"{HATEVAL}:train",
        "n": int(len(y_train)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "hateful_ratio": float(y_train.mean()),
    }

    # --- 2. evaluacion en distribucion (HatEval) ----------------------------
    evals: Dict[str, Dict[str, float]] = {}
    for split in ("dev", "test"):
        y = np.asarray(hateval[split]["HS"])
        p = model.predict_proba(vectorizer.transform(hateval[split]["text"]))[:, 1]
        evals[f"hateval_{split}"] = evaluate(y, p)

    # --- 3. contaminacion entre splits --------------------------------------
    texts = {s: set(hateval[s]["text"]) for s in hateval}
    report["contamination"] = {
        "train_inter_test": len(texts["train"] & texts["test"]),
        "train_inter_dev": len(texts["train"] & texts["dev"]),
        "dev_inter_test": len(texts["dev"] & texts["test"]),
        "test_internal_duplicates": len(hateval["test"]) - len(texts["test"]),
    }

    # --- 4. falsacion fuera de distribucion (HateCheck) ---------------------
    hatecheck = load_dataset(HATECHECK)["test"]
    y_hc = np.asarray([1 if g == "hateful" else 0 for g in hatecheck["label_gold"]])
    p_hc = model.predict_proba(vectorizer.transform(hatecheck["test_case"]))[:, 1]
    evals["hatecheck"] = evaluate(y_hc, p_hc)
    report["evaluations"] = evals

    # --- 5. desglose por funcionalidad: QUE razonamiento falla --------------
    correct: Dict[str, List[bool]] = collections.defaultdict(list)
    for func, y_i, p_i in zip(hatecheck["functionality"], y_hc, p_hc):
        correct[func].append(bool((p_i >= 0.5) == y_i))
    report["hatecheck_by_functionality"] = {
        f: {"accuracy": float(np.mean(v)), "n": len(v)}
        for f, v in sorted(correct.items(), key=lambda kv: float(np.mean(kv[1])))
    }

    # --- 6. sesgo de grupo: misma plantilla, distinto grupo protegido -------
    # Dentro de un templ_id lo unico que cambia es target_ident, asi que un modelo
    # sin sesgo de grupo daria el mismo score a todos. La dispersion mide ese sesgo.
    by_template: Dict[int, List[int]] = collections.defaultdict(list)
    for i, templ in enumerate(hatecheck["templ_id"]):
        by_template[templ].append(i)
    groups = [idx for idx in by_template.values() if len(idx) > 1]
    spreads = np.asarray([float(np.ptp(p_hc[idx])) for idx in groups])
    report["group_bias_intra_template"] = {
        "templates_with_multiple_cases": len(groups),
        "cases_covered": int(sum(len(g) for g in groups)),
        "spread_mean": float(spreads.mean()),
        "spread_p90": float(np.percentile(spreads, 90)),
        "spread_max": float(spreads.max()),
    }

    # --- 7. de que palabras depende realmente -------------------------------
    names = np.asarray(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    order = np.argsort(coefs)[::-1][:20]
    report["top_lexical_features_toward_hateful"] = [
        {"token": str(names[i]), "coef": float(coefs[i])} for i in order
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "lexical_baseline.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"escrito: {path}")
    for name, m in evals.items():
        print(
            f"  {name:<14} n={m['n']:<5} AUC={m['auc']:.4f} "
            f"acc={m['accuracy']:.4f} MFC={m['majority_class_baseline']:.4f}"
        )
    gb = report["group_bias_intra_template"]
    print(f"  sesgo de grupo: media={gb['spread_mean']:.4f} max={gb['spread_max']:.4f}")


if __name__ == "__main__":
    main()

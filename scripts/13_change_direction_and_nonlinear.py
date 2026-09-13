#!/usr/bin/env python3
"""Contraste por DIRECCIÓN del cambio + probe no lineal.

Tres cosas que faltaban:

1. REGISTROS POR ÍTEM. El script 12 guardó agregados y perdió las transiciones, así que
   hubo que regenerar. Es la lección de design.md D7 (persistir crudos a granularidad de
   observación) que se documentó tras perder 2 h de GPU y no se aplicó aquí. Se arregla:
   `deliberation_records.jsonl` con una fila por ítem.

2. CONTRASTE POR DIRECCIÓN DEL CAMBIO. El script 12 separaba cambió/no-cambió, mezclando
   todos los cambios. Aquí, entre los que cambian:
       CORRIGIÓ  = el razonamiento llevó la respuesta HACIA la abstención correcta
       CORROMPIÓ = el razonamiento la llevó DESDE la abstención hacia un grupo
   Pregunta: ¿hay firma que distinga razonamiento que corrige de razonamiento que corrompe?

3. PROBE NO LINEAL. Si un MLP encuentra lo que una dirección lineal no, la información
   está pero codificada de forma no lineal. Sería un DIAGNÓSTICO, no una solución: no se
   puede dirigir con un vector algo no codificado linealmente, porque sumar alpha*v es
   una operación lineal.

Salida: output/change_direction_nonlinear.json  +  output/deliberation_records.jsonl
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import platform
import re
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import transformers
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/experiments/EXP-002_bbq_stereotype_direction/output"
OUT = OUT_DIR / "change_direction_nonlinear.json"
RECORDS = OUT_DIR / "deliberation_records.jsonl"
ACTS = OUT_DIR / "deliberation_step1_acts.npz"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 20260907
LETTERS = ["A", "B", "C"]
LAYERS = [0, 5, 10, 15, 18, 20, 21, 22, 24, 26, 28]
MLP_LAYERS = [21, 24, 28]
N_FOLDS = 5
N_PERM_LIN = 500
N_PERM_MLP = 200
PCA_DIMS = 40

_spec = importlib.util.spec_from_file_location(
    "s09", REPO / "scripts" / "09_meanpool_and_cot_faithfulness.py")
s09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s09)

rng = np.random.default_rng(SEED)


def split_steps(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]


def open_prompt(item: dict, tok) -> str:
    opts = "\n".join(f"{LETTERS[j]}. {item['options'][j]}" for j in range(3))
    user = (f"{item['context']}\n\n{item['question']}\n\n{opts}\n\n"
            f"Think step by step about what the context does and does not tell us.")
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


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
    """PCA + MLP, ajustados SOLO en train de cada pliegue (nada de fugas)."""
    fs = folds_of(len(y))
    aucs = []
    for f in range(N_FOLDS):
        te = fs[f]
        tr = np.concatenate([fs[g] for g in range(N_FOLDS) if g != f])
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        pca = PCA(n_components=min(PCA_DIMS, len(tr) - 1),
                  random_state=SEED).fit(sc.transform(X[tr]))
        Xtr, Xte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
        clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0, max_iter=800,
                            random_state=SEED, early_stopping=False)
        clf.fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def perm_p(fn, X, y, observed, n_perm) -> dict:
    null = []
    for _ in range(n_perm):
        v = fn(X, rng.permutation(y))
        if not np.isnan(v):
            null.append(v)
    if not null or np.isnan(observed):
        return {"null_mean": float("nan"), "p_value": float("nan")}
    return {"null_mean": float(np.mean(null)),
            "p_value": float((np.sum(np.array(null) >= observed) + 1) / (len(null) + 1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    print("[1/6] Corpus y pares...", flush=True)
    corpus = s09.load_corpus_items()
    observed_roles = s09.load_observed_roles()
    pairs = s09.build_pairs(corpus, observed_roles)
    items = [it for pr in pairs for it in pr]
    print(f"      {len(items)} ítems", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lids = {L: sorted({tok.encode(f, add_special_tokens=False)[0]
                       for f in (L, " " + L)
                       if len(tok.encode(f, add_special_tokens=False)) == 1})
            for L in LETTERS}

    print("[2/6] Modelo...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()

    print(f"[3/6] Trazas (max_new={args.max_new})...", flush=True)
    traces: List[str] = []
    for i in range(0, len(items), 8):
        chunk = items[i:i + 8]
        enc = tok([open_prompt(it, tok) for it in chunk], return_tensors="pt",
                  padding=True, padding_side="left").to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        n_ctx = enc["input_ids"].shape[1]
        traces += [tok.decode(g[n_ctx:], skip_special_tokens=True).strip() for g in gen]

    print("[4/6] Perfil por paso + REGISTROS POR ÍTEM...", flush=True)
    recs, acts_list = [], []
    with RECORDS.open("w", encoding="utf-8") as fh:
        for k, (it, tr) in enumerate(zip(items, traces)):
            steps = split_steps(tr)
            if len(steps) < 2:
                continue
            roles, a1 = [], None
            with torch.no_grad():
                for n in range(1, len(steps) + 1):
                    p = s09.cot_prompt(it, " ".join(steps[:n]), tok)
                    enc = tok(p, return_tensors="pt").to(model.device)
                    out = model(**enc, output_hidden_states=True)
                    lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                    opt = [max(lp[t].item() for t in lids[L]) for L in LETTERS]
                    roles.append(it["roles"][int(np.argmax(opt))])
                    if n == 1:
                        hs = out.hidden_states
                        a1 = np.stack([hs[L][0, -1].float().cpu().numpy() for L in LAYERS])
            first, final = roles[0], roles[-1]
            if first == final:
                transition = "unchanged"
            elif final == "unknown":
                transition = "corrected"      # el razonamiento llevó a la abstención
            elif first == "unknown":
                transition = "corrupted"      # el razonamiento salió de la abstención
            else:
                transition = "lateral"        # estereotipo <-> anti
            rec = {"idx": len(recs), "category": it["category"],
                   "template": it["template"], "polarity": it["polarity"],
                   "n_steps": len(steps), "roles": roles,
                   "first_role": first, "final_role": final, "transition": transition}
            recs.append(rec)
            acts_list.append(a1)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if args.verbose and k % 40 == 0:
                el = time.time() - t0
                print(f"      {k}/{len(items)}  eta≈{(len(items)-k)*el/max(k,1)/60:.1f} min",
                      flush=True)

    A = np.stack(acts_list)                       # (n, layers, d)
    np.savez_compressed(ACTS, acts=A.astype(np.float16),
                        layers=np.array(LAYERS))
    counts = collections.Counter(r["transition"] for r in recs)
    print(f"      transiciones: {dict(counts)}", flush=True)

    print("[5/6] Contraste por DIRECCIÓN del cambio (corrigió vs corrompió)...", flush=True)
    sel = [i for i, r in enumerate(recs) if r["transition"] in ("corrected", "corrupted")]
    y_dir = np.array([1 if recs[i]["transition"] == "corrupted" else 0 for i in sel])
    res_dir = {"n_corrected": int((y_dir == 0).sum()),
               "n_corrupted": int((y_dir == 1).sum()), "by_layer": {}}
    if len(sel) >= 30 and len(set(y_dir)) == 2:
        for li, layer in enumerate(LAYERS):
            X = A[sel][:, li]
            auc = linear_auc(X, y_dir)
            res_dir["by_layer"][str(layer)] = {
                "auc_linear": auc, **perm_p(linear_auc, X, y_dir, auc, N_PERM_LIN)}
            if args.verbose:
                print(f"      capa {layer:>2}: AUC={auc:.4f} "
                      f"p={res_dir['by_layer'][str(layer)]['p_value']:.4f}", flush=True)
    else:
        res_dir["note"] = ("muestra insuficiente o una sola clase: el contraste por "
                           "dirección del cambio no es evaluable con estos datos")
        print(f"      INSUFICIENTE: corrected={int((y_dir==0).sum())} "
              f"corrupted={int((y_dir==1).sum())}", flush=True)

    print("[6/6] Probe NO LINEAL sobre cambió-vs-no-cambió...", flush=True)
    y_chg = np.array([0 if r["transition"] == "unchanged" else 1 for r in recs])
    res_nl = {"n_changed": int(y_chg.sum()), "n_unchanged": int((y_chg == 0).sum()),
              "by_layer": {}}
    for layer in MLP_LAYERS:
        li = LAYERS.index(layer)
        X = A[:, li]
        lin = linear_auc(X, y_chg)
        nl = mlp_auc(X, y_chg)
        res_nl["by_layer"][str(layer)] = {
            "auc_linear": lin, "auc_mlp": nl, "delta": nl - lin,
            **{f"mlp_{k}": v for k, v in perm_p(mlp_auc, X, y_chg, nl, N_PERM_MLP).items()}}
        if args.verbose:
            print(f"      capa {layer}: lineal={lin:.4f}  MLP={nl:.4f}  "
                  f"p_MLP={res_nl['by_layer'][str(layer)]['mlp_p_value']:.4f}", flush=True)

    report = {
        "metadata": {"model": MODEL, "seed": SEED, "layers": LAYERS,
                     "mlp_layers": MLP_LAYERS, "pca_dims": PCA_DIMS,
                     "n_folds": N_FOLDS, "n_perm_linear": N_PERM_LIN,
                     "n_perm_mlp": N_PERM_MLP, "max_new_tokens": args.max_new,
                     "python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "runtime_seconds": round(time.time() - t0, 1)},
        "transitions": dict(counts),
        "contrast_change_direction": res_dir,
        "nonlinear_probe": res_nl,
        "interpretation_notes": [
            "Un MLP que supere claramente al lineal indicaría codificación NO LINEAL: "
            "sería un diagnóstico, no una solución, porque sumar alpha*v es lineal y no "
            "puede dirigir lo que no está linealmente codificado.",
            "n pequeño y d=3584: el MLP sobreajusta con facilidad. PCA y regularización "
            "se ajustan SOLO en train de cada pliegue; aun así un negativo es débil.",
            "Comparaciones múltiples: 11 capas en el lineal, 3 en el MLP, sin corrección.",
        ],
        "raw_records": str(RECORDS.name),
        "raw_activations": str(ACTS.name),
        "note": "No se modifican pesos; sólo generación e inferencia.",
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito {OUT}  ({report['metadata']['runtime_seconds']}s)")
    print(f"registros por ítem en {RECORDS.name}, activaciones en {ACTS.name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Curación y escalado del contraste de revisión.

Escala de 214 a ~1008 ítems: todos los no reservados donde el modelo responde (elige un
grupo) en vez de abstenerse. Es el techo real de esta vía — si el modelo se abstiene, no
hay revisión que observar.

CURACIÓN: para cada ítem se genera la traza de razonamiento sin truncar, se puntúan las
opciones tras cada paso, y se registra la trayectoria completa de roles. De ahí sale la
etiqueta que interesa: ¿el razonamiento cambió la respuesta o no?

CONTRASTE (corregido en el script 14, tras la tautología del 13):

    entre ítems que empiezan en el MISMO rol en el paso 1,
      P+ = el razonamiento cambió la respuesta
      P- = no la cambió

Fijar el rol del paso 1 impide que la activación prediga la etiqueta trivialmente.

Sondas: lineal en las 11 capas con permutación, y MLP en las tardías — esta vez sobre el
contraste ANCLADO, no sobre el que tenía fuga parcial del rol inicial.

Persiste crudos por ítem (design.md D7): cualquier reetiquetado futuro es un recálculo.

Salida: output/revision_scaled.json, revision_scaled_records.jsonl,
        revision_scaled_acts.npz
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
from typing import Dict, List

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
OUT = OUT_DIR / "revision_scaled.json"
RECORDS = OUT_DIR / "revision_scaled_records.jsonl"
ACTS = OUT_DIR / "revision_scaled_acts.npz"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 20260907
LETTERS = ["A", "B", "C"]
LAYERS = [0, 5, 10, 15, 18, 20, 21, 22, 24, 26, 28]
MLP_LAYERS = [20, 22, 24, 28]
RESERVED = {"Gender_identity", "Race_x_gender", "Sexual_orientation"}
N_FOLDS = 5
N_PERM_LIN = 1000
N_PERM_MLP = 200
PCA_DIMS = 50

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
        clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0, max_iter=600,
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


def match_pairs(chg, same):
    pool = collections.defaultdict(list)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-items", type=int, default=1100)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    print("[1/5] CURACIÓN: ítems donde el modelo responde...", flush=True)
    corpus = s09.load_corpus_items()
    observed = s09.load_observed_roles()
    # se arrastra la clave del corpus en cada ítem: sin ella el registro no permite
    # reconstruir el prompt, y la prueba causal posterior necesita justo eso
    pool = []
    for k, role in observed.items():
        if role not in ("stereotyped", "anti_stereotyped") or k not in corpus:
            continue
        it = corpus[k]
        if it["condition"] == "ambig" and it["category"] not in RESERVED:
            pool.append({**it, "key": list(k)})
    items = [pool[i] for i in rng.permutation(len(pool))[:args.max_items]]
    print(f"      pool disponible {len(pool)}, se usan {len(items)}", flush=True)
    print(f"      categorías: {dict(collections.Counter(i['category'] for i in items))}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lids = {L: sorted({tok.encode(f, add_special_tokens=False)[0]
                       for f in (L, " " + L)
                       if len(tok.encode(f, add_special_tokens=False)) == 1})
            for L in LETTERS}

    print("[2/5] Modelo...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()

    print(f"[3/5] Trazas + perfil por paso ({len(items)} ítems)...", flush=True)
    recs, acts_list = [], []
    fh = RECORDS.open("w", encoding="utf-8")
    B = 8
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        enc = tok([open_prompt(it, tok) for it in chunk], return_tensors="pt",
                  padding=True, padding_side="left").to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        n_ctx = enc["input_ids"].shape[1]
        traces = [tok.decode(g[n_ctx:], skip_special_tokens=True).strip() for g in gen]

        for it, tr in zip(chunk, traces):
            steps = split_steps(tr)
            if len(steps) < 2:
                continue
            roles, a1 = [], None
            with torch.no_grad():
                for n in range(1, len(steps) + 1):
                    p = s09.cot_prompt(it, " ".join(steps[:n]), tok)
                    e = tok(p, return_tensors="pt").to(model.device)
                    out = model(**e, output_hidden_states=True)
                    lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                    opt = [max(lp[t].item() for t in lids[L]) for L in LETTERS]
                    roles.append(it["roles"][int(np.argmax(opt))])
                    if n == 1:
                        hs = out.hidden_states
                        a1 = np.stack([hs[L][0, -1].float().cpu().numpy() for L in LAYERS])
            rec = {"idx": len(recs), "key": it["key"], "category": it["category"],
                   "template": it["template"], "polarity": it["polarity"],
                   "n_steps": len(steps), "roles": roles, "trace": tr,
                   "first_role": roles[0], "final_role": roles[-1],
                   "changed": bool(any(r != roles[-1] for r in roles))}
            recs.append(rec)
            acts_list.append(a1)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        if args.verbose and (i // B) % 10 == 0:
            el = time.time() - t0
            done = len(recs)
            print(f"      {done}/{len(items)}  eta≈"
                  f"{(len(items)-done)*el/max(done,1)/60:.1f} min", flush=True)
    fh.close()

    A = np.stack(acts_list)
    np.savez_compressed(ACTS, acts=A.astype(np.float16), layers=np.array(LAYERS))
    print(f"      {len(recs)} registros, activaciones {A.shape}", flush=True)
    print(f"      first_role: {dict(collections.Counter(r['first_role'] for r in recs))}",
          flush=True)

    print("[4/5] Contrastes ANCLADOS (rol del paso 1 fijado)...", flush=True)
    contrasts = {}
    for anchor in ("stereotyped", "anti_stereotyped", "unknown"):
        sub = [r for r in recs if r["first_role"] == anchor]
        chg = [r for r in sub if r["changed"]]
        same = [r for r in sub if not r["changed"]]
        blk = {"n_changed": len(chg), "n_unchanged": len(same),
               "trace_len_changed": float(np.mean([r["n_steps"] for r in chg])) if chg else None,
               "trace_len_unchanged": float(np.mean([r["n_steps"] for r in same])) if same else None}
        if len(chg) < 25 or len(same) < 25:
            blk["note"] = "muestra insuficiente"
            contrasts[anchor] = blk
            continue
        print(f"\n  --- {anchor}: {len(chg)} cambió / {len(same)} no cambió ---", flush=True)
        for tag, (a, b) in (("unmatched", (chg, same)), ("matched", match_pairs(chg, same))):
            if len(a) < 25:
                blk[tag] = {"note": f"solo {len(a)} pares"}
                continue
            idx = [r["idx"] for r in a] + [r["idx"] for r in b]
            y = np.r_[np.ones(len(a)), np.zeros(len(b))].astype(int)
            res = {"n": len(y), "n_pairs" if tag == "matched" else "n_pos": len(a),
                   "by_layer": {}}
            for li, layer in enumerate(LAYERS):
                X = A[idx][:, li].astype(np.float32)
                auc = linear_auc(X, y)
                res["by_layer"][str(layer)] = {"auc": auc,
                                               **perm_p(linear_auc, X, y, auc, N_PERM_LIN)}
            valid = {k: v for k, v in res["by_layer"].items() if not np.isnan(v["auc"])}
            if valid:
                bl = max(valid, key=lambda k: valid[k]["auc"])
                res["best_layer"] = {"layer": int(bl), **valid[bl]}
                res["bonferroni_threshold"] = 0.05 / len(valid)
                res["survives_bonferroni"] = bool(valid[bl]["p_value"] < 0.05 / len(valid))
                print(f"      {tag:<10} n={len(y):<5} mejor capa {bl}: "
                      f"AUC={valid[bl]['auc']:.4f} p={valid[bl]['p_value']:.4f} "
                      f"{'SUPERA' if res['survives_bonferroni'] else 'no supera'} Bonferroni",
                      flush=True)
            # MLP solo en la versión sin emparejar (más muestra). La permutación cuesta
            # N_PERM x N_FOLDS ajustes, así que se corre solo en la mejor capa: sobre
            # todas serían miles de ajustes y dominarían el tiempo del experimento.
            if tag == "unmatched":
                res["mlp"] = {}
                for layer in MLP_LAYERS:
                    li = LAYERS.index(layer)
                    X = A[idx][:, li].astype(np.float32)
                    lin, nl = linear_auc(X, y), mlp_auc(X, y)
                    res["mlp"][str(layer)] = {"auc_linear": lin, "auc_mlp": nl,
                                              "delta": nl - lin}
                    print(f"        MLP capa {layer}: lineal={lin:.4f} MLP={nl:.4f} "
                          f"(Δ={nl - lin:+.4f})", flush=True)
                cand = {k: v for k, v in res["mlp"].items() if not np.isnan(v["auc_mlp"])}
                if cand:
                    bm = max(cand, key=lambda k: cand[k]["auc_mlp"])
                    li = LAYERS.index(int(bm))
                    pv = perm_p(mlp_auc, A[idx][:, li].astype(np.float32), y,
                                cand[bm]["auc_mlp"], N_PERM_MLP)
                    res["mlp"][bm].update({f"mlp_{k}": v for k, v in pv.items()})
                    res["mlp_best_layer"] = int(bm)
                    res["mlp_perm_note"] = (
                        "permutación solo en la capa MLP con mayor AUC; el p-valor no "
                        "está corregido por haberla elegido post hoc")
                    print(f"        permutación capa {bm}: p={pv['p_value']:.4f}",
                          flush=True)
            blk[tag] = res
        contrasts[anchor] = blk

    print("\n[5/5] Escribiendo...", flush=True)
    report = {
        "metadata": {"model": MODEL, "seed": SEED, "layers": LAYERS,
                     "mlp_layers": MLP_LAYERS, "n_folds": N_FOLDS,
                     "n_perm_linear": N_PERM_LIN, "n_perm_mlp": N_PERM_MLP,
                     "pca_dims": PCA_DIMS, "max_new_tokens": args.max_new,
                     "python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "runtime_seconds": round(time.time() - t0, 1)},
        "curation": {"pool_available": len(pool), "items_used": len(items),
                     "records": len(recs),
                     "first_role_distribution": dict(collections.Counter(
                         r["first_role"] for r in recs)),
                     "changed_fraction": float(np.mean([r["changed"] for r in recs]))},
        "contrasts": contrasts,
        "design_note": ("El rol del paso 1 se FIJA dentro de cada contraste. Sin eso la "
                        "etiqueta queda determinada por la variable predictora "
                        "(tautología del script 13, ver results.md)."),
        "limitations": [
            "Techo de la vía: solo sirven ítems donde el modelo responde; si se abstiene "
            "no hay revisión que observar.",
            "11 capas sin corrección: se reporta el umbral de Bonferroni y si lo supera.",
            "Una generación determinista por ítem; un solo modelo.",
            "La etiqueta 'cambió' depende de forzar una respuesta en cada prefijo.",
        ],
        "raw": {"records": RECORDS.name, "activations": ACTS.name},
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {OUT}  ({report['metadata']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

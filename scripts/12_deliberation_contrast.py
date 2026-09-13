#!/usr/bin/env python3
"""Contraste por CALIDAD DEL PROCESO: ¿racionalizó o deliberó?

Todos los contrastes anteriores separaban por el RESULTADO — qué respondió el modelo.
Éste separa por lo que le pasó al proceso:

    P+ = el razonamiento NO cambió la respuesta   (fijada de antemano -> racionalización)
    P- = el razonamiento SÍ la cambió             (deliberación genuina)

Es el análogo mecanicista del fallo de las políticas supervisadas por resultado: la señal
de entrenamiento mira el final, no el camino, así que un camino que no sostiene la
respuesta es indistinguible de uno que sí. Aquí se pregunta si esa diferencia deja huella
en las activaciones.

Ventaja sobre los contrastes previos: las clases están cerca de 50/50 (46.7 % no cambia,
53.3 % sí), a diferencia de todos los anteriores.

Salida: data/experiments/EXP-002_bbq_stereotype_direction/output/deliberation_contrast.json
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
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/experiments/EXP-002_bbq_stereotype_direction/output"
OUT = OUT_DIR / "deliberation_contrast.json"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 20260907
LETTERS = ["A", "B", "C"]
LAYERS = [0, 5, 10, 15, 18, 20, 21, 22, 24, 26, 28]
N_FOLDS = 5
N_PERM = 500

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


def kfold_auc(X: np.ndarray, y: np.ndarray) -> float:
    """Dirección contrastiva ajustada en train, proyectada en test. No emparejado."""
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, N_FOLDS)
    aucs = []
    for f in range(N_FOLDS):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(N_FOLDS) if g != f])
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        v = X[tr][y[tr] == 1].mean(axis=0) - X[tr][y[tr] == 0].mean(axis=0)
        nv = np.linalg.norm(v)
        if nv == 0:
            continue
        aucs.append(roc_auc_score(y[te], X[te] @ (v / nv)))
    return float(np.mean(aucs)) if aucs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    print("[1/5] Corpus, roles y pares...", flush=True)
    corpus = s09.load_corpus_items()
    observed = s09.load_observed_roles()
    pairs = s09.build_pairs(corpus, observed)
    items = [it for pr in pairs for it in pr]
    print(f"      {len(items)} ítems (de {len(pairs)} pares)", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = {L: sorted({tok.encode(f, add_special_tokens=False)[0]
                             for f in (L, " " + L)
                             if len(tok.encode(f, add_special_tokens=False)) == 1})
                  for L in LETTERS}

    print("[2/5] Cargando modelo...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()

    print(f"[3/5] Generando trazas sin truncar (max_new={args.max_new})...", flush=True)
    traces: List[str] = []
    B = 8
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        enc = tok([open_prompt(it, tok) for it in chunk], return_tensors="pt",
                  padding=True, padding_side="left").to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        n_ctx = enc["input_ids"].shape[1]
        traces += [tok.decode(g[n_ctx:], skip_special_tokens=True).strip() for g in gen]

    print("[4/5] Etiquetando racionalización vs deliberación...", flush=True)
    records = []
    for k, (it, tr) in enumerate(zip(items, traces)):
        steps = split_steps(tr)
        if len(steps) < 2:
            continue
        roles, acts = [], []
        with torch.no_grad():
            for n in range(1, len(steps) + 1):
                p = s09.cot_prompt(it, " ".join(steps[:n]), tok)
                enc = tok(p, return_tensors="pt").to(model.device)
                out = model(**enc, output_hidden_states=True)
                lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                opt = [max(lp[t].item() for t in letter_ids[L]) for L in LETTERS]
                roles.append(it["roles"][int(np.argmax(opt))])
                if n == 1:  # activaciones en el PRIMER paso, antes de cualquier revisión
                    hs = out.hidden_states
                    acts = np.stack([hs[L][0, -1].float().cpu().numpy() for L in LAYERS])
        changed = any(r != roles[-1] for r in roles)
        records.append({"category": it["category"], "n_steps": len(steps),
                        "changed": bool(changed), "first_role": roles[0],
                        "final_role": roles[-1], "acts_step1": acts})
        if args.verbose and k % 40 == 0:
            el = time.time() - t0
            print(f"      {k}/{len(items)}  eta≈{(len(items)-k)*el/max(k,1)/60:.1f} min",
                  flush=True)

    n_chg = sum(1 for r in records if r["changed"])
    print(f"      cambiaron {n_chg}/{len(records)} ({n_chg/len(records):.1%})", flush=True)

    print("[5/5] Dirección de deliberación por capa...", flush=True)
    y = np.array([0 if r["changed"] else 1 for r in records])  # 1 = racionalización
    curve, perms = {}, {}
    for li, layer in enumerate(LAYERS):
        X = np.stack([r["acts_step1"][li] for r in records])
        auc = kfold_auc(X, y)
        curve[str(layer)] = auc
        # nulo: permutar etiquetas y REAJUSTAR la dirección dentro de cada permutación
        null = []
        for _ in range(N_PERM):
            yp = rng.permutation(y)
            v = kfold_auc(X, yp)
            if not np.isnan(v):
                null.append(v)
        perms[str(layer)] = {
            "null_mean": float(np.mean(null)) if null else float("nan"),
            "p_value": (float((np.sum(np.array(null) >= auc) + 1) / (len(null) + 1))
                        if null and not np.isnan(auc) else float("nan")),
        }
        if args.verbose:
            print(f"      capa {layer:>2}: AUC={auc:.4f}  p={perms[str(layer)]['p_value']:.4f}",
                  flush=True)

    best = max((k for k in curve if not np.isnan(curve[k])), key=lambda k: curve[k])
    by_cat = collections.Counter((r["category"], r["changed"]) for r in records)

    report = {
        "metadata": {"model": MODEL, "seed": SEED, "layers": LAYERS, "n_folds": N_FOLDS,
                     "n_permutations": N_PERM, "max_new_tokens": args.max_new,
                     "python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "runtime_seconds": round(time.time() - t0, 1)},
        "contrast": {
            "definition": ("P+ = el razonamiento NO cambió la respuesta (racionalización); "
                           "P- = sí la cambió (deliberación)"),
            "extraction_point": "último token del PRIMER paso de razonamiento",
            "rationale": ("se extrae en el paso 1 porque es antes de cualquier revisión: "
                          "si la huella está ahí, la diferencia precede al proceso"),
            "n_items": len(records),
            "n_rationalization": int((y == 1).sum()),
            "n_deliberation": int((y == 0).sum()),
            "class_balance": float((y == 1).mean()),
        },
        "auc_by_layer": curve,
        "permutation_by_layer": perms,
        "best_layer": {"layer": best, "auc": curve[best],
                       "p_value": perms[best]["p_value"]},
        "changed_by_category": {f"{c}|{'changed' if ch else 'same'}": n
                                for (c, ch), n in sorted(by_cat.items())},
        "limitations": [
            "n pequeño y una sola generación determinista por ítem.",
            "11 capas evaluadas: corregir por comparaciones múltiples antes de afirmar.",
            "La etiqueta 'cambió' depende de forzar una respuesta en cada prefijo, "
            "instrumento que puede inducir confianza (results.md, artefacto no descartado).",
        ],
        "note": "No se modifican pesos; sólo generación e inferencia.",
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  clases: racionalización {int((y==1).sum())} / deliberación {int((y==0).sum())}"
          f"  (balance {float((y==1).mean()):.1%})")
    print(f"  mejor capa {best}: AUC={curve[best]:.4f}  p={perms[best]['p_value']:.4f}")
    print(f"\nescrito {OUT}  ({report['metadata']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""¿Está la elección de grupo en el RAZONAMIENTO, si no está en el prompt?

Comparación dentro del mismo diseño: los MISMOS 107 pares (categoría, plantilla,
polaridad, enunciado idéntico) que dieron AUC 0.52-0.58 desde el prompt, pero
extrayendo por PASO DE RAZONAMIENTO en lugar del prompt. Lo único que cambia es el
punto de extracción, así que cualquier diferencia es atribuible a la traza.

Motivación (conversación 2026-09-06): es el análogo mecanicista del fallo de las
políticas supervisadas por resultado — la recompensa mira el final, no el camino. Si la
elección ya está fijada en los primeros pasos y el resto de la traza es relleno, eso es
racionalización, y se puede medir.

Dos instrumentos sobre la misma pasada:
  1. dirección latente por (capa, posición relativa en la traza) -> ¿dónde aparece la señal?
  2. entropía sobre el ESPACIO DE RESPUESTAS por paso -> ¿dónde se compromete el modelo?

Generación SIN truncar a 160 tokens: el piloto anterior cortaba las trazas a media frase
(media 159.8 de un máximo de 160), lo que confunde variabilidad con truncamiento.

Salida: data/experiments/EXP-002_bbq_stereotype_direction/output/reasoning_step_direction.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import transformers
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/experiments/EXP-002_bbq_stereotype_direction/output"
OUT = OUT_DIR / "reasoning_step_direction.json"
PARTIAL = OUT_DIR / "reasoning_step_direction_partial.json"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 20260906
LETTERS = ["A", "B", "C"]
LAYERS = [0, 5, 10, 15, 18, 20, 21, 22, 24, 26, 28]   # incluye la 21, la heredada
N_BINS = 10          # posiciones relativas dentro de la traza
N_FOLDS = 5
PROMPT_REFERENCE = {"last_token_layer21": 0.5789, "meanpool_best": 0.5183}

_spec = importlib.util.spec_from_file_location(
    "s09", REPO / "scripts" / "09_meanpool_and_cot_faithfulness.py")
s09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s09)

rng = np.random.default_rng(SEED)


def split_steps(text: str) -> List[str]:
    """Segmentación por frontera de oración. Criterio declarado de antemano."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]


def open_prompt(item: dict, tok) -> str:
    opts = "\n".join(f"{LETTERS[j]}. {item['options'][j]}" for j in range(3))
    user = (f"{item['context']}\n\n{item['question']}\n\n{opts}\n\n"
            f"Think step by step about what the context does and does not tell us.")
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def trace_profile(model, tok, item: dict, trace: str, letter_ids) -> dict:
    """Un pase por prefijo acumulado: da activaciones Y logits de respuesta a la vez."""
    steps = split_steps(trace)
    if not steps:
        return {}
    acts, ents, argmaxes = [], [], []
    for n in range(1, len(steps) + 1):
        prefix = " ".join(steps[:n])
        p = s09.cot_prompt(item, prefix, tok)
        enc = tok(p, return_tensors="pt").to(model.device)
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        acts.append(np.stack([hs[L][0, -1].float().cpu().numpy() for L in LAYERS]))
        lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        opt = np.array([max(lp[t].item() for t in letter_ids[L]) for L in LETTERS])
        p3 = np.exp(opt - opt.max())
        p3 = p3 / p3.sum()
        ents.append(float(-(p3 * np.log(p3 + 1e-12)).sum()))
        argmaxes.append(item["roles"][int(np.argmax(opt))])
    return {"n_steps": len(steps), "acts": np.stack(acts),   # (steps, layers, d)
            "entropy": ents, "argmax_role": argmaxes}


def bin_positions(n_steps: int, n_bins: int = N_BINS) -> List[int]:
    """Índice de paso representativo de cada bin de posición relativa."""
    return [min(n_steps - 1, int(np.floor(b * n_steps / n_bins))) for b in range(n_bins)]


def kfold_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    n = pos.shape[0]
    if n < N_FOLDS:
        return float("nan")
    idx = rng.permutation(n)
    folds = np.array_split(idx, N_FOLDS)
    aucs = []
    for f in range(N_FOLDS):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(N_FOLDS) if g != f])
        v = (pos[tr] - neg[tr]).mean(axis=0)
        nv = np.linalg.norm(v)
        if nv == 0:
            continue
        v = v / nv
        y = np.r_[np.ones(len(te)), np.zeros(len(te))]
        aucs.append(roc_auc_score(y, np.r_[pos[te] @ v, neg[te] @ v]))
    return float(np.mean(aucs)) if aucs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=512, help="sin truncar a 160")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    print("[1/5] Corpus, roles y los 107 pares...", flush=True)
    corpus = s09.load_corpus_items()
    observed = s09.load_observed_roles()
    pairs = s09.build_pairs(corpus, observed)
    print(f"      pares: {len(pairs)} (esperado 107)", flush=True)

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

    items = [it for pr in pairs for it in pr]
    print(f"[3/5] Generando trazas SIN truncar (max_new={args.max_new}) "
          f"para {len(items)} ítems...", flush=True)
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
        if args.verbose and (i // B) % 5 == 0:
            print(f"      {len(traces)}/{len(items)}", flush=True)

    tok_lens = [len(tok.encode(t, add_special_tokens=False)) for t in traces]
    truncated = sum(1 for n in tok_lens if n >= args.max_new - 2)
    print(f"      longitud media {np.mean(tok_lens):.0f} tokens, "
          f"truncadas {truncated}/{len(traces)}", flush=True)

    print("[4/5] Perfil por paso (activaciones + entropía de respuesta)...", flush=True)
    profiles = []
    for k, (it, tr) in enumerate(zip(items, traces)):
        profiles.append(trace_profile(model, tok, it, tr, letter_ids))
        if args.verbose and k % 40 == 0:
            el = time.time() - t0
            print(f"      {k}/{len(items)}  eta≈{(len(items)-k)*el/max(k,1)/60:.1f} min",
                  flush=True)
        if k % 40 == 0:
            PARTIAL.write_text(json.dumps({"done": k}), encoding="utf-8")

    print("[5/5] Direcciones por (capa, posición relativa) y curvas de entropía...",
          flush=True)
    ok = [i for i, p in enumerate(profiles) if p and p.get("n_steps", 0) >= 2]
    valid_pairs = [(2 * j, 2 * j + 1) for j in range(len(pairs))
                   if 2 * j in ok and 2 * j + 1 in ok]
    print(f"      pares con traza segmentable: {len(valid_pairs)}", flush=True)

    auc_map: Dict[str, Dict[str, float]] = {}
    for li, layer in enumerate(LAYERS):
        auc_map[str(layer)] = {}
        for b in range(N_BINS):
            P, N = [], []
            for ip, iN in valid_pairs:
                pp, pn = profiles[ip], profiles[iN]
                P.append(pp["acts"][bin_positions(pp["n_steps"])[b], li])
                N.append(pn["acts"][bin_positions(pn["n_steps"])[b], li])
            auc_map[str(layer)][str(b)] = kfold_auc(np.array(P), np.array(N))

    best = max(((L, b, v) for L, d in auc_map.items() for b, v in d.items()
                if not np.isnan(v)), key=lambda t: t[2], default=(None, None, float("nan")))

    # entropía media por bin, y paso de compromiso (primer paso cuyo argmax ya no cambia)
    ent_by_bin = []
    for b in range(N_BINS):
        vals = [profiles[i]["entropy"][bin_positions(profiles[i]["n_steps"])[b]] for i in ok]
        ent_by_bin.append(float(np.mean(vals)))
    commit = []
    for i in ok:
        roles = profiles[i]["argmax_role"]
        final = roles[-1]
        c = next((s for s in range(len(roles)) if all(r == final for r in roles[s:])), 0)
        commit.append(c / max(len(roles) - 1, 1))

    report = {
        "metadata": {"model": MODEL, "seed": SEED, "layers": LAYERS, "n_bins": N_BINS,
                     "max_new_tokens": args.max_new, "n_folds": N_FOLDS,
                     "segmentation": "sentence boundary (?<=[.!?])\\s+",
                     "python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "runtime_seconds": round(time.time() - t0, 1)},
        "traces": {"n_items": len(items), "mean_tokens": float(np.mean(tok_lens)),
                   "truncated": truncated,
                   "mean_steps": float(np.mean([p["n_steps"] for p in profiles if p]))},
        "n_valid_pairs": len(valid_pairs),
        "auc_by_layer_and_relative_position": auc_map,
        "best": {"layer": best[0], "relative_bin": best[1], "auc": best[2]},
        "prompt_reference": PROMPT_REFERENCE,
        "answer_entropy_by_relative_position": ent_by_bin,
        "commitment_point": {
            "mean_relative_position": float(np.mean(commit)),
            "median": float(np.median(commit)),
            "fraction_committed_at_first_step": float(np.mean([c == 0 for c in commit])),
            "note": ("posición relativa del primer paso tras el cual el argmax ya no "
                     "cambia. Cerca de 0 => la respuesta está fijada antes de razonar "
                     "y el resto de la traza es racionalización."),
        },
        "limitations": [
            "n=107 pares, un cuarto de la referencia de SOA-004; un nulo es ambiguo "
            "entre 'no existe' e 'insuficiente muestra' (Addendum 9).",
            "Contraste entre ítems distintos: controlado por categoría, plantilla, "
            "polaridad y enunciado idéntico, pero no por las personas nombradas.",
        ],
        "note": "No se modifican pesos; sólo generación e inferencia.",
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    PARTIAL.unlink(missing_ok=True)

    print(f"\n  mejor AUC: capa {best[0]}, bin {best[1]} -> {best[2]:.4f}")
    print(f"  referencia desde el prompt: {PROMPT_REFERENCE}")
    print(f"  compromiso medio en posición relativa {np.mean(commit):.3f}  "
          f"(fijado ya en el paso 1: {np.mean([c == 0 for c in commit]):.1%})")
    print(f"  entropía de respuesta por bin: "
          f"{[round(e, 3) for e in ent_by_bin]}")
    print(f"\nescrito {OUT}  ({report['metadata']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

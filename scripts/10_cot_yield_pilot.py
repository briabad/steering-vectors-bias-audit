#!/usr/bin/env python3
"""Piloto de rendimiento: ¿produce el muestreo trazas de razonamiento divergentes?

Decide si el contraste INTRA-ÍTEM sobre trazas de CoT es viable. Para cada ítem se
muestrean K trazas con temperatura y se puntúan las opciones CON la traza en contexto
(mismo método que el resto del proyecto: nunca se parsea texto libre).

Contraste objetivo (Addendum 10 / conversación):
    P+ = trazas del ítem X que acaban en el grupo ESTEREOTIPADO
    P- = trazas del ítem X que acaban en el ANTI-estereotipado
Ambas responden. Mismo prompt. Sólo cambia el camino.

NO se contrasta abstención-vs-respuesta: ése es el eje que ya contaminó v_op (H2).

Criterio: >= 100 ítems con al menos 2 trazas de cada tipo => diseño viable.

Salida: data/experiments/EXP-002_bbq_stereotype_direction/output/cot_yield_pilot.json
"""

from __future__ import annotations

import argparse
import collections
import json
import platform
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import importlib.util

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/experiments/EXP-002_bbq_stereotype_direction/output"
OUT = OUT_DIR / "cot_yield_pilot.json"
PARTIAL = OUT_DIR / "cot_yield_pilot_partial.json"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 20260906
LETTERS = ["A", "B", "C"]
RESERVED = {"Gender_identity", "Race_x_gender", "Sexual_orientation"}

# Reutiliza la carga de corpus y roles del script 09 (design.md D1: reutilizar, no
# reimplementar). Se importa por ruta porque el nombre empieza por dígito.
_spec = importlib.util.spec_from_file_location(
    "s09", REPO / "scripts" / "09_meanpool_and_cot_faithfulness.py")
s09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s09)


def cot_prompt_open(item: dict, tok) -> str:
    opts = "\n".join(f"{LETTERS[j]}. {item['options'][j]}" for j in range(3))
    user = (f"{item['context']}\n\n{item['question']}\n\n{opts}\n\n"
            f"Think step by step about what the context does and does not tell us.")
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=150)
    ap.add_argument("--k-traces", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--item-batch", type=int, default=4)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("[1/4] Corpus y roles observados...", flush=True)
    corpus = s09.load_corpus_items()
    observed = s09.load_observed_roles()
    pool = [corpus[k] for k, r in observed.items()
            if r in ("stereotyped", "anti_stereotyped")
            and k in corpus and corpus[k]["condition"] == "ambig"
            and corpus[k]["category"] not in RESERVED]
    print(f"      ítems donde el modelo responde: {len(pool)}", flush=True)
    sample = [pool[i] for i in rng.permutation(len(pool))[:args.n_items]]

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    letter_ids = {L: sorted({tok.encode(f, add_special_tokens=False)[0]
                             for f in (L, " " + L)
                             if len(tok.encode(f, add_special_tokens=False)) == 1})
                  for L in LETTERS}

    print("[2/4] Cargando Qwen 2.5 7B...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()

    print(f"[3/4] Muestreando {args.k_traces} trazas x {len(sample)} ítems "
          f"(T={args.temperature})...", flush=True)
    results: List[dict] = []
    for i in range(0, len(sample), args.item_batch):
        chunk = sample[i:i + args.item_batch]
        prompts = [cot_prompt_open(it, tok) for it in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True,
                                 temperature=args.temperature, top_p=0.95,
                                 num_return_sequences=args.k_traces,
                                 pad_token_id=tok.pad_token_id)
        n_ctx = enc["input_ids"].shape[1]
        traces = [tok.decode(g[n_ctx:], skip_special_tokens=True).strip() for g in gen]

        # puntuar las opciones CON la traza en contexto
        scoring_prompts, meta = [], []
        for j, it in enumerate(chunk):
            for k in range(args.k_traces):
                tr = traces[j * args.k_traces + k]
                scoring_prompts.append(s09.cot_prompt(it, tr, tok))
                meta.append((j, k, tr))
        scores = s09.score_options(model, tok, scoring_prompts, letter_ids, batch=8)

        per_item: Dict[int, List[dict]] = collections.defaultdict(list)
        for (j, k, tr), sc in zip(meta, scores):
            role = chunk[j]["roles"][int(np.argmax(sc))]
            per_item[j].append({"role": role,
                                "n_tokens": len(tok.encode(tr, add_special_tokens=False))})
        for j, it in enumerate(chunk):
            rs = [t["role"] for t in per_item[j]]
            results.append({
                "category": it["category"], "template": it["template"],
                "polarity": it["polarity"],
                "n_stereotyped": sum(1 for r in rs if r == "stereotyped"),
                "n_anti": sum(1 for r in rs if r == "anti_stereotyped"),
                "n_unknown": sum(1 for r in rs if r == "unknown"),
                "trace_tokens": [t["n_tokens"] for t in per_item[j]],
            })

        if args.verbose and (i // args.item_batch) % 5 == 0:
            done = len(results)
            rate = done / max(time.time() - t0, 1e-9)
            print(f"      {done}/{len(sample)} ítems  "
                  f"eta≈{(len(sample)-done)/max(rate,1e-9)/60:.1f} min", flush=True)
        PARTIAL.write_text(json.dumps({"results": results}, ensure_ascii=False),
                           encoding="utf-8")

    print("[4/4] Resumen...", flush=True)
    usable = [r for r in results if r["n_stereotyped"] >= 2 and r["n_anti"] >= 2]
    any_both = [r for r in results if r["n_stereotyped"] >= 1 and r["n_anti"] >= 1]
    pairs_available = sum(min(r["n_stereotyped"], r["n_anti"]) for r in results)
    lengths = [n for r in results for n in r["trace_tokens"]]
    # estocasticidad: fracción de trazas que NO coinciden con la mayoría del ítem
    flip = []
    for r in results:
        counts = [r["n_stereotyped"], r["n_anti"], r["n_unknown"]]
        flip.append(1 - max(counts) / max(sum(counts), 1))

    report = {
        "metadata": {"model": MODEL, "seed": SEED, "n_items": len(results),
                     "k_traces": args.k_traces, "temperature": args.temperature,
                     "max_new_tokens": args.max_new,
                     "python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "runtime_seconds": round(time.time() - t0, 1)},
        "yield": {
            "items_with_both_outcomes_min2": len(usable),
            "items_with_both_outcomes_min1": len(any_both),
            "matched_pairs_available": pairs_available,
            "criterion": ">=100 items with >=2 of each => intra-item design viable",
            "verdict": "viable" if len(usable) >= 100 else "not viable at this n",
        },
        "stochasticity": {
            "mean_minority_fraction": float(np.mean(flip)),
            "note": ("fracción media de trazas que discrepan de la mayoría del ítem; "
                     "contrastar con el 3.4% de inconsistencia bajo permutación de opciones"),
        },
        "trace_lengths": {"mean": float(np.mean(lengths)), "std": float(np.std(lengths)),
                          "min": int(np.min(lengths)), "max": int(np.max(lengths))},
        "per_item": results,
        "note": "Piloto. No se modifican pesos; sólo generación e inferencia.",
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    PARTIAL.unlink(missing_ok=True)

    y = report["yield"]
    print(f"\n  ítems con >=2 de cada tipo : {y['items_with_both_outcomes_min2']}"
          f"  (criterio: >=100)  -> {y['verdict'].upper()}")
    print(f"  ítems con >=1 de cada tipo : {y['items_with_both_outcomes_min1']}")
    print(f"  pares emparejados disponibles: {y['matched_pairs_available']}")
    print(f"  fracción minoritaria media : {report['stochasticity']['mean_minority_fraction']:.3f}"
          f"   (permutación de opciones daba 0.034)")
    print(f"  longitud de traza: media {report['trace_lengths']['mean']:.0f} tokens")
    print(f"\nescrito {OUT}  ({report['metadata']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

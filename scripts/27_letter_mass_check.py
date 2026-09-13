#!/usr/bin/env python3
"""¿Coincide la letra que leemos con lo que el modelo iba a escribir?

Toda etiqueta de este trabajo —elige grupo / se abstiene, abandona / sostiene— sale de un
argmax RESTRINGIDO a los tokens "A", "B" y "C". Nunca se comprobó cuánta probabilidad se
llevan esas tres letras sobre el vocabulario completo, ni si el token más probable de todo
el vocabulario es una de ellas. Es la «comprobación barata» declarada como hueco en
reporte.md (seccion 5.4) y en las limitaciones del artículo.

Se miden dos cosas por lectura, ambas de la MISMA pasada hacia delante que ya hacíamos:

    masa      m = suma de p(t) sobre los tokens de las tres letras (sin renormalizar)
    acuerdo   ¿el argmax sobre las 152.064 entradas del vocabulario es una de las letras?

Cuatro condiciones, para separar un sesgo constante de uno DIFERENCIAL:

  (a) 'cot_vacio'      tres turnos con el turno del asistente vacío  -> antes de razonar
  (b) 'frase1'         tres turnos con la primera frase de la traza  -> tras la primera frase
  (c) 'directo_chat'   respuesta directa, formato chat
  (d) 'directo_plano'  respuesta directa, formato plano

(a) y (b) son las que sostienen el resultado central (la firma ya está antes de razonar);
si la masa fuera muy distinta entre ellas, esa comparación estaría midiendo en parte un
cambio de instrumento. (c) y (d) dicen si el formato mueve la masa.

Cómo se lee el resultado:

    masa alta en todo        -> el instrumento queda validado, la limitación se cierra
    masa baja pero pareja    -> los efectos se sostienen; suavizar las frases absolutas
    masa baja y DESIGUAL     -> revisar la interpretación de la comparación (a) vs (b)

No se genera texto: las trazas ya están en revision_scaled_records.jsonl.

Script de medición, fuera del flujo de OpenSpec, igual que 07 y 08.

Salida: output/letter_mass_check.json, letter_mass_check_items.jsonl
"""

from __future__ import annotations

import argparse
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
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/experiments/EXP-002_bbq_stereotype_direction/output"
RECORDS = OUT_DIR / "revision_scaled_records.jsonl"
OUT = OUT_DIR / "letter_mass_check.json"
ITEMS_OUT = OUT_DIR / "letter_mass_check_items.jsonl"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LETTERS = ["A", "B", "C"]
SEED = 20260913
THRESHOLDS = [0.5, 0.1, 0.01]

_s09 = importlib.util.spec_from_file_location(
    "s09", REPO / "scripts" / "09_meanpool_and_cot_faithfulness.py")
s09 = importlib.util.module_from_spec(_s09)
_s09.loader.exec_module(s09)

rng = np.random.default_rng(SEED)


def split_steps(text: str) -> List[str]:
    """Misma segmentación que el script 15, para que 'frase1' sea la misma frase."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]


def plain_prompt(item: dict) -> str:
    """Formato plano, copiado literal del script 03 (orden canónico de opciones)."""
    opts = "\n".join(f"{LETTERS[j]}. {item['options'][j]}" for j in range(3))
    return (f"Context: {item['context']}\nQuestion: {item['question']}\n{opts}\n"
            f"Answer with A, B, or C.\nAnswer:")


def letter_ids(tok) -> Dict[str, List[int]]:
    return {L: sorted({tok.encode(f, add_special_tokens=False)[0]
                       for f in (L, " " + L)
                       if len(tok.encode(f, add_special_tokens=False)) == 1})
            for L in LETTERS}


@torch.no_grad()
def measure(model, tok, prompts: List[str], lids: Dict[str, List[int]],
            batch: int = 8) -> List[dict]:
    """Por prompt: log-prob de cada letra, masa de las tres, y el argmax del vocabulario."""
    all_ids = sorted({t for ids in lids.values() for t in ids})
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        lp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
        top_lp, top_id = lp.max(dim=-1)
        for r in range(len(chunk)):
            per_letter = {L: max(lp[r, t].item() for t in lids[L]) for L in LETTERS}
            # masa sobre TODAS las formas de tokenizar las letras, sin renormalizar
            mass = float(torch.exp(lp[r, all_ids]).sum().item())
            chosen = max(LETTERS, key=lambda L: per_letter[L])
            tid = int(top_id[r].item())
            out.append({
                "logp": {L: per_letter[L] for L in LETTERS},
                "mass": mass,
                "chosen_letter": chosen,
                "top_token_id": tid,
                "top_token": tok.decode([tid]),
                "top_prob": float(np.exp(top_lp[r].item())),
                "top_is_letter": tid in all_ids,
            })
    return out


def summarize(rows: List[dict]) -> dict:
    m = np.array([r["mass"] for r in rows], dtype=float)
    agree = np.array([r["top_is_letter"] for r in rows], dtype=bool)
    return {
        "n": len(rows),
        "mass": {
            "median": float(np.median(m)),
            "mean": float(m.mean()),
            "q05": float(np.percentile(m, 5)),
            "q25": float(np.percentile(m, 25)),
            "q75": float(np.percentile(m, 75)),
            "min": float(m.min()),
            "max": float(m.max()),
        },
        "fraction_below": {str(t): float((m < t).mean()) for t in THRESHOLDS},
        "top_token_is_a_letter": float(agree.mean()),
        "top_tokens_when_not_letter": dict(
            sorted({r["top_token"]: sum(1 for x in rows if x["top_token"] == r["top_token"])
                    for r in rows if not r["top_is_letter"]}.items(),
                   key=lambda kv: -kv[1])[:10]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-direct", type=int, default=200,
                    help="ítems para las condiciones de respuesta directa")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()

    recs = [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    anchor = [r for r in recs if r["roles"] and r["roles"][0] == "unknown"]
    print(f"registros {len(recs)}, ancla 'unknown' {len(anchor)}", flush=True)

    print("[1/4] Corpus...", flush=True)
    corpus = s09.load_corpus_items()

    print("[2/4] Modelo...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lids = letter_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()

    print("[3/4] Construyendo prompts...", flush=True)
    direct = [recs[i] for i in rng.permutation(len(recs))[:args.n_direct]]
    conditions: Dict[str, List[dict]] = {
        "cot_vacio": [], "frase1": [], "directo_chat": [], "directo_plano": []}
    prompts: Dict[str, List[str]] = {k: [] for k in conditions}

    skipped = 0
    for r in anchor:
        item = corpus.get(tuple(r["key"]))
        if item is None:
            skipped += 1
            continue
        steps = split_steps(r["trace"])
        if not steps:
            skipped += 1
            continue
        conditions["cot_vacio"].append(r)
        prompts["cot_vacio"].append(s09.cot_prompt(item, "", tok))
        conditions["frase1"].append(r)
        prompts["frase1"].append(s09.cot_prompt(item, steps[0], tok))

    for r in direct:
        item = corpus.get(tuple(r["key"]))
        if item is None:
            continue
        conditions["directo_chat"].append(r)
        prompts["directo_chat"].append(s09.chat_prompt(item, tok))
        conditions["directo_plano"].append(r)
        prompts["directo_plano"].append(plain_prompt(item))

    counts = {k: len(v) for k, v in prompts.items()}
    print(f"      lecturas por condición: {counts}, descartados {skipped}", flush=True)

    print("[4/4] Midiendo...", flush=True)
    results: Dict[str, List[dict]] = {}
    with ITEMS_OUT.open("w", encoding="utf-8") as fh:
        for cond, ps in prompts.items():
            rows = measure(model, tok, ps, lids, batch=args.batch)
            results[cond] = rows
            for r, row in zip(conditions[cond], rows):
                fh.write(json.dumps({"condition": cond, "idx": r["idx"],
                                     "category": r["category"], **row},
                                    ensure_ascii=False) + "\n")
            s = summarize(rows)
            print(f"   {cond:>14}  n={s['n']:>4}  masa mediana={s['mass']['median']:.4f}"
                  f"  <0.5: {s['fraction_below']['0.5']:.1%}"
                  f"  argmax es letra: {s['top_token_is_a_letter']:.1%}", flush=True)

    report = {
        "metadata": {
            "model": MODEL, "seed": SEED, "n_direct": args.n_direct,
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "question": ("¿la letra leída por argmax restringido coincide con lo que el modelo "
                     "iba a escribir, y cuánta masa de probabilidad llevan las tres letras?"),
        "by_condition": {k: summarize(v) for k, v in results.items()},
    }

    # la comparación que importa: misma población, antes de razonar vs tras la frase 1
    paired = {}
    a = {r["idx"]: row for r, row in zip(conditions["cot_vacio"], results["cot_vacio"])}
    b = {r["idx"]: row for r, row in zip(conditions["frase1"], results["frase1"])}
    common = sorted(set(a) & set(b))
    if common:
        da = np.array([b[i]["mass"] - a[i]["mass"] for i in common])
        paired = {
            "n_common": len(common),
            "median_mass_before": float(np.median([a[i]["mass"] for i in common])),
            "median_mass_after_sentence1": float(np.median([b[i]["mass"] for i in common])),
            "median_difference": float(np.median(da)),
            "items_where_agreement_differs": int(sum(
                1 for i in common if a[i]["top_is_letter"] != b[i]["top_is_letter"])),
            "items_where_letter_differs": int(sum(
                1 for i in common if a[i]["chosen_letter"] != b[i]["chosen_letter"])),
        }
    report["paired_before_vs_sentence1"] = paired

    report["interpretation_rule"] = {
        "masa alta en todas las condiciones":
            "el instrumento queda validado; la limitación se cierra",
        "masa baja pero pareja entre condiciones":
            "los efectos medidos se sostienen (mismo instrumento en todos los brazos); "
            "hay que suavizar las afirmaciones absolutas sobre lo que el modelo respondería",
        "masa desigual entre 'cot_vacio' y 'frase1'":
            "la comparación que sostiene el resultado central mide en parte un cambio de "
            "instrumento; revisar su interpretación",
    }
    report["limitations"] = [
        "Orden canónico de opciones en las cuatro condiciones: no se promedian las tres "
        "permutaciones, así que la masa no se estima sobre el mismo consolidado que el rol.",
        "No se compara contra generación libre: esta prueba dice si el argmax global es una "
        "letra, no qué respondería el modelo si escribiera una frase entera.",
        "Las condiciones directas usan una muestra aleatoria de los registros, no los 1008.",
    ]
    report["metadata"]["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito {OUT}")
    print(f"escrito {ITEMS_OUT}  ({report['metadata']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

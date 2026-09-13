#!/usr/bin/env python3
"""Confirmación pareada del efecto causal de v_aband.

El script 16 encuentra, con alpha calibrado para no degenerar:

    alpha_ef = -0.125   ambiguo +0.090 P(unknown)   disambig +0.033   <- sin coste
    alpha_ef = -0.250   ambiguo +0.140 P(unknown)   disambig -0.083   <- con coste
    brazo aleatorio (5 semillas)  maximo +0.02, dispersion 0.02

con dosis-respuesta monotona al cruzar cero. Pero n=100 y, sobre todo, el script 16 SOLO
GUARDO AGREGADOS: sin resultados por item no se puede hacer McNemar, que es el test que
corresponde porque los items son LOS MISMOS con y sin intervencion. Un contraste de
proporciones no pareado tira potencia a la basura.

Esto es exactamente la regla D7 que ya me habia costado dos veces. Aqui se persiste todo
por item antes de agregar nada.

DISENO
  - Mismos items en todos los brazos; McNemar exacto (binomial sobre los discordantes).
  - Dos dosis: -0.125 (la ventana sin coste) y -0.250 (mayor efecto, con coste).
  - Brazo aleatorio pareado a -0.125, 5 semillas, para calibrar el nulo empiricamente.
  - Metrica primaria: acierto BBQ por item en ambiguo (rol final == 'unknown').
  - Control disambig obligatorio, tambien por item y pareado.

Salida: output/steering_confirmation.json, steering_confirmation_items.jsonl
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import platform
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import transformers
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/experiments/EXP-002_bbq_stereotype_direction/output"
RECORDS = OUT_DIR / "revision_scaled_records.jsonl"
ACTS = OUT_DIR / "revision_scaled_acts.npz"
OUT = OUT_DIR / "steering_confirmation.json"
ITEMS_OUT = OUT_DIR / "steering_confirmation_items.jsonl"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 20260908
LAYER = 22
DOSES = [-0.125, -0.250]
RANDOM_DOSE = -0.125
RANDOM_SEEDS = [20260908 + i for i in range(5)]
RESERVED = {"Gender_identity", "Race_x_gender", "Sexual_orientation"}
LEN_LO, LEN_HI = 0.70, 1.25

_s16 = importlib.util.spec_from_file_location(
    "s16", REPO / "scripts" / "16_revision_steering.py")
s16 = importlib.util.module_from_spec(_s16)
_s16.loader.exec_module(s16)
s09 = s16.s09
LETTERS = s16.LETTERS

rng = np.random.default_rng(SEED)


def mcnemar(base: List[bool], treat: List[bool]) -> dict:
    """Exacto (binomial sobre discordantes). Bilateral."""
    b = sum(1 for x, y in zip(base, treat) if x and not y)   # se rompio
    c = sum(1 for x, y in zip(base, treat) if y and not x)   # se arreglo
    n = b + c
    p = 1.0 if n == 0 else float(stats.binomtest(c, n, 0.5).pvalue)
    return {"n_broken": b, "n_fixed": c, "n_discordant": n, "p_value": p,
            "delta": (c - b) / len(base)}


def run(model, tok, lids, amb, dis, steerer, delta, tag) -> dict:
    """Genera CON intervencion, lee SIN intervencion, y devuelve por item."""
    a_tr = s16.generate(model, tok, amb, steerer, delta)
    d_tr = s16.generate(model, tok, dis, steerer, delta)
    rows = []
    for it, tr in zip(amb, a_tr):
        steps = s16.split_steps(tr)
        if len(steps) < 2:
            rows.append({"arm": tag, "cond": "ambig", "key": None, "usable": False})
            continue
        roles = [it["roles"][s16.read_choice(model, tok, lids, it, " ".join(steps[:n]))]
                 for n in range(1, len(steps) + 1)]
        rows.append({"arm": tag, "cond": "ambig", "category": it["category"],
                     "usable": True, "first": roles[0], "final": roles[-1],
                     "correct": roles[-1] == "unknown", "n_steps": len(steps),
                     "rep": s16.repetition_rate(tr)})
    for it, tr in zip(dis, d_tr):
        steps = s16.split_steps(tr)
        if len(steps) < 2:
            rows.append({"arm": tag, "cond": "disambig", "usable": False})
            continue
        c = s16.read_choice(model, tok, lids, it, tr)
        rows.append({"arm": tag, "cond": "disambig", "category": it["category"],
                     "usable": True, "correct": c == it["label"],
                     "n_steps": len(steps), "rep": s16.repetition_rate(tr)})
    return {"rows": rows}


def agg(rows, cond):
    ok = [r for r in rows if r["cond"] == cond and r["usable"]]
    if not ok:
        return {"n": 0}
    return {"n": len(ok), "accuracy": float(np.mean([r["correct"] for r in ok])),
            "mean_steps": float(np.mean([r["n_steps"] for r in ok])),
            "mean_rep": float(np.mean([r["rep"] for r in ok])),
            "n_unusable": sum(1 for r in rows
                              if r["cond"] == cond and not r["usable"])}


def paired(base_rows, arm_rows, cond):
    """Empareja por posicion: mismos items, mismo orden, en todos los brazos."""
    b = [r for r in base_rows if r["cond"] == cond]
    a = [r for r in arm_rows if r["cond"] == cond]
    pairs = [(x["correct"], y["correct"]) for x, y in zip(b, a)
             if x["usable"] and y["usable"]]
    if len(pairs) < 10:
        return {"note": f"solo {len(pairs)} pares utilizables"}
    return {"n_paired": len(pairs),
            **mcnemar([p[0] for p in pairs], [p[1] for p in pairs])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ambig", type=int, default=200)
    ap.add_argument("--n-disambig", type=int, default=120)
    ap.add_argument("--population", choices=["anchor", "fresh"], default="anchor",
                    help=("anchor: items del ancla 'unknown' del script 15 (subpoblacion "
                          "donde el modelo aun se abstenia tras el paso 1). "
                          "fresh: muestra sin preseleccionar de BBQ ambiguo -- la que "
                          "decide si esto es un guardarrail o solo una palanca sobre un "
                          "subgrupo"))
    ap.add_argument("--n-random", type=int, default=len(RANDOM_SEEDS))
    ap.add_argument("--suffix", default="", help="sufijo de los ficheros de salida")
    args = ap.parse_args()
    t0 = time.time()
    global OUT, ITEMS_OUT
    if args.suffix:
        OUT = OUT_DIR / f"steering_confirmation{args.suffix}.json"
        ITEMS_OUT = OUT_DIR / f"steering_confirmation_items{args.suffix}.jsonl"

    print("[1/4] v_aband desde pares emparejados (identico al script 16)...", flush=True)
    recs = [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    z = np.load(ACTS)
    A, layers = z["acts"], [int(x) for x in z["layers"]]
    li = layers.index(LAYER)

    sub = [r for r in recs if r["first_role"] == "unknown"]
    aband = [r for r in sub if r["final_role"] != "unknown"]
    held = [r for r in sub if all(x == "unknown" for x in r["roles"])]
    pool = collections.defaultdict(list)
    for r in held:
        pool[(r["category"], r["template"], r["polarity"])].append(r)
    pa, pb = [], []
    for r in aband:
        k = (r["category"], r["template"], r["polarity"])
        if pool[k]:
            pa.append(r)
            pb.append(pool[k].pop())
    Xa = A[[r["idx"] for r in pa]][:, li].astype(np.float32)
    Xb = A[[r["idx"] for r in pb]][:, li].astype(np.float32)
    v = (Xa - Xb).mean(axis=0)
    v = v / np.linalg.norm(v)
    scale = float(np.median(np.linalg.norm(A[:, li].astype(np.float32), axis=1)))
    v_aband = v * scale
    print(f"      {len(pa)} pares, capa {LAYER}, norma {scale:.2f}", flush=True)

    used = {r["idx"] for r in pa} | {r["idx"] for r in pb}
    disjoint = [r for r in sub if r["idx"] not in used]
    test_recs = [disjoint[i] for i in rng.permutation(len(disjoint))[:args.n_ambig]]

    print("[2/4] Modelo y conjuntos...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lids = {L: sorted({tok.encode(f, add_special_tokens=False)[0]
                       for f in (L, " " + L)
                       if len(tok.encode(f, add_special_tokens=False)) == 1})
            for L in LETTERS}
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    steerer = s16.Steerer(model, LAYER)

    corpus = s09.load_corpus_items()
    if args.population == "anchor":
        amb = [corpus[tuple(r["key"])] for r in test_recs]
        pop_note = ("ancla 'unknown' del script 15: items donde el modelo AUN se abstenia "
                    "tras el paso 1; disjuntos de los pares que construyen el vector")
    else:
        # sin preseleccionar: se excluyen solo los items que construyen el vector
        vec_keys = {tuple(r["key"]) for r in pa} | {tuple(r["key"]) for r in pb}
        apool = [(k, it) for k, it in corpus.items()
                 if it["condition"] == "ambig" and it["category"] not in RESERVED
                 and k not in vec_keys]
        amb = [apool[i][1] for i in rng.permutation(len(apool))[:args.n_ambig]]
        pop_note = ("muestra SIN preseleccionar de BBQ ambiguo (8 categorias no "
                    "reservadas), excluyendo solo los items que construyen el vector")
    dpool = [it for it in corpus.values()
             if it["condition"] == "disambig" and it["category"] not in RESERVED]
    dis = [dpool[i] for i in rng.permutation(len(dpool))[:args.n_disambig]]
    print(f"      poblacion: {args.population} -- {pop_note}", flush=True)
    print(f"      ambiguo {len(amb)}, disambig {len(dis)}", flush=True)

    print("[3/4] Brazos...", flush=True)
    fh = ITEMS_OUT.open("w", encoding="utf-8")
    arms: Dict[str, list] = {}

    def do(tag, delta):
        r = run(model, tok, lids, amb, dis, steerer, delta, tag)["rows"]
        arms[tag] = r
        for row in r:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        a, d = agg(r, "ambig"), agg(r, "disambig")
        print(f"      {tag:<22} ambig acc={a.get('accuracy'):.4f} "
              f"pasos={a.get('mean_steps'):.2f} | disambig acc={d.get('accuracy'):.4f}",
              flush=True)
        return r

    do("baseline", None)
    for dose in DOSES:
        do(f"direction_{dose}", dose * v_aband)
    for sd in RANDOM_SEEDS[:args.n_random]:
        r0 = np.random.default_rng(sd).normal(size=v_aband.shape[0])
        do(f"random_{sd}", RANDOM_DOSE * r0 / np.linalg.norm(r0) * scale)
    fh.close()
    steerer.remove()

    print("[4/4] McNemar pareado...", flush=True)
    base = arms["baseline"]
    base_steps = agg(base, "ambig")["mean_steps"]
    report = {
        "metadata": {"model": MODEL, "seed": SEED, "layer": LAYER,
                     "n_pairs_direction": len(pa), "n_ambig": len(amb),
                     "n_disambig": len(dis), "doses": DOSES,
                     "population": args.population, "population_note": pop_note,
                     "random_dose": RANDOM_DOSE,
                     "random_seeds": RANDOM_SEEDS[:args.n_random],
                     "residual_norm_median": scale,
                     "python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "scipy": __import__("scipy").__version__},
        "test": "McNemar exacto (binomial bilateral sobre discordantes), items pareados",
        "baseline": {"ambig": agg(base, "ambig"), "disambig": agg(base, "disambig")},
        "arms": {},
    }
    for tag, rows in arms.items():
        if tag == "baseline":
            continue
        a = agg(rows, "ambig")
        report["arms"][tag] = {
            "ambig": a, "disambig": agg(rows, "disambig"),
            "mcnemar_ambig": paired(base, rows, "ambig"),
            "mcnemar_disambig": paired(base, rows, "disambig"),
            "within_window": bool(LEN_LO * base_steps <= a.get("mean_steps", -1)
                                  <= LEN_HI * base_steps),
        }
        m = report["arms"][tag]["mcnemar_ambig"]
        md = report["arms"][tag]["mcnemar_disambig"]
        print(f"      {tag:<22} ambig Δ={m.get('delta', float('nan')):+.4f} "
              f"p={m.get('p_value', float('nan')):.4f} "
              f"(+{m.get('n_fixed')}/-{m.get('n_broken')}) | "
              f"disambig Δ={md.get('delta', float('nan')):+.4f} "
              f"p={md.get('p_value', float('nan')):.4f}", flush=True)

    rnd = [report["arms"][t]["mcnemar_ambig"].get("delta", np.nan)
           for t in report["arms"] if t.startswith("random_")]
    rnd = [x for x in rnd if not np.isnan(x)]
    if rnd:
        report["random_null"] = {
            "n_seeds": len(rnd), "mean": float(np.mean(rnd)),
            "sd": float(np.std(rnd, ddof=1)) if len(rnd) > 1 else None,
            "max": float(np.max(rnd)), "min": float(np.min(rnd)),
            "note": "nulo empirico del efecto en ambiguo a la misma dosis y norma",
        }
    report["limitations"] = [
        "La CAPA (22) se eligio mirando los mismos datos en el script 17; el sesgo de "
        "seleccion de capa persiste aunque el vector se ajuste en pares disjuntos.",
        "El vector se construye con 31 pares: estimacion ruidosa.",
        "5 semillas aleatorias dan un nulo descriptivo, no una distribucion.",
        "Una generacion determinista por item y brazo; un solo modelo.",
        "El conjunto disambig no esta emparejado item a item con el ambiguo: mide dano "
        "colateral, no un efecto comparable.",
        (pop_note if args.population == "fresh"
         else "Los items ambiguos vienen del ancla 'unknown' del script 15, no de una "
              "muestra fresca de BBQ: el efecto se afirma sobre esa subpoblacion."),
    ]
    report["metadata"]["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {OUT}  ({report['metadata']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Perfil de ESPECIFICIDAD por profundidad, sobre los mismos items.

PREGUNTA. No «que capa da mas efecto» —esa pregunta ya demostro estar mal planteada
(results.md III.15)— sino como se intercambian MAGNITUD y ESPECIFICIDAD a lo largo de las
capas, y si hay un optimo.

Los tres puntos disponibles apuntan a que si:

    capa 18   efecto +0.216   max aleatorio +0.128   ratio 1.69
    capa 22   efecto +0.150   max aleatorio +0.040   ratio 3.75
    capa 24   efecto +0.017   sin control

El efecto DECRECE con la profundidad y la especificidad CRECE. Si se confirma, es un
resultado sobre como funciona el steering, no sobre nuestro vector.

POR QUE IMPORTA MAS ALLA DE ESTE TRABAJO. SOA-007 elige la capa por efectividad del
steering (Apendice B.4), que es exactamente el criterio que aqui fallo: favorece capas donde
cualquier perturbacion mueve el comportamiento. Si el perfil se confirma, hay evidencia
medida de que ese criterio esta sesgado, con la correccion propuesta.

DOS FALLOS DE III.15 QUE ESTE DISEÑO CORRIGE

  1. Dosis del brazo de control no emparejada. Alli las aleatorias corrieron solo a -0.25
     mientras la direccion se probaba tambien a -0.125, dejando ese punto sin control.
     Aqui hay aleatorias en CADA dosis.
  2. Capas medidas sobre conjuntos distintos (18 en 125 items, 22 en otros 200), luego no
     comparables punto a punto. Aqui las tres capas usan LOS MISMOS 125 items de
     confirmacion del split ya registrado.

Las mismas SEMILLAS aleatorias en todas las capas: la direccion unitaria es identica y solo
cambia el reescalado por la norma de la capa, de modo que el brazo de control tambien es
comparable entre capas.

Reutiliza de III.15, sin regenerar: la linea base y, para la capa 18, la direccion en ambas
dosis y las aleatorias a -0.25. Se VERIFICA que la linea base coincide antes de usarla.

Salida: output/specificity_profile.json, specificity_profile_items.jsonl
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/experiments/EXP-002_bbq_stereotype_direction/output"
RECORDS = OUT_DIR / "revision_scaled_records.jsonl"
ACTS = OUT_DIR / "revision_scaled_acts.npz"
SPLIT = OUT_DIR / "layer_sweep_split.json"
PREV_ITEMS = OUT_DIR / "layer_sweep_items.jsonl"
OUT = OUT_DIR / "specificity_profile.json"
ITEMS_OUT = OUT_DIR / "specificity_profile_items.jsonl"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 20260910
NEW_LAYERS = [20, 22]          # la 18 se reutiliza de III.15
COMPLETE_LAYER = 18            # solo le faltan las aleatorias a -0.125
DOSES = [-0.125, -0.250]
RANDOM_SEEDS = [20260911 + i for i in range(3)]   # mismas semillas en todas las capas
RESERVED = {"Gender_identity", "Race_x_gender", "Sexual_orientation"}

_s18 = importlib.util.spec_from_file_location(
    "s18", REPO / "scripts" / "18_steering_confirmation.py")
s18 = importlib.util.module_from_spec(_s18)
_s18.loader.exec_module(s18)
s16, s09 = s18.s16, s18.s09

rng = np.random.default_rng(SEED)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-disambig", type=int, default=80)
    args = ap.parse_args()
    t0 = time.time()

    print("[1/5] Split registrado y vector...", flush=True)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    conf_keys = [tuple(k) for k in split["confirm_keys"]]
    print(f"      confirmacion: {len(conf_keys)} ítems (split seed {split['split_seed']})",
          flush=True)

    recs = [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    z = np.load(ACTS)
    A, layers = z["acts"], [int(x) for x in z["layers"]]
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
    ia = [r["idx"] for r in pa]
    ib = [r["idx"] for r in pb]

    vecs, scales = {}, {}
    for layer in sorted(set(NEW_LAYERS + [COMPLETE_LAYER])):
        li = layers.index(layer)
        v = (A[ia][:, li].astype(np.float32) - A[ib][:, li].astype(np.float32)).mean(axis=0)
        v = v / np.linalg.norm(v)
        sc = float(np.median(np.linalg.norm(A[:, li].astype(np.float32), axis=1)))
        vecs[layer] = v * sc
        scales[layer] = sc
        print(f"      capa {layer}: norma mediana {sc:.2f}", flush=True)

    print("[2/5] Modelo y conjuntos...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lids = {L: sorted({tok.encode(f, add_special_tokens=False)[0]
                       for f in (L, " " + L)
                       if len(tok.encode(f, add_special_tokens=False)) == 1})
            for L in s16.LETTERS}
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()

    corpus = s09.load_corpus_items()
    conf = [corpus[k] for k in conf_keys]
    dpool = [it for it in corpus.values()
             if it["condition"] == "disambig" and it["category"] not in RESERVED]
    # mismo muestreo de disambig que el script 25, para que sea el mismo conjunto
    dis = [dpool[i] for i in np.random.default_rng(split["split_seed"]
                                                  ).permutation(len(dpool))[:args.n_disambig]]
    print(f"      {len(conf)} ambiguos + {len(dis)} disambig", flush=True)

    fh = ITEMS_OUT.open("w", encoding="utf-8")
    arms = {}

    def do(tag, layer, delta):
        st = s16.Steerer(model, layer)
        rows = s18.run(model, tok, lids, conf, dis, st, delta, tag)["rows"]
        st.remove()
        arms[tag] = rows
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        a, d = s18.agg(rows, "ambig"), s18.agg(rows, "disambig")
        print(f"      {tag:<30} ambig={a.get('accuracy'):.4f} | "
              f"disambig={d.get('accuracy'):.4f}", flush=True)
        return rows

    print("[3/5] Linea base y verificacion contra III.15...", flush=True)
    base = do("baseline", NEW_LAYERS[0], None)
    prev = [json.loads(l) for l in PREV_ITEMS.read_text(encoding="utf-8").splitlines() if l.strip()]
    prev_arms = collections.defaultdict(list)
    for r in prev:
        prev_arms[r["arm"]].append(r)
    pb_rows = [r for r in prev_arms.get("phase2_baseline", []) if r["cond"] == "ambig"]
    nb_rows = [r for r in base if r["cond"] == "ambig"]
    match = None
    if len(pb_rows) == len(nb_rows):
        match = sum(1 for x, y in zip(pb_rows, nb_rows)
                    if x["usable"] == y["usable"]
                    and (not x["usable"] or x["correct"] == y["correct"]))
        print(f"      linea base coincide con III.15 en {match}/{len(nb_rows)} ítems",
              flush=True)
        if match < len(nb_rows) * 0.95:
            print("      AVISO: la generacion determinista no reprodujo la corrida previa; "
                  "los brazos reutilizados NO son comparables", flush=True)

    print("[4/5] Brazos...", flush=True)
    for layer in NEW_LAYERS:
        for dose in DOSES:
            do(f"L{layer}_direction_{dose}", layer, dose * vecs[layer])
        for dose in DOSES:
            for sd in RANDOM_SEEDS:
                r0 = np.random.default_rng(sd).normal(size=vecs[layer].shape[0])
                do(f"L{layer}_random_{sd}_{dose}", layer,
                   dose * r0 / np.linalg.norm(r0) * scales[layer])
    # completar la capa 18: solo le faltan las aleatorias a -0.125
    for sd in RANDOM_SEEDS:
        r0 = np.random.default_rng(sd).normal(size=vecs[COMPLETE_LAYER].shape[0])
        do(f"L{COMPLETE_LAYER}_random_{sd}_-0.125", COMPLETE_LAYER,
           -0.125 * r0 / np.linalg.norm(r0) * scales[COMPLETE_LAYER])
    fh.close()

    print("[5/5] Perfil...", flush=True)
    # incorporar los brazos reutilizables de III.15 para la capa 18
    for tag, rows in prev_arms.items():
        if tag.startswith("phase2_direction_"):
            arms[f"L18_direction_{tag.split('_')[-1]}"] = rows
        elif tag.startswith("phase2_random_"):
            arms[f"L18_random_{tag.split('_')[-1]}_-0.25"] = rows

    report = {
        "metadata": {"model": MODEL, "seed": SEED, "split_seed": split["split_seed"],
                     "layers_measured": NEW_LAYERS, "layer_completed": COMPLETE_LAYER,
                     "doses": DOSES, "random_seeds": RANDOM_SEEDS,
                     "n_ambig": len(conf), "n_disambig": len(dis),
                     "residual_norms": scales,
                     "baseline_match_with_III15": match,
                     "python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "design": ("mismas capas sobre LOS MISMOS items, aleatorias en CADA dosis, mismas "
                   "semillas en todas las capas"),
        "baseline": {"ambig": s18.agg(base, "ambig"),
                     "disambig": s18.agg(base, "disambig")},
        "profile": {},
    }
    print(f"\n{'capa':>5}{'dosis':>9}{'Δambig':>10}{'p':>9}{'máx alea':>10}"
          f"{'ratio':>8}{'Δdisamb':>10}{'p':>9}")
    print("-" * 70)
    for layer in sorted(set(NEW_LAYERS + [COMPLETE_LAYER])):
        for dose in DOSES:
            dtag = f"L{layer}_direction_{dose}"
            if dtag not in arms:
                continue
            md = s18.paired(base, arms[dtag], "ambig")
            mdd = s18.paired(base, arms[dtag], "disambig")
            rnd = []
            for sd in RANDOM_SEEDS:
                rt = f"L{layer}_random_{sd}_{dose}"
                if rt in arms:
                    m = s18.paired(base, arms[rt], "ambig")
                    if "delta" in m:
                        rnd.append(m["delta"])
            row = {"delta_ambig": md.get("delta"), "p_ambig": md.get("p_value"),
                   "delta_disambig": mdd.get("delta"), "p_disambig": mdd.get("p_value"),
                   "n_random": len(rnd),
                   "random_max": float(np.max(rnd)) if rnd else None,
                   "random_min": float(np.min(rnd)) if rnd else None,
                   "random_spread": float(np.max(rnd) - np.min(rnd)) if len(rnd) > 1 else None}
            if rnd and row["random_max"] and row["random_max"] > 0:
                row["specificity_ratio"] = row["delta_ambig"] / row["random_max"]
            row["composite"] = (row["delta_ambig"]
                                - max(0.0, -(row["delta_disambig"] or 0.0)))
            report["profile"][f"L{layer}_{dose}"] = row
            print(f"{layer:>5}{dose:>9.3f}{row['delta_ambig']:>10.4f}"
                  f"{row['p_ambig']:>9.4f}"
                  f"{(row['random_max'] if row['random_max'] is not None else float('nan')):>10.4f}"
                  f"{(row.get('specificity_ratio') or float('nan')):>8.2f}"
                  f"{row['delta_disambig']:>10.4f}{row['p_disambig']:>9.4f}")

    report["limitations"] = [
        "3 semillas aleatorias por capa y dosis: el maximo es un estimador ruidoso, y el "
        "ratio de especificidad hereda ese ruido.",
        "Los brazos de la capa 18 a dosis -0.25 provienen de III.15; su comparabilidad "
        "depende de que la linea base coincida (se verifica y se reporta).",
        "125 items ambiguos: suficiente para el efecto principal, ajustado para el coste "
        "en disambig.",
        "Poblacion: subpoblacion en riesgo del ancla 'unknown', 2% del corpus ambiguo.",
        "Un solo modelo; una generacion determinista por item y brazo.",
        "El perfil describe TRES capas, no una funcion continua de la profundidad.",
    ]
    report["metadata"]["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito {OUT}  ({report['metadata']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""¿En que capa se debe sumar el vector? Barrido causal con desarrollo / confirmacion.

MOTIVO. La capa 22 se eligio por la meseta del AUC EMPAREJADO, y ese criterio es ciego:
dentro del par la categoria esta fijada, asi que su componente se cancela y capas que
difieren precisamente en ella parecen equivalentes. Fuera del emparejamiento la capa 20 da
0.63 y la 22 cae a 0.57 (results.md III.12). SOA-007 (Apendice B.4) usa el criterio
correcto: elegir la capa por EFECTIVIDAD DEL STEERING, no por AUC de sonda.

POR QUE NO BASTA CON REPETIR EN LA CAPA 20. La capa 20 se identifico mirando los mismos
items sobre los que se mediria el efecto. Reportar la mejor de dos capas probadas sobre el
mismo conjunto es la maldicion del ganador con una capa en lugar de un AUC, agravada porque
el resultado se conoce de antemano. De ahi la particion.

DISEÑO

  FASE 0  particion UNICA con semilla registrada, ESCRITA A DISCO antes de tocar la GPU
            245 items disponibles -> 120 desarrollo / 125 confirmacion

  FASE 1  capas candidatas {18, 20, 21, 22, 24}, declaradas de antemano
            una sola dosis, el MISMO alpha en todas las capas
            sin brazo aleatorio ni disambig: es un instrumento de SELECCION, no una
            afirmacion, y gastar controles aqui seria tirar GPU
            -> se elige la capa con mayor efecto y se congela

  FASE 2  SOLO esa capa, protocolo completo del script 18, sobre los 125 items que la
            fase 1 NUNCA vio. Este p-valor si es la afirmacion.

DECISIONES QUE HACEN COMPARABLES LAS CAPAS
  - v se construye por capa con los MISMOS 31 pares, y se reescala por la norma mediana
    DE ESA CAPA. Sin eso, alpha significaria magnitudes distintas segun la profundidad y el
    barrido mediria magnitud, no capa.
  - MISMO alpha en todas. Si una capa degenera, se baja alpha PARA TODAS: un alpha por capa
    confundiria capa con magnitud.

LIMITACION DECLARADA ANTES DE CORRER. 120 items para elegir entre cinco capas es poco: con
efectos de 0.10-0.15 el orden entre capas vecinas sera incierto. La fase 2 confirma que «la
capa elegida funciona», NO que «esta es la capa optima». Solo la primera queda demostrada.

Salida: output/layer_sweep.json, layer_sweep_split.json, layer_sweep_items.jsonl
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
OUT = OUT_DIR / "layer_sweep.json"
SPLIT_OUT = OUT_DIR / "layer_sweep_split.json"
ITEMS_OUT = OUT_DIR / "layer_sweep_items.jsonl"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SPLIT_SEED = 20260911          # semilla de la particion, registrada
CANDIDATE_LAYERS = [18, 20, 21, 22, 24]
PHASE1_ALPHA = -0.25           # dosis unica de la fase 1
PHASE2_DOSES = [-0.125, -0.250]
RANDOM_SEEDS = [20260911 + i for i in range(5)]
RANDOM_DOSE = -0.250
N_DEV = 120
CALIB_N = 16
LEN_LO, LEN_HI = 0.70, 1.25
REP_MARGIN = 0.05
RESERVED = {"Gender_identity", "Race_x_gender", "Sexual_orientation"}

_s18 = importlib.util.spec_from_file_location(
    "s18", REPO / "scripts" / "18_steering_confirmation.py")
s18 = importlib.util.module_from_spec(_s18)
_s18.loader.exec_module(s18)
s16, s09 = s18.s16, s18.s09


def build_vector(A, layers, layer, ia, ib):
    """Vector de esa capa, reescalado por la norma mediana DE ESA CAPA."""
    li = layers.index(layer)
    Xa = A[ia][:, li].astype(np.float32)
    Xb = A[ib][:, li].astype(np.float32)
    v = (Xa - Xb).mean(axis=0)
    v = v / np.linalg.norm(v)
    scale = float(np.median(np.linalg.norm(A[:, li].astype(np.float32), axis=1)))
    return v * scale, scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-disambig", type=int, default=80)
    args = ap.parse_args()
    t0 = time.time()

    # ---------------------------------------------------------------- FASE 0
    print("[FASE 0] Particion, antes de tocar la GPU...", flush=True)
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
    used = {r["idx"] for r in pa} | {r["idx"] for r in pb}
    ia = [r["idx"] for r in pa]
    ib = [r["idx"] for r in pb]

    avail = [r for r in sub if r["idx"] not in used]
    rng_split = np.random.default_rng(SPLIT_SEED)
    perm = rng_split.permutation(len(avail))
    dev_recs = [avail[i] for i in perm[:N_DEV]]
    conf_recs = [avail[i] for i in perm[N_DEV:]]

    split_doc = {
        "split_seed": SPLIT_SEED,
        "n_pairs_vector": len(pa),
        "n_available": len(avail),
        "n_dev": len(dev_recs), "n_confirm": len(conf_recs),
        "dev_keys": [r["key"] for r in dev_recs],
        "confirm_keys": [r["key"] for r in conf_recs],
        "note": ("escrito ANTES de cualquier medicion; es el compromiso que impide "
                 "repartir de nuevo si el resultado no gusta"),
    }
    SPLIT_OUT.write_text(json.dumps(split_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"      {len(pa)} pares para el vector; {len(avail)} disponibles", flush=True)
    print(f"      desarrollo {len(dev_recs)} / confirmacion {len(conf_recs)}", flush=True)
    print(f"      particion escrita en {SPLIT_OUT.name}", flush=True)

    # ---------------------------------------------------------------- modelo
    print("[SETUP] Modelo...", flush=True)
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
    dev = [corpus[tuple(r["key"])] for r in dev_recs]
    conf = [corpus[tuple(r["key"])] for r in conf_recs]
    rng = np.random.default_rng(SPLIT_SEED)
    dpool = [it for it in corpus.values()
             if it["condition"] == "disambig" and it["category"] not in RESERVED]
    dis = [dpool[i] for i in rng.permutation(len(dpool))[:args.n_disambig]]

    vecs = {}
    for layer in CANDIDATE_LAYERS:
        v, sc = build_vector(A, layers, layer, ia, ib)
        vecs[layer] = v
        print(f"      capa {layer}: norma mediana del residual = {sc:.2f}", flush=True)

    fh = ITEMS_OUT.open("w", encoding="utf-8")
    report = {
        "metadata": {"model": MODEL, "split_seed": SPLIT_SEED,
                     "candidate_layers": CANDIDATE_LAYERS,
                     "phase1_alpha": PHASE1_ALPHA, "phase2_doses": PHASE2_DOSES,
                     "random_seeds": RANDOM_SEEDS, "n_pairs_vector": len(pa),
                     "n_dev": len(dev), "n_confirm": len(conf), "n_disambig": len(dis),
                     "python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "design": ("fase 1 elige la capa en desarrollo; fase 2 mide en items que la fase 1 "
                   "nunca vio. El p-valor de la fase 1 NO es una afirmacion."),
        "declared_before_running": (
            "120 items para elegir entre 5 capas es poco: el orden entre capas vecinas sera "
            "incierto. La fase 2 confirma que la capa elegida FUNCIONA, no que sea OPTIMA."),
        "split_file": SPLIT_OUT.name,
    }

    # -------------------------------------------------- calibracion compartida
    print(f"\n[CALIB] Guarda de dos lados con alpha={PHASE1_ALPHA} en todas las capas...",
          flush=True)
    st = s16.Steerer(model, CANDIDATE_LAYERS[0])
    base_cal = s18.agg(s18.run(model, tok, lids, dev[:CALIB_N], [], st, None, "cal")["rows"],
                       "ambig")
    st.remove()
    lo, hi = base_cal["mean_steps"] * LEN_LO, base_cal["mean_steps"] * LEN_HI
    print(f"      ventana: {lo:.2f}-{hi:.2f} pasos, repeticion < "
          f"{base_cal['mean_rep'] + REP_MARGIN:.4f}", flush=True)

    alpha = PHASE1_ALPHA
    while True:
        calib, ok_all = {}, True
        for layer in CANDIDATE_LAYERS:
            st = s16.Steerer(model, layer)
            a = s18.agg(s18.run(model, tok, lids, dev[:CALIB_N], [], st,
                                alpha * vecs[layer], "cal")["rows"], "ambig")
            st.remove()
            ok = (a["n"] > 0 and lo <= a["mean_steps"] <= hi
                  and a["mean_rep"] < base_cal["mean_rep"] + REP_MARGIN)
            calib[str(layer)] = {"mean_steps": a.get("mean_steps"),
                                 "mean_rep": a.get("mean_rep"), "acceptable": bool(ok)}
            print(f"      capa {layer}: pasos={a.get('mean_steps'):.2f} "
                  f"rep={a.get('mean_rep'):.4f} {'ok' if ok else 'DEGRADA'}", flush=True)
            ok_all = ok_all and ok
        if ok_all or abs(alpha) < 0.03:
            break
        alpha /= 2
        print(f"      alguna capa degenera -> se baja alpha PARA TODAS a {alpha}", flush=True)
    report["calibration"] = {"alpha_used": alpha, "by_layer": calib,
                             "baseline_steps": base_cal["mean_steps"],
                             "window": [lo, hi],
                             "rule": ("si una capa degenera se baja alpha para todas: un "
                                      "alpha por capa confundiria capa con magnitud")}
    print(f"      alpha compartido: {alpha}", flush=True)

    # ---------------------------------------------------------------- FASE 1
    print(f"\n[FASE 1] Barrido en desarrollo ({len(dev)} items)...", flush=True)
    st = s16.Steerer(model, CANDIDATE_LAYERS[0])
    base_rows = s18.run(model, tok, lids, dev, [], st, None, "phase1_baseline")["rows"]
    st.remove()
    for r in base_rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    base_acc = s18.agg(base_rows, "ambig")
    print(f"      baseline acc={base_acc['accuracy']:.4f} "
          f"pasos={base_acc['mean_steps']:.2f}", flush=True)

    phase1 = {}
    for layer in CANDIDATE_LAYERS:
        st = s16.Steerer(model, layer)
        rows = s18.run(model, tok, lids, dev, [], st, alpha * vecs[layer],
                       f"phase1_layer{layer}")["rows"]
        st.remove()
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        a = s18.agg(rows, "ambig")
        m = s18.paired(base_rows, rows, "ambig")
        phase1[str(layer)] = {"ambig": a, "mcnemar": m,
                              "within_window": bool(lo <= a.get("mean_steps", -1) <= hi)}
        print(f"      capa {layer}: acc={a['accuracy']:.4f} "
              f"Δ={m.get('delta', float('nan')):+.4f} "
              f"(+{m.get('n_fixed')}/-{m.get('n_broken')}) "
              f"pasos={a['mean_steps']:.2f}", flush=True)
    report["phase1"] = {"baseline": base_acc, "by_layer": phase1,
                        "note": ("p-valores NO reportados como afirmacion: estan "
                                 "seleccionados sobre 5 comparaciones")}

    valid = {k: v for k, v in phase1.items()
             if v["within_window"] and not np.isnan(v["mcnemar"].get("delta", np.nan))}
    if not valid:
        raise SystemExit("ABORTA: ninguna capa quedo dentro de la ventana")
    chosen = int(max(valid, key=lambda k: valid[k]["mcnemar"]["delta"]))
    report["chosen_layer"] = chosen
    report["chosen_reason"] = "mayor delta en desarrollo entre las capas dentro de ventana"
    print(f"\n  ==> capa elegida y CONGELADA: {chosen}", flush=True)

    # ---------------------------------------------------------------- FASE 2
    print(f"\n[FASE 2] Confirmacion en capa {chosen} ({len(conf)} items nunca vistos)...",
          flush=True)
    st = s16.Steerer(model, chosen)
    v_ch = vecs[chosen]
    scale_ch = float(np.linalg.norm(v_ch))

    arms = {}

    def do(tag, delta):
        rows = s18.run(model, tok, lids, conf, dis, st, delta, tag)["rows"]
        arms[tag] = rows
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        a, d = s18.agg(rows, "ambig"), s18.agg(rows, "disambig")
        print(f"      {tag:<24} ambig acc={a.get('accuracy'):.4f} | "
              f"disambig acc={d.get('accuracy'):.4f}", flush=True)

    do("phase2_baseline", None)
    for dose in PHASE2_DOSES:
        do(f"phase2_direction_{dose}", dose * v_ch)
    for sd in RANDOM_SEEDS:
        r0 = np.random.default_rng(sd).normal(size=v_ch.shape[0])
        do(f"phase2_random_{sd}", RANDOM_DOSE * r0 / np.linalg.norm(r0) * scale_ch)
    st.remove()
    fh.close()

    print("\n[FASE 2] McNemar pareado...", flush=True)
    b2 = arms["phase2_baseline"]
    p2 = {"baseline": {"ambig": s18.agg(b2, "ambig"),
                       "disambig": s18.agg(b2, "disambig")}, "arms": {}}
    for tag, rows in arms.items():
        if tag == "phase2_baseline":
            continue
        p2["arms"][tag] = {
            "ambig": s18.agg(rows, "ambig"), "disambig": s18.agg(rows, "disambig"),
            "mcnemar_ambig": s18.paired(b2, rows, "ambig"),
            "mcnemar_disambig": s18.paired(b2, rows, "disambig")}
        m = p2["arms"][tag]["mcnemar_ambig"]
        md = p2["arms"][tag]["mcnemar_disambig"]
        print(f"      {tag:<24} ambig Δ={m.get('delta', float('nan')):+.4f} "
              f"p={m.get('p_value', float('nan')):.4f} "
              f"(+{m.get('n_fixed')}/-{m.get('n_broken')}) | "
              f"disambig Δ={md.get('delta', float('nan')):+.4f} "
              f"p={md.get('p_value', float('nan')):.4f}", flush=True)
    report["phase2"] = p2

    rnd = [p2["arms"][t]["mcnemar_ambig"].get("delta", np.nan)
           for t in p2["arms"] if "random" in t]
    rnd = [x for x in rnd if not np.isnan(x)]
    dirs = [p2["arms"][t] for t in p2["arms"] if "direction" in t]
    if rnd and dirs:
        best = max(dirs, key=lambda x: x["mcnemar_ambig"].get("delta", -9))
        report["verdict"] = {
            "layer": chosen,
            "delta_ambig": best["mcnemar_ambig"]["delta"],
            "p_ambig": best["mcnemar_ambig"]["p_value"],
            "delta_disambig": best["mcnemar_disambig"].get("delta"),
            "p_disambig": best["mcnemar_disambig"].get("p_value"),
            "random_max": float(np.max(rnd)), "random_mean": float(np.mean(rnd)),
            "random_sd": float(np.std(rnd, ddof=1)) if len(rnd) > 1 else None,
            "exceeds_random": bool(best["mcnemar_ambig"]["delta"] > np.max(rnd)),
            "reference_layer22_main": {"delta": 0.150, "p": 0.0000, "n": 200,
                                       "note": "resultado previo, protocolo equivalente"},
            "claim": ("la capa elegida FUNCIONA sobre items no vistos; NO se afirma que sea "
                      "la capa optima"),
        }
    report["limitations"] = [
        "120 items para elegir entre 5 capas: el orden entre capas vecinas es incierto.",
        "El vector usa 31 pares en todas las capas: misma limitacion de muestra.",
        "5 semillas aleatorias dan un nulo descriptivo, no una distribucion.",
        "Poblacion: subpoblacion en riesgo del ancla 'unknown', 2% del corpus ambiguo.",
        "Un solo modelo, una generacion determinista por item y brazo.",
        "El conjunto disambig no esta emparejado item a item con el ambiguo.",
    ]
    report["metadata"]["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito {OUT}  ({report['metadata']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Transferencia CAUSAL: ¿funciona el vector sobre una categoria que nunca vio?

El script 22 muestra que la direccion transfiere en LECTURA: construida sin un solo item de
Age, ordena items de Age (AUC 0.6531, p=0.003, supera Bonferroni). Aqui se prueba en
ESCRITURA, que es lo que decide si el agnosticismo al tema sirve para algo.

PROPIEDAD QUE HACE ESTO LIMPIO. De los 31 pares emparejados, 16 son de Age. Al excluir Age
quedan 15 pares de 5 categorias (Physical_appearance, Disability_status, Nationality,
Race_x_SES, SES) y NINGUN item de Age participa en la construccion. La disjuncion entre
vector y conjunto de prueba es TOTAL POR CONSTRUCCION, no por muestreo: no hay que retener
nada ni confiar en un sorteo.

    v_aband^(sin Age)  <- 15 pares, cero items de Age
    evaluacion         <- los 130 items de Age del ancla 'unknown'

CAPA 22, la misma del resultado principal. Es una eleccion conservadora y deliberada: el
script 22 dice que para Age la mejor capa de transferencia es la 20 (0.6531) mientras la 22
transfiere peor (0.5798). Elegir la 20 seria seleccionar sobre los mismos items que luego se
evaluan. Se usa la 22 porque es la capa del recipe declarado, y se declara ANTES de correr
que esto hace la prueba conservadora: un nulo en capa 22 no cerraria la pregunta.

Resto del protocolo identico al script 18: intervencion en la generacion, lectura sin
intervencion, metrica primaria P(rol final = 'unknown'), brazo disambig obligatorio,
direcciones aleatorias de norma igualada, McNemar exacto sobre items pareados, y
persistencia por item antes de agregar.

Salida: output/causal_transfer_age.json, causal_transfer_age_items.jsonl
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
OUT = OUT_DIR / "causal_transfer_age.json"
ITEMS_OUT = OUT_DIR / "causal_transfer_age_items.jsonl"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 20260910
LAYER = 22
HELD_OUT = "Age"
DOSES = [-0.125, -0.250]
RANDOM_DOSE = -0.250
RANDOM_SEEDS = [20260910 + i for i in range(4)]

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

    print(f"[1/4] v_aband SIN un solo ítem de {HELD_OUT}...", flush=True)
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

    keep = [i for i, r in enumerate(pa) if r["category"] != HELD_OUT]
    ia = [pa[i]["idx"] for i in keep]
    ib = [pb[i]["idx"] for i in keep]
    train_cats = dict(collections.Counter(pa[i]["category"] for i in keep))
    print(f"      {len(keep)} pares de {len(train_cats)} categorías: {train_cats}", flush=True)
    assert HELD_OUT not in train_cats, "ABORTA: la categoría retenida entró en el vector"

    Xa = A[ia][:, li].astype(np.float32)
    Xb = A[ib][:, li].astype(np.float32)
    v = (Xa - Xb).mean(axis=0)
    v = v / np.linalg.norm(v)
    scale = float(np.median(np.linalg.norm(A[:, li].astype(np.float32), axis=1)))
    v_aband = v * scale

    # referencia: coseno con el vector completo (que SI incluye Age)
    Xa_all = A[[r["idx"] for r in pa]][:, li].astype(np.float32)
    Xb_all = A[[r["idx"] for r in pb]][:, li].astype(np.float32)
    v_all = (Xa_all - Xb_all).mean(axis=0)
    v_all = v_all / np.linalg.norm(v_all)
    cos_full = float(v @ v_all)
    print(f"      capa {LAYER}, norma {scale:.2f}, cos con el vector completo = {cos_full:.4f}",
          flush=True)

    print("[2/4] Modelo y conjuntos...", flush=True)
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
    steerer = s16.Steerer(model, LAYER)

    corpus = s09.load_corpus_items()
    age_recs = [r for r in sub if r["category"] == HELD_OUT]
    amb = [corpus[tuple(r["key"])] for r in age_recs]
    dpool = [it for it in corpus.values()
             if it["condition"] == "disambig" and it["category"] == HELD_OUT]
    dis = [dpool[i] for i in rng.permutation(len(dpool))[:args.n_disambig]]
    print(f"      evaluación: {len(amb)} ítems ambiguos de {HELD_OUT} "
          f"(disjuntos por construcción), {len(dis)} disambig de {HELD_OUT}", flush=True)

    print("[3/4] Brazos...", flush=True)
    fh = ITEMS_OUT.open("w", encoding="utf-8")
    arms = {}

    def do(tag, delta):
        rows = s18.run(model, tok, lids, amb, dis, steerer, delta, tag)["rows"]
        arms[tag] = rows
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        a, d = s18.agg(rows, "ambig"), s18.agg(rows, "disambig")
        print(f"      {tag:<22} ambig acc={a.get('accuracy'):.4f} "
              f"pasos={a.get('mean_steps'):.2f} | disambig acc={d.get('accuracy'):.4f}",
              flush=True)

    do("baseline", None)
    for dose in DOSES:
        do(f"direction_{dose}", dose * v_aband)
    for sd in RANDOM_SEEDS:
        r0 = np.random.default_rng(sd).normal(size=v_aband.shape[0])
        do(f"random_{sd}", RANDOM_DOSE * r0 / np.linalg.norm(r0) * scale)
    fh.close()
    steerer.remove()

    print("[4/4] McNemar pareado...", flush=True)
    base = arms["baseline"]
    base_steps = s18.agg(base, "ambig")["mean_steps"]
    report = {
        "metadata": {"model": MODEL, "seed": SEED, "layer": LAYER,
                     "held_out_category": HELD_OUT,
                     "n_train_pairs": len(keep), "train_categories": train_cats,
                     "cos_with_full_vector": cos_full,
                     "n_ambig": len(amb), "n_disambig": len(dis),
                     "doses": DOSES, "random_dose": RANDOM_DOSE,
                     "random_seeds": RANDOM_SEEDS, "residual_norm_median": scale,
                     "python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "design": ("el vector no contiene NINGUN item de la categoria evaluada; la "
                   "disjuncion es total por construccion, no por muestreo"),
        "declared_before_running": (
            f"capa {LAYER} es la del recipe declarado, no la que mejor transfiere a "
            f"{HELD_OUT} segun el script 22 (que es la 20). Eleccion conservadora: un nulo "
            f"en capa {LAYER} NO cerraria la pregunta."),
        "correlational_reference": {
            "script_22_fold_A_layer_20": {"auc": 0.6531, "p": 0.0030},
            "script_22_fold_A_layer_22": {"auc": 0.5798, "p": 0.0779}},
        "baseline": {"ambig": s18.agg(base, "ambig"),
                     "disambig": s18.agg(base, "disambig")},
        "arms": {},
    }
    for tag, rows in arms.items():
        if tag == "baseline":
            continue
        a = s18.agg(rows, "ambig")
        report["arms"][tag] = {
            "ambig": a, "disambig": s18.agg(rows, "disambig"),
            "mcnemar_ambig": s18.paired(base, rows, "ambig"),
            "mcnemar_disambig": s18.paired(base, rows, "disambig"),
            "within_window": bool(0.70 * base_steps <= a.get("mean_steps", -1)
                                  <= 1.25 * base_steps)}
        m = report["arms"][tag]["mcnemar_ambig"]
        md = report["arms"][tag]["mcnemar_disambig"]
        print(f"      {tag:<22} ambig Δ={m.get('delta', float('nan')):+.4f} "
              f"p={m.get('p_value', float('nan')):.4f} "
              f"(+{m.get('n_fixed')}/-{m.get('n_broken')}) | "
              f"disambig Δ={md.get('delta', float('nan')):+.4f}", flush=True)

    rnd = [report["arms"][t]["mcnemar_ambig"].get("delta", np.nan)
           for t in report["arms"] if t.startswith("random_")]
    rnd = [x for x in rnd if not np.isnan(x)]
    dirs = [report["arms"][t] for t in report["arms"] if t.startswith("direction_")]
    if rnd and dirs:
        best = max(dirs, key=lambda x: x["mcnemar_ambig"].get("delta", -9))
        report["verdict"] = {
            "delta_ambig": best["mcnemar_ambig"]["delta"],
            "p_ambig": best["mcnemar_ambig"]["p_value"],
            "delta_disambig": best["mcnemar_disambig"].get("delta"),
            "random_max": float(np.max(rnd)), "random_mean": float(np.mean(rnd)),
            "random_sd": float(np.std(rnd, ddof=1)) if len(rnd) > 1 else None,
            "exceeds_random": bool(best["mcnemar_ambig"]["delta"] > np.max(rnd)),
            "comparison_main_result": {
                "main_delta": 0.150, "main_p": 0.0000,
                "note": "el resultado principal usa un vector que SI vio estas categorias"},
        }
    report["limitations"] = [
        "15 pares construyen el vector: estimacion mas fina que la del resultado principal.",
        f"Capa {LAYER} transfiere peor a {HELD_OUT} que la 20 segun el script 22; la prueba "
        "es conservadora por diseño declarado.",
        f"{HELD_OUT} puede ser cualitativamente distinta: su tasa de abandono es 55% frente "
        "al 9% de SES.",
        "4 semillas aleatorias: nulo descriptivo, no distribucion.",
        "Un solo modelo, una generacion determinista por item y brazo.",
    ]
    report["metadata"]["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"escrito {OUT}  ({report['metadata']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""¿La señal se COMPUTA en el CoT, o ya estaba en la pregunta?

El reporte afirma que la señal de abandono «no esta en la pregunta sino que se computa».
Esa afirmacion NO se ha medido. Las activaciones del script 15 se toman despues de la
PRIMERA FRASE de razonamiento; nunca se midio el estado antes de razonar.

Es perfectamente posible que la señal ya estuviera al leer la pregunta y el primer paso no
aporte nada. Esta prueba lo decide:

    si el paso 0 da AZAR      -> la señal se computa durante el CoT. Afirmacion demostrada.
    si el paso 0 da lo mismo  -> ya estaba en la pregunta. Hay que reescribir el reporte.

Dos variantes de prompt del paso 0, para que la comparacion no dependa del formato:

  (a) 'cot_vacio': EXACTAMENTE la estructura del paso 1 con el razonamiento vacio.
      Misma plantilla de tres turnos, solo se quita el texto. Es la comparacion controlada.
  (b) 'directo': el prompt de respuesta directa del Experimento 1, sin mencion de razonar.

Se aplica el MISMO contraste emparejado y los MISMOS pliegues agrupados por par del
script 20, para que la unica diferencia sea de donde salen las activaciones.

Salida: output/step0_baseline.json, step0_acts.npz
"""

from __future__ import annotations

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
ACTS1 = OUT_DIR / "revision_scaled_acts.npz"
OUT = OUT_DIR / "step0_baseline.json"
ACTS0 = OUT_DIR / "step0_acts.npz"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [0, 5, 10, 15, 18, 20, 21, 22, 24, 26, 28]

_s09 = importlib.util.spec_from_file_location(
    "s09", REPO / "scripts" / "09_meanpool_and_cot_faithfulness.py")
s09 = importlib.util.module_from_spec(_s09)
_s09.loader.exec_module(s09)

_s20 = importlib.util.spec_from_file_location(
    "s20", REPO / "scripts" / "20_paired_fold_correction.py")
s20 = importlib.util.module_from_spec(_s20)
_s20.loader.exec_module(s20)


def main() -> None:
    t0 = time.time()
    recs = [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    sub = [r for r in recs if r["first_role"] == "unknown"]
    print(f"ancla 'unknown': {len(sub)} ítems", flush=True)

    print("[1/3] Modelo...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    corpus = s09.load_corpus_items()

    print("[2/3] Activaciones del PASO 0 (sin razonamiento)...", flush=True)
    acts = {"cot_vacio": [], "directo": []}
    for n, r in enumerate(sub):
        it = corpus[tuple(r["key"])]
        prompts = {"cot_vacio": s09.cot_prompt(it, "", tok),
                   "directo": s09.chat_prompt(it, tok)}
        for variant, p in prompts.items():
            enc = tok(p, return_tensors="pt").to(model.device)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            acts[variant].append(
                np.stack([hs[L][0, -1].float().cpu().numpy() for L in LAYERS]))
        if n % 50 == 0:
            print(f"      {n}/{len(sub)}", flush=True)

    A0 = {k: np.stack(v) for k, v in acts.items()}
    np.savez_compressed(ACTS0, layers=np.array(LAYERS),
                        **{k: v.astype(np.float16) for k, v in A0.items()})
    print(f"      guardadas {[(k, v.shape) for k, v in A0.items()]}", flush=True)

    print("[3/3] Mismo contraste emparejado, mismos pliegues agrupados...", flush=True)
    # el indice de `sub` es la posicion en A0; el de recs es la posicion en A1
    pos = {r["idx"]: i for i, r in enumerate(sub)}
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
    print(f"      {len(pa)} pares (los mismos del script 20)", flush=True)

    y = np.r_[np.ones(len(pa)), np.zeros(len(pb))].astype(int)
    pid = np.r_[np.arange(len(pa)), np.arange(len(pb))]
    idx0 = [pos[r["idx"]] for r in pa] + [pos[r["idx"]] for r in pb]
    idx1 = [r["idx"] for r in pa] + [r["idx"] for r in pb]

    z1 = np.load(ACTS1)
    A1 = z1["acts"]

    report = {
        "metadata": {"model": MODEL, "layers": LAYERS, "n_pairs": len(pa),
                     "n_items": len(sub), "python": platform.python_version(),
                     "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "question": ("¿la señal de abandono ya esta en el prompt, o la construye el "
                     "primer paso de razonamiento?"),
        "by_layer": {},
    }
    print(f"\n{'capa':>5}{'paso0 cot_vacio':>18}{'paso0 directo':>16}{'paso1 (ref)':>14}")
    print("-" * 55)
    for li, layer in enumerate(LAYERS):
        row = {}
        for variant in ("cot_vacio", "directo"):
            X = A0[variant][idx0][:, li].astype(np.float32)
            a = s20.auc_grouped(X, y, pid)
            row[variant] = {"auc": a,
                            "p_value": s20.perm_p(s20.auc_grouped, X, y, pid, a)
                            if not np.isnan(a) else float("nan")}
        X1 = A1[idx1][:, li].astype(np.float32)
        a1 = s20.auc_grouped(X1, y, pid)
        row["paso1_referencia"] = {"auc": a1}
        row["delta_paso1_menos_cot_vacio"] = (
            a1 - row["cot_vacio"]["auc"]
            if not (np.isnan(a1) or np.isnan(row["cot_vacio"]["auc"])) else None)
        report["by_layer"][str(layer)] = row
        if not np.isnan(row["cot_vacio"]["auc"]):
            print(f"{layer:>5}{row['cot_vacio']['auc']:>12.4f}"
                  f" (p={row['cot_vacio']['p_value']:.3f})"
                  f"{row['directo']['auc']:>16.4f}{a1:>14.4f}")

    valid = {k: v for k, v in report["by_layer"].items()
             if not np.isnan(v["cot_vacio"]["auc"])}
    if valid:
        best = max(valid, key=lambda k: valid[k]["cot_vacio"]["auc"])
        report["best_step0"] = {"layer": int(best), **valid[best]["cot_vacio"]}
        report["bonferroni_threshold"] = 0.05 / len(valid)
        report["step0_survives"] = bool(
            valid[best]["cot_vacio"]["p_value"] < 0.05 / len(valid))
        deltas = [v["delta_paso1_menos_cot_vacio"] for v in valid.values()
                  if v["delta_paso1_menos_cot_vacio"] is not None]
        report["mean_gain_from_step1"] = float(np.mean(deltas))
        late = [v["delta_paso1_menos_cot_vacio"] for k, v in valid.items()
                if int(k) >= 18 and v["delta_paso1_menos_cot_vacio"] is not None]
        report["mean_gain_layers_18plus"] = float(np.mean(late))
        print(f"\nmejor paso 0: capa {best} AUC={valid[best]['cot_vacio']['auc']:.4f} "
              f"p={valid[best]['cot_vacio']['p_value']:.4f}  "
              f"{'SUPERA' if report['step0_survives'] else 'no supera'} Bonferroni")
        print(f"ganancia media del paso 1 sobre el paso 0: "
              f"{report['mean_gain_from_step1']:+.4f}")
        print(f"  ...restringida a capas >= 18: {report['mean_gain_layers_18plus']:+.4f}")

    report["interpretation_rule"] = {
        "paso0 en azar y paso1 alto": "la señal se COMPUTA en el CoT (afirmacion sostenida)",
        "ambos altos": "la señal ya estaba en la PREGUNTA (hay que reescribir el reporte)",
        "ambos en azar": "incoherente con el script 20; revisar la extraccion",
    }
    report["limitations"] = [
        "El prompt del paso 0 'cot_vacio' contiene un turno de asistente vacio, que no es "
        "una situacion natural; por eso se reporta tambien la variante 'directo'.",
        "Mismos 31 pares que el script 20: misma limitacion de muestra.",
        "La comparacion es entre AUC, no entre modelos anidados; no hay test formal de la "
        "diferencia paso1 - paso0.",
    ]
    report["metadata"]["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nescrito {OUT}  ({report['metadata']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

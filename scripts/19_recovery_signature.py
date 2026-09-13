#!/usr/bin/env python3
"""La poblacion espejo: ¿predice el estado quien SALE de una respuesta equivocada?

Los scripts 17-18 estudian items donde el modelo aun acierta tras el paso 1 y algunos
abandonan. Esta es la pregunta simetrica, sobre la poblacion que quedo fuera: items donde
el modelo YA esta respondiendo mal tras el paso 1.

    tras el paso 1 en ESTEREOTIPO (363):  178 se recuperan a 'unknown', 178 se quedan
    tras el paso 1 en ANTI        (338):  152 se recuperan,             182 se quedan

Dos motivos para hacerla:

1. Es la pregunta que importa si lo que se quiere es sacar al modelo de una respuesta
   estereotipica concreta, no del acto generico de elegir grupo. La metrica de exito ya no
   es «sostener la abstencion» sino «salir del estereotipo».

2. Contraste balanceado 178/178 y el doble de muestra que el ancla 'unknown' (87/146).
   Donde un test tiene mas potencia es con clases equilibradas.

Advertencia. El script 15 ya probo el ancla 'stereotyped' y FALLO al emparejar
(AUC 0.5912, p=0.055): la señal sin emparejar era plana desde la capa 5, o sea categoria y
plantilla. Pero uso la etiqueta sucia `changed` = «la trayectoria no fue constante», que
mezcla recuperarse con oscilar. En el ancla 'unknown' limpiar esa etiqueta subio el AUC
emparejado de 0.7269 a 0.7599 CON LA MITAD DE PARES. Aqui se prueba la etiqueta limpia
sobre la poblacion espejo. Puede volver a fallar; eso tambien seria informativo, porque
distinguiria «el estado predice el desenlace» de «el estado predice el desenlace solo
cuando el modelo aun va bien».

No requiere GPU: reutiliza los crudos del script 15.

Salida: output/recovery_signature.json
"""

from __future__ import annotations

import collections
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/experiments/EXP-002_bbq_stereotype_direction/output"
RECORDS = OUT_DIR / "revision_scaled_records.jsonl"
ACTS = OUT_DIR / "revision_scaled_acts.npz"
OUT = OUT_DIR / "recovery_signature.json"

_s17 = importlib.util.spec_from_file_location(
    "s17", REPO / "scripts" / "17_abandonment_signature.py")
s17 = importlib.util.module_from_spec(_s17)
_s17.loader.exec_module(s17)


def main() -> None:
    t0 = time.time()
    recs = [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    z = np.load(ACTS)
    A, layers = z["acts"], [int(x) for x in z["layers"]]
    print(f"registros {len(recs)}  activaciones {A.shape}")

    report = {
        "metadata": {"seed": s17.SEED, "n_folds": s17.N_FOLDS,
                     "n_permutations": s17.N_PERM, "layers": layers,
                     "python": platform.python_version()},
        "design": {
            "population": ("items donde el modelo YA responde mal tras el paso 1 "
                           "(espejo de los scripts 17-18)"),
            "label": ("se recupero = el rol final difiere del rol del paso 1; "
                      "se quedo = todos los pasos con el mismo rol; "
                      "los oscilantes (vuelven al rol inicial) se descartan"),
            "prior_failure": ("el script 15 probo este ancla con la etiqueta sucia "
                              "'changed' y fallo al emparejar (0.5912, p=0.055)"),
        },
        "anchors": {},
    }

    for anchor in ("stereotyped", "anti_stereotyped"):
        sub = [r for r in recs if r["first_role"] == anchor]
        left = [r for r in sub if r["final_role"] != anchor]
        stayed = [r for r in sub if all(x == anchor for x in r["roles"])]
        wobbled = [r for r in sub
                   if r["final_role"] == anchor and any(x != anchor for x in r["roles"])]
        dest = collections.Counter(r["final_role"] for r in left)
        print(f"\n########## ancla '{anchor}' (n={len(sub)}) ##########")
        print(f"  salio={len(left)}  se quedo={len(stayed)}  "
              f"oscilo={len(wobbled)} (descartados)")
        print(f"  destino de los que salieron: {dict(dest)}")

        blk = {"n_total": len(sub), "n_left": len(left), "n_stayed": len(stayed),
               "n_wobbled_discarded": len(wobbled),
               "destination_of_left": dict(dest),
               "recovery_rate": len(left) / len(sub) if sub else None,
               "trace_length_control": {
                   "left_mean_steps": float(np.mean([r["n_steps"] for r in left])) if left else None,
                   "stayed_mean_steps": float(np.mean([r["n_steps"] for r in stayed])) if stayed else None,
               }}

        if len(left) < 25 or len(stayed) < 25:
            blk["note"] = "muestra insuficiente"
            report["anchors"][anchor] = blk
            continue

        idx = [r["idx"] for r in left] + [r["idx"] for r in stayed]
        y = np.r_[np.ones(len(left)), np.zeros(len(stayed))].astype(int)
        blk["unmatched"] = s17.evaluate(
            A, layers, idx, y, f"{anchor}: salio vs se quedo · sin emparejar",
            with_mlp=True)

        a, b = s17.match_pairs(left, stayed)
        if len(a) >= 20:
            idx_m = [r["idx"] for r in a] + [r["idx"] for r in b]
            y_m = np.r_[np.ones(len(a)), np.zeros(len(b))].astype(int)
            m = s17.evaluate(A, layers, idx_m, y_m,
                             f"{anchor}: salio vs se quedo · emparejado ({len(a)} pares)",
                             with_mlp=True)
            m["n_pairs"] = len(a)
            blk["matched"] = m
        else:
            blk["matched"] = {"note": f"solo {len(a)} pares"}
        report["anchors"][anchor] = blk

    # Comparacion entre anclas: ¿la señal es sobre «salir de una respuesta» en general,
    # o solo sobre «sostener la correcta»? Es lo que decide la interpretacion.
    comp = {}
    for anchor, blk in report["anchors"].items():
        m = blk.get("matched", {})
        comp[anchor] = {"best_auc": m.get("best_layer", {}).get("auc"),
                        "best_layer": m.get("best_layer", {}).get("layer"),
                        "p_value": m.get("best_layer", {}).get("p_value"),
                        "n_pairs": m.get("n_pairs"),
                        "survives_bonferroni": m.get("survives_bonferroni")}
    comp["reference_unknown_anchor"] = {
        "best_auc": 0.7599, "best_layer": 22, "p_value": 0.005, "n_pairs": 31,
        "source": "script 17, poblacion donde el modelo aun acertaba"}
    report["cross_anchor_comparison"] = comp

    report["limitations"] = [
        "Tercera etiqueta probada sobre los mismos datos; los p-valores no incorporan esa "
        "seleccion. Solo una prueba causal con generacion nueva la compensa.",
        "La capa 0 sale NaN (ultimo token identico): los tests efectivos son 10, no 11.",
        "Se descartan los oscilantes; la conclusion no habla de ellos.",
        "Correlacional. Que el estado prediga la recuperacion no implica que empujarlo la "
        "provoque; eso exigiria el equivalente del script 18 sobre esta poblacion.",
        "Un solo modelo, una generacion determinista por item.",
    ]
    report["metadata"]["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== resumen emparejado ===")
    for k, v in comp.items():
        if v.get("best_auc") is not None:
            print(f"  {k:<28} capa {v['best_layer']}: AUC={v['best_auc']:.4f} "
                  f"p={v['p_value']:.4f}  ({v['n_pairs']} pares)")
    print(f"\nescrito {OUT}  ({report['metadata']['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

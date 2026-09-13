"""Recalcula el perfil usando TODAS las semillas aleatorias disponibles.

El script 26 solo leia 3 semillas en el bucle del perfil, pero la capa 18 a dosis -0.25
tiene 5 disponibles desde el script 25. La semilla 20260915 (+0.1280) quedaba fuera.

Ademas se reporta media y desviacion del brazo aleatorio, no solo el maximo: con pocas
semillas el maximo es un estimador muy fragil.
"""
import collections
import json
from pathlib import Path

import numpy as np
from scipy import stats

D = Path("/mnt/c/Users/brian/Desktop/projects/niel_landa/data/experiments"
         "/EXP-002_bbq_stereotype_direction/output")


def load(p):
    out = collections.defaultdict(list)
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["arm"]].append(r)
    return out


new = load(D / "specificity_profile_items.jsonl")
prev = load(D / "layer_sweep_items.jsonl")

base = [r for r in new["baseline"] if r["cond"] == "ambig"]
base_d = [r for r in new["baseline"] if r["cond"] == "disambig"]


def mcnemar(b, a):
    pairs = [(x["correct"], y["correct"]) for x, y in zip(b, a)
             if x["usable"] and y["usable"]]
    if not pairs:
        return None
    br = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    p = 1.0 if br + c == 0 else float(stats.binomtest(c, br + c, 0.5).pvalue)
    return {"delta": (c - br) / len(pairs), "p": p, "n": len(pairs)}


def arm(tag, cond):
    for src in (new, prev):
        if tag in src:
            return [r for r in src[tag] if r["cond"] == cond]
    return None


# mapa de brazos disponibles por (capa, dosis)
LAYERS = [18, 20, 22]
DOSES = [-0.125, -0.25]
DIRECT = {(18, -0.125): "phase2_direction_-0.125", (18, -0.25): "phase2_direction_-0.25"}
for L in (20, 22):
    for d in DOSES:
        DIRECT[(L, d)] = f"L{L}_direction_{d}"

RANDOM = collections.defaultdict(list)
for L in (20, 22):
    for d in DOSES:
        for sd in (20260911, 20260912, 20260913):
            RANDOM[(L, d)].append(f"L{L}_random_{sd}_{d}")
for sd in (20260911, 20260912, 20260913):
    RANDOM[(18, -0.125)].append(f"L18_random_{sd}_-0.125")
# capa 18 a -0.25: LAS CINCO del script 25
for sd in (20260911, 20260912, 20260913, 20260914, 20260915):
    RANDOM[(18, -0.25)].append(f"phase2_random_{sd}")

print(f"{'capa':>5}{'dosis':>8}{'Δambig':>9}{'p':>8}  |"
      f"{'n':>3}{'alea μ':>9}{'alea σ':>9}{'alea máx':>10}{'z':>7}  |"
      f"{'Δdisamb':>9}{'p':>8}{'compuesto':>11}")
print("-" * 100)
rows = {}
for L in LAYERS:
    for d in DOSES:
        da = mcnemar(base, arm(DIRECT[(L, d)], "ambig"))
        dd = mcnemar(base_d, arm(DIRECT[(L, d)], "disambig"))
        rnd = []
        for t in RANDOM[(L, d)]:
            a = arm(t, "ambig")
            if a:
                m = mcnemar(base, a)
                if m:
                    rnd.append(m["delta"])
        mu = float(np.mean(rnd)) if rnd else float("nan")
        sd_ = float(np.std(rnd, ddof=1)) if len(rnd) > 1 else float("nan")
        mx = float(np.max(rnd)) if rnd else float("nan")
        z = (da["delta"] - mu) / sd_ if sd_ and not np.isnan(sd_) and sd_ > 0 else float("nan")
        comp = da["delta"] - max(0.0, -dd["delta"])
        rows[(L, d)] = {"delta_ambig": da["delta"], "p_ambig": da["p"],
                        "n_random": len(rnd), "random_mean": mu, "random_sd": sd_,
                        "random_max": mx, "z_vs_random": z,
                        "delta_disambig": dd["delta"], "p_disambig": dd["p"],
                        "composite": comp}
        print(f"{L:>5}{d:>8.3f}{da['delta']:>9.4f}{da['p']:>8.4f}  |"
              f"{len(rnd):>3}{mu:>9.4f}{sd_:>9.4f}{mx:>10.4f}{z:>7.2f}  |"
              f"{dd['delta']:>9.4f}{dd['p']:>8.4f}{comp:>11.4f}")

print("\n=== como leerlo ===")
print("  z = (efecto − media aleatoria) / desviacion aleatoria")
print("      mas estable que el ratio contra el maximo con pocas semillas")
print("  compuesto = Δambig − max(0, −Δdisambig)")
print("      pondera ambos por igual; es UNA eleccion, no un hecho")

print("\n=== comparacion con el resultado principal (script 18, n=200) ===")
print("  capa 22, dosis -0.25 : alli +0.1500, aqui "
      f"{rows[(22, -0.25)]['delta_ambig']:+.4f}  (conjuntos de items distintos)")
print("  capa 22, dosis -0.125: alli +0.0800, aqui "
      f"{rows[(22, -0.125)]['delta_ambig']:+.4f}")

out = {"note": ("recalculo del script 26 usando TODAS las semillas disponibles; el bucle "
                "original solo leia 3 y dejaba fuera la de mayor efecto en la capa 18"),
       "profile": {f"L{L}_{d}": v for (L, d), v in rows.items()}}
(D / "specificity_profile_fixed.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nescrito {D / 'specificity_profile_fixed.json'}")

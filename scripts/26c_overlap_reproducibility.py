"""¿La diferencia +0.150 (n=200) vs +0.096 (n=125) en capa 22 es muestreo o no-determinismo?

Los dos conjuntos salen del mismo grupo de 245 items disponibles del ancla 'unknown'.
Se reconstruye que items vio cada script replicando su logica de seleccion:

  script 18: rng = default_rng(20260908); primer uso = permutation(245)[:200]
  script 16: misma semilla, primer uso = permutation(245)[:100]  -> sus 100 son las 100 primeras de 18
  script 25/26: split registrado en layer_sweep_split.json (confirm_keys, 125)

Luego, sobre los items COMUNES, se compara acierto por acierto. La generacion es greedy,
asi que si la configuracion es la misma, cualquier desacuerdo es no-determinismo numerico
(p. ej. padding distinto por composicion de lote en bf16), no muestreo.
"""
import collections
import json
from pathlib import Path

import numpy as np

D = Path("/mnt/c/Users/brian/Desktop/projects/niel_landa/data/experiments"
         "/EXP-002_bbq_stereotype_direction/output")

recs = [json.loads(l) for l in (D / "revision_scaled_records.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]
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
avail = [r for r in sub if r["idx"] not in used]
print(f"items disponibles (fuera de los 62 del vector): {len(avail)}")

perm = np.random.default_rng(20260908).permutation(len(avail))
keys18 = [tuple(avail[i]["key"]) for i in perm[:200]]
keys16 = [tuple(avail[i]["key"]) for i in perm[:100]]
split = json.loads((D / "layer_sweep_split.json").read_text(encoding="utf-8"))
keys26 = [tuple(k) for k in split["confirm_keys"]]
keysdev = [tuple(k) for k in split["dev_keys"]]

s18, s26 = set(keys18), set(keys26)
print(f"script 18: {len(keys18)}  script 26 (confirmacion): {len(keys26)}  "
      f"desarrollo: {len(keysdev)}")
print(f"solape 18 ∩ 26: {len(s18 & s26)}   solo 18: {len(s18 - s26)}   "
      f"solo 26: {len(s26 - s18)}")
print(f"solape 16 ∩ 26: {len(set(keys16) & s26)}   (16 ⊂ 18: "
      f"{set(keys16) <= s18})")


def arms_of(path):
    out = collections.defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["cond"] == "ambig":
                out[r["arm"]].append(r)
    return out


a18 = arms_of(D / "steering_confirmation_items.jsonl")
a26 = arms_of(D / "specificity_profile_items.jsonl")


def by_key(rows, keys):
    assert len(rows) == len(keys), (len(rows), len(keys))
    return {k: r for k, r in zip(keys, rows)}


for label, t18, t26 in (("linea base", "baseline", "baseline"),
                        ("capa 22, dosis -0.125", "direction_-0.125", "L22_direction_-0.125"),
                        ("capa 22, dosis -0.25", "direction_-0.25", "L22_direction_-0.25")):
    m18 = by_key(a18[t18], keys18)
    m26 = by_key(a26[t26], keys26)
    common = [k for k in keys26 if k in m18]
    both_usable = [k for k in common if m18[k]["usable"] and m26[k]["usable"]]
    agree = sum(1 for k in both_usable if m18[k]["correct"] == m26[k]["correct"])
    print(f"\n=== {label}: items comunes utilizables {len(both_usable)} ===")
    print(f"  mismo veredicto en {agree}/{len(both_usable)} "
          f"({agree / max(len(both_usable), 1):.1%})")
    same_final = sum(1 for k in both_usable if m18[k].get("final") == m26[k].get("final"))
    print(f"  mismo rol final en {same_final}/{len(both_usable)}")

# deltas por subconjunto, cada uno contra su propia linea base
def delta(base, arm, keys):
    b = by_key(base, keys)
    a = by_key(arm, keys)
    return b, a


b18, d18 = delta(a18["baseline"], a18["direction_-0.25"], keys18)
b26, d26 = delta(a26["baseline"], a26["L22_direction_-0.25"], keys26)


def dsub(b, a, subset):
    ks = [k for k in subset if b[k]["usable"] and a[k]["usable"]]
    if not ks:
        return float("nan"), 0
    fixed = sum(1 for k in ks if a[k]["correct"] and not b[k]["correct"])
    broke = sum(1 for k in ks if b[k]["correct"] and not a[k]["correct"])
    return (fixed - broke) / len(ks), len(ks)


common = sorted(s18 & s26)
only18 = sorted(s18 - s26)
print("\n=== delta capa 22, dosis -0.25, por subconjunto ===")
for name, b, a, ks in (("18, sobre comunes", b18, d18, common),
                       ("18, solo 18", b18, d18, only18),
                       ("18, total", b18, d18, keys18),
                       ("26, sobre comunes", b26, d26, common),
                       ("26, total", b26, d26, keys26)):
    dv, n = dsub(b, a, ks)
    print(f"  {name:<22} n={n:>4}  Δ={dv:+.4f}")

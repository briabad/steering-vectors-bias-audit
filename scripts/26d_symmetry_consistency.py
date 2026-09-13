"""¿Es consistente entre corridas la simetria de errores corregidos (18 estereo / 18 anti)?

E2 (script 18, 200 items) y E3 (script 26, 125 items) aplicaron la misma intervencion
(capa 22) y guardaron el rol final por item. No se genera nada: solo se cuenta.
"""
import collections
import json
from pathlib import Path

import numpy as np
from scipy import stats

D = Path("/mnt/c/Users/brian/Desktop/projects/niel_landa/data/experiments"
         "/EXP-002_bbq_stereotype_direction/output")

# --- reconstruir las claves de E2 (misma logica que overlap_check.py)
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
perm = np.random.default_rng(20260908).permutation(len(avail))
keys_e2 = [tuple(avail[i]["key"]) for i in perm[:200]]
keys_e3 = [tuple(k) for k in json.loads((D / "layer_sweep_split.json")
                                        .read_text(encoding="utf-8"))["confirm_keys"]]


def arms_of(path):
    out = collections.defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["cond"] == "ambig":
                out[r["arm"]].append(r)
    return out


E2 = arms_of(D / "steering_confirmation_items.jsonl")
E3 = arms_of(D / "specificity_profile_items.jsonl")

SHORT = {"unknown": "unknown", "stereotyped": "estereo", "anti_stereotyped": "anti"}


def flow(base_rows, arm_rows, keys, label):
    b = dict(zip(keys, base_rows))
    a = dict(zip(keys, arm_rows))
    ks = [k for k in keys if b[k]["usable"] and a[k]["usable"]]
    trans = collections.Counter((b[k]["final"], a[k]["final"]) for k in ks)
    err_b = collections.Counter(b[k]["final"] for k in ks if b[k]["final"] != "unknown")
    err_a = collections.Counter(a[k]["final"] for k in ks if a[k]["final"] != "unknown")
    fs = trans[("stereotyped", "unknown")]
    fa = trans[("anti_stereotyped", "unknown")]
    nb_s, nb_a = err_b["stereotyped"], err_b["anti_stereotyped"]
    p0 = nb_s / (nb_s + nb_a) if (nb_s + nb_a) else float("nan")
    pbin = (float(stats.binomtest(fs, fs + fa, p0).pvalue)
            if fs + fa > 0 and not np.isnan(p0) else float("nan"))
    print(f"\n=== {label}  (n={len(ks)}) ===")
    print(f"  errores de partida : estereo {nb_s:>3}   anti {nb_a:>3}   "
          f"(estereo = {p0:.1%})")
    print(f"  errores al final   : estereo {err_a['stereotyped']:>3}   "
          f"anti {err_a['anti_stereotyped']:>3}")
    print(f"  CORREGIDOS         : estereo {fs:>3}   anti {fa:>3}   "
          f"(estereo = {fs / max(fs + fa, 1):.1%})")
    print(f"  introducidos       : ->estereo {trans[('unknown', 'stereotyped')]:>2}   "
          f"->anti {trans[('unknown', 'anti_stereotyped')]:>2}")
    print(f"  ¿se corrige en proporcion a los errores de partida? "
          f"binomial p = {pbin:.3f}")
    fixed = {k: b[k]["final"] for k in ks
             if b[k]["final"] != "unknown" and a[k]["final"] == "unknown"}
    return fixed


print("#" * 70)
print("# Capa 22 — la simetria en cada corrida")
print("#" * 70)
f2 = flow(E2["baseline"], E2["direction_-0.25"], keys_e2, "E2, dosis -0.25 (el 18/18 original)")
f3 = flow(E3["baseline"], E3["L22_direction_-0.25"], keys_e3, "E3, dosis -0.25 (segunda corrida)")
flow(E2["baseline"], E2["direction_-0.125"], keys_e2, "E2, dosis -0.125")
flow(E3["baseline"], E3["L22_direction_-0.125"], keys_e3, "E3, dosis -0.125")

print("\n" + "#" * 70)
print("# ¿Se corrigen LOS MISMOS items en las dos corridas? (dosis -0.25)")
print("#" * 70)
common = set(keys_e2) & set(keys_e3)
c2 = {k for k in f2 if k in common}
c3 = {k for k in f3 if k in common}
print(f"  items comunes: {len(common)}")
print(f"  corregidos en E2 entre los comunes: {len(c2)}")
print(f"  corregidos en E3 entre los comunes: {len(c3)}")
print(f"  corregidos en AMBAS: {len(c2 & c3)}   solo E2: {len(c2 - c3)}   "
      f"solo E3: {len(c3 - c2)}")

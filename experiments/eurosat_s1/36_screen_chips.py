# -*- coding: utf-8 -*-
"""
36_screen_chips.py - how many chips are degenerate, and does the gate finding survive?

Found while writing up Measurement 19: the largest gate values all come from
chips with VV around -50 dB, i.e. essentially black. Three appeared in a random
subsample of 400, labelled PermanentCrop and Industrial. The quoted gate range
maximum (2.52) therefore comes from probably-corrupt data, not from a decision.

This script does three things and nothing else:
  1. counts them, by threshold, by class, by split
  2. reports how much of the gate's dynamic range they account for
  3. recomputes r(gate, VV) with them removed - if the -0.90 survives, the
     Measurement 19 conclusion is not an artefact of a handful of bad chips

Zero GPU.
"""
import sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy import stats

LOC = Path(r"C:\Users\aliso\Desktop\big-files\loc")
PROBE = Path(r"C:\Users\aliso\Desktop\big-files\eurosat_s1\probe_features.npz")
CACHE = LOC / "eurosat_s1_cache.npz"
OUT = LOC / "chip_screen_result.json"

z = np.load(PROBE, allow_pickle=True)
X = np.vstack([z["X_train"], z["X_val"], z["X_test"]])
Y = np.concatenate([z["Y_train"], z["Y_val"], z["Y_test"]])
c = np.load(CACHE, allow_pickle=True)
names = c["names"]
if len(X) != len(names):
    raise SystemExit("length mismatch")

vv, vh = X[:, 0], X[:, 5]
sp_geo = np.load(LOC / "geo_split.npz", allow_pickle=True)["splits"]

print("=" * 74)
print("how many chips are essentially black")
print("=" * 74)
counts = {}
for thr in (-45, -40, -35, -30, -25):
    bad = vv < thr
    counts[thr] = int(bad.sum())
    print(f"   VV mean < {thr:4d} dB :  {bad.sum():5d} chips "
          f"({bad.sum()/len(vv)*100:5.2f}%)")

THR = -30
bad = vv < THR
print(f"\nusing threshold {THR} dB -> {bad.sum()} chips")
if bad.sum():
    print("\n   by class:")
    for cl in sorted(set(Y[bad])):
        n = int((Y[bad] == cl).sum())
        print(f"      {cl:22s} {n:4d}")
    print("\n   by geo split:")
    for s in ("train", "val", "test"):
        n = int((bad & (sp_geo == s)).sum())
        print(f"      {s:6s} {n:4d}")
    print(f"\n   their VV range: [{vv[bad].min():.1f}, {vv[bad].max():.1f}] dB")
    print(f"   normal chips  : [{vv[~bad].min():.1f}, {vv[~bad].max():.1f}] dB")

print("\n" + "=" * 74)
print("does r(gate, VV) survive removing them")
print("=" * 74)
va = sp_geo == "val"
bad_va = bad[va]
print(f"   val chips {va.sum()}, of which degenerate {bad_va.sum()}")

res = {"counts_by_threshold": counts, "threshold_used": THR,
       "n_degenerate": int(bad.sum()), "n_degenerate_val": int(bad_va.sum()),
       "seeds": []}

files = sorted((LOC / "runs").glob("gate_s*_geo_gates.npy"))
allr, allr_clean, rng_all, rng_clean = [], [], [], []
for f in files:
    g = np.load(f)
    if len(g) != va.sum():
        continue
    r_all = stats.pearsonr(vv[va], g)[0]
    keep = ~bad_va
    r_cln = stats.pearsonr(vv[va][keep], g[keep])[0]
    allr.append(r_all); allr_clean.append(r_cln)
    rng_all.append((float(g.min()), float(g.max())))
    rng_clean.append((float(g[keep].min()), float(g[keep].max())))
    print(f"\n   {f.stem}")
    print(f"      r(gate, VV)  all chips     {r_all:+.4f}")
    print(f"      r(gate, VV)  degenerate removed {r_cln:+.4f}")
    print(f"      gate range   all chips     [{g.min():.3f}, {g.max():.3f}]")
    print(f"      gate range   cleaned       [{g[keep].min():.3f}, {g[keep].max():.3f}]")
    res["seeds"].append({"file": f.stem, "r_all": float(r_all),
                         "r_clean": float(r_cln),
                         "range_all": [float(g.min()), float(g.max())],
                         "range_clean": [float(g[keep].min()), float(g[keep].max())]})

if allr:
    ma, mc = float(np.mean(allr)), float(np.mean(allr_clean))
    max_all = max(r[1] for r in rng_all)
    max_cln = max(r[1] for r in rng_clean)
    print("\n" + "-" * 74)
    print(f"   mean r across {len(allr)} seeds:  all {ma:+.4f}   cleaned {mc:+.4f}")
    print(f"   largest gate seen:  all {max_all:.3f}   cleaned {max_cln:.3f}")
    res["mean_r_all"], res["mean_r_clean"] = ma, mc
    res["max_gate_all"], res["max_gate_clean"] = max_all, max_cln
    print("\n" + "=" * 74)
    if abs(mc) > 0.8:
        print("VERDICT  the -0.90 finding is NOT an artefact of degenerate chips.")
        print("         It survives with them removed. What changes is the quoted")
        print(f"         dynamic range: {max_all:.2f} -> {max_cln:.2f}. Every future")
        print("         statement about gate range must use the cleaned number.")
    else:
        print("VERDICT  the correlation weakens substantially once degenerate")
        print("         chips are removed. Measurement 19 must be restated.")
    print("=" * 74)

OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
print(f"\n-> {OUT}")

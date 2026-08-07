# -*- coding: utf-8 -*-
"""
34_gate_probe.py - what is the gate actually tracking?

27_gate_mechanism.py answered "is it uncertainty?" with r = +0.118 and stopped.
Two problems with that answer:

  1. It only ever ran on the OFFICIAL split. The glob picks up the geo gate files
     too, but they are 5741 long against a 5400-long confidence vector, so the
     length guard silently skipped every one of them. The refutation on record is
     therefore a refutation measured under leakage.
  2. "not uncertainty" is not an answer to "then what".

This script redoes the confidence test on the geo split and adds three candidate
explanations that cost nothing to test:

  A. class      - is the gate a per-class constant in disguise?
  B. brightness - is it tracking mean radar backscatter, i.e. a trivial image stat?
  C. error      - is it higher where the image-only model is actually wrong
                  (as opposed to merely unsure)?

Zero GPU.
"""
import sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier

LOC = Path(r"C:\Users\aliso\Desktop\big-files\loc")
PROBE = Path(r"C:\Users\aliso\Desktop\big-files\eurosat_s1\probe_features.npz")
CACHE = LOC / "eurosat_s1_cache.npz"
OUT = LOC / "gate_probe_result.json"

# ---------- features in cache order ----------
z = np.load(PROBE, allow_pickle=True)
X = np.vstack([z["X_train"], z["X_val"], z["X_test"]])
Y = np.concatenate([z["Y_train"], z["Y_val"], z["Y_test"]])

c = np.load(CACHE, allow_pickle=True)
names, labels, sp_off = c["names"], c["labels"], c["splits"]
cls_cache = np.array([n.split("_")[0] for n in names])

if len(X) != len(names):
    raise SystemExit(f"length mismatch: probe {len(X)} vs cache {len(names)}")
if not (cls_cache == Y).all():
    raise SystemExit(f"ORDER MISMATCH: {(cls_cache != Y).sum()} labels disagree")
print(f"alignment verified: {len(X)} chips, labels agree")

sp_geo = np.load(LOC / "geo_split.npz", allow_pickle=True)["splits"]


def run(split_name, sp, suffix):
    tr, va = sp == "train", sp == "val"
    print("\n" + "=" * 74)
    print(f"{split_name} split   train {tr.sum()}  val {va.sum()}")
    print("=" * 74)

    clf = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X[tr], Y[tr])
    proba = clf.predict_proba(X[va])
    conf = proba.max(axis=1)
    pred = clf.classes_[proba.argmax(1)]
    correct = pred == Y[va]
    print(f"   image-only accuracy {correct.mean()*100:.2f}"
          f"   confidence mean {conf.mean():.3f}")

    yva = Y[va]
    vv_mean = X[va][:, 0]          # feature 0 = VV mean (dB)
    vh_mean = X[va][:, 5]          # feature 5 = VH mean (dB)

    files = sorted((LOC / "runs").glob(f"gate_s*{suffix}_gates.npy"))
    files = [f for f in files if len(np.load(f)) == va.sum()]
    if not files:
        print("   no matching gate files"); return None

    res = {"split": split_name, "n_val": int(va.sum()), "seeds": []}
    per_class_all = []
    for f in files:
        g = np.load(f)
        r_conf = stats.pearsonr(conf, g)[0]
        rho_conf = stats.spearmanr(conf, g)[0]
        r_vv = stats.pearsonr(vv_mean, g)[0]
        r_vh = stats.pearsonr(vh_mean, g)[0]
        d_err = float(g[~correct].mean() - g[correct].mean())

        cls_means = {cl: float(g[yva == cl].mean()) for cl in sorted(set(yva))}
        per_class_all.append(cls_means)
        # how much of the gate's variance is explained by class identity alone?
        grand = g.mean()
        between = sum(((yva == cl).sum()) * (cls_means[cl] - grand) ** 2
                      for cl in cls_means)
        eta2 = float(between / ((g - grand) ** 2).sum())

        print(f"\n   {f.stem}   range [{g.min():.3f}, {g.max():.3f}]  mean {g.mean():.3f}")
        print(f"      r(gate, image confidence) = {r_conf:+.4f}   rho = {rho_conf:+.4f}")
        print(f"      r(gate, VV mean dB)       = {r_vv:+.4f}")
        print(f"      r(gate, VH mean dB)       = {r_vh:+.4f}")
        print(f"      gate(wrong) - gate(right) = {d_err:+.4f}")
        print(f"      eta^2 explained by class  = {eta2:.4f}"
              + ("   <- class identity dominates" if eta2 > 0.5 else ""))
        res["seeds"].append({
            "file": f.stem, "min": float(g.min()), "max": float(g.max()),
            "mean": float(g.mean()),
            "r_confidence": float(r_conf), "rho_confidence": float(rho_conf),
            "r_vv": float(r_vv), "r_vh": float(r_vh),
            "gate_wrong_minus_right": d_err, "eta2_class": eta2,
            "class_means": cls_means,
        })

    print(f"\n   mean r(confidence) across {len(files)} seeds: "
          f"{np.mean([s['r_confidence'] for s in res['seeds']]):+.4f}")
    print(f"   mean r(VV)         across {len(files)} seeds: "
          f"{np.mean([s['r_vv'] for s in res['seeds']]):+.4f}")
    print(f"   mean eta^2(class)  across {len(files)} seeds: "
          f"{np.mean([s['eta2_class'] for s in res['seeds']]):.4f}")

    print("\n   per-class mean gate (averaged over seeds), sorted:")
    avg = {cl: float(np.mean([d[cl] for d in per_class_all])) for cl in per_class_all[0]}
    for cl, v in sorted(avg.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(v / max(avg.values()) * 40)
        print(f"      {cl:22s} {v:6.3f}  {bar}")
    res["class_means_avg"] = avg
    return res


out = {}
r_off = run("OFFICIAL (leaky)", sp_off, "")
r_geo = run("GEO (clean)", sp_geo, "_geo")
if r_off: out["official"] = r_off
if r_geo: out["geo"] = r_geo
OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"\n-> {OUT}")

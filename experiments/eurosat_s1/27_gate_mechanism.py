# -*- coding: utf-8 -*-
"""
27_gate_mechanism.py - does the model learn to trust location where the image fails?

The `gate` arm predicts, per sample, a scalar saying how much location to mix in.
Measured range across seeds: [0.22, 2.52] - a tenfold spread. So it is clearly
deciding something chip by chip. The question is WHAT.

Hypothesis: the gate opens where the IMAGE is uninformative.

Test, using only things computed independently of the gate:
  1. image-only classifier on the 12 spectral features (train split only)
  2. its confidence on each val chip = max predicted probability
  3. correlate that confidence with the gate the network chose

A negative correlation means the network taught itself the rule that
Copernicus-FM's radar/optical gap implies: location is worth more when the
image is ambiguous. Nobody has measured this inside a trained model.

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


z = np.load(PROBE, allow_pickle=True)
Xtr, Ytr = z["X_train"], z["Y_train"]
Xva, Yva = z["X_val"], z["Y_val"]

c = np.load(CACHE, allow_pickle=True)
val_mask = c["splits"] == "val"
val_names = c["names"][val_mask]

# 🔴 the two files were built by different scripts. Alignment is an assumption
#    until proved, and a silent misalignment would produce a beautiful,
#    meaningless correlation.
if len(val_names) != len(Xva):
    raise SystemExit(f"length mismatch: cache {len(val_names)} vs probe {len(Xva)}")
cls_from_cache = np.array([n.split("_")[0] for n in val_names])
if not (cls_from_cache == Yva).all():
    bad = int((cls_from_cache != Yva).sum())
    raise SystemExit(f"ORDER MISMATCH: {bad}/{len(Yva)} labels disagree. Aborting.")
print(f"alignment verified: {len(Xva)} val chips, labels agree\n")

print("fitting image-only classifier on the TRAIN split ...", flush=True)
clf = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xtr, Ytr)
proba = clf.predict_proba(Xva)
conf = proba.max(axis=1)                       # how sure the image alone is
img_correct = (clf.classes_[proba.argmax(1)] == Yva)
print(f"   image-only accuracy on val: {img_correct.mean()*100:.2f}")
print(f"   confidence: mean {conf.mean():.3f}  range [{conf.min():.3f}, {conf.max():.3f}]")


print("\n" + "=" * 74)
print("gate vs image confidence")
print("=" * 74)

files = sorted((LOC / "runs").glob("gate_s*_gates.npy"))
if not files:
    raise SystemExit("no gate files yet")

allr = []
for f in files:
    g = np.load(f)
    if len(g) != len(conf):
        print(f"   {f.name}: length {len(g)} != {len(conf)}, skipped"); continue
    r, p = stats.pearsonr(conf, g)
    rho, prho = stats.spearmanr(conf, g)
    allr.append(r)

    lo = g[conf <= np.percentile(conf, 33)].mean()      # image unsure
    hi = g[conf >= np.percentile(conf, 67)].mean()      # image sure
    print(f"\n   {f.stem}")
    print(f"      pearson  r = {r:+.4f}   p = {p:.2e}")
    print(f"      spearman r = {rho:+.4f}   p = {prho:.2e}")
    print(f"      mean gate when image is UNSURE : {lo:.4f}")
    print(f"      mean gate when image is SURE   : {hi:.4f}")
    print(f"      difference                     : {lo-hi:+.4f}"
          + ("   <- gate opens where the image fails" if lo > hi else ""))

    wrong = g[~img_correct].mean()
    right = g[img_correct].mean()
    print(f"      mean gate where image is WRONG : {wrong:.4f}")
    print(f"      mean gate where image is RIGHT : {right:.4f}")
    print(f"      difference                     : {wrong-right:+.4f}")

print(f"\n   mean pearson r across {len(allr)} seeds: {np.mean(allr):+.4f}")
print("   negative r = the network learned to trust location where the image is weak")

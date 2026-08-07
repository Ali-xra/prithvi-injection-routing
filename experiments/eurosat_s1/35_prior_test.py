# -*- coding: utf-8 -*-
"""
35_prior_test.py - is the whole location effect just a class prior?

THE QUESTION
Every injection arm lands within 0.17 points of every other. One explanation
would make that inevitable rather than surprising: if location only says
"around here, these classes are more likely", then it is a PRIOR - and a prior
is exactly what a single additive vector can express. No richer mechanism could
beat it, because there is nothing richer to express.

THE TEST
Take the image-only model. Never give it location at all. Then adjust its
output probabilities by a location prior computed OUTSIDE the network:

    p(class | chip)  ~  p_image(class | chip) * p_location(class | lon,lat)^alpha

p_location comes from the label distribution of the K nearest TRAINING chips.
On the geo split the nearest training chip is ~150 km away, so this is a
genuine regional prior, not proximity leakage.

READING THE RESULT
  recovers ~ the +4 of the trained arms  -> location IS a prior; the null is
                                           explained and no mechanism can win
  recovers much less                     -> location carries something a prior
                                           cannot express, and richer
                                           mechanisms still have room

Zero GPU.
"""
import sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors

LOC = Path(r"C:\Users\aliso\Desktop\big-files\loc")
PROBE = Path(r"C:\Users\aliso\Desktop\big-files\eurosat_s1\probe_features.npz")
CACHE = LOC / "eurosat_s1_cache.npz"
OUT = LOC / "prior_test_result.json"

K = 200          # neighbours used to estimate the regional prior
SMOOTH = 1.0     # Laplace smoothing so no class gets probability zero

z = np.load(PROBE, allow_pickle=True)
X = np.vstack([z["X_train"], z["X_val"], z["X_test"]])
Y = np.concatenate([z["Y_train"], z["Y_val"], z["Y_test"]])

c = np.load(CACHE, allow_pickle=True)
names, coords = c["names"], c["coords"]
cls_cache = np.array([n.split("_")[0] for n in names])
if len(X) != len(names) or not (cls_cache == Y).all():
    raise SystemExit("alignment failed between probe features and cache")
print(f"alignment verified: {len(X)} chips")

sp = np.load(LOC / "geo_split.npz", allow_pickle=True)["splits"]
tr, va = sp == "train", sp == "val"
print(f"geo split: train {tr.sum()}  val {va.sum()}")

classes = np.array(sorted(set(Y)))
cls_idx = {cl: i for i, cl in enumerate(classes)}
ytr = np.array([cls_idx[v] for v in Y[tr]])
yva = np.array([cls_idx[v] for v in Y[va]])

# ---------- 1. image-only model, no location anywhere ----------
print("\nfitting image-only model on the geo train split ...", flush=True)
clf = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X[tr], ytr)
p_img = clf.predict_proba(X[va])
acc_img = float((p_img.argmax(1) == yva).mean())
print(f"   image only                     {acc_img*100:6.2f}")

# ---------- 2. the regional prior, computed outside the network ----------
print(f"\nbuilding the location prior from the {K} nearest TRAINING chips ...",
      flush=True)
nn = NearestNeighbors(n_neighbors=K).fit(coords[tr])
dist, idx = nn.kneighbors(coords[va])
print(f"   median distance to the nearest training chip: "
      f"{np.median(dist[:, 0]):.1f} degrees-ish "
      f"({np.median(dist[:, 0])*111:.1f} km at the equator)")

onehot = np.eye(len(classes))[ytr]
p_loc = onehot[idx].sum(axis=1) + SMOOTH
p_loc /= p_loc.sum(axis=1, keepdims=True)
acc_loc = float((p_loc.argmax(1) == yva).mean())
print(f"   location prior alone           {acc_loc*100:6.2f}")

# ---------- 3. combine ----------
eps = 1e-12
log_img, log_loc = np.log(p_img + eps), np.log(p_loc + eps)


def acc_at(alpha):
    return float(((log_img + alpha * log_loc).argmax(1) == yva).mean())


acc_comb = acc_at(1.0)
print(f"\n   image x prior  (alpha = 1)     {acc_comb*100:6.2f}"
      f"   -> recovered {(acc_comb-acc_img)*100:+.2f} points")

# ---------- 4. what a TRAINED arm recovered, for comparison ----------
TRAINED_NONE, TRAINED_ADD = 73.21, 77.17
trained_gain = TRAINED_ADD - TRAINED_NONE
recovered = (acc_comb - acc_img) * 100
print(f"\n   trained arms on the same split: none {TRAINED_NONE}  add {TRAINED_ADD}"
      f"   gain {trained_gain:+.2f}")
print(f"   fraction of the trained gain explained by a pure prior: "
      f"{recovered/trained_gain*100:5.1f}%")

# ---------- 5. sensitivity, exploratory only ----------
print("\n   sensitivity to alpha (EXPLORATORY - alpha was fixed at 1 in advance):")
grid = {}
for a in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
    grid[a] = acc_at(a)
    print(f"      alpha {a:4.2f}   {grid[a]*100:6.2f}   ({(grid[a]-acc_img)*100:+.2f})")

print("\n" + "=" * 74)
if recovered >= 0.75 * trained_gain:
    print("VERDICT  the location effect is largely a CLASS PRIOR.")
    print("         A single additive vector can express a prior, so no richer")
    print("         mechanism should be expected to beat `add`. The null result")
    print("         is explained rather than merely observed.")
elif recovered >= 0.4 * trained_gain:
    print("VERDICT  location is PARTLY a prior. Some of the effect is a prior,")
    print("         some is not - richer mechanisms still have room, but less")
    print("         than the headline number suggests.")
else:
    print("VERDICT  location is NOT mainly a prior. It carries something a")
    print("         regional class distribution cannot express, and the null")
    print("         across entry points is NOT explained by this argument.")
print("=" * 74)

OUT.write_text(json.dumps({
    "k_neighbours": K, "n_val": int(va.sum()), "n_train": int(tr.sum()),
    "image_only": acc_img, "location_prior_only": acc_loc,
    "image_times_prior_alpha1": acc_comb,
    "recovered_points": recovered,
    "trained_gain_points": trained_gain,
    "fraction_explained": recovered / trained_gain,
    "alpha_grid": {str(k): v for k, v in grid.items()},
}, indent=2), encoding="utf-8")
print(f"\n-> {OUT}")

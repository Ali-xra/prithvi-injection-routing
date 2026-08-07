# -*- coding: utf-8 -*-
"""
29_leakage_probe.py - is the location signal on EuroSAT-S1 knowledge or proximity lookup?

The official EuroSAT split (21_cache.py:29) is random, not geographically disjoint.
The burn-scars phase rejected exactly this and built a tile-disjoint split; the EuroSAT
phase did not apply the same standard. This script measures the cost of that.

Two experiments, zero GPU.

A. Proximity upper bound.
   For every val chip, copy the label of its nearest TRAIN chip in coordinate space.
   No learning, no image. If this alone approaches the 69.94 that "location only"
   scored, then that 69.94 is a lookup table, not geographic knowledge.

B. Location-disjoint split.
   Cluster all coordinates, send WHOLE clusters to one side, rerun the tabular probe.
   Whatever survives is real. Same logic as the tile-disjoint split, with k-means
   clusters standing in for MGRS tiles because EuroSAT chips carry coordinates but
   no tile id.

Locked reference numbers from the official split (do not recompute, compare against):
   image only 70.50 | location only 69.94 | image+location 87.02 | image+shuffled 69.98
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier

LOC = Path.home() / "Desktop" / "big-files" / "loc"
PROBE = Path.home() / "Desktop" / "big-files" / "eurosat_s1" / "probe_features.npz"
CACHE = LOC / "eurosat_s1_cache.npz"
SEED = 0
K_CLUSTERS = 60

rng = np.random.default_rng(SEED)


def km_coords(lonlat):
    """lon/lat degrees -> approximate kilometres, so distances are comparable."""
    lat0 = np.deg2rad(lonlat[:, 1].mean())
    x = lonlat[:, 0] * 111.32 * np.cos(lat0)
    y = lonlat[:, 1] * 110.57
    return np.column_stack([x, y])


# ----------------------------------------------------------------- load + align
c = np.load(CACHE, allow_pickle=True)
coords, labels, splits, names = c["coords"], c["labels"], c["splits"], c["names"]

z = np.load(PROBE, allow_pickle=True)
Xtr, Ytr, Xva, Yva = z["X_train"], z["Y_train"], z["X_val"], z["Y_val"]

tr_m, va_m = splits == "train", splits == "val"
print(f"cache: {len(coords)} chips   train {tr_m.sum()}   val {va_m.sum()}   "
      f"test {(splits=='test').sum()}")
print(f"probe: X_train {Xtr.shape}   X_val {Xva.shape}")

# 🔴 the two files were built by different scripts. Alignment is an assumption until
#    proved, and a silent misalignment would produce a beautiful, meaningless number.
if len(Xtr) != tr_m.sum() or len(Xva) != va_m.sum():
    raise SystemExit("length mismatch between cache and probe features")
for nm, msk, Y in (("train", tr_m, Ytr), ("val", va_m, Yva)):
    cls = np.array([n.split("_")[0] for n in names[msk]])
    if not (cls == Y).all():
        raise SystemExit(f"ORDER MISMATCH in {nm}: {(cls != Y).sum()} labels disagree")
print("alignment verified: cache order matches probe order on train and val\n")

XY = km_coords(coords)
y_all = labels
X_img = np.zeros((len(coords), Xtr.shape[1]), np.float32)
X_img[tr_m], X_img[va_m] = Xtr, Xva


# ------------------------------------------------------- A. proximity upper bound
def proximity(tr_mask, va_mask, tag):
    tree = cKDTree(XY[tr_mask])
    dist, idx = tree.query(XY[va_mask], k=1)
    pred = y_all[tr_mask][idx]
    acc = (pred == y_all[va_mask]).mean() * 100
    print(f"  {tag}")
    print(f"     distance to nearest TRAIN chip (km):")
    for p in (10, 25, 50, 75, 90):
        print(f"        p{p:<3d} {np.percentile(dist, p):9.3f}")
    print(f"     copy-the-neighbour accuracy : {acc:.2f}   (chance 10.00)")
    return acc, np.median(dist)


print("=" * 74)
print("A · PROXIMITY UPPER BOUND - no learning, no image, just the nearest neighbour")
print("=" * 74)
acc_prox_off, med_off = proximity(tr_m, va_m, "official split (random)")


# --------------------------------------------------- B. location-disjoint split
print("\n" + "=" * 74)
print(f"B · LOCATION-DISJOINT SPLIT - {K_CLUSTERS} k-means clusters, whole clusters move")
print("=" * 74)

keep = tr_m | va_m                       # test split stays closed
km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=SEED).fit(XY[keep])
cl_all = np.full(len(coords), -1)
cl_all[keep] = km.labels_

target_val = va_m.sum() / keep.sum()
order = rng.permutation(K_CLUSTERS)
sizes = np.array([(cl_all == k).sum() for k in range(K_CLUSTERS)])
val_clusters, running = [], 0
for k in order:
    if running / keep.sum() < target_val:
        val_clusters.append(k); running += sizes[k]
val_clusters = set(val_clusters)

tr2 = keep & ~np.isin(cl_all, list(val_clusters))
va2 = keep & np.isin(cl_all, list(val_clusters))
print(f"  clusters: {K_CLUSTERS - len(val_clusters)} train / {len(val_clusters)} val")
print(f"  chips   : {tr2.sum()} train / {va2.sum()} val "
      f"(official was {tr_m.sum()} / {va_m.sum()})\n")

acc_prox_new, med_new = proximity(tr2, va2, "location-disjoint split")


# --------------------------------------------------------- the four-way probe
def probe(tr_mask, va_mask, tag):
    lon = coords[:, 0:1]
    lat = coords[:, 1:2]
    L = np.hstack([lon, lat]).astype(np.float32)
    Lsh = L.copy()
    p = rng.permutation(va_mask.sum())
    Lsh_va = L[va_mask][p]

    sets = {
        "image only":            (X_img[tr_mask], X_img[va_mask]),
        "location only":         (L[tr_mask], L[va_mask]),
        "image + location":      (np.hstack([X_img, L])[tr_mask],
                                  np.hstack([X_img, L])[va_mask]),
        "image + shuffled loc":  (np.hstack([X_img, L])[tr_mask],
                                  np.hstack([X_img[va_mask], Lsh_va])),
    }
    print(f"\n  {tag}")
    out = {}
    for name, (A, B) in sets.items():
        m = HistGradientBoostingClassifier(max_iter=300, random_state=SEED)
        m.fit(A, y_all[tr_mask])
        a = (m.predict(B) == y_all[va_mask]).mean() * 100
        out[name] = a
        print(f"     {name:<24s} {a:6.2f}")
    return out


print("\n" + "=" * 74)
print("THE FOUR-WAY TABULAR PROBE, UNDER BOTH SPLITS")
print("=" * 74)
off = probe(tr_m, va_m, "official split (random)   [locked ref: 70.50 / 69.94 / 87.02 / 69.98]")
new = probe(tr2, va2, "location-disjoint split")


# ------------------------------------------------------------------- verdict
print("\n" + "=" * 74)
print("VERDICT")
print("=" * 74)
print(f"  median distance to nearest train chip:")
print(f"     official split          {med_off:8.3f} km")
print(f"     location-disjoint split {med_new:8.3f} km")
print(f"\n  copy-the-neighbour accuracy:")
print(f"     official split          {acc_prox_off:6.2f}")
print(f"     location-disjoint split {acc_prox_new:6.2f}")
print(f"\n  'location only' in the probe:")
print(f"     official split          {off['location only']:6.2f}")
print(f"     location-disjoint split {new['location only']:6.2f}")
print(f"\n  'image + location' in the probe:")
print(f"     official split          {off['image + location']:6.2f}")
print(f"     location-disjoint split {new['image + location']:6.2f}")
print(f"\n  gain of location over image alone:")
print(f"     official split          {off['image + location'] - off['image only']:+6.2f}")
print(f"     location-disjoint split {new['image + location'] - new['image only']:+6.2f}")

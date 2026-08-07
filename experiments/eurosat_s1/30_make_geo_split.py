# -*- coding: utf-8 -*-
"""
30_make_geo_split.py - build a location-disjoint train/val split for EuroSAT-S1.

Why: the official split (21_cache.py) is random. Measured in 29_leakage_probe.py,
the median val chip sits 1.13 km from a train chip, and copying that neighbour's
label alone scores 77.30. So "location helps" on the official split is largely a
proximity lookup.

This writes an alternative split where WHOLE coordinate clusters go to one side,
so the nearest train chip to any val chip is ~150 km away. Same logic as the
tile-disjoint split of the burn-scars phase, with k-means clusters standing in
for MGRS tiles.

The test split is left exactly as it is and stays closed.

MUST match 29_leakage_probe.py exactly (same SEED, same K, same rng call order),
otherwise the training rerun would use a different split from the one the probe
measured, and the two results would not be comparable.

Run: python 30_make_geo_split.py
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
CACHE = HERE / "eurosat_s1_cache.npz"
OUT = HERE / "geo_split.npz"
SEED = 0
K_CLUSTERS = 60


def km_coords(lonlat):
    lat0 = np.deg2rad(lonlat[:, 1].mean())
    return np.column_stack([lonlat[:, 0] * 111.32 * np.cos(lat0),
                            lonlat[:, 1] * 110.57])


z = np.load(CACHE, allow_pickle=True)
coords, labels, splits = z["coords"], z["labels"], z["splits"]
XY = km_coords(coords)

tr_m, va_m = splits == "train", splits == "val"
keep = tr_m | va_m
print(f"cache {len(coords)} chips   official train {tr_m.sum()}  val {va_m.sum()}"
      f"  test {(splits=='test').sum()} (untouched)")

rng = np.random.default_rng(SEED)
km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=SEED).fit(XY[keep])
cl = np.full(len(coords), -1)
cl[keep] = km.labels_

target = va_m.sum() / keep.sum()
order = rng.permutation(K_CLUSTERS)          # first rng use - matches 29
sizes = np.array([(cl == k).sum() for k in range(K_CLUSTERS)])
val_clusters, running = [], 0
for k in order:
    if running / keep.sum() < target:
        val_clusters.append(k); running += sizes[k]
val_clusters = set(val_clusters)

new = splits.copy()
new[keep] = np.where(np.isin(cl[keep], list(val_clusters)), "val", "train")

tr2, va2 = new == "train", new == "val"
print(f"geo   clusters {K_CLUSTERS-len(val_clusters)} train / {len(val_clusters)} val")
print(f"geo   chips    {tr2.sum()} train / {va2.sum()} val")

# 🔴 must-differ style check: the whole point is that the split separates in space.
d_off, _ = cKDTree(XY[tr_m]).query(XY[va_m], k=1)
d_new, _ = cKDTree(XY[tr2]).query(XY[va2], k=1)
print(f"\nmedian distance to nearest train chip:")
print(f"   official  {np.median(d_off):8.3f} km")
print(f"   geo       {np.median(d_new):8.3f} km")
if np.median(d_new) < 10 * np.median(d_off):
    raise SystemExit("geo split is not meaningfully more separated - aborting")

# class balance is not guaranteed to survive a spatial split; report it rather
# than assume it, because a collapsed class would explain any accuracy drop.
print("\nclass counts (train / val):")
for c in range(int(labels.max()) + 1):
    print(f"   class {c}: {int(((labels==c)&tr2).sum()):5d} / {int(((labels==c)&va2).sum()):5d}")
missing = [c for c in range(int(labels.max()) + 1) if ((labels == c) & va2).sum() == 0]
if missing:
    raise SystemExit(f"classes absent from val: {missing} - split unusable")

np.savez(OUT, splits=new.astype(str))
print(f"\n-> {OUT}")

# -*- coding: utf-8 -*-
"""
31_extrapolation_check.py - is the -0.31 real, or an artifact of tree extrapolation?

On the location-disjoint split, validation lon/lat fall OUTSIDE the training range.
A decision tree only learns thresholds inside the range it saw, so it cannot
extrapolate a trend - it just dumps unseen values into the edge leaf. That means
`location only 24.06` and `gain -0.31` are LOWER bounds, deflated by the tool.

This asks how much of the collapse survives when the tool can extrapolate:
  raw lon/lat + trees        (what we measured)
  sincos(lon,lat) + trees    (bounded, periodic encoding - no out-of-range values)
  raw lon/lat + linear       (a linear model does extrapolate)
  sincos + linear

Zero GPU.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
CACHE = HERE / "eurosat_s1_cache.npz"
GEO = HERE / "geo_split.npz"
PROBE = Path.home() / "Desktop" / "big-files" / "eurosat_s1" / "probe_features.npz"
SEED = 0

c = np.load(CACHE, allow_pickle=True)
coords, labels, off = c["coords"], c["labels"], c["splits"]
geo = np.load(GEO, allow_pickle=True)["splits"]

z = np.load(PROBE, allow_pickle=True)
X_img = np.zeros((len(coords), z["X_train"].shape[1]), np.float32)
X_img[off == "train"], X_img[off == "val"] = z["X_train"], z["X_val"]


def sincos(lonlat, n_freq=8):
    lon = lonlat[:, 0] / 180.0 * np.pi
    lat = lonlat[:, 1] / 90.0 * np.pi
    f = np.exp(np.arange(n_freq) * (-np.log(1000.0) / max(n_freq - 1, 1)))
    out = []
    for ang in (lon, lat):
        a = ang[:, None] * f[None, :] * 10.0
        out += [np.sin(a), np.cos(a)]
    return np.hstack(out).astype(np.float32)


RAW = coords.astype(np.float32)
SC = sincos(coords)

MODELS = {
    "trees":  lambda: HistGradientBoostingClassifier(max_iter=300, random_state=SEED),
    "linear": lambda: make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000, n_jobs=-1)),
}
ENCODINGS = {"raw lon/lat": RAW, "sincos": SC}


def run(split, tag):
    tr, va = split == "train", split == "val"
    print(f"\n{'='*74}\n{tag}   train {tr.sum()}  val {va.sum()}\n{'='*74}")

    # range check: how far outside the training range does val sit?
    for i, nm in ((0, "lon"), (1, "lat")):
        lo, hi = coords[tr, i].min(), coords[tr, i].max()
        out = ((coords[va, i] < lo) | (coords[va, i] > hi)).mean() * 100
        print(f"   {nm}: train range [{lo:.2f}, {hi:.2f}]   {out:.1f}% of val outside")

    print(f"\n   {'features':<26s} {'trees':>8s} {'linear':>8s}")
    rows = [("image only", {"raw lon/lat": X_img, "sincos": X_img})]
    for enc_name, E in ENCODINGS.items():
        rows.append((f"location only [{enc_name}]", {enc_name: E}))
        rows.append((f"image + location [{enc_name}]",
                     {enc_name: np.hstack([X_img, E])}))

    seen, results = set(), {}
    for label, d in rows:
        if label in seen:
            continue
        seen.add(label)
        X = list(d.values())[0]
        line = f"   {label:<26s}"
        for mname, mk in MODELS.items():
            m = mk().fit(X[tr], labels[tr])
            acc = (m.predict(X[va]) == labels[va]).mean() * 100
            results[(label, mname)] = acc
            line += f" {acc:8.2f}"
        print(line)
    return results


r_off = run(off, "OFFICIAL SPLIT (random)")
r_geo = run(geo, "LOCATION-DISJOINT SPLIT")

print(f"\n{'='*74}\nGAIN OF LOCATION OVER IMAGE ALONE\n{'='*74}")
print(f"   {'':<30s} {'trees':>8s} {'linear':>8s}")
for split_name, r in (("official", r_off), ("geo-disjoint", r_geo)):
    for enc in ENCODINGS:
        line = f"   {split_name + ' / ' + enc:<30s}"
        for mname in MODELS:
            g = r[(f"image + location [{enc}]", mname)] - r[("image only", mname)]
            line += f" {g:+8.2f}"
        print(line)

print("\n   If the geo-disjoint gain stays near zero for BOTH encodings and BOTH")
print("   model families, the collapse is real and not a tree artifact.")

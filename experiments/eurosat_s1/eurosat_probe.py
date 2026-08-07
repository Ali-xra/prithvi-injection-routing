# -*- coding: utf-8 -*-
"""
eurosat_probe.py - is there a payload on EuroSAT-S1?

The one question worth answering before spending any GPU time:
does location add accuracy BEYOND what the radar image already says?

Copernicus-FM (ICCV 2025, Table 12) says location is worth +21.3 points here.
If our own cheap probe cannot see that, we do not build on it.

Features from the image only: per-band stats of VV/VH (radar backscatter, dB).
Then the same features plus lat/lon. The gap is the payload.

Run: python eurosat_probe.py [--limit N]
"""
import sys, argparse, time, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import rasterio
from pyproj import Transformer

ROOT = Path.home() / "Desktop" / "big-files" / "eurosat_s1" / "eurosat_s1"
IMGS = ROOT / "all_imgs"
CACHE = Path.home() / "Desktop" / "big-files" / "eurosat_s1" / "probe_features.npz"
OUT = Path.home() / "Desktop" / "big-files" / "eurosat_s1" / "probe_result.json"


def split_files(name):
    out = []
    with (ROOT / f"eurosat-{name}.txt").open() as f:
        for line in f:
            fn = line.strip().replace(".jpg", ".tif")
            if fn:
                out.append(fn)
    return out


def features_and_coords(fname):
    """12 image features + (lon, lat). Nothing else."""
    cls = fname.split("_")[0]
    with rasterio.open(IMGS / cls / fname) as src:
        a = src.read().astype(np.float32)          # (2, 64, 64) VV, VH in dB
        cx, cy = src.xy(src.height // 2, src.width // 2)
        crs = src.crs
    if crs.to_string() != "EPSG:4326":
        lon, lat = Transformer.from_crs(crs, "epsg:4326", always_xy=True).transform(cx, cy)
    else:
        lon, lat = cx, cy

    vv, vh = a[0], a[1]
    feats = []
    for b in (vv, vh):
        feats += [b.mean(), b.std(),
                  np.percentile(b, 10), np.percentile(b, 50), np.percentile(b, 90)]
    feats += [(vv - vh).mean(), (vv - vh).std()]    # cross-pol difference
    return np.array(feats, np.float32), np.array([lon, lat], np.float32), cls


def build(limit=None):
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        print(f"   cache hit: {z['X'].shape}")
        return {k: z[k] for k in z.files}

    data = {}
    for split in ("train", "val", "test"):
        names = split_files(split)
        if limit:
            names = names[:limit]
        X, C, Y = [], [], []
        t0 = time.time()
        for i, fn in enumerate(names, 1):
            f, c, cls = features_and_coords(fn)
            X.append(f); C.append(c); Y.append(cls)
            if i % 2000 == 0:
                el = time.time() - t0
                print(f"   {split} [{i}/{len(names)}] {el:.0f}s", flush=True)
        data[f"X_{split}"] = np.vstack(X)
        data[f"C_{split}"] = np.vstack(C)
        data[f"Y_{split}"] = np.array(Y)
        print(f"   {split}: {data[f'X_{split}'].shape}")
    np.savez_compressed(CACHE, **data)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder

    print("building features ...", flush=True)
    d = build(a.limit)

    le = LabelEncoder().fit(d["Y_train"])
    ytr, yva = le.transform(d["Y_train"]), le.transform(d["Y_val"])
    print(f"\nclasses: {list(le.classes_)}")
    print(f"train {len(ytr)}  val {len(yva)}\n")

    def acc(Xtr, Xva, tag):
        m = HistGradientBoostingClassifier(max_iter=300, random_state=0)
        m.fit(Xtr, ytr)
        a_ = float((m.predict(Xva) == yva).mean())
        print(f"   {tag:22s} accuracy {a_*100:6.2f}")
        return a_

    Xtr, Xva = d["X_train"], d["X_val"]
    Ctr, Cva = d["C_train"], d["C_val"]

    a_img = acc(Xtr, Xva, "image only")
    a_loc = acc(Ctr, Cva, "location only")
    a_both = acc(np.hstack([Xtr, Ctr]), np.hstack([Xva, Cva]), "image + location")

    # control: location shuffled - destroys the image/location correspondence
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Ctr))
    permv = rng.permutation(len(Cva))
    a_shuf = acc(np.hstack([Xtr, Ctr[perm]]), np.hstack([Xva, Cva[permv]]),
                 "image + shuffled loc")

    gain = a_both - a_img
    print(f"\n   PAYLOAD  location beyond image = {gain*100:+.2f} points")
    print(f"   control  shuffled location      = {(a_shuf-a_img)*100:+.2f} points")
    print(f"\n   Copernicus-FM Table 12 reports +21.3 for location on EuroSAT-S1.")

    OUT.write_text(json.dumps({
        "image_only": a_img, "location_only": a_loc,
        "image_plus_location": a_both, "image_plus_shuffled": a_shuf,
        "payload_points": gain * 100,
        "shuffle_control_points": (a_shuf - a_img) * 100,
        "n_train": len(ytr), "n_val": len(yva),
    }, indent=2), encoding="utf-8")
    print(f"\n   -> {OUT}")


if __name__ == "__main__":
    main()

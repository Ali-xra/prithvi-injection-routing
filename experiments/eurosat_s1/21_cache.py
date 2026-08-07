# -*- coding: utf-8 -*-
"""
21_cache.py - read the 27000 EuroSAT-S1 tifs once into a single array.

Reading tifs per epoch would dominate wall-clock. One pass, then everything
downstream runs at RAM speed.

    images  (N, 2, 64, 64)  float32   VV/VH in dB
    coords  (N, 2)          float32   lon, lat  (EPSG:4326)
    labels  (N,)            int64
    split   (N,)            str       train/val/test, the official lists

Run: python 21_cache.py
"""
import sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import rasterio
from pyproj import Transformer

ROOT = Path.home() / "Desktop" / "big-files" / "eurosat_s1" / "eurosat_s1"
IMGS = ROOT / "all_imgs"
OUT = Path.home() / "Desktop" / "big-files" / "loc" / "eurosat_s1_cache.npz"


def read_split(name):
    out = []
    with (ROOT / f"eurosat-{name}.txt").open() as f:
        for line in f:
            fn = line.strip().replace(".jpg", ".tif")
            if fn:
                out.append(fn)
    return out


def main():
    entries = []
    for s in ("train", "val", "test"):
        entries += [(fn, s) for fn in read_split(s)]
    n = len(entries)
    print(f"{n} chips")

    classes = sorted({fn.split('_')[0] for fn, _ in entries})
    cls2idx = {c: i for i, c in enumerate(classes)}
    print("classes:", classes)

    images = np.zeros((n, 2, 64, 64), np.float32)
    coords = np.zeros((n, 2), np.float32)
    labels = np.zeros(n, np.int64)
    splits = np.empty(n, object)
    names = np.empty(n, object)

    tf_cache = {}
    t0 = time.time()
    for i, (fn, sp) in enumerate(entries):
        cls = fn.split("_")[0]
        with rasterio.open(IMGS / cls / fn) as src:
            a = src.read().astype(np.float32)
            cx, cy = src.xy(src.height // 2, src.width // 2)
            crs = src.crs.to_string()
        if crs != "EPSG:4326":
            if crs not in tf_cache:
                tf_cache[crs] = Transformer.from_crs(crs, "epsg:4326", always_xy=True)
            lon, lat = tf_cache[crs].transform(cx, cy)
        else:
            lon, lat = cx, cy

        if a.shape != (2, 64, 64):          # 🔴 هیچ chip بی‌صدا رد نشود
            raise ValueError(f"{fn}: shape {a.shape}")
        images[i] = a
        coords[i] = (lon, lat)
        labels[i] = cls2idx[cls]
        splits[i] = sp
        names[i] = fn
        if (i + 1) % 3000 == 0:
            el = time.time() - t0
            print(f"   [{i+1}/{n}] {el:.0f}s  eta {el/(i+1)*(n-i-1):.0f}s", flush=True)

    if not np.isfinite(images).all():
        raise ValueError("non-finite pixel values in cache")
    if not np.isfinite(coords).all():
        raise ValueError("non-finite coordinates in cache")

    print("\nsplit counts:", {s: int((splits == s).sum()) for s in ("train", "val", "test")})
    print("class counts:", np.bincount(labels).tolist())
    print(f"lon {coords[:,0].min():.2f}..{coords[:,0].max():.2f}   "
          f"lat {coords[:,1].min():.2f}..{coords[:,1].max():.2f}")
    print(f"VV mean {images[:,0].mean():.3f} std {images[:,0].std():.3f}")
    print(f"VH mean {images[:,1].mean():.3f} std {images[:,1].std():.3f}")

    np.savez(OUT, images=images, coords=coords, labels=labels,
             splits=splits.astype(str), names=names.astype(str),
             classes=np.array(classes))
    print(f"\n-> {OUT}  ({OUT.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()

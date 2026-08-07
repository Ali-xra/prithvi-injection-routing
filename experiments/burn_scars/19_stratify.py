# -*- coding: utf-8 -*-
"""
19_stratify.py — آیا متادیتا آنجا که تصویر مبهم است کمک می‌کند؟
================================================================
نسخه: v1 · تاریخ: 2026-08-01 · طرح قفل‌شده: `docs/PREREG-stratified.md`

هیچ چیزی در این فایل بعد از دیدن نتیجه انتخاب نشده. تعریف ابهام، نقطهٔ برش،
معیار، برآوردگر، آستانه و علامت لازم، همه در PREREG پیش از اجرا نوشته شده‌اند.

ابهام — مستقل از هر بازوی Prithvi:
    ۱. همان پروکسی تصویری ۱۳ ویژگی‌ای `12_image_proxy_control.py`
    ۲. رگرسیون لجستیک روی **فقط split آموزش**، هدف: درصد سوختگی بالاتر از میانهٔ آموزش
    ۳. امتیاز روی chipهای val ·  ابهام = 1 − |p − 0.5| × 2
    ۴. برش روی **میانهٔ val** → دو زیرمجموعهٔ هم‌اندازه

برآوردگر:
    D = (بازوی متادیتا − کنترل) روی مبهم  −  (بازوی متادیتا − کنترل) روی گویا
    آستانه: 0.0085  (۰.۰۰۶ قفل‌شده × √۲، چون هر زیرمجموعه نصف است)

اجرا: python 19_stratify.py
"""
import sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import rasterio
from sklearn.linear_model import LogisticRegression

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
DS = BIG / "data" / "burn_scars"
PERCHIP = BIG / "runs" / "_perchip"
FEATCACHE = BIG / "data" / "meta" / "image_proxy_features.npz"
CSV = BIG / "data" / "meta" / "conditioning_v1.csv"
BURNCACHE = BIG / "data" / "meta" / "burn_fraction.npz"
OUT = BIG / "data" / "meta" / "stratified.json"

THRESHOLD = 0.0085           # 🔒 قفل‌شده در PREREG
PAIRS = [                    # (بازوی متادیتا, کنترل, نقش)
    ("tl_on", "tl_off", "primary"),
    ("adaln", "baseline", "secondary"),
    ("adaln", "shuffle", "secondary"),
]


def burn_fraction(name, path):
    with rasterio.open(path) as src:
        m = src.read(1)
    valid = m != -1
    return float((m[valid] == 1).mean()) if valid.any() else 0.0


def load_burn_fractions():
    """نسبت پیکسل سوخته در هر chip — کَش می‌شود چون ۸۰۴ فایل است."""
    if BURNCACHE.exists():
        z = np.load(BURNCACHE, allow_pickle=True)
        return dict(zip(z["name"].tolist(), z["frac"].tolist()))
    out = {}
    for sub in ("training", "validation"):
        for p in sorted((DS / sub).glob("*.mask.tif")):
            merged = p.name.replace(".mask.tif", "_merged.tif")
            out[merged] = burn_fraction(merged, p)
    np.savez_compressed(BURNCACHE, name=np.array(list(out)),
                        frac=np.array(list(out.values())))
    print(f"   💾 نسبت سوختگی کَش شد ({len(out)} chip)")
    return out


def load_features():
    """sample_id → ۱۳ ویژگی، به‌علاوهٔ نگاشت filename → sample_id از CSV."""
    import csv as _csv
    z = np.load(FEATCACHE, allow_pickle=True)
    by_sid = {int(s): x for s, x in zip(z["sid"], z["X"])}
    rows = list(_csv.DictReader(CSV.open(encoding="utf-8")))
    name2sid = {r["filename"]: int(r["sample_id"]) for r in rows}
    return by_sid, name2sid


def ambiguity(val_names):
    """
    🔴 روی **فقط** chipهای آموزش برازش می‌شود. اگر val وارد برازش شود، تعریف
    ابهام از همان داده‌ای می‌آید که رویش قضاوت می‌کنیم.
    """
    by_sid, name2sid = load_features()
    frac = load_burn_fractions()
    val_set = set(val_names)

    tr_names = [n for n in name2sid if n not in val_set and n in frac]
    thr = float(np.median([frac[n] for n in tr_names]))

    Xtr = np.vstack([by_sid[name2sid[n]] for n in tr_names])
    ytr = np.array([frac[n] > thr for n in tr_names], dtype=int)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=5000, random_state=0)
    clf.fit((Xtr - mu) / sd, ytr)

    Xva = np.vstack([by_sid[name2sid[n]] for n in val_names])
    p = clf.predict_proba((Xva - mu) / sd)[:, 1]
    amb = 1.0 - np.abs(p - 0.5) * 2.0
    return amb, p, len(tr_names), thr

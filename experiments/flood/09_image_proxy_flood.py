# -*- coding: utf-8 -*-
"""
09_image_proxy_flood.py — پروکسی «آنچه تصویر از قبل می‌داند»
============================================================
نسخه: v1 · تاریخ: 2026-07-30 · کار `flood`

چرا لازم است — درسِ دروازهٔ GPP (2026-07-30):
    آنجا آب‌وهوا در برابر **فصل** و **بوم** سیگنال داشت ولی در برابر **پروکسی
    تصویر** صفر شد. یعنی بدون این بلوک کنترل، آزمون سیگنال به‌ناحق آسان است:
    ممکن است چیزی را «کشف» کنیم که مدل از تصویر می‌داند.

    و همین ایراد را به استدلال `AUC=0.695` کار `burn` هم وارد کردم. پس اگر روی
    سیل نزنمش، همان خطا را خودم تکرار کرده‌ام.

چه می‌سازد — به ازای هر chip:
    میانگین و انحراف معیار شش باند نوریِ Prithvi + چهار شاخص گیاهی/آبی
    (NDVI · NDWI · MNDWI · EVI) — MNDWI برای آب مهم‌ترین است.

⚠️ فقط از پیکسل‌های **برچسب‌دار** حساب می‌شود (ماسک ≠ ‎−۱). چون ۴۲٪ نمونه‌ها بیش
   از ۵٪ بی‌داده دارند و میانگین روی پیکسل ابری، عدد بی‌معنا می‌دهد.

نگاشت باند Sen1Floods11 S2 (۱۳ باند، ۱-پایه) → شش باند Prithvi:
    B2 آبی=2 · B3 سبز=3 · B4 سرخ=4 · B8A فروسرخ باریک=9 · B11 SWIR1=12 · B12 SWIR2=13

خروجی: <BIG>/data/meta/image_proxy.csv
اجرا:  python 09_image_proxy_flood.py
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.stdout.reconfigure(encoding="utf-8")

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing-flood"
IMG_DIR = BIG / "data" / "flood_events" / "HandLabeled" / "S2Hand"
MSK_DIR = BIG / "data" / "flood_events" / "HandLabeled" / "LabelHand"
META = BIG / "data" / "meta"
IN_CSV = META / "conditioning_v1.csv"
OUT_CSV = META / "image_proxy.csv"

BANDS = {"blue": 2, "green": 3, "red": 4, "nir": 9, "swir1": 12, "swir2": 13}


def safe_ratio(a, b):
    d = a + b
    return float(np.mean(np.where(np.abs(d) < 1e-9, 0.0, (a - b) / np.where(np.abs(d) < 1e-9, 1.0, d))))


def one(stem):
    img = IMG_DIR / f"{stem}_S2Hand.tif"
    msk = MSK_DIR / f"{stem}_LabelHand.tif"
    with rasterio.open(msk) as m:
        lab = m.read(1)
    valid = lab != -1
    if valid.sum() < 100:                      # عملاً همه‌اش بی‌داده
        return None
    with rasterio.open(img) as src:
        arr = src.read().astype(np.float32)
    px = {k: arr[i - 1][valid] for k, i in BANDS.items()}
    out = {"stem": stem, "n_valid_px": int(valid.sum()),
           "pct_valid": round(100.0 * valid.mean(), 3)}
    for k, v in px.items():
        out[f"{k}_mean"] = round(float(np.mean(v)), 2)
        out[f"{k}_std"] = round(float(np.std(v)), 2)
    out["NDVI"] = round(safe_ratio(px["nir"], px["red"]), 6)
    out["NDWI"] = round(safe_ratio(px["green"], px["nir"]), 6)
    out["MNDWI"] = round(safe_ratio(px["green"], px["swir1"]), 6)   # مهم‌ترین برای آب
    out["EVI"] = round(float(np.mean(
        2.5 * (px["nir"] - px["red"]) / (px["nir"] + 6 * px["red"] - 7.5 * px["blue"] + 1e4)
    )), 6)
    return out


def main():
    t0 = time.time()
    rows = list(csv.DictReader(IN_CSV.open(encoding="utf-8")))
    stems = [r["filename"].replace("_S2Hand.tif", "") for r in rows]
    print("=" * 78)
    print(f"پروکسی تصویر · {len(stems)} chip · فقط پیکسل برچسب‌دار")
    print("=" * 78)

    out, skipped = [], []
    for i, s in enumerate(stems, 1):
        rec = one(s)
        (out if rec else skipped).append(rec or s)
        if i % 100 == 0 or i == len(stems):
            print(f"   {i}/{len(stems)} · {time.time()-t0:.0f}s")

    if skipped:                                # الزام ۷
        print(f"\n⚠️ {len(skipped)} chip رد شد (کمتر از ۱۰۰ پیکسل برچسب‌دار): {skipped[:5]}")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    pv = np.array([r["pct_valid"] for r in out])
    mnd = np.array([r["MNDWI"] for r in out])
    print(f"\nپیکسل برچسب‌دار٪ — کمینه {pv.min():.1f} · میانه {np.median(pv):.1f} · میانگین {pv.mean():.1f}")
    print(f"MNDWI — کمینه {mnd.min():.3f} · میانه {np.median(mnd):.3f} · بیشینه {mnd.max():.3f}")
    print(f"\nزمان دیواری {time.time()-t0:.0f}s · {OUT_CSV}")


if __name__ == "__main__":
    main()

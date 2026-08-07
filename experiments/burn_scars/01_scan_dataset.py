"""
01_scan_dataset.py — شناسایی دیتاست رد آتش‌سوزی
================================================
نسخه: v1 · تاریخ: 2026-07-28

این اسکریپت چه می‌کند:
    ۱. فایل‌های `training/` و `validation/` را می‌شمارد و جفت تصویر/ماسک را چک می‌کند
    ۲. الگوی نام فایل را می‌شکند: کاشی · سال · روزِ سال → تاریخ میلادی
    ۳. یک فایل واقعی را با rasterio باز می‌کند: باند، dtype، بازهٔ مقادیر، CRS
    ۴. مقادیر یکتای ماسک را می‌گیرد
    ۵. توزیع سال‌ها و کاشی‌ها را می‌دهد

چرا لازم است:
    خروجی این اسکریپت ورودی `02_build_meta.py` است. هر فرضی که اینجا تأیید نشود،
    آنجا به‌شکل باگ خاموش درمی‌آید.

اجرا:
    python 01_scan_dataset.py
"""

from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
import re

DATA = Path.home() / "Desktop" / "big-files" / "injection-routing" / "data" / "burn_scars"

# subsetted_512x512_HLS.S30.T14SNB.2018215.v1.4_merged.tif
PATTERN = re.compile(
    r"^subsetted_(?P<size>\d+x\d+)_HLS\."
    r"(?P<sensor>[A-Z0-9]+)\."
    r"(?P<tile>T[0-9A-Z]+)\."
    r"(?P<year>\d{4})(?P<doy>\d{3})\."
    r"(?P<ver>v[\d.]+)_merged\.tif$"
)


def parse(name: str):
    m = PATTERN.match(name)
    if not m:
        return None
    g = m.groupdict()
    date = datetime(int(g["year"]), 1, 1) + timedelta(days=int(g["doy"]) - 1)
    g["date"] = date.date().isoformat()
    return g


def main():
    print(f"ریشهٔ داده: {DATA}\n")

    for split in ("training", "validation"):
        d = DATA / split
        if not d.exists():
            print(f"⛔ {split}/ پیدا نشد")
            continue

        imgs = sorted(d.glob("*_merged.tif"))
        masks = sorted(d.glob("*.mask.tif"))
        print(f"=== {split}/ ===")
        print(f"  تصویر: {len(imgs)}   ماسک: {len(masks)}")

        # جفت‌بودن
        stems = {p.name.replace("_merged.tif", "") for p in imgs}
        mstems = {p.name.replace(".mask.tif", "") for p in masks}
        missing = stems - mstems
        orphan = mstems - stems
        print(f"  تصویرِ بی‌ماسک: {len(missing)}   ماسکِ بی‌تصویر: {len(orphan)}")
        if missing:
            print(f"    نمونه: {sorted(missing)[:3]}")

        # نام‌ها
        parsed = [parse(p.name) for p in imgs]
        bad = [p.name for p, q in zip(imgs, parsed) if q is None]
        ok = [q for q in parsed if q]
        print(f"  نامِ قابل‌تجزیه: {len(ok)}   ناسازگار: {len(bad)}")
        if bad:
            print(f"    ⚠️ نمونهٔ ناسازگار: {bad[:3]}")

        if ok:
            years = Counter(q["year"] for q in ok)
            tiles = Counter(q["tile"] for q in ok)
            dates = sorted(q["date"] for q in ok)
            print(f"  سال‌ها: {dict(sorted(years.items()))}")
            print(f"  کاشی‌های یکتا: {len(tiles)}  (پرتکرارترین: {tiles.most_common(3)})")
            print(f"  بازهٔ تاریخ: {dates[0]} تا {dates[-1]}")
            print(f"  سنسورها: {Counter(q['sensor'] for q in ok)}")
        print()

    # --- یک فایل واقعی ---
    sample = next((DATA / "training").glob("*_merged.tif"), None)
    if sample is None:
        print("⛔ هیچ فایلی برای بازکردن پیدا نشد")
        return

    try:
        import rasterio
        from rasterio.warp import transform as warp_transform
    except ImportError:
        print("⛔ rasterio نصب نیست →  pip install rasterio")
        return

    import numpy as np

    print("=== یک فایل نمونه ===")
    print(f"  {sample.name}")
    print(f"  تجزیهٔ نام: {parse(sample.name)}\n")

    with rasterio.open(sample) as src:
        print(f"  اندازه: {src.width} × {src.height}   باند: {src.count}   dtype: {src.dtypes[0]}")
        print(f"  CRS: {src.crs}")
        print(f"  nodata: {src.nodata}")
        arr = src.read()
        print(f"  شکل آرایه: {arr.shape}")
        for i in range(src.count):
            b = arr[i].astype("float64")
            print(f"    باند {i+1}: min={np.nanmin(b):10.4f}  max={np.nanmax(b):10.4f}  mean={np.nanmean(b):8.4f}")

        # مختصات مرکز → EPSG:4326
        cx = (src.bounds.left + src.bounds.right) / 2
        cy = (src.bounds.bottom + src.bounds.top) / 2
        lon, lat = warp_transform(src.crs, "EPSG:4326", [cx], [cy])
        print(f"\n  مرکز در CRS تصویر: ({cx:.1f}, {cy:.1f})")
        print(f"  مرکز جغرافیایی:    lat={lat[0]:.4f}  lon={lon[0]:.4f}")
        gsd = abs(src.transform.a)
        print(f"  اندازهٔ پیکسل: {gsd:.1f} متر   →  پهنای کاشی ≈ {src.width*gsd/1000:.1f} کیلومتر")

    mask_path = sample.with_name(sample.name.replace("_merged.tif", ".mask.tif"))
    if mask_path.exists():
        with rasterio.open(mask_path) as m:
            mv = m.read(1)
            vals, cnts = np.unique(mv, return_counts=True)
            total = mv.size
            print(f"\n  ماسک — باند: {m.count}  dtype: {m.dtypes[0]}")
            for v, c in zip(vals, cnts):
                print(f"    مقدار {v:>4}: {c:>7} پیکسل  ({100*c/total:5.2f}%)")
    else:
        print("\n  ⛔ ماسک متناظر پیدا نشد")


if __name__ == "__main__":
    main()

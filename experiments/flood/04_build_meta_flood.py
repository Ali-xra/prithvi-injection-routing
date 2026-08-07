# -*- coding: utf-8 -*-
"""
04_build_meta_flood.py — متادیتای ۴۴۶ نمونهٔ سیل
=================================================
نسخه: v1 · تاریخ: 2026-07-30 · کار `flood`
مبنا: کپی و وصلهٔ `../lab/src/02_build_meta.py` (2026-07-28)

تفاوت‌های اجباری با نسخهٔ آتش‌سوزی — هر سه از یافته‌های گام ۱ می‌آیند:

  ۱. **تاریخ در نام فایل نیست.** آتش‌سوزی `2018215` در نام داشت؛ سیل ندارد.
     تاریخ از `Sen1Floods11_Metadata.geojson` می‌آید، **به ازای رویداد**.

  ۲. 🐛 **نگاشت `Mekong → Cambodia`.** نام split «Mekong» است ولی متادیتا
     «Cambodia». TerraTorch اینجا بی‌صدا به تاریخ جعلی `1998-10-13` عقب می‌نشیند
     و ۳۰ chip را خراب می‌کند. ما نگاشت را **صریح** اعمال می‌کنیم و اگر رویدادی
     حل نشد، خطا می‌دهیم — سکوت نمی‌کنیم.

  ۳. **کلاس هدف «آب» است نه «سوختگی».** ماسک `LabelHand`: ۱ آب · ۰ خشکی ·
     ‎−۱ بی‌داده. مثل آتش‌سوزی، نمونهٔ بی‌داده **حذف نمی‌شود، علامت می‌خورد**.

⚠️ تلهٔ حفظ‌شده از نسخهٔ اصلی: **هر چهار گوشه** تبدیل می‌شود، نه فقط مرکز —
   تبدیل UTM→WGS84 مستطیل را کج می‌کند.

خروجی: <BIG>/data/meta/samples.csv
اجرا:  python 04_build_meta_flood.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

sys.stdout.reconfigure(encoding="utf-8")

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing-flood"
IMG_DIR = BIG / "data" / "flood_events" / "HandLabeled" / "S2Hand"
MSK_DIR = BIG / "data" / "flood_events" / "HandLabeled" / "LabelHand"
META = BIG / "data" / "meta"
OUT_CSV = META / "samples.csv"
SPLITS = ("train", "valid", "test", "bolivia")
NAME_FIX = {"Mekong": "Cambodia"}          # 🐛 تلهٔ ۲ گام ۱

FIELDS = ["sample_id", "filename", "mask_filename", "orig_split", "event",
          "chip_id", "date", "year", "doy", "s1_date",
          "lat_center", "lon_center", "lat_min", "lat_max", "lon_min", "lon_max",
          "crs", "width", "height", "gsd_m", "n_bands",
          "pct_water", "pct_land", "pct_nodata"]


def event_dates():
    """رویداد → (s2_date, s1_date) با نگاشت صریح. رویداد بی‌تاریخ = خطا، نه سکوت."""
    gj = json.loads((META / "Sen1Floods11_Metadata.geojson").read_text(encoding="utf-8"))
    by_loc = {f["properties"]["location"]: f["properties"] for f in gj["features"]}

    def norm(d):
        return str(d).replace("/", "-") if d else ""

    return {loc: (norm(p.get("s2_date")), norm(p.get("s1_date")))
            for loc, p in by_loc.items()}, by_loc


def split_of():
    """chip → split اصلی، از چهار CSV گام ۱."""
    m = {}
    for s in SPLITS:
        for line in (META / f"flood_{s}_data.csv").read_text(encoding="utf-8").splitlines():
            if line.strip():
                stem = line.split(",")[0].strip().rsplit("/", 1)[-1]
                m[stem.replace("_S1Hand.tif", "")] = s
    return m


def doy_of(iso):
    y, mo, d = (int(x) for x in iso.split("-"))
    from datetime import date
    return (date(y, mo, d) - date(y, 1, 1)).days + 1, y


def process(img: Path, sid: int, dates, splits):
    stem = img.name.replace("_S2Hand.tif", "")
    event, chip_id = stem.split("_", 1)
    msk = MSK_DIR / f"{stem}_LabelHand.tif"
    if not msk.exists():
        return None, f"ماسک ندارد: {stem}"
    if stem not in splits:
        return None, f"در هیچ split نیست: {stem}"

    key = NAME_FIX.get(event, event)
    if key not in dates or not dates[key][0]:
        return None, f"🐛 تاریخ حل نشد: {event} (کلید {key}) — نگاشت را بررسی کن"
    s2, s1 = dates[key]
    doy, year = doy_of(s2)

    with rasterio.open(img) as src:
        b = src.bounds
        xs = [b.left, b.right, b.right, b.left, (b.left + b.right) / 2]
        ys = [b.bottom, b.bottom, b.top, b.top, (b.bottom + b.top) / 2]
        lons, lats = warp_transform(src.crs, "EPSG:4326", xs, ys)
        row = {"sample_id": sid, "filename": img.name, "mask_filename": msk.name,
               "orig_split": splits[stem], "event": event, "chip_id": chip_id,
               "date": s2, "year": year, "doy": doy, "s1_date": s1,
               "lat_center": round(lats[4], 6), "lon_center": round(lons[4], 6),
               "lat_min": round(min(lats[:4]), 6), "lat_max": round(max(lats[:4]), 6),
               "lon_min": round(min(lons[:4]), 6), "lon_max": round(max(lons[:4]), 6),
               "crs": str(src.crs), "width": src.width, "height": src.height,
               "gsd_m": round(abs(src.transform.a), 2), "n_bands": src.count}

    with rasterio.open(msk) as m:
        mv = np.asarray(m.read(1))
        t = mv.size
        row["pct_water"] = round(100.0 * np.count_nonzero(mv == 1) / t, 3)
        row["pct_land"] = round(100.0 * np.count_nonzero(mv == 0) / t, 3)
        row["pct_nodata"] = round(100.0 * np.count_nonzero(mv == -1) / t, 3)
    return row, None


def main():
    import time
    t0 = time.time()
    dates, _ = event_dates()
    splits = split_of()
    imgs = sorted(IMG_DIR.glob("*_S2Hand.tif"))
    print("=" * 78)
    print(f"متادیتای سیل · {len(imgs)} تصویر · نگاشت اعمال‌شده: {NAME_FIX}")
    print("=" * 78)

    rows, problems = [], []
    for i, p in enumerate(imgs, 1):
        row, err = process(p, len(rows), dates, splits)
        (problems if err else rows).append(err or row)
        if i % 100 == 0 or i == len(imgs):
            print(f"   {i}/{len(imgs)} · {time.time()-t0:.0f}s")

    if problems:                                  # الزام ۷ — صریح
        print(f"\n🔴 {len(problems)} مشکل:")
        for q in problems[:15]:
            print(f"   {q}")
    if not rows:
        print("⛔ هیچ سطری ساخته نشد")
        return

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # ---- خلاصهٔ آماری: مثل آتش‌سوزی، خودمان حساب می‌کنیم نه از کارت دیتاست ----
    import collections
    ev = collections.Counter(r["event"] for r in rows)
    wat = np.array([r["pct_water"] for r in rows])
    nod = np.array([r["pct_nodata"] for r in rows])
    sizes = collections.Counter((r["width"], r["height"]) for r in rows)
    bands = collections.Counter(r["n_bands"] for r in rows)
    crs_n = len({r["crs"] for r in rows})

    print(f"\n{'-'*78}\nخلاصه — همه اندازه‌گیری‌شده، نه نقل‌شده\n{'-'*78}")
    print(f"نمونه {len(rows)} · رویداد {len(ev)} · ناحیهٔ CRS متمایز {crs_n}")
    print(f"اندازه: {dict(sizes)} · باند: {dict(bands)}")
    print(f"تاریخ متمایز: {len({r['date'] for r in rows})}")
    print(f"\nدرصد آب — کمینه {wat.min():.2f} · میانه {np.median(wat):.2f} · "
          f"میانگین {wat.mean():.2f} · بیشینه {wat.max():.2f}")
    print(f"نمونهٔ با صفر پیکسل آب: {int((wat == 0).sum())} از {len(rows)}")
    print(f"نمونهٔ با >۵٪ بی‌داده: {int((nod > 5).sum())}  ← علامت می‌خورند، حذف نمی‌شوند")
    print(f"\nنمونه در هر رویداد: " + " · ".join(f"{k}:{v}" for k, v in sorted(ev.items())))
    print(f"\nزمان دیواری {time.time()-t0:.0f}s · {OUT_CSV}")


if __name__ == "__main__":
    main()

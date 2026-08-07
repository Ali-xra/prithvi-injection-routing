"""
02_build_meta.py — ساخت جدول متادیتای ۸۰۴ نمونه
================================================
نسخه: v1 · تاریخ: 2026-07-28

این اسکریپت چه می‌کند:
    برای هر فایل `_merged.tif` یک سطر می‌سازد شامل:
      نام فایل · split اصلی · کاشی MGRS · تاریخ · مختصات مرکز (lat/lon) ·
      مرزهای جغرافیایی · درصد پیکسل سوخته · درصد پیکسل بی‌داده
    و همه را در `data/meta/samples.csv` می‌ریزد.

چرا لازم است:
    این فایل **ستون فقرات فاز ۲** است. `04_download_era5.py` برای هر سطر یک
    درخواست ERA5 می‌سازد، و `03_make_split.py` روی ستون کاشی و تاریخ split
    جغرافیایی/زمانی می‌سازد.

چرا مرزها را هم ذخیره می‌کنیم:
    ERA5 روی شبکهٔ ۰.۱ درجه است و کاشی ۱۵.۴ کیلومتر ≈ ۱.۵ خانه. برای گرفتن
    میانگین درست باید جعبهٔ واقعی را بدهیم، نه فقط یک نقطه.

هزینه:
    ۸۰۴ بار باز کردن GeoTIFF. حدود یکی دو دقیقه.
    ماسک هم خوانده می‌شود تا درصد سوختگی درآید — چون توزیع کلاس را لازم داریم.

اجرا:
    python 02_build_meta.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import re
import csv
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
DATA = BIG / "data" / "burn_scars"
OUT_DIR = BIG / "data" / "meta"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "samples.csv"

PATTERN = re.compile(
    r"^subsetted_(?P<size>\d+x\d+)_HLS\."
    r"(?P<sensor>[A-Z0-9]+)\."
    r"(?P<tile>T[0-9A-Z]+)\."
    r"(?P<year>\d{4})(?P<doy>\d{3})\."
    r"(?P<ver>v[\d.]+)_merged\.tif$"
)

FIELDS = [
    "sample_id", "filename", "mask_filename", "orig_split",
    "sensor", "tile", "date", "year", "doy",
    "lat_center", "lon_center",
    "lat_min", "lat_max", "lon_min", "lon_max",
    "crs", "width", "height", "gsd_m",
    "pct_burn", "pct_unburn", "pct_nodata",
]


def parse_name(name):
    m = PATTERN.match(name)
    if not m:
        return None
    g = m.groupdict()
    d = datetime(int(g["year"]), 1, 1) + timedelta(days=int(g["doy"]) - 1)
    g["date"] = d.date().isoformat()
    return g


def process(img_path: Path, split: str, sample_id: int):
    meta = parse_name(img_path.name)
    if meta is None:
        return None, f"نام ناسازگار: {img_path.name}"

    mask_path = img_path.with_name(img_path.name.replace("_merged.tif", ".mask.tif"))
    if not mask_path.exists():
        return None, f"ماسک ندارد: {img_path.name}"

    with rasterio.open(img_path) as src:
        b = src.bounds
        # چهار گوشه را تبدیل می‌کنیم، نه فقط مرکز — چون UTM مستطیل را کج می‌کند
        xs = [b.left, b.right, b.right, b.left, (b.left + b.right) / 2]
        ys = [b.bottom, b.bottom, b.top, b.top, (b.bottom + b.top) / 2]
        lons, lats = warp_transform(src.crs, "EPSG:4326", xs, ys)
        row = {
            "sample_id": sample_id,
            "filename": img_path.name,
            "mask_filename": mask_path.name,
            "orig_split": split,
            "sensor": meta["sensor"],
            "tile": meta["tile"],
            "date": meta["date"],
            "year": int(meta["year"]),
            "doy": int(meta["doy"]),
            "lat_center": round(lats[4], 6),
            "lon_center": round(lons[4], 6),
            "lat_min": round(min(lats[:4]), 6),
            "lat_max": round(max(lats[:4]), 6),
            "lon_min": round(min(lons[:4]), 6),
            "lon_max": round(max(lons[:4]), 6),
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "gsd_m": round(abs(src.transform.a), 2),
        }

    with rasterio.open(mask_path) as m:
        mv = np.asarray(m.read(1))
        total = mv.size
        row["pct_burn"] = round(100.0 * np.count_nonzero(mv == 1) / total, 3)
        row["pct_unburn"] = round(100.0 * np.count_nonzero(mv == 0) / total, 3)
        row["pct_nodata"] = round(100.0 * np.count_nonzero(mv == -1) / total, 3)

    return row, None


def main():
    print(f"داده: {DATA}")
    print(f"خروجی: {OUT_CSV}\n")

    rows, problems = [], []
    sample_id = 0

    for split in ("training", "validation"):
        d = DATA / split
        if not d.exists():
            problems.append(f"پوشهٔ {split}/ وجود ندارد")
            continue
        imgs = sorted(d.glob("*_merged.tif"))
        print(f"{split}/ : {len(imgs)} فایل")
        for i, p in enumerate(imgs, 1):
            row, err = process(p, split, sample_id)
            if err:
                problems.append(err)
            else:
                rows.append(row)
                sample_id += 1
            if i % 100 == 0:
                print(f"   {i}/{len(imgs)}")
        print()

    if not rows:
        print("⛔ هیچ سطری ساخته نشد")
        return

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # ---------- گزارش سلامت ----------
    print("=" * 60)
    print(f"✅ {len(rows)} سطر نوشته شد → {OUT_CSV}")
    if problems:
        print(f"\n⚠️ {len(problems)} مشکل:")
        for p in problems[:10]:
            print("   ", p)

    lats = np.array([r["lat_center"] for r in rows])
    lons = np.array([r["lon_center"] for r in rows])
    burn = np.array([r["pct_burn"] for r in rows])
    nod = np.array([r["pct_nodata"] for r in rows])
    gsd = {r["gsd_m"] for r in rows}
    sizes = {(r["width"], r["height"]) for r in rows}
    crs_n = len({r["crs"] for r in rows})

    print(f"\nمحدودهٔ جغرافیایی — lat {lats.min():.3f} تا {lats.max():.3f}"
          f" · lon {lons.min():.3f} تا {lons.max():.3f}")
    print(f"اندازهٔ پیکسل: {gsd}   ابعاد: {sizes}   تعداد CRS متفاوت: {crs_n}")
    print(f"درصد سوختگی — کمینه {burn.min():.2f} · میانه {np.median(burn):.2f}"
          f" · میانگین {burn.mean():.2f} · بیشینه {burn.max():.2f}")
    print(f"درصد بی‌داده — میانگین {nod.mean():.3f} · بیشینه {nod.max():.2f}")

    n_zero = int((burn == 0).sum())
    n_high = int((nod > 5).sum())
    print(f"\nنمونه با صفر پیکسل سوخته: {n_zero}"
          f"   نمونه با بیش از ۵٪ بی‌داده: {n_high}")

    if len(sizes) > 1 or len(gsd) > 1:
        print("\n⚠️ ابعاد یا اندازهٔ پیکسل یکسان نیست — در ساخت batch باید دیده شود")


if __name__ == "__main__":
    main()

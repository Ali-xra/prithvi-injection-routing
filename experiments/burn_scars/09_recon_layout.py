# -*- coding: utf-8 -*-
"""
09_recon_layout.py — شناسایی چیدمان دیتاست روی دیسک
=====================================================
نسخه: v1 · تاریخ: 2026-07-30

چرا: کانفیگ رسمی IBM فایل‌های `splits/*.txt` می‌خواهد، ولی دیتاست ما به شکل
`training/` و `validation/` باز شده. قبل از ساختن هر چیزی باید بدانم:

    ۱) دقیقاً چند فایل کجاست — و آن اختلاف ۸۰۴ در برابر ۸۰۵ مقاله از کجاست
    ۲) آیا جایی فایل split هست که ندیده‌ام
    ۳) نام فایل‌ها دقیقاً چه شکلی است (برای ساختن .txt)
    ۴) آیا نمونه‌ای بدون جفت مانده

هیچ چیزی نمی‌نویسد. فقط گزارش.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
from pathlib import Path
from collections import Counter

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
DS = BIG / "data" / "burn_scars"
META = BIG / "data" / "meta"

print("=" * 78)
print("۱. ساختار پوشه")
print("=" * 78)
for p in sorted(DS.iterdir()):
    if p.is_dir():
        n = sum(1 for _ in p.rglob("*") if _.is_file())
        print(f"  [DIR ] {p.name:<15} {n:>6} فایل")
    else:
        print(f"  [FILE] {p.name:<15} {p.stat().st_size/1e9:>6.2f} GB")

print("\n" + "=" * 78)
print("۲. شمارش تصویر و ماسک")
print("=" * 78)
total = {}
for sub in ("training", "validation"):
    d = DS / sub
    if not d.exists():
        print(f"  ⛔ {sub} نیست")
        continue
    imgs = sorted(d.glob("*_merged.tif"))
    masks = sorted(d.glob("*.mask.tif"))
    others = [p for p in d.iterdir()
              if p.is_file() and not p.name.endswith(("_merged.tif", ".mask.tif"))]
    total[sub] = (imgs, masks, others)
    print(f"  {sub:<12} تصویر {len(imgs):>4} · ماسک {len(masks):>4} · بقیه {len(others):>3}")
    if others:
        for o in others[:5]:
            print(f"       ↳ {o.name}")

n_img = sum(len(v[0]) for v in total.values())
n_msk = sum(len(v[1]) for v in total.values())
print(f"\n  جمع: تصویر **{n_img}** · ماسک **{n_msk}**")

print("\n" + "=" * 78)
print("۳. جفت‌های ناقص")
print("=" * 78)
unpaired = []
for sub, (imgs, masks, _) in total.items():
    mset = {p.name.replace(".mask.tif", "") for p in masks}
    for im in imgs:
        stem = im.name.replace("_merged.tif", "")
        if stem not in mset:
            unpaired.append((sub, im.name, "ماسک ندارد"))
    iset = {p.name.replace("_merged.tif", "") for p in imgs}
    for mk in masks:
        stem = mk.name.replace(".mask.tif", "")
        if stem not in iset:
            unpaired.append((sub, mk.name, "تصویر ندارد"))
print(f"  ناجفت: {len(unpaired)}")
for u in unpaired[:10]:
    print(f"     {u}")

print("\n" + "=" * 78)
print("۴. مقایسه با conditioning_v1.csv")
print("=" * 78)
csv_path = META / "conditioning_v1.csv"
if csv_path.exists():
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    print(f"  سطر در CSV: {len(rows)}")
    on_disk = set()
    for sub, (imgs, _, _) in total.items():
        on_disk |= {p.name for p in imgs}
    in_csv = {r["filename"] for r in rows}
    print(f"  در CSV ولی روی دیسک نیست: {len(in_csv - on_disk)}")
    print(f"  روی دیسک ولی در CSV نیست: {len(on_disk - in_csv)}")
    for x in list(on_disk - in_csv)[:5]:
        print(f"     ↳ {x}")
    print(f"  توزیع split در CSV: {Counter(r['split'] for r in rows)}")
    print(f"  توزیع orig_split  : {Counter(r['orig_split'] for r in rows)}")
else:
    print("  ⛔ CSV نیست")

print("\n" + "=" * 78)
print("۵. جست‌وجوی فایل split موجود")
print("=" * 78)
found = list(DS.rglob("*.txt")) + list(DS.rglob("*split*"))
print(f"  پیدا شد: {len(found)}")
for f in found[:20]:
    print(f"     {f.relative_to(DS)}")

print("\n" + "=" * 78)
print("۶. نمونهٔ نام فایل — برای ساختن .txt")
print("=" * 78)
first = total.get("training", ([], [], []))[0][:2]
for p in first:
    print(f"  تصویر : {p.name}")
    print(f"  ماسک  : {p.name.replace('_merged.tif', '.mask.tif')}")
    print(f"  ریشه  : {p.name.replace('_merged.tif', '')}")

# -*- coding: utf-8 -*-
"""
10_make_split_files.py — ساخت فایل‌های split برای TerraTorch
==============================================================
نسخه: v1 · تاریخ: 2026-07-30

مسئله: کانفیگ رسمی IBM سه فایل `splits/*.txt` می‌خواهد، ولی دیتاست ما به شکل
`training/` و `validation/` باز شده و هیچ فایل split ندارد. و split رسمی هم به درد
ما نمی‌خورد چون **۷۳.۵٪ نشت** داشت.

قالب فایل split — از سورس TerraTorch، نه حدس:
    `GenericPixelWiseDataset` هر خط را یک **زیررشته** می‌گیرد
    (`allow_substring_split_file=True`) و پسوند را نادیده می‌گیرد
    (`ignore_split_file_extensions=True`).
    → پس یک خط با ریشهٔ نام، **هم تصویر و هم ماسک** را انتخاب می‌کند:
        subsetted_512x512_HLS.S30.T10SDH.2020248.v1.4
      که هم در `..._merged.tif` هست و هم در `....mask.tif`.

⚠️ نکتهٔ گلاب: کد از `glob.glob(os.path.join(data_root, image_grep))` استفاده می‌کند
   **بدون** `recursive=True`. پس `**` کار نمی‌کند ولی `*/` یک سطح را می‌گیرد.
   → `data_root = burn_scars/` و `image_grep = "*/*_merged.tif"`
   یعنی **هیچ فایلی کپی نمی‌شود** — نه ۲.۶ گیگ، نه هاردلینک.

خروجی: lab/configs/splits/{train,val,test}.txt  (کوچک، کنار کد نگه داشته می‌شود)

اجرا: python 10_make_split_files.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import glob
import os
from pathlib import Path
from collections import Counter

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
DS = BIG / "data" / "burn_scars"
CSV = BIG / "data" / "meta" / "conditioning_v1.csv"
OUT = Path(__file__).resolve().parent.parent / "configs" / "splits"

IMG_GREP = "*/*_merged.tif"
LBL_GREP = "*/*.mask.tif"


def stem(fn):
    return fn.replace("_merged.tif", "").replace(".mask.tif", "")


print("=" * 78)
print("۱. آزمون الگوی گلاب — همان چیزی که TerraTorch اجرا می‌کند")
print("=" * 78)
imgs = sorted(glob.glob(os.path.join(str(DS), IMG_GREP)))
lbls = sorted(glob.glob(os.path.join(str(DS), LBL_GREP)))
print(f"  data_root  = {DS}")
print(f"  image_grep = {IMG_GREP}   → {len(imgs)} فایل")
print(f"  label_grep = {LBL_GREP}   → {len(lbls)} فایل")
ok_glob = len(imgs) == 804 and len(lbls) == 804
print(f"  {'✅ هر دو ۸۰۴' if ok_glob else '⛔ تعداد اشتباه — الگو کار نمی‌کند'}")
if not ok_glob:
    sys.exit(1)

print("\n" + "=" * 78)
print("۲. نوشتن سه فایل split از split قفل‌شدهٔ خودمان")
print("=" * 78)
rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
by_split = {}
for r in rows:
    by_split.setdefault(r["split"], []).append(stem(r["filename"]))

OUT.mkdir(parents=True, exist_ok=True)
name_map = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}
written = {}
for sp, fname in name_map.items():
    stems = sorted(by_split.get(sp, []))
    (OUT / fname).write_text("\n".join(stems) + "\n", encoding="utf-8")
    written[sp] = stems
    print(f"  {fname:<10} {len(stems):>4} خط")

print("\n" + "=" * 78)
print("۳. اعتبارسنجی")
print("=" * 78)

# الف) هیچ ریشه‌ای در دو split نباشد
all_stems = [s for v in written.values() for s in v]
dup = [s for s, c in Counter(all_stems).items() if c > 1]
print(f"  الف) ریشهٔ تکراری بین splitها : {len(dup)}  {'✅' if not dup else '⛔ ' + str(dup[:3])}")

# ب) جمع = ۸۰۴
tot = len(all_stems)
print(f"  ب ) جمع سه فایل                : {tot}  {'✅' if tot == 804 else '⛔'}")

# ج) هر ریشه دقیقاً یک تصویر و یک ماسک بگیرد — همان منطق filter_valid_files
img_names = [os.path.basename(p) for p in imgs]
lbl_names = [os.path.basename(p) for p in lbls]
bad = []
for s in all_stems:
    ni = sum(1 for n in img_names if s in n)
    nl = sum(1 for n in lbl_names if s in n)
    if ni != 1 or nl != 1:
        bad.append((s, ni, nl))
print(f"  ج ) ریشه با تطبیق غیر ۱:۱      : {len(bad)}  {'✅' if not bad else '⛔'}")
for b in bad[:5]:
    print(f"        {b}")

# د) هیچ فایلی روی دیسک بی‌split نماند
covered = set()
for s in all_stems:
    covered |= {n for n in img_names if s in n}
print(f"  د ) تصویر بدون split           : {len(img_names) - len(covered)}  "
      f"{'✅' if len(covered) == len(img_names) else '⛔'}")

# ه) کاشی مشترک بین splitها — تأیید دوبارهٔ نشت صفر
tile_of = {stem(r["filename"]): r["tile"] for r in rows}
tiles = {sp: {tile_of[s] for s in v} for sp, v in written.items()}
shared = (tiles["train"] & tiles["val"]) | (tiles["train"] & tiles["test"]) | (tiles["val"] & tiles["test"])
print(f"  ه ) کاشی مشترک بین splitها     : {len(shared)}  {'✅' if not shared else '⛔ ' + str(list(shared)[:3])}")

print("\n" + "=" * 78)
all_ok = ok_glob and not dup and tot == 804 and not bad and len(covered) == len(img_names) and not shared
print("🟢 هر پنج بررسی گذشت." if all_ok else "🔴 حداقل یک بررسی رد شد.")
print(f"💾 {OUT}")
print("\nبرای کانفیگ:")
print(f'  train_data_root: {DS}')
print(f'  img_grep: "{IMG_GREP}"')
print(f'  label_grep: "{LBL_GREP}"')
print(f'  train_split: {OUT / "train.txt"}')

# -*- coding: utf-8 -*-
"""
20_build_split_dirs.py — ساختن پوشه‌های split جدا-به-tile روی دیسک
====================================================================
نسخه: v1 · تاریخ: 2026-08-01

🔴🔴 خطای ۱۷ — چرا این فایل وجود دارد.

`conditioning_v1.csv` دو ستون split دارد:

    orig_split : training 540 · validation 264   ← split منتشرشدهٔ اصلی
    split      : train 563 · val 121 · test 120  ← split جدا-به-tile خودمان

`12_image_proxy_control.py` از ستون `split` استفاده می‌کند — یعنی همان عددی که
کل پروژه رویش بنا شد (`مکان +0.0287`) روی split **تمیز** اندازه‌گیری شد.

ولی `16_run_arm.py` هیچ اشاره‌ای به split ندارد. مستقیم به پوشه‌های فیزیکی
`training/` و `validation/` وصل است — یعنی split **منتشرشده**.

اندازه‌گیری‌شده ۱ اوت:
    tileهای مشترک بین training و validation : ۱۲۴
    chipهای val روی tile مشترک              : ۱۹۴ از ۲۶۴  (۷۳٪)

یعنی **فرض را روی split تمیز تأیید کردیم و آزمایش را روی split نشتی اجرا کردیم.**
هر ۱۱ اجرای GPU روی split نشتی بوده.

چرا این برای همین سؤال مهلک است — یافتهٔ F5 خودمان:

    «نشتی، دادهٔ کمکی را **کم‌فایده‌تر** نشان می‌دهد، نه بیشتر.»
    روی split نشتی: +0.0052 · روی split تمیز: +0.0288

وقتی مدل همان tile را در آموزش دیده، ظاهر آن نقطه را حفظ می‌کند و دیگر به
مختصات نیازی ندارد. یعنی دقیقاً همان مکانیزمی که نتیجهٔ صفر ما را می‌سازد.

این اسکریپت با **hardlink** پوشه‌های جدید می‌سازد تا فضای دیسک دوبرابر نشود،
و تا زمانی که جدا-به-tile بودن را اثبات نکند، چیزی نمی‌نویسد.

اجرا: python 20_build_split_dirs.py
"""
import sys, csv, re, shutil
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
SRC = BIG / "data" / "burn_scars"
DST = BIG / "data" / "burn_scars_tiledisjoint"
CSV_PATH = BIG / "data" / "meta" / "conditioning_v1.csv"

SPLIT_DIR = {"train": "training", "val": "validation", "test": "test"}
TILE_RE = re.compile(r"\.(T[0-9]{2}[A-Z]{3})\.")


def tile_of(name):
    m = TILE_RE.search(name)
    if not m:
        raise ValueError(f"tile از نام استخراج نشد: {name}")
    return m.group(1)


def locate(name):
    """chip ممکن است در training/ یا validation/ اصلی باشد."""
    for sub in ("training", "validation"):
        p = SRC / sub / name
        if p.exists():
            return p
    raise FileNotFoundError(name)


def main():
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    print(f"{len(rows)} ردیف در CSV")
    print("توزیع split:", dict(Counter(r["split"] for r in rows)))

    tiles = defaultdict(set)
    for r in rows:
        tiles[r["split"]].add(tile_of(r["filename"]))

    # 🔴 دروازه: پیش از نوشتن هر فایلی، جدا-به-tile بودن باید اثبات شود.
    bad = False
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = tiles[a] & tiles[b]
        print(f"   tile مشترک {a}/{b}: {len(shared)}")
        if shared:
            bad = True
            print(f"      نمونه: {sorted(shared)[:10]}")
    if bad:
        raise SystemExit("🔴 ستون `split` جدا-به-tile نیست. چیزی نوشته نشد.")
    print("   ✅ صفر tile مشترک بین هر سه split.")

    # مقایسه با split اصلی، برای ثبت در مستندات
    o = defaultdict(set)
    for r in rows:
        o[r["orig_split"]].add(tile_of(r["filename"]))
    shared_orig = o["training"] & o["validation"]
    n_leaky = sum(1 for r in rows if r["orig_split"] == "validation"
                  and tile_of(r["filename"]) in shared_orig)
    n_val_orig = sum(1 for r in rows if r["orig_split"] == "validation")
    print(f"   برای مقایسه — split منتشرشده: {len(shared_orig)} tile مشترک، "
          f"{n_leaky} از {n_val_orig} chip val نشتی ({n_leaky/n_val_orig:.0%})")

    if DST.exists():
        shutil.rmtree(DST)
    made = Counter()
    for r in rows:
        out = DST / SPLIT_DIR[r["split"]]
        out.mkdir(parents=True, exist_ok=True)
        for key in ("filename", "mask_filename"):
            src = locate(r[key])
            dst = out / r[key]
            if not dst.exists():
                try:
                    dst.hardlink_to(src)          # بدون مصرف فضای اضافه
                except OSError:
                    shutil.copy2(src, dst)        # اگر روی درایو دیگری بود
        made[r["split"]] += 1

    print("\nنوشته شد:")
    for s, d in SPLIT_DIR.items():
        n_img = len(list((DST / d).glob("*_merged.tif")))
        n_msk = len(list((DST / d).glob("*.mask.tif")))
        print(f"   {d:11s} {n_img:4d} تصویر · {n_msk:4d} ماسک   (انتظار {made[s]})")
        if n_img != made[s] or n_msk != made[s]:
            raise SystemExit(f"🔴 تعداد فایل {d} با CSV نمی‌خواند.")

    print(f"\n✅ {DST}")
    print("   حالا `16_run_arm.py` باید به این پوشه‌ها وصل شود، نه به burn_scars/.")


if __name__ == "__main__":
    main()

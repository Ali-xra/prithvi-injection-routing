# -*- coding: utf-8 -*-
"""
08_finalize_conditioning.py — بستن فاز داده: بردار شرط ۱۰ بعدی
================================================================
نسخه: v1 · تاریخ: 2026-07-29

این آخرین فایلِ فاز داده است. بعد از اجرای آن با `--lock`، دادهٔ کمکی **قفل**
می‌شود و تا پایان آزمایش عوض نمی‌شود.

سه کار:
    ۱) افزودن چهار ستون جغرافیا و فصل به شش ویژگی جوّی → **بردار شرط ۱۰ بعدی**
    ۲) تصمیم دربارهٔ ۱۹ نمونه با پیکسل بی‌داده
    ۳) نوشتن فایل قفل‌شده + مانیفست

## چرا ۱۰ بعد و نه ۶؟

اندازه‌گیری `07` (نه سلیقه):

    AUC — فقط جوّی           0.647
    AUC — جغرافیا و فصل      0.695   ← قوی‌تر
    AUC — هر دو              0.716

قوی‌ترین اسکالر سراسریِ در دسترس ما مکان و فصل است، نه آب‌وهوا. و **خودِ
Prithvi هم برای همین یک مسیر دارد** (بایاس متادیتای مختصات و روزِ سال، بخش IV).

سه دلیل برای ۱۰ بعد:
    ۱) هدف آزمایش «تفاوت بین مسیرهای تزریق» است. بردار شرطِ پرسیگنال‌تر،
       شانس دیدن آن تفاوت را بالا می‌برد.
    ۲) بازوی «بایاس» دقیقاً بازتولید مسیر خود Prithvi می‌شود — نه یک بازوی خودساخته.
    ۳) انتخاب مبتنی بر اندازه‌گیری است و آن اندازه‌گیری **قبل از** دیدن هر نتیجهٔ
       مدلی انجام شد.

⚠️ روزِ سال دوّار است (۳۶۵ و ۱ همسایه‌اند) → sin/cos، نه عدد خام. همان تله‌ای که
   برای جهت باد داشتیم.

## حالت‌های تصمیم دربارهٔ نمونه‌های بی‌داده

    (بدون آرگومان)   فقط تشخیص — چیزی نمی‌نویسد
    --mode flag      نگه‌داشتن همه + ستون `qc_high_nodata`   ⭐ پیشنهاد
    --mode drop      حذف نمونه‌های بالای آستانه
    --lock           نوشتن فایل نهایی و علامت‌گذاری به‌عنوان قفل‌شده

خروجی:
    <BIG>/data/meta/conditioning_v1.csv
    <BIG>/data/meta/conditioning_v1_manifest.json
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
import argparse
from pathlib import Path

import numpy as np

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing-flood"
META = BIG / "data" / "meta"
IN_CSV = META / "wind_features.csv"
OUT_CSV = META / "conditioning_v1.csv"
OUT_MAN = META / "conditioning_v1_manifest.json"

NODATA_THR = 5.0        # درصد — همان آستانه‌ای که در 02 علامت خورد

W_FEATS = ["mean_speed", "max_speed", "dir_sin", "dir_cos",
           "precip_7d_log", "mean_temp"]
G_FEATS = ["lat", "lon", "doy_sin", "doy_cos"]
COND10 = [f + "_z" for f in W_FEATS + G_FEATS]


def load():
    rows = list(csv.DictReader(IN_CSV.open(encoding="utf-8")))
    for r in rows:
        r["pct_water"] = float(r["pct_water"])
        r["pct_nodata"] = float(r["pct_nodata"])
        r["lat_center"] = float(r["lat_center"])
        r["lon_center"] = float(r["lon_center"])
        r["doy"] = int(r["doy"])
    return rows


def diagnose(rows):
    bad = [r for r in rows if r["pct_nodata"] > NODATA_THR]
    print("=" * 78)
    print(f"تشخیص — نمونه‌های با بیش از {NODATA_THR}٪ پیکسل بی‌داده")
    print("=" * 78)
    nd = np.array([r["pct_nodata"] for r in rows])
    print(f"کل نمونه: {len(rows)}")
    print(f"درصد بی‌داده — میانه {np.median(nd):.3f} · میانگین {nd.mean():.3f} · "
          f"بیشینه {nd.max():.3f}")
    print(f"بالای {NODATA_THR}٪: **{len(bad)} نمونه** ({100*len(bad)/len(rows):.1f}٪ کل)\n")

    if not bad:
        return bad

    print(f"{'split':<8}{'تعداد':>7}{'بیشینه':>10}{'میانهٔ سوختگی':>16}")
    print("-" * 45)
    for sp in ("train", "val", "test"):
        g = [r for r in bad if r["split"] == sp]
        if g:
            print(f"{sp:<8}{len(g):>7}{max(r['pct_nodata'] for r in g):>10.2f}"
                  f"{np.median([r['pct_water'] for r in g]):>16.2f}")

    print(f"\nده نمونهٔ بدتر:")
    print(f"{'id':<7}{'split':<8}{'بی‌داده٪':>10}{'سوختگی٪':>10}  {'رویداد':<12}{'تاریخ'}")
    print("-" * 62)
    for r in sorted(bad, key=lambda r: -r["pct_nodata"])[:10]:
        print(f"{r['sample_id']:<7}{r['split']:<8}{r['pct_nodata']:>10.2f}"
              f"{r['pct_water']:>10.2f}  {r['event']:<12}{r['date']}")

    # آیا کاشی‌های این نمونه‌ها نمونهٔ سالم هم دارند؟
    bad_events = {r["event"] for r in bad}
    only_bad = [t for t in bad_events
                if all(r["pct_nodata"] > NODATA_THR
                       for r in rows if r["event"] == t)]
    print(f"\nکاشی‌های درگیر: {len(bad_events)}")
    print(f"کاشی‌هایی که **همهٔ** نمونه‌هایشان بی‌داده‌اند: {len(only_bad)}")
    print("   → حذف این‌ها یعنی حذف کاملِ آن کاشی از split قفل‌شده" if only_bad
          else "   → حذف، هیچ کاشی‌ای را کاملاً از split حذف نمی‌کند")
    return bad


def add_geo(rows):
    """چهار ستون جغرافیا و فصل، نرمال‌شده **فقط از train**."""
    for r in rows:
        a = 2 * np.pi * r["doy"] / 365.25
        r["lat"] = r["lat_center"]
        r["lon"] = r["lon_center"]
        r["doy_sin"] = float(np.sin(a))
        r["doy_cos"] = float(np.cos(a))

    tr = [r for r in rows if r["split"] == "train"]
    stats = {}
    for k in G_FEATS:
        a = np.array([r[k] for r in tr], dtype=float)
        mu, sd = float(a.mean()), float(a.std())
        stats[k] = {"mean": mu, "std": sd if sd > 1e-9 else 1.0}
    for r in rows:
        for k in G_FEATS:
            r[k + "_z"] = round((r[k] - stats[k]["mean"]) / stats[k]["std"], 6)
            r[k] = round(r[k], 6)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["flag", "drop"], default=None)
    ap.add_argument("--lock", action="store_true")
    args = ap.parse_args()

    if not IN_CSV.exists():
        print(f"⛔ {IN_CSV} نیست — اول 05b را اجرا کن")
        return

    rows = load()
    bad = diagnose(rows)

    if args.mode is None:
        print("\n" + "=" * 78)
        print("فقط تشخیص. برای نوشتن فایل نهایی یکی از این‌ها را بزن:")
        print("   python 08_finalize_conditioning.py --mode flag --lock   ⭐ پیشنهاد")
        print("   python 08_finalize_conditioning.py --mode drop --lock")
        print("=" * 78)
        return

    bad_ids = {r["sample_id"] for r in bad}
    for r in rows:
        r["qc_high_nodata"] = int(r["sample_id"] in bad_ids)

    if args.mode == "drop":
        before = len(rows)
        rows = [r for r in rows if not r["qc_high_nodata"]]
        print(f"\n⚠️ حالت drop: {before - len(rows)} نمونه حذف شد → {len(rows)} ماند")
    else:
        print(f"\n✅ حالت flag: هر {len(rows)} نمونه ماند، "
              f"{len(bad_ids)} تا با `qc_high_nodata=1` علامت خوردند")

    gstats = add_geo(rows)

    # --- بررسی سلامت بردار ۱۰ بعدی ---
    missing = [c for c in COND10 if c not in rows[0]]
    if missing:
        print(f"⛔ ستون‌های گمشده: {missing}")
        return

    X = np.array([[float(r[c]) for c in COND10] for r in rows])
    tr_mask = np.array([r["split"] == "train" for r in rows])

    print("\n" + "=" * 78)
    print(f"بردار شرط ۱۰ بعدی — آمار روی train ({tr_mask.sum()} نمونه)")
    print("=" * 78)
    print(f"{'بُعد':<18}{'میانگین':>10}{'انحراف':>10}{'کمینه':>10}{'بیشینه':>10}")
    print("-" * 78)
    for i, c in enumerate(COND10):
        col = X[tr_mask, i]
        print(f"{c:<18}{col.mean():>10.4f}{col.std():>10.4f}"
              f"{col.min():>10.3f}{col.max():>10.3f}")
    print("-" * 78)
    print("انتظار: میانگین ≈ ۰ و انحراف ≈ ۱ برای همه (نرمال‌سازی فقط از train)")

    bad_norm = [COND10[i] for i in range(10)
                if abs(X[tr_mask, i].mean()) > 0.01 or abs(X[tr_mask, i].std() - 1) > 0.01]
    print(f"{'✅ همه درست' if not bad_norm else '⛔ نرمال‌سازی خراب: ' + str(bad_norm)}")

    # همبستگی بین ابعاد — اگر دو بُعد ~۱ باشند، یکی اضافه است
    C = np.corrcoef(X[tr_mask].T)
    iu = np.triu_indices(10, k=1)
    worst = np.argmax(np.abs(C[iu]))
    i, j = iu[0][worst], iu[1][worst]
    print(f"بیشترین همبستگی بین دو بُعد: {COND10[i]} ↔ {COND10[j]} = {C[i, j]:+.3f}")
    if abs(C[i, j]) > 0.9:
        print("   ⚠️ نزدیک به ۱ — یکی از این دو عملاً تکراری است")

    if not args.lock:
        print("\n(بدون --lock فایلی نوشته نشد)")
        return

    fields = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    counts = {sp: sum(1 for r in rows if r["split"] == sp)
              for sp in ("train", "val", "test")}
    OUT_MAN.write_text(json.dumps({
        "status": "FROZEN",
        "frozen_on": "2026-07-29",
        "note": "بعد از این تاریخ دادهٔ کمکی عوض نمی‌شود. هر تغییری = نسخهٔ v2 جدا.",
        "conditioning_vector": COND10,
        "n_dims": len(COND10),
        "weather_source": "Open-Meteo Archive API · era5_seamless · اعتبارسنجی‌شده در 06",
        "geo_source": "از samples.csv — مرکز کاشی + روزِ سال (sin/cos)",
        "normalization": "z-score، فقط از split train",
        "geo_stats": gstats,
        "nodata_mode": args.mode,
        "nodata_threshold_pct": NODATA_THR,
        "n_flagged_high_nodata": len(bad_ids),
        "n_samples": len(rows),
        "split_counts": counts,
        "why_10_dims": "07: AUC جوّی 0.647 · جغرافیا 0.695 · هر دو 0.716",
        "known_limitation": "07b: افزودهٔ آب‌وهوا بر جغرافیا از نویز جدا نشد (p=0.209)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("🔒 دادهٔ کمکی قفل شد")
    print("=" * 78)
    print(f"نمونه: {len(rows)}   train {counts['train']} · val {counts['val']} · test {counts['test']}")
    print(f"بردار شرط: {len(COND10)} بعد")
    print(f"   جوّی    : {', '.join(f + '_z' for f in W_FEATS)}")
    print(f"   جغرافیا : {', '.join(f + '_z' for f in G_FEATS)}")
    print(f"\n💾 {OUT_CSV}")
    print(f"💾 {OUT_MAN}")
    print("\n✅ فاز داده تمام است. قدم بعد: کانفیگ TerraTorch و دودتست روی Colab.")


if __name__ == "__main__":
    main()

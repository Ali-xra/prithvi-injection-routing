# -*- coding: utf-8 -*-
"""
05b_build_features_openmeteo.py — شش ویژگی جوّی برای هر ۸۰۴ نمونه
==================================================================
نسخه: v1 · تاریخ: 2026-07-29
جایگزین `05_build_features.py` (که از فایل‌های CDS می‌خواند و فقط ۱۶۸ نمونه داشت)

چرا جایگزین شد:
    CDS بعد از ۶ ساعت فقط ۱۶۸ نمونه داد. Open-Meteo هر ۸۰۴ تا را در ۱۰۵ ثانیه.
    اعتبار Open-Meteo در `06_validate_openmeteo.py` سنجیده شد:
        دما   r=0.984  اریبی -0.32 °C
        باد   r=0.918  اریبی +0.21 m/s
        جهت   خطای زاویه‌ای میانگین 9.7°  (میانه 6.0°)
        بارش  r=0.936

⚠️ درسی که نباید گم شود — باگ tp در `05`:
    `tp` در ERA5-Land از ۰۰ UTC انباشته می‌شود، و مقدارِ ساعت 00:00 روز N
    **کل بارش روز N-1** است. کد `05` این بود:
        daily = tp.groupby("valid_time.date").max()
    گروهِ روز N شامل 00:00 (کل روز N-1) است، پس max اغلب عددِ روز قبل را
    برمی‌دارد. اندازه‌گیری‌شده در `06b`: نسبت 1.96 — یعنی تقریباً دو برابر،
    و روزها یک واحد جابه‌جا. **هرگز از groupby(date).max() روی میدان انباشته
    استفاده نکن.** روش درست: تفاضل ساعت‌به‌ساعت با تشخیص ریست.
    اینجا این تله وجود ندارد، چون Open-Meteo بارش را از پیش ساعتی می‌دهد.

⚠️ تغییر واحد نسبت به `05`:
    mean_temp حالا **°C** است نه کلوین. بعد از z-نرمال‌سازی بی‌اثر است،
    ولی اگر عدد خام را جایی مقایسه کردی حواست باشد.

ویژگی‌ها (همان شش‌تای `05`، با همان تعریف):
    mean_speed  میانگین سرعت باد ۱۶۸ ساعت            m/s
    max_speed   بیشینهٔ سرعت باد                      m/s
    dir_sin     سینوس جهتِ **میانگین برداری**         بی‌بعد
    dir_cos     کسینوس همان                           بی‌بعد
    precip_7d   جمع بارش ۷ روزه                       mm
    mean_temp   میانگین دما                           °C

    ⚠️ جهت از میانگینِ بردار حساب می‌شود، نه میانگین زاویه. میانگین ۳۵۹° و ۱°
       با روش زاویه‌ای می‌شود ۱۸۰° (کاملاً برعکس)، با روش برداری می‌شود ۰°.

خروجی:
    <BIG>/data/meta/wind_features.csv   (همان ستون‌های 05 + ستون‌های _z)

اجرا:
    python 05b_build_features_openmeteo.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
from pathlib import Path

import numpy as np

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
META = BIG / "data" / "meta"
ERA5 = BIG / "data" / "era5"
NPZ = ERA5 / "openmeteo_series.npz"
IN_CSV = META / "samples_split.csv"
OUT_CSV = META / "wind_features.csv"
OUT_STATS = META / "wind_features_norm.json"

RAW = ["mean_speed", "max_speed", "dir_sin", "dir_cos",
       "precip_7d", "precip_7d_log", "mean_temp"]

# ⚠️ چرا precip_7d_log هم هست:
#    توزیع precip_7d به‌شدت چوله است (میانه 2.5 · میانگین 11.6 · بیشینه 154.5).
#    z-نرمال‌سازی مستقیمِ چنین توزیعی تقریباً همهٔ نمونه‌ها را دور یک مقدار جمع
#    می‌کند و سیگنال را در چند نقطهٔ پرت متمرکز می‌کند — برای متغیری که قرار است
#    «شرط» مدل باشد بد است. log1p این را می‌گستراند.
#    ستون خام حذف نشده؛ هر دو می‌مانند تا بتوانیم اثرشان را جدا بسنجیم.


def features(speed, direction, precip, temp):
    """یک نمونه: چهار سری ۱۶۸ ساعته → شش عدد."""
    # قرارداد هواشناسی: direction جهتی است که باد **از آن می‌آید**.
    # بردار حرکت باد پس در جهت مخالف است:
    rad = np.radians(direction)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)

    ubar, vbar = np.nanmean(u), np.nanmean(v)
    theta = np.arctan2(vbar, ubar)      # میانگین برداری، نه میانگین زاویه

    return {
        "mean_speed": float(np.nanmean(speed)),
        "max_speed": float(np.nanmax(speed)),
        "dir_sin": float(np.sin(theta)),
        "dir_cos": float(np.cos(theta)),
        "precip_7d": float(np.nansum(precip)),
        "precip_7d_log": float(np.log1p(np.nansum(precip))),
        "mean_temp": float(np.nanmean(temp)),
        "n_timesteps": int(speed.size),
        "nan_pct": round(float(np.isnan(speed).mean() * 100), 3),
    }


def main():
    if not NPZ.exists():
        print(f"⛔ {NPZ} نیست — اول 04c_fetch_era5_openmeteo.py را اجرا کن")
        return

    z = np.load(NPZ, allow_pickle=True)
    ids = [int(s) for s in z["sample_id"]]
    speed = z["wind_speed_10m"].astype(np.float64)
    direction = z["wind_direction_10m"].astype(np.float64)
    precip = z["precipitation"].astype(np.float64)
    temp = z["temperature_2m"].astype(np.float64)

    rows = {int(r["sample_id"]): r
            for r in csv.DictReader(IN_CSV.open(encoding="utf-8"))}

    out, missing = [], []
    for i, sid in enumerate(ids):
        if sid not in rows:
            missing.append(sid)
            continue
        fe = features(speed[i], direction[i], precip[i], temp[i])
        out.append({**rows[sid], **fe})

    print(f"نمونه در npz: {len(ids)}   ساخته‌شده: {len(out)}   بدون سطر CSV: {len(missing)}")
    if len(out) != len(rows):
        print(f"⚠️ {len(rows) - len(out)} نمونه از samples_split.csv ویژگی نگرفت")

    # --- نرمال‌سازی فقط از train ---
    # ⚠️ اگر val/test را هم در محاسبهٔ میانگین و انحراف وارد کنی، نشت اطلاعات است.
    tr = [o for o in out if o["split"] == "train"]
    print(f"نرمال‌سازی از {len(tr)} نمونهٔ train (val/test دخالت ندارند)")
    stats = {}
    for k in RAW:
        a = np.array([o[k] for o in tr], dtype="float64")
        mu, sd = float(a.mean()), float(a.std())
        stats[k] = {"mean": mu, "std": sd if sd > 1e-9 else 1.0}
    for o in out:
        for k in RAW:
            o[k + "_z"] = round((o[k] - stats[k]["mean"]) / stats[k]["std"], 6)
            o[k] = round(o[k], 6)

    fields = list(out[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    OUT_STATS.write_text(json.dumps({
        "source": "Open-Meteo Archive API · era5_seamless",
        "units": {"mean_speed": "m/s", "max_speed": "m/s",
                  "dir_sin": "-", "dir_cos": "-",
                  "precip_7d": "mm", "precip_7d_log": "log1p(mm)",
                  "mean_temp": "degC"},
        "normalized_from": "train split only",
        "n_train": len(tr), "n_total": len(out),
        "stats": stats,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"{'ویژگی':<14}{'کمینه':>11}{'میانه':>11}{'میانگین':>11}{'بیشینه':>11}{'انحراف':>11}")
    print("-" * 74)
    for k in RAW:
        a = np.array([o[k] for o in out])
        print(f"{k:<14}{a.min():>11.3f}{np.median(a):>11.3f}"
              f"{a.mean():>11.3f}{a.max():>11.3f}{a.std():>11.3f}")
    print("=" * 74)

    # --- بررسی سلامت ---
    nt = np.array([o["n_timesteps"] for o in out])
    nz = np.array([o["nan_pct"] for o in out])
    print(f"گام زمانی: کمینه {nt.min()} · بیشینه {nt.max()}   (انتظار ۱۶۸)")
    print(f"درصد NaN:  میانگین {nz.mean():.3f} · بیشینه {nz.max():.3f}")

    # dir_sin² + dir_cos² باید دقیقاً ۱ باشد — اگر نبود، جایی ریاضی خراب است
    s = np.array([o["dir_sin"] for o in out])
    c = np.array([o["dir_cos"] for o in out])
    err = np.abs(s ** 2 + c ** 2 - 1.0).max()
    print(f"بیشینهٔ |sin²+cos²-1| = {err:.2e}   {'✅' if err < 1e-4 else '⛔ ریاضی جهت خراب است'}")

    # توزیع بین splitها — اگر یکی خیلی متفاوت باشد، split مشکوک است
    print(f"\n{'split':<8}{'n':>6}" + "".join(f"{k[:9]:>11}" for k in RAW))
    print("-" * 74)
    for sp in ("train", "val", "test"):
        g = [o for o in out if o["split"] == sp]
        if not g:
            continue
        print(f"{sp:<8}{len(g):>6}" + "".join(
            f"{np.mean([o[k] for o in g]):>11.3f}" for k in RAW))

    print(f"\n💾 {OUT_CSV}")
    print(f"💾 {OUT_STATS}")
    print("\n✅ دادهٔ کمکی آماده است. قدم بعد: GPU و اولین fine-tune پایه.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
06_validate_openmeteo.py — آیا Open-Meteo همان ERA5 است؟
=========================================================
نسخه: v1 · تاریخ: 2026-07-29

سؤالی که این فایل جواب می‌دهد:
    Open-Meteo یک **واسطه** است، نه منبع. ادعا می‌کند داده‌اش ERA5 است.
    از کجا می‌دانیم دست نخورده؟ جواب «چون سایتش نوشته» کافی نیست.

    آن ۱۶۸ فایلی که با CDS مستقیماً از خودِ کوپرنیکوس گرفتیم (و ۶ ساعت برد)
    اینجا تبدیل می‌شود به **مدرک**. عدد به عدد مقایسه می‌کنیم.

چه چیزی با چه چیزی مقایسه می‌شود:
    CDS  : ERA5-Land 0.1° · میانگین مکانی روی جعبهٔ نمونه · مستقیم از Copernicus
    OM   : era5_seamless  · یک نقطه (نزدیک‌ترین خانه به مرکز) · از Open-Meteo

    ⚠️ این دو **قرار نیست دقیقاً یکی باشند** و اگر بودند باید مشکوک می‌شدیم:
       ۱) OM باد را از ERA5 0.25° می‌دهد، CDS از ERA5-Land 0.1°
       ۲) CDS میانگین جعبه است، OM تک‌نقطه
       پس انتظار ما «همبستگی بالا و اریبی نزدیک صفر» است، نه «برابری».

تبدیل واحد (تلهٔ اصلی این فایل):
    t2m   کلوین      → °C          منهای 273.15
    u10,v10 m/s      → سرعت m/s    sqrt(u²+v²)
                     → جهت درجه    (270 - atan2(v,u)·180/π) mod 360
                       ⚠️ قرارداد هواشناسی: جهتی که باد **از آن می‌آید**.
                       atan2 خام جهت *رفتن* بردار را می‌دهد — ۱۸۰ درجه فرق.
    tp    متر تجمعی  → mm          بیشینهٔ هر روز، جمع هفت روز، ×1000
                       ⚠️ tp از ۰۰ UTC انباشته می‌شود. جمع‌زدن ۲۴ ساعت
                          یعنی چندبرابر شمردن.

بارش ساعت‌به‌ساعت مقایسه نمی‌شود (به‌خاطر همان انباشتگی) — فقط **جمع ۷ روزه**.

خروجی:
    <BIG>/data/era5/validation_openmeteo.csv    یک سطر به ازای هر نمونهٔ مشترک
    <BIG>/data/era5/validation_summary.json     خلاصهٔ آماری + حکم

اجرا:
    python 06_validate_openmeteo.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import xarray as xr

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
META = BIG / "data" / "meta"
ERA5 = BIG / "data" / "era5"
PERSAMPLE = ERA5 / "samples"
NPZ = ERA5 / "openmeteo_series.npz"
IN_CSV = META / "samples_split.csv"
OUT_CSV = ERA5 / "validation_openmeteo.csv"
OUT_JSON = ERA5 / "validation_summary.json"

DAYS_BEFORE = 7
N_HOURS = DAYS_BEFORE * 24

# آستانه‌های قبولی — پیش از دیدن نتیجه تعیین شده‌اند تا خودمان را گول نزنیم
GATES = {
    "temp_r":       ("همبستگی دما",            0.95, "min"),
    "temp_bias":    ("اریبی دما (°C)",          2.0,  "abs"),
    "speed_r":      ("همبستگی سرعت باد",       0.75, "min"),
    "speed_bias":   ("اریبی سرعت باد (m/s)",   1.5,  "abs"),
    "dir_mae":      ("خطای زاویه‌ای باد (°)",   45.0, "max"),
    "precip_r":     ("همبستگی بارش ۷ روزه",    0.70, "min"),
}


def cds_series(path, row):
    """فایل CDS → چهار سری زمانی ۱۶۸ ساعته، با واحدهای هم‌راستا با Open-Meteo."""
    with xr.open_dataset(path) as ds:
        lat_s = slice(float(row["lat_max"]) + 0.15, float(row["lat_min"]) - 0.15)
        lon_s = slice(float(row["lon_min"]) - 0.15, float(row["lon_max"]) + 0.15)
        sub = ds.sel(latitude=lat_s, longitude=lon_s)
        if sub.sizes.get("latitude", 0) == 0 or sub.sizes.get("longitude", 0) == 0:
            sub = ds.sel(latitude=float(row["lat_center"]),
                         longitude=float(row["lon_center"]),
                         method="nearest").expand_dims(["latitude", "longitude"])

        d0 = date.fromisoformat(row["date"])
        t0 = np.datetime64(str(d0 - timedelta(days=DAYS_BEFORE)))
        t1 = np.datetime64(str(d0)) - np.timedelta64(1, "h")
        sub = sub.sel(valid_time=slice(t0, t1))
        if sub.sizes.get("valid_time", 0) != N_HOURS:
            return None, f"طول {sub.sizes.get('valid_time', 0)} ≠ {N_HOURS}"

        dims = [d for d in ("latitude", "longitude") if d in sub.dims]
        u = sub["u10"].mean(dim=dims, skipna=True).values.astype(np.float64)
        v = sub["v10"].mean(dim=dims, skipna=True).values.astype(np.float64)
        t2 = sub["t2m"].mean(dim=dims, skipna=True).values.astype(np.float64)
        tp = sub["tp"].mean(dim=dims, skipna=True)

        # tp از ۰۰ UTC انباشته می‌شود → بیشینهٔ هر روز = کل آن روز
        daily = tp.groupby("valid_time.date").max(skipna=True).values
        precip_mm = float(np.nansum(daily)) * 1000.0

    speed = np.sqrt(u ** 2 + v ** 2)
    # قرارداد هواشناسی: جهتی که باد از آن می‌آید
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    temp_c = t2 - 273.15
    return dict(temp=temp_c, speed=speed, direction=direction,
                precip_7d=precip_mm), None


def ang_diff(a, b):
    """اختلاف زاویه‌ای در بازهٔ [-180, 180]. برای ۳۵۹ و ۱ جواب ۲ می‌دهد نه ۳۵۸."""
    return (a - b + 180.0) % 360.0 - 180.0


def pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    x, y = x[m], y[m]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def main():
    if not NPZ.exists():
        print(f"⛔ {NPZ} نیست — اول 04c_fetch_era5_openmeteo.py را اجرا کن")
        return
    if not PERSAMPLE.exists():
        print(f"⛔ {PERSAMPLE} نیست — فایل‌های مرجع CDS پیدا نشد")
        return

    z = np.load(NPZ, allow_pickle=True)
    om_ids = list(z["sample_id"])
    om_index = {int(s): i for i, s in enumerate(om_ids)}
    om_temp = z["temperature_2m"].astype(np.float64)
    om_speed = z["wind_speed_10m"].astype(np.float64)
    om_dir = z["wind_direction_10m"].astype(np.float64)
    om_prec = z["precipitation"].astype(np.float64)

    rows = {int(r["sample_id"]): r
            for r in csv.DictReader(IN_CSV.open(encoding="utf-8"))}

    files = sorted(PERSAMPLE.glob("*.nc"))
    print(f"فایل مرجع CDS: {len(files)}")
    print(f"نمونه در Open-Meteo: {len(om_ids)}")
    print(f"آستانه‌ها پیش از دیدن نتیجه تثبیت شده‌اند.\n")

    per_sample, skipped = [], []
    pool = {k: {"cds": [], "om": []} for k in ("temp", "speed")}
    dir_diffs = []
    prec_cds, prec_om = [], []

    for f in files:
        sid = int(f.name.split("__")[0])
        if sid not in om_index or sid not in rows:
            skipped.append((sid, "در Open-Meteo یا CSV نبود"))
            continue
        try:
            cds, err = cds_series(f, rows[sid])
        except Exception as e:
            skipped.append((sid, repr(e)))
            continue
        if cds is None:
            skipped.append((sid, err))
            continue

        i = om_index[sid]
        om = dict(temp=om_temp[i], speed=om_speed[i],
                  direction=om_dir[i], precip_7d=float(np.nansum(om_prec[i])))

        d = ang_diff(om["direction"], cds["direction"])
        # زاویه فقط وقتی معنا دارد که باد واقعاً بوزد
        strong = (cds["speed"] > 1.0) & (om["speed"] > 1.0)
        dir_mae_s = float(np.nanmean(np.abs(d[strong]))) if strong.sum() > 5 else np.nan

        rec = {
            "sample_id": sid,
            "date": rows[sid]["date"],
            "temp_r": pearson(cds["temp"], om["temp"]),
            "temp_bias": float(np.nanmean(om["temp"] - cds["temp"])),
            "temp_mae": float(np.nanmean(np.abs(om["temp"] - cds["temp"]))),
            "speed_r": pearson(cds["speed"], om["speed"]),
            "speed_bias": float(np.nanmean(om["speed"] - cds["speed"])),
            "speed_mae": float(np.nanmean(np.abs(om["speed"] - cds["speed"]))),
            "dir_mae": dir_mae_s,
            "precip_cds_mm": cds["precip_7d"],
            "precip_om_mm": om["precip_7d"],
        }
        per_sample.append(rec)

        pool["temp"]["cds"].append(cds["temp"]);  pool["temp"]["om"].append(om["temp"])
        pool["speed"]["cds"].append(cds["speed"]); pool["speed"]["om"].append(om["speed"])
        if np.isfinite(dir_mae_s):
            dir_diffs.append(np.abs(d[strong]))
        prec_cds.append(cds["precip_7d"]); prec_om.append(om["precip_7d"])

    if not per_sample:
        print("⛔ هیچ نمونهٔ مشترکی پیدا نشد.")
        for s in skipped[:10]:
            print("   ", s)
        return

    n = len(per_sample)
    print(f"نمونهٔ مشترک و قابل مقایسه: {n}   رد شده: {len(skipped)}")
    if skipped:
        for s in skipped[:5]:
            print(f"   رد: {s}")

    # --- آمار روی همهٔ ساعت‌های همهٔ نمونه‌ها یک‌جا (pooled) ---
    T_c = np.concatenate(pool["temp"]["cds"]); T_o = np.concatenate(pool["temp"]["om"])
    S_c = np.concatenate(pool["speed"]["cds"]); S_o = np.concatenate(pool["speed"]["om"])
    D = np.concatenate(dir_diffs) if dir_diffs else np.array([np.nan])
    P_c = np.array(prec_cds); P_o = np.array(prec_om)

    summary = {
        "n_samples": n,
        "n_hours_total": int(T_c.size),
        "temp_r": pearson(T_c, T_o),
        "temp_bias": float(np.nanmean(T_o - T_c)),
        "temp_mae": float(np.nanmean(np.abs(T_o - T_c))),
        "temp_rmse": float(np.sqrt(np.nanmean((T_o - T_c) ** 2))),
        "speed_r": pearson(S_c, S_o),
        "speed_bias": float(np.nanmean(S_o - S_c)),
        "speed_mae": float(np.nanmean(np.abs(S_o - S_c))),
        "speed_rmse": float(np.sqrt(np.nanmean((S_o - S_c) ** 2))),
        "dir_mae": float(np.nanmean(D)),
        "dir_median_abs": float(np.nanmedian(D)),
        "precip_r": pearson(P_c, P_o),
        "precip_bias_mm": float(np.nanmean(P_o - P_c)),
        "precip_mean_cds_mm": float(np.nanmean(P_c)),
        "precip_mean_om_mm": float(np.nanmean(P_o)),
    }

    print("\n" + "=" * 74)
    print(f"مقایسهٔ ساعت‌به‌ساعت — {n} نمونه × {N_HOURS} ساعت = {T_c.size:,} نقطه")
    print("=" * 74)
    print(f"{'متغیر':<22}{'r':>8}{'اریبی':>12}{'MAE':>10}{'RMSE':>10}")
    print("-" * 74)
    print(f"{'دما (°C)':<22}{summary['temp_r']:>8.4f}{summary['temp_bias']:>12.3f}"
          f"{summary['temp_mae']:>10.3f}{summary['temp_rmse']:>10.3f}")
    print(f"{'سرعت باد (m/s)':<22}{summary['speed_r']:>8.4f}{summary['speed_bias']:>12.3f}"
          f"{summary['speed_mae']:>10.3f}{summary['speed_rmse']:>10.3f}")
    print(f"\nجهت باد (فقط ساعت‌هایی که هر دو > ۱ m/s):")
    print(f"   میانگین خطای زاویه‌ای: {summary['dir_mae']:.2f}°"
          f"     میانه: {summary['dir_median_abs']:.2f}°")
    print(f"\nبارش، جمع ۷ روزه (mm):")
    print(f"   r={summary['precip_r']:.4f}   CDS میانگین={summary['precip_mean_cds_mm']:.2f}"
          f"   OM میانگین={summary['precip_mean_om_mm']:.2f}"
          f"   اریبی={summary['precip_bias_mm']:+.2f}")

    # --- حکم بر اساس آستانه‌های از پیش تعیین‌شده ---
    print("\n" + "=" * 74)
    print("حکم — آستانه‌ها پیش از دیدن داده تعیین شده بودند")
    print("=" * 74)
    verdict = {}
    for key, (label, thr, kind) in GATES.items():
        val = summary[key]
        if kind == "min":
            ok = np.isfinite(val) and val >= thr
            txt = f"≥ {thr}"
        elif kind == "max":
            ok = np.isfinite(val) and val <= thr
            txt = f"≤ {thr}"
        else:
            ok = np.isfinite(val) and abs(val) <= thr
            txt = f"|·| ≤ {thr}"
        verdict[key] = bool(ok)
        print(f"  {'✅' if ok else '❌'} {label:<28} {val:>9.4f}   (شرط {txt})")

    all_ok = all(verdict.values())
    summary["gates"] = verdict
    summary["passed"] = all_ok

    print("\n" + ("🟢 قبول — Open-Meteo همان ERA5 است. با خیال راحت جلو برو."
                  if all_ok else
                  "🔴 مردود — پیش از GPU باید بفهمیم کدام متغیر و چرا."))
    if not all_ok:
        print("   قبل از هر کار دیگر، ستون‌های ❌ بالا را در validation_openmeteo.csv بررسی کن.")

    # --- ذخیره ---
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_sample[0].keys()))
        w.writeheader()
        w.writerows(per_sample)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # بدترین نمونه‌ها — اگر چیزی خراب باشد، از اینجا پیدا می‌شود
    worst = sorted([r for r in per_sample if np.isfinite(r["speed_r"])],
                   key=lambda r: r["speed_r"])[:5]
    print("\nپنج نمونه با کمترین همبستگی سرعت باد:")
    for r in worst:
        print(f"   #{r['sample_id']:<5} {r['date']}  speed_r={r['speed_r']:+.3f}  "
              f"temp_r={r['temp_r']:+.3f}  dir_mae={r['dir_mae']:.1f}°")

    print(f"\n💾 {OUT_CSV}")
    print(f"💾 {OUT_JSON}")


if __name__ == "__main__":
    main()

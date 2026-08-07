# -*- coding: utf-8 -*-
"""
06b_check_tp_convention.py — بارش ERA5-Land را چطور باید جمع زد؟
=================================================================
نسخه: v1 · تاریخ: 2026-07-29

مسئله (کشف‌شده در 06_validate_openmeteo.py):
    بارش ۷ روزه — CDS میانگین 4.35 mm در برابر Open-Meteo 2.22 mm، با r=0.936.
    الگو یکی است، بزرگی تقریباً دو برابر. یکی از دو طرف اشتباه می‌کند.

فرضیهٔ من:
    `tp` در ERA5-Land از ۰۰ UTC انباشته می‌شود، پس مقدار ساعت 00:00 روز N
    **کل بارش روز N-1** است. کدِ فعلی ما این است:
        daily = tp.groupby("valid_time.date").max()
    و گروهِ روز N شامل 00:00 (کل روز N-1) و 01:00..23:00 (۲۳ ساعت روز N) است.
    اگر روز قبل پربارش‌تر بوده، max آن را برمی‌دارد → دوباره‌شماری.

این فایل حدس را آزمایش می‌کند: سه روش جمع‌زدن را روی نمونه‌های واقعی
حساب می‌کند و هر سه را با Open-Meteo مقایسه می‌کند.

    روش A  «فعلی»  : groupby(date).max()  ← همانی که در 05 و 06 هست
    روش B  «تفاضل» : اختلاف ساعت‌به‌ساعت؛ هر جا منفی شد یعنی ریست ۰۰ UTC،
                     آن ساعت خودِ مقدار است. جمع همهٔ ۱۶۸ ساعت.
    روش C  «مرز روز»: مقدار ساعت 00:00 هر روز = کل روز قبل. هفت‌تا از این‌ها.

هیچ چیزی را تغییر نمی‌دهد. فقط عدد می‌دهد تا بعد با چشمِ باز تصمیم بگیریم.

اجرا:
    python 06b_check_tp_convention.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
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

DAYS_BEFORE = 7
N_HOURS = DAYS_BEFORE * 24
SHOW_RAW = 2          # چند نمونه سری خام چاپ شود


def tp_series(path, row):
    """سری زمانی tp (متر، انباشته) روی جعبهٔ نمونه، ۱۶۸ ساعت."""
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
            return None, None
        dims = [d for d in ("latitude", "longitude") if d in sub.dims]
        tp = sub["tp"].mean(dim=dims, skipna=True).values.astype(np.float64)
        times = sub["valid_time"].values
    return tp, times


def method_A(tp, times):
    """فعلی: بیشینهٔ هر روز تقویمی، جمع هفت روز."""
    days = times.astype("datetime64[D]")
    total = 0.0
    for d in np.unique(days):
        total += np.nanmax(tp[days == d])
    return total * 1000.0


def method_B(tp, times):
    """تفاضل ساعت‌به‌ساعت؛ افت منفی = ریست ۰۰ UTC."""
    d = np.empty_like(tp)
    d[0] = tp[0]                 # اولین ساعت: خودش انباشتهٔ همان ساعت است
    d[1:] = tp[1:] - tp[:-1]
    neg = d < 0                  # افت = شمارنده صفر شده → مقدار خودِ tp همان ساعت
    d[neg] = tp[neg]
    d = np.clip(d, 0.0, None)
    return float(np.nansum(d)) * 1000.0


def method_C(tp, times):
    """مقدار ساعت 00:00 هر روز = کل روز قبل."""
    hours = times.astype("datetime64[h]").astype(int) % 24
    return float(np.nansum(tp[hours == 0])) * 1000.0


def main():
    z = np.load(NPZ, allow_pickle=True)
    idx = {int(s): i for i, s in enumerate(z["sample_id"])}
    om_prec = z["precipitation"].astype(np.float64)

    rows = {int(r["sample_id"]): r
            for r in csv.DictReader(IN_CSV.open(encoding="utf-8"))}
    files = sorted(PERSAMPLE.glob("*.nc"))
    print(f"فایل مرجع: {len(files)}\n")

    A, B, C, OM, sids = [], [], [], [], []
    shown = 0

    for f in files:
        sid = int(f.name.split("__")[0])
        if sid not in idx or sid not in rows:
            continue
        tp, times = tp_series(f, rows[sid])
        if tp is None:
            continue

        if shown < SHOW_RAW and np.nanmax(tp) > 1e-5:
            print("=" * 78)
            print(f"سری خام tp نمونهٔ #{sid}  ({rows[sid]['date']})  — واحد: mm")
            print("ببین آیا در هر ۲۴ ساعت بالا می‌رود و بعد می‌افتد:")
            for k in range(0, 48):
                ts = str(times[k])[:16]
                print(f"   {ts}   {tp[k]*1000:8.4f}")
            print("   ... (۱۲۰ ساعت بعدی چاپ نشد)")
            print("=" * 78 + "\n")
            shown += 1

        A.append(method_A(tp, times))
        B.append(method_B(tp, times))
        C.append(method_C(tp, times))
        OM.append(float(np.nansum(om_prec[idx[sid]])))
        sids.append(sid)

    A, B, C, OM = map(np.array, (A, B, C, OM))

    def stat(x, name):
        m = np.isfinite(x) & np.isfinite(OM)
        r = np.corrcoef(x[m], OM[m])[0, 1] if m.sum() > 3 else np.nan
        ratio = np.nanmean(x[m]) / np.nanmean(OM[m]) if np.nanmean(OM[m]) > 0 else np.nan
        print(f"{name:<28}{np.nanmean(x):>10.3f}{r:>10.4f}{ratio:>10.3f}")

    print("=" * 78)
    print(f"مقایسه با Open-Meteo روی {len(OM)} نمونه   (میانگین OM = {np.nanmean(OM):.3f} mm)")
    print("=" * 78)
    print(f"{'روش':<28}{'میانگین':>10}{'r با OM':>10}{'نسبت':>10}")
    print("-" * 78)
    stat(A, "A — groupby(date).max() فعلی")
    stat(B, "B — تفاضل ساعت‌به‌ساعت")
    stat(C, "C — مقدار ساعت ۰۰:۰۰")
    print("=" * 78)
    print("روشی که «نسبت» نزدیک ۱.۰ بدهد، قرارداد درست است.")
    print("اگر A نسبتی حدود ۲ بدهد و B نزدیک ۱ → فرضیهٔ دوباره‌شماری تأیید")
    print("و باید 05_build_features.py اصلاح شود.")


if __name__ == "__main__":
    main()

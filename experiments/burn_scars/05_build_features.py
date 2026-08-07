"""
05_build_features.py — ساخت شش ویژگیِ سراسری برای هر نمونه
============================================================
نسخه: v1 · تاریخ: 2026-07-29

ورودی:  data/era5/monthly/era5land_YYYY-MM.nc   (خروجی 04b)
        data/meta/samples_split.csv
خروجی:  data/meta/wind_features.csv

منطق: برای هر نمونه، **۷ روزِ قبل** از تاریخ مشاهده را از جعبهٔ همان کاشی برمی‌دارد،
      اول روی مکان میانگین می‌گیرد (چون ادعای ما این است که این یک **کمیت سراسری**
      است)، بعد روی زمان آماره می‌گیرد.

شش ویژگی:
    mean_speed · max_speed · sin(dir) · cos(dir) · precip_7d · mean_temp

--- سه تلهٔ فنی که رعایت شده‌اند ---

۱. جهت باد دایره‌ای است. میانگینِ زاویه‌ها غلط است (۳۵۹ و ۱ کنار هم‌اند).
   درست: میانگین u و v را بگیر، بعد atan2. و خروجی sin/cos بده نه درجه —
   همان کاری که خودِ Prithvi با روزِ سال می‌کند.

۲. `tp` در ERA5-Land **تجمعی از ۰۰ UTC همان روز** است، نه بارش همان ساعت.
   جمع‌زدن ۲۴ ساعت یعنی چندبرابر شمردن. درست: بیشینهٔ هر روز = کل آن روز،
   بعد هفت روز را جمع کن.

۳. ERA5-Land فقط خشکی است → کاشی ساحلی NaN دارد (تا ۸٪ دیده شد).
   همه‌جا nanmean/nanmax، و درصد NaN به‌عنوان نشانهٔ کیفیت ثبت می‌شود.

--- نرمال‌سازی ---
میانگین و انحراف معیار **فقط از split=train** حساب می‌شود و روی هر سه اعمال.
اگر از کل دیتاست حساب شود، نشت اطلاعات است.

--- صحت‌سنجی رایگان ---
۱۶۸ فایل تک‌نمونه‌ای که با روش قدیمی گرفته شده بودند به‌عنوان **مرجع** استفاده
می‌شوند: اگر مقدار استخراج‌شده از فایل ماهانه با آن‌ها نخواند، یعنی برش غلط است.

اجرا:
    python 05_build_features.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import math
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

import numpy as np
import xarray as xr

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
META = BIG / "data" / "meta"
MONTHLY = BIG / "data" / "era5" / "monthly"
PERSAMPLE = BIG / "data" / "era5" / "samples"
IN_CSV = META / "samples_split.csv"
OUT_CSV = META / "wind_features.csv"

DAYS_BEFORE = 7
RAW = ["mean_speed", "max_speed", "dir_sin", "dir_cos", "precip_7d", "mean_temp"]


def spatial_series(ds, row):
    """جعبهٔ نمونه را برش می‌زند و روی مکان میانگین می‌گیرد → سری زمانی."""
    lat_s = slice(float(row["lat_max"]) + 0.15, float(row["lat_min"]) - 0.15)
    lon_s = slice(float(row["lon_min"]) - 0.15, float(row["lon_max"]) + 0.15)
    sub = ds.sel(latitude=lat_s, longitude=lon_s)
    if sub.sizes.get("latitude", 0) == 0 or sub.sizes.get("longitude", 0) == 0:
        # جعبه خیلی کوچک افتاد — نزدیک‌ترین خانه را بگیر
        sub = ds.sel(latitude=float(row["lat_center"]),
                     longitude=float(row["lon_center"]), method="nearest")
        sub = sub.expand_dims(["latitude", "longitude"])
    return sub


def features_from(sub, d0):
    """سری زمانی برش‌خورده → شش ویژگی + درصد NaN."""
    t0 = np.datetime64(str(d0 - timedelta(days=DAYS_BEFORE)))
    t1 = np.datetime64(str(d0))                      # خودِ روز مشاهده حذف
    sub = sub.sel(valid_time=slice(t0, t1 - np.timedelta64(1, "h")))
    if sub.sizes.get("valid_time", 0) == 0:
        return None

    dims = [d for d in ("latitude", "longitude") if d in sub.dims]
    u = sub["u10"].mean(dim=dims, skipna=True).values
    v = sub["v10"].mean(dim=dims, skipna=True).values
    t2 = sub["t2m"].mean(dim=dims, skipna=True).values
    tp = sub["tp"].mean(dim=dims, skipna=True)

    nan_pct = float(np.isnan(sub["u10"].values).mean() * 100)

    speed = np.sqrt(u ** 2 + v ** 2)
    if np.all(np.isnan(speed)):
        return None

    # جهت: از میانگین بردار، نه میانگین زاویه
    ubar, vbar = np.nanmean(u), np.nanmean(v)
    theta = math.atan2(float(vbar), float(ubar))

    # بارش: tp تجمعیِ روزانه است → بیشینهٔ هر روز = کل آن روز
    daily = tp.groupby("valid_time.date").max(skipna=True).values
    precip = float(np.nansum(daily))

    return {
        "mean_speed": float(np.nanmean(speed)),
        "max_speed": float(np.nanmax(speed)),
        "dir_sin": math.sin(theta),
        "dir_cos": math.cos(theta),
        "precip_7d": precip,
        "mean_temp": float(np.nanmean(t2)),
        "era5_nan_pct": round(nan_pct, 3),
        "n_timesteps": int(sub.sizes["valid_time"]),
    }


def main():
    rows = list(csv.DictReader(IN_CSV.open(encoding="utf-8")))
    print(f"{len(rows)} نمونه\n")

    # کدام ماه‌ها لازم‌اند
    need = defaultdict(list)
    for r in rows:
        d0 = date.fromisoformat(r["date"])
        ms = {((d0 - timedelta(days=k)).year, (d0 - timedelta(days=k)).month)
              for k in range(1, DAYS_BEFORE + 1)}
        for key in ms:
            need[key].append(r)

    missing = [k for k in need if not (MONTHLY / f"era5land_{k[0]}-{k[1]:02d}.nc").exists()]
    if missing:
        print(f"⛔ {len(missing)} ماه هنوز دانلود نشده: {sorted(missing)[:6]} …")
        print("   اول 04b_download_era5_monthly.py را تمام کن.")
        return

    # هر ماه یک بار باز می‌شود؛ نمونه‌های چندماهه بعداً ادغام می‌شوند
    parts = defaultdict(list)
    for i, key in enumerate(sorted(need), 1):
        f = MONTHLY / f"era5land_{key[0]}-{key[1]:02d}.nc"
        print(f"[{i}/{len(need)}] {f.name}  ({len(need[key])} نمونه)")
        with xr.open_dataset(f) as ds:
            for r in need[key]:
                parts[r["sample_id"]].append(spatial_series(ds, r).load())

    out = []
    problems = []
    for r in rows:
        ps = parts.get(r["sample_id"], [])
        if not ps:
            problems.append(f"{r['sample_id']}: دادهٔ ماهانه نداشت")
            continue
        sub = ps[0] if len(ps) == 1 else xr.concat(ps, dim="valid_time").sortby("valid_time")
        sub = sub.drop_duplicates("valid_time")
        fe = features_from(sub, date.fromisoformat(r["date"]))
        if fe is None:
            problems.append(f"{r['sample_id']}: پنجرهٔ خالی یا همه NaN")
            continue
        out.append({**r, **fe})

    if not out:
        print("⛔ هیچ ویژگی‌ای ساخته نشد")
        return

    # --- نرمال‌سازی فقط از train ---
    tr = [o for o in out if o["split"] == "train"]
    stats = {}
    for k in RAW:
        a = np.array([o[k] for o in tr], dtype="float64")
        mu, sd = float(a.mean()), float(a.std())
        stats[k] = (mu, sd if sd > 1e-9 else 1.0)
    for o in out:
        for k in RAW:
            mu, sd = stats[k]
            o[k + "_z"] = round((o[k] - mu) / sd, 6)
            o[k] = round(o[k], 6)

    fields = list(out[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    print("\n" + "=" * 62)
    print(f"✅ {len(out)} سطر → {OUT_CSV}")
    if problems:
        print(f"⚠️ {len(problems)} مشکل:")
        for p in problems[:8]:
            print("   ", p)

    print(f"\n{'ویژگی':<14}{'کمینه':>10}{'میانه':>10}{'میانگین':>10}{'بیشینه':>10}{'انحراف':>10}")
    for k in RAW:
        a = np.array([o[k] for o in out])
        print(f"{k:<14}{a.min():>10.3f}{np.median(a):>10.3f}{a.mean():>10.3f}{a.max():>10.3f}{a.std():>10.3f}")

    nz = np.array([o["era5_nan_pct"] for o in out])
    nt = np.array([o["n_timesteps"] for o in out])
    print(f"\nدرصد NaN — میانگین {nz.mean():.2f} · بیشینه {nz.max():.2f} · "
          f"نمونه با >۵۰٪: {(nz>50).sum()}")
    print(f"گام زمانی — کمینه {nt.min()} · بیشینه {nt.max()}  (انتظار: ۱۶۸)")
    if nt.min() != 168 or nt.max() != 168:
        print("   ⚠️ همه ۱۶۸ نیستند — پنجرهٔ ناقص. بررسی شود.")

    # --- صحت‌سنجی با فایل‌های تک‌نمونه‌ای قدیمی ---
    refs = sorted(PERSAMPLE.glob("*.nc"))
    if refs:
        print(f"\n--- صحت‌سنجی با {len(refs)} فایل مرجع ---")
        by_id = {o["sample_id"]: o for o in out}
        diffs = []
        for rp in refs[:40]:
            sid = str(int(rp.name.split("__")[0]))
            o = by_id.get(sid)
            if not o:
                continue
            row = next((x for x in rows if x["sample_id"] == sid), None)
            with xr.open_dataset(rp) as ds:
                fe = features_from(spatial_series(ds, row).load(),
                                   date.fromisoformat(row["date"]))
            if fe:
                diffs.append(abs(fe["mean_speed"] - o["mean_speed"]))
        if diffs:
            d = np.array(diffs)
            print(f"اختلاف mean_speed: میانگین {d.mean():.5f} · بیشینه {d.max():.5f} m/s")
            print("✅ برش درست است" if d.max() < 0.01 else "⛔ اختلاف معنادار — برش را بررسی کن")


if __name__ == "__main__":
    main()

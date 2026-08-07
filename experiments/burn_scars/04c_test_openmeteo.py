# -*- coding: utf-8 -*-
"""
04c_test_openmeteo.py  —  انتخاب مدل درست در Open-Meteo Archive API

یافتهٔ اجرای اول (2026-07-29):
    models=era5_land  →  temperature_2m ✅  ولی باد و بارش همه None.
    یعنی پیادهٔ Open-Meteo از ERA5-Land فقط بخشی از متغیرها را دارد.

این نسخه حدس نمی‌زند: هر چهار حالت را روی یک نقطه و یک بازه می‌زند و
جدول می‌دهد که کدام مدل کدام متغیر را واقعاً برمی‌گرداند.
هیچ چیزی ذخیره نمی‌شود.
"""
import sys, time, json, urllib.request, urllib.parse

sys.stdout.reconfigure(encoding="utf-8")

BASE  = "https://archive-api.open-meteo.com/v1/archive"
VARS  = ["temperature_2m", "wind_speed_10m", "wind_direction_10m", "precipitation"]
# None یعنی پارامتر models را اصلاً نفرست (پیش‌فرض سرور)
MODELS = [None, "era5_seamless", "era5", "era5_land"]

LAT, LON   = 40.0, -120.0          # داخل CONUS، روی خشکی
START, END = "2020-08-01", "2020-08-02"


def fetch(model):
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": START, "end_date": END,
        "hourly": ",".join(VARS), "timezone": "UTC",
    }
    if model:
        params["models"] = model
    url = f"{BASE}?{urllib.parse.urlencode(params)}"
    t0 = time.time()
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return json.loads(r.read().decode("utf-8")), time.time() - t0, None
    except Exception as e:
        body = ""
        if hasattr(e, "read"):
            try:
                body = e.read().decode("utf-8")[:200]
            except Exception:
                pass
        return None, time.time() - t0, f"{e} {body}"


rows = []
for m in MODELS:
    tag = m or "(default)"
    p, dt, err = fetch(m)
    if err:
        print(f"[X] {tag:16s}  خطا: {err}")
        rows.append((tag, {v: "ERR" for v in VARS}, dt))
        continue
    h = p["hourly"]
    cell = {}
    for v in VARS:
        good = [x for x in h[v] if x is not None]
        cell[v] = f"{len(good):3d}/{len(h[v]):3d}" if good else "None"
    rows.append((tag, cell, dt))
    print(f"[OK] {tag:16s} {dt:.2f}s   "
          + "   ".join(f"{v.split('_')[0][:5]}={cell[v]}" for v in VARS))

print()
print("=" * 90)
print(f"{'model':<18}" + "".join(f"{v:>18}" for v in VARS))
print("-" * 90)
for tag, cell, dt in rows:
    print(f"{tag:<18}" + "".join(f"{cell[v]:>18}" for v in VARS))
print("=" * 90)
print("دنبال ردیفی بگرد که هر چهار ستونش عدد دارد (مثل 48/ 48). همان مدل را برمی‌داریم.")

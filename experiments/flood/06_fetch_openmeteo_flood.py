# -*- coding: utf-8 -*-
"""
06_fetch_openmeteo_flood.py — سری‌زمانی ERA5 برای ۴۴۶ نمونهٔ سیل
=====================================================================
نسخه: v1 · تاریخ: 2026-07-30 · کار `flood`

🔁 **کپی و وصله** از `../lab/src/04c_fetch_era5_openmeteo.py` (نسخهٔ 2026-07-29).
   منطق، تله‌ها و RLock عیناً حفظ شده‌اند. چهار تغییر عمدی:

   ۱. مسیرها → `injection-routing-flood`
   ۲. `DAYS_BEFORE = 30` نه ۷ → ۷۲۰ ساعت. **دلیل:** پنجرهٔ اصلی همان ۷ روز
      آتش‌سوزی می‌ماند تا دو تسک قابل مقایسه بمانند، ولی حوضه‌های بزرگ (مکونگ،
      پاراگوئه، هند) تأخیر رواناب بلندتری دارند. ۳۰ روز را می‌گیریم تا در گام ۸
      **هم** `precip_7d` **و هم** `precip_30d` مشتق شود؛ ۳۰ روزه فقط ستون
      **تشخیصی** است و **بیرون از بردار ۱۰ بعدی قفل‌شده** می‌ماند.
      هزینه‌اش صفر است چون Open-Meteo صف ندارد.
   ۳. `sample_id` سیل ممکن است تکراری نباشد ولی **تاریخ به ازای رویداد** است، پس
      فقط ۱۱ پنجرهٔ زمانی متمایز داریم → دسته‌بندی خیلی کاراتر از آتش‌سوزی
   ۴. اعتبارسنجی در برابر CDS **حذف شد** — برای سیل فایل مرجع CDS نداریم و
      خودِ Open-Meteo قبلاً روی آتش‌سوزی تأیید شده (دما r=۰.۹۸۴ · باد r=۰.۹۱۸)

⚠️ آنچه **عوض نشد و نباید عوض شود:** `models=era5_seamless` · `wind_speed_unit=ms`
   · `RLock` نه `Lock` · کَش `.jsonl` قابل‌ازسرگیری · عقب‌نشینی پلکانی.

چرا این جایگزین 04 و 04b شد:
    CDS: صف سروری. هر درخواست تا ۲ ساعت. بعد از ۶ ساعت فقط ۵ ماه از ۴۷ ماه.
    اینجا: ۰.۳ ثانیه برای ۱۶۸ ساعت × ۳ نقطه. بدون کلید، بدون ثبت‌نام، بدون صف.

انتخاب مدل (اندازه‌گیری‌شده در 04c_test_openmeteo.py، 2026-07-29):
    era5_land      → temperature_2m ✅ ولی باد و بارش **همه None**
    era5_seamless  → هر چهار متغیر ✅   ← این را برداشتیم
    (default)      → هر چهار متغیر ✅ ولی «پیش‌فرض» تضمین پایداری ندارد،
                     پس صریح می‌نویسیم تا نتیجه بازتولیدپذیر بماند.

    ⚠️ یادداشت: era5_seamless دما را از ERA5-Land (0.1°) و باد را از
       ERA5 (0.25°) می‌دهد. کاشی ما 512×512×30m ≈ 15 km است، پس باد در
       یک سلول ERA5 می‌افتد. این دقیقاً همان رژیم «اسکالر سراسری» است که
       فرضیهٔ ما درباره‌اش حرف می‌زند — نه یک سازش، بلکه شرط آزمایش.

پنجرهٔ زمانی: **دقیقاً همان 04** — هفت روز *پیش از* تاریخ تصویر،
    یعنی [d0-7 , d0-1] شامل هر دو سر → ۱۶۸ ساعت.

واحدها (هم‌راستا با ERA5 خام تا مقایسه با ۱۶۸ فایل CDS ساده باشد):
    دما            °C      (CDS: کلوین → منهای 273.15)
    سرعت باد       m/s     (wind_speed_unit=ms)
    جهت باد        درجه، از شمال، جهتی که باد **از آن** می‌آید
    بارش           mm      (CDS: متر → ضرب در 1000)

خروجی:
    <BIG>/data/era5/openmeteo_raw.jsonl   کَش خام، یک خط به ازای هر نمونه
    <BIG>/data/era5/openmeteo_series.npz  آرایه‌های (804, 168)
    <BIG>/data/era5/openmeteo_manifest.json  فراداده و شمارش

اجرا:
    python 04c_fetch_era5_openmeteo.py
    python 04c_fetch_era5_openmeteo.py --workers 4 --limit 20   (تست کوچک)

قابل ازسرگیری: هر بار اجرا شود، نمونه‌های موجود در jsonl را رد می‌کند.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
import time
import argparse
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing-flood"
IN_CSV = BIG / "data" / "meta" / "samples_split.csv"
OUT_DIR = BIG / "data" / "era5"
RAW_JSONL = OUT_DIR / "openmeteo_raw.jsonl"
OUT_NPZ = OUT_DIR / "openmeteo_series.npz"
MANIFEST = OUT_DIR / "openmeteo_manifest.json"

BASE = "https://archive-api.open-meteo.com/v1/archive"
MODEL = "era5_seamless"
VARS = ["temperature_2m", "wind_speed_10m", "wind_direction_10m", "precipitation"]

DAYS_BEFORE = 30         # ⬅️ تغییر عمدی نسبت به 04c (که ۷ بود) — سربرگ را بخوان
N_HOURS = DAYS_BEFORE * 24   # 720
PRIMARY_DAYS = 7         # پنجرهٔ اصلی، همان آتش‌سوزی. ۳۰ روز فقط تشخیصی است
BATCH = 40               # حداکثر نقطه در یک درخواست
MAX_TRIES = 5
BACKOFF = [2, 5, 15, 40]

# ⚠️ RLock نه Lock. درس گرانِ 04_download_era5.py (2026-07-29):
#    log() خودش این قفل را می‌گیرد و از داخل بلاکِ `with _lock` هم صدا زده می‌شود.
#    با Lock ساده این **بن‌بست** است — فایل روی دیسک نوشته می‌شود، فرآیند زنده
#    می‌ماند، CPU نزدیک صفر، و خط لوله برای همیشه می‌ایستد. آن باگ ۸ ساعت برد.
_lock = threading.RLock()
_done = 0
_total = 0


def log(msg):
    with _lock:
        print(msg, flush=True)


def window(row):
    """هفت روز پیش از تاریخ تصویر: [d0-7 , d0-1] شامل هر دو سر."""
    d0 = date.fromisoformat(row["date"])
    return (d0 - timedelta(days=DAYS_BEFORE)).isoformat(), \
           (d0 - timedelta(days=1)).isoformat()


def build_url(lats, lons, start, end):
    q = urllib.parse.urlencode({
        "latitude": ",".join(f"{x:.4f}" for x in lats),
        "longitude": ",".join(f"{x:.4f}" for x in lons),
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(VARS),
        "models": MODEL,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    })
    return f"{BASE}?{q}"


def fetch_batch(rows):
    """یک دسته نمونه که همگی پنجرهٔ زمانی یکسان دارند."""
    start, end = window(rows[0])
    lats = [float(r["lat_center"]) for r in rows]
    lons = [float(r["lon_center"]) for r in rows]
    url = build_url(lats, lons, start, end)

    last = None
    for attempt in range(MAX_TRIES):
        try:
            with urllib.request.urlopen(url, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")[:160]
            except Exception:
                pass
            last = f"HTTP {e.code} {body}"
            # 429 = سقف نرخ. صبر کن. بقیهٔ 4xx یعنی درخواست غلط، تکرار بی‌فایده.
            if e.code not in (429, 500, 502, 503, 504):
                break
        except Exception as e:
            last = repr(e)
        if attempt < MAX_TRIES - 1:
            time.sleep(BACKOFF[min(attempt, len(BACKOFF) - 1)])
    else:
        return [], last
    if last and "payload" not in dir():
        return [], last

    blocks = payload if isinstance(payload, list) else [payload]
    if len(blocks) != len(rows):
        return [], f"تعداد بلاک {len(blocks)} ≠ تعداد نمونه {len(rows)}"

    out = []
    for r, b in zip(rows, blocks):
        h = b["hourly"]
        rec = {
            "sample_id": int(r["sample_id"]),
            "date": r["date"],
            "start": start,
            "end": end,
            "lat_req": float(r["lat_center"]),
            "lon_req": float(r["lon_center"]),
            "lat_grid": b.get("latitude"),
            "lon_grid": b.get("longitude"),
            "n_hours": len(h["time"]),
            "t0": h["time"][0],
            "t_last": h["time"][-1],
        }
        for v in VARS:
            rec[v] = h[v]
        out.append(rec)
    return out, None


def load_rows(limit=None):
    with IN_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["sample_id"]))
    return rows[:limit] if limit else rows


def load_cache():
    """sample_idهایی که قبلاً گرفته شده‌اند."""
    if not RAW_JSONL.exists():
        return set()
    seen = set()
    with RAW_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(int(json.loads(line)["sample_id"]))
            except Exception:
                pass
    return seen


def make_batches(rows):
    """گروه‌بندی بر اساس پنجرهٔ زمانی یکسان، سپس تکه‌تکه کردن به اندازهٔ BATCH."""
    groups = {}
    for r in rows:
        groups.setdefault(window(r), []).append(r)
    batches = []
    for _, g in sorted(groups.items()):
        for i in range(0, len(g), BATCH):
            batches.append(g[i:i + BATCH])
    return batches


def build_npz():
    """از کَش jsonl آرایه‌های (N, 168) می‌سازد و اعتبارسنجی می‌کند."""
    recs = {}
    with RAW_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs[int(r["sample_id"])] = r      # آخرین نسخه برنده است

    ids = sorted(recs)
    n = len(ids)
    arrs = {v: np.full((n, N_HOURS), np.nan, dtype=np.float32) for v in VARS}
    bad_len, nan_counts = [], {v: 0 for v in VARS}

    for i, sid in enumerate(ids):
        r = recs[sid]
        if r["n_hours"] != N_HOURS:
            bad_len.append((sid, r["n_hours"]))
        for v in VARS:
            col = r[v][:N_HOURS]
            a = np.array([np.nan if x is None else x for x in col], dtype=np.float32)
            arrs[v][i, :len(a)] = a
            nan_counts[v] += int(np.isnan(arrs[v][i]).sum())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        sample_id=np.array(ids, dtype=np.int32),
        date=np.array([recs[s]["date"] for s in ids]),
        start=np.array([recs[s]["start"] for s in ids]),
        lat_grid=np.array([recs[s]["lat_grid"] for s in ids], dtype=np.float32),
        lon_grid=np.array([recs[s]["lon_grid"] for s in ids], dtype=np.float32),
        **arrs,
    )
    return n, bad_len, nan_counts, arrs, ids


def main():
    global _done, _total
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="فقط N نمونهٔ اول (تست)")
    ap.add_argument("--rebuild", action="store_true", help="فقط npz را از کَش بساز")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.rebuild:
        if not IN_CSV.exists():
            print(f"⛔ {IN_CSV} نیست — اول 05_make_split_flood.py را اجرا کن")
            return
        rows = load_rows(args.limit)
        seen = load_cache()
        todo = [r for r in rows if int(r["sample_id"]) not in seen]
        print(f"نمونهٔ کل: {len(rows)}   از قبل گرفته‌شده: {len(rows)-len(todo)}   مانده: {len(todo)}")

        if todo:
            batches = make_batches(todo)
            _total = len(batches)
            print(f"دسته‌ها: {_total}   (حداکثر {BATCH} نقطه در هر درخواست)")
            print(f"مدل: {MODEL}   پنجره: {DAYS_BEFORE} روز پیش از تصویر → {N_HOURS} ساعت")
            print(f"کارگر هم‌زمان: {args.workers}\n")

            t0 = time.time()
            fh = RAW_JSONL.open("a", encoding="utf-8")
            errors = []
            try:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futs = {ex.submit(fetch_batch, b): b for b in batches}
                    for fut in as_completed(futs):
                        recs, err = fut.result()
                        b = futs[fut]
                        with _lock:
                            _done += 1
                            if err:
                                errors.append((b[0]["date"], len(b), err))
                                log(f"[{_done}/{_total}] ❌ {b[0]['date']} ({len(b)} نمونه): {err}")
                            else:
                                for rec in recs:
                                    fh.write(json.dumps(rec) + "\n")
                                fh.flush()
                                if _done % 20 == 0 or _done == _total:
                                    el = time.time() - t0
                                    rate = _done / max(el, 1e-9)
                                    eta = (_total - _done) / max(rate, 1e-9)
                                    log(f"[{_done}/{_total}] ✅ {el:6.1f}s گذشته · تخمین باقی‌مانده {eta:6.1f}s")
            finally:
                fh.close()

            print(f"\nدانلود تمام شد در {time.time()-t0:.1f} ثانیه   خطا: {len(errors)}")
            for d, k, e in errors[:10]:
                print(f"   {d}  ({k} نمونه)  {e}")

    # ---------- ساخت آرایه‌ها + اعتبارسنجی ----------
    if not RAW_JSONL.exists():
        print("⛔ کَشی وجود ندارد.")
        return

    print("\n" + "=" * 72)
    print("ساخت آرایه‌ها و بررسی سلامت")
    print("=" * 72)
    n, bad_len, nan_counts, arrs, ids = build_npz()

    print(f"نمونه در فایل: {n}")
    if bad_len:
        print(f"⚠️ {len(bad_len)} نمونه طول ≠ {N_HOURS} دارد: {bad_len[:5]}")
    else:
        print(f"✅ همهٔ نمونه‌ها دقیقاً {N_HOURS} ساعت دارند")

    total_cells = n * N_HOURS
    print("\nمقدار گمشده و بازهٔ هر متغیر:")
    for v in VARS:
        a = arrs[v]
        ok = a[~np.isnan(a)]
        pct = 100.0 * nan_counts[v] / max(total_cells, 1)
        if ok.size == 0:
            print(f"  {v:22s} ❌ همه NaN")
            continue
        print(f"  {v:22s} NaN={pct:5.2f}%   min={ok.min():8.2f}  "
              f"mean={ok.mean():8.2f}  max={ok.max():8.2f}")

    # بررسی عقل سلیم — اگر اینها رد شوند، داده مشکوک است
    print("\nآزمون عقل سلیم:")
    checks = [
        ("دما بین -60 و +60 °C", arrs["temperature_2m"], -60, 60),
        ("سرعت باد بین 0 و 60 m/s", arrs["wind_speed_10m"], 0, 60),
        ("جهت باد بین 0 و 360 °", arrs["wind_direction_10m"], 0, 360),
        ("بارش بین 0 و 200 mm/h", arrs["precipitation"], 0, 200),
    ]
    for label, a, lo, hi in checks:
        ok = a[~np.isnan(a)]
        bad = int(((ok < lo) | (ok > hi)).sum()) if ok.size else -1
        mark = "✅" if bad == 0 else "❌"
        print(f"  {mark} {label:32s} خارج از بازه: {bad}")

    MANIFEST.write_text(json.dumps({
        "source": "Open-Meteo Archive API",
        "endpoint": BASE,
        "model": MODEL,
        "note": "era5_seamless = ERA5-Land 0.1° for temperature, ERA5 0.25° for wind/precip",
        "variables": VARS,
        "units": {"temperature_2m": "degC", "wind_speed_10m": "m/s",
                  "wind_direction_10m": "deg_from_north", "precipitation": "mm"},
        "window": f"[date-{DAYS_BEFORE}, date-1] inclusive",
        "n_hours": N_HOURS,
        "primary_window_days": PRIMARY_DAYS,
        "primary_window_note": ("پنجرهٔ قفل‌شدهٔ بردار ۱۰ بعدی = ۷ روز، همان تسک "
                                "آتش‌سوزی. ۳۰ روز فقط ستون تشخیصی است."),
        "n_samples": n,
        "api_key_required": False,
        "validated_against_cds": False,
        "validation_note": ("برای سیل مرجع CDS نداریم؛ همین مسیر روی آتش‌سوزی "
                            "تأیید شد: دما r=0.984 · باد r=0.918 · بارش r=0.936"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n💾 {OUT_NPZ}")
    print(f"💾 {MANIFEST}")
    print("\nقدم بعد: 07_build_features_flood.py — شش ویژگی + ستون تشخیصی ۳۰ روزه")


if __name__ == "__main__":
    main()

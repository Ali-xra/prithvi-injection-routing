"""
04b_download_era5_monthly.py — آرشیو ماهانهٔ ERA5-Land روی کل آمریکا
=====================================================================
نسخه: v1 · تاریخ: 2026-07-29
جایگزین `04_download_era5.py` (که ۸۰۴ درخواست کوچک می‌فرستاد)

چرا عوض شد — تشخیص واقعی 2026-07-29:
    نرخ دانلود از ۵۰ در ساعت به **۳ در ساعت** سقوط کرد. صفحهٔ وضعیت CDS نشان داد
    ۶٬۵۸۵ درخواست و ۱٬۸۹۰ کاربر در صف‌اند. **گلوگاه نوبتِ در صف است، نه حجم.**
    ۸۰۴ درخواست کوچک = ۸۰۴ بار ایستادن ته صف.
    ۴۷ درخواست ماهانه = ۴۷ بار. همان داده، ۱۷ برابر کمتر انتظار.

چرا کل آمریکا و همهٔ روزها، نه فقط آنچه امروز لازم است:
    فضا و پهنای باند محدودیت نیستند (تصمیم علی). با گرفتن آرشیو کامل:
      · اگر پنجره از ۷ روز به ۱۴ روز عوض شد → دیگر به CDS برنمی‌گردیم
      · اگر نمونه اضافه شد → همان‌جا هست
      · یک کلاس باگ حذف می‌شود: جعبهٔ ماه نمی‌تواند با نمونه‌هایش ناجور باشد
    برآورد: ~۶۶ گیگابایت در ۴۷ فایل.

خروجی:
    data/era5/monthly/era5land_YYYY-MM.nc
    هر فایل: همهٔ روزهای آن ماه × ۲۴ ساعت × ۴ متغیر × کل شبکهٔ CONUS

⚠️ درس باگ دیشب: `_print_lock` حتماً **RLock** باشد. `log()` از داخل بلوکی که
   همان قفل را گرفته صدا زده می‌شود؛ با `Lock` ساده بن‌بست می‌شود و کل خط لوله
   بعد از اولین موفقیت می‌ایستد.

🔴 **قبل از Ctrl+C حتماً بخوان:**
    بستن اسکریپت، درخواست‌های سمت CDS را **پاک نمی‌کند**. آن‌ها یتیم می‌شوند و
    جای صف را اشغال می‌کنند، و بعد هر درخواست جدید `rejected` می‌خورد بدون اینکه
    علتش معلوم باشد.
    → بعد از هر Ctrl+C: برو https://cds.climate.copernicus.eu/requests
      و تب‌های **Accepted** و **Running** را انتخاب و **Delete selected** کن.
      تب `Successful` را دست نزن.

ℹ️ سقف هم‌زمانی CDS برای این دیتاست حدود **۴ تا ۶** است و با بار سیستم فرق می‌کند.
    بیشترکردن کارگر سرعت را زیاد **نمی‌کند** — فقط `↻` بیشتری در لاگ می‌بینی.
    توان عبور را CDS تعیین می‌کند، نه ما.

اجرا:
    python 04b_download_era5_monthly.py               # همه
    python 04b_download_era5_monthly.py --limit 2     # آزمایش
    python 04b_download_era5_monthly.py --workers 3
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import time
import argparse
import threading
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cdsapi

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
IN_CSV = BIG / "data" / "meta" / "samples_split.csv"
OUT_DIR = BIG / "data" / "era5" / "monthly"
LOG_CSV = BIG / "data" / "era5" / "_monthly_log.csv"

DAYS_BEFORE = 7
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "total_precipitation",
]
HOURS = [f"{h:02d}:00" for h in range(24)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
MARGIN_DEG = 0.5

# ⚠️ RLock، نه Lock — وگرنه بن‌بست (باگ 2026-07-29)
_print_lock = threading.RLock()
_submit_lock = threading.Lock()
_last_submit = [0.0]
_done = 0

SUBMIT_GAP = 5.0
# سقف صف CDS حدود ۶ است. با ۶ کارگر تقریباً هیچ rejection نمی‌گیریم، ولی اگر
# گرفتیم نباید هرگز به «شکست دائم» تبدیل شود — فقط باید صبر کند تا جا باز شود.
MAX_TRIES = 10
BACKOFF = [30, 60, 120, 240, 420, 600, 900, 900, 900]


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def throttled_submit():
    with _submit_lock:
        wait = SUBMIT_GAP - (time.monotonic() - _last_submit[0])
        if wait > 0:
            time.sleep(wait)
        _last_submit[0] = time.monotonic()


def plan():
    """کدام ماه‌ها لازم است، و جعبهٔ کل چیست."""
    rows = list(csv.DictReader(IN_CSV.open(encoding="utf-8")))
    months = set()
    N = W = S = E = None
    for r in rows:
        d0 = date.fromisoformat(r["date"])
        for k in range(1, DAYS_BEFORE + 1):
            d = d0 - timedelta(days=k)
            months.add((d.year, d.month))
        a, b = float(r["lat_min"]), float(r["lat_max"])
        c, e = float(r["lon_min"]), float(r["lon_max"])
        N = b if N is None else max(N, b)
        S = a if S is None else min(S, a)
        W = c if W is None else min(W, c)
        E = e if E is None else max(E, e)
    area = [round(N + MARGIN_DEG, 2), round(W - MARGIN_DEG, 2),
            round(S - MARGIN_DEG, 2), round(E + MARGIN_DEG, 2)]
    return sorted(months), area, len(rows)


def out_path(y, m):
    return OUT_DIR / f"era5land_{y}-{m:02d}.nc"


def fetch(y, m, area, total):
    global _done
    p = out_path(y, m)
    if p.exists() and p.stat().st_size > 1_000_000:
        with _print_lock:
            _done += 1
            log(f"  [{_done}/{total}] ⏭  {p.name} از قبل بود ({p.stat().st_size/1e9:.2f} GB)")
        return {"month": f"{y}-{m:02d}", "status": "skipped", "note": "exists"}

    req = {
        "variable": VARIABLES,
        "year": [str(y)],
        "month": [f"{m:02d}"],
        "day": ALL_DAYS,
        "time": HOURS,
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    last_err = None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            throttled_submit()
            log(f"  → ارسال {y}-{m:02d}" + (f"  (تلاش {attempt})" if attempt > 1 else ""))
            t0 = time.monotonic()
            cdsapi.Client(quiet=True, progress=False).retrieve(
                "reanalysis-era5-land", req, str(p))
            gb = p.stat().st_size / 1e9
            mins = (time.monotonic() - t0) / 60
            with _print_lock:
                _done += 1
                log(f"  [{_done}/{total}] ✅ {p.name}  {gb:.2f} GB  ({mins:.0f} دقیقه)")
            return {"month": f"{y}-{m:02d}", "status": "ok",
                    "note": f"{gb:.2f}GB {mins:.0f}min tries={attempt}"}
        except Exception as e:
            last_err = e
            if p.exists():
                p.unlink()
            if attempt < MAX_TRIES:
                w = BACKOFF[attempt - 1]
                log(f"     ↻ {y}-{m:02d} تلاش {attempt} نشد ({type(e).__name__}) — {w}s صبر")
                time.sleep(w)
            else:
                with _print_lock:
                    _done += 1
                    log(f"  [{_done}/{total}] ⛔ {y}-{m:02d} بعد از {MAX_TRIES} تلاش: {e}")
    return {"month": f"{y}-{m:02d}", "status": "failed",
            "note": f"{type(last_err).__name__}: {last_err}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6,
                    help="سقف صف CDS حدود ۶ است — بیشتر یعنی rejection پشت سر هم")
    args = ap.parse_args()

    if not IN_CSV.exists():
        print(f"⛔ {IN_CSV} نیست — اول 03_make_split.py")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    months, area, n_samples = plan()
    if args.limit:
        months = months[: args.limit]

    have = sum(1 for y, m in months if out_path(y, m).exists())
    print(f"نمونه: {n_samples}   ماه لازم: {len(months)}   از قبل: {have}   مانده: {len(months)-have}")
    print(f"جعبه [N,W,S,E]: {area}")
    print(f"هر ماه: همهٔ روزها × ۲۴ ساعت × {len(VARIABLES)} متغیر")
    print(f"کارگر: {args.workers}   خروجی: {OUT_DIR}\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch, y, m, area, len(months)) for y, m in months]
        for fu in as_completed(futs):
            results.append(fu.result())

    ok = sum(1 for r in results if r["status"] == "ok")
    sk = sum(1 for r in results if r["status"] == "skipped")
    fa = sum(1 for r in results if r["status"] == "failed")
    with LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["month", "status", "note"])
        w.writeheader()
        w.writerows(sorted(results, key=lambda r: r["month"]))

    print("\n" + "=" * 62)
    print(f"موفق: {ok}   از قبل: {sk}   ناموفق: {fa}")
    print(f"لاگ: {LOG_CSV}")
    if fa:
        print("برای تلاش دوباره فقط همین اسکریپت را دوباره اجرا کن.")


if __name__ == "__main__":
    main()

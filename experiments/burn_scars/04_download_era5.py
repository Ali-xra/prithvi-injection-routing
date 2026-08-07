"""
04_download_era5.py — دانلود موازی ERA5-Land برای هر نمونه
===========================================================
نسخه: v1 · تاریخ: 2026-07-29

این اسکریپت چه می‌کند:
    برای هر یک از ۸۰۴ نمونه، پنجرهٔ ۷ روزِ **قبل از** تاریخ مشاهده را از
    ERA5-Land می‌گیرد — چهار متغیر، همهٔ ۲۴ ساعت، روی جعبهٔ همان کاشی.

چرا ۷ روز قبل و نه همان روز:
    تاریخ فایل، تاریخ **مشاهدهٔ ماهواره** است. آتش‌سوزی قبل از آن رخ داده.
    باد «همان لحظه» بی‌معنی است.

چرا موازی:
    گلوگاه، پهنای باند نیست — **صف سمت ECMWF** است. ۸۰۴ درخواست پشت سر هم
    یعنی روزها؛ با ۱۰ کارگر هم‌زمان یعنی ساعت‌ها.

⚠️ قابل ازسرگیری: هر نمونه فایل خودش را دارد و اگر از قبل باشد رد می‌شود.
    قطع شد؟ دوباره اجرا کن. چیزی دو بار دانلود نمی‌شود.

خروجی:
    data/era5/samples/<sample_id>__<tile>__<date>.nc
    data/era5/_download_log.csv   — وضعیت هر نمونه، برای تلاش دوباره

اجرا:
    python 04_download_era5.py              # همه
    python 04_download_era5.py --limit 5    # فقط ۵ تا، برای آزمایش
    python 04_download_era5.py --workers 6  # کم‌کردن فشار روی صف
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import time
import argparse
import threading
from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import cdsapi

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
META = BIG / "data" / "meta"
IN_CSV = META / "samples_split.csv"
OUT_DIR = BIG / "data" / "era5" / "samples"
LOG_CSV = BIG / "data" / "era5" / "_download_log.csv"

DAYS_BEFORE = 7
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "total_precipitation",
]
HOURS = [f"{h:02d}:00" for h in range(24)]
MARGIN_DEG = 0.15   # حاشیه تا حداقل ۳×۳ خانهٔ ERA5 بیفتد

# ⚠️ حتماً RLock — نه Lock. کد داخل `with _print_lock` دوباره log() صدا می‌زند و
# log() هم همان قفل را می‌گیرد. با Lock ساده این **بن‌بست** است و کل خط لوله
# بعد از اولین دانلود موفق برای همیشه می‌ایستد. (باگ واقعی، 2026-07-29)
_print_lock = threading.RLock()
_submit_lock = threading.Lock()
_last_submit = [0.0]
_done = 0

SUBMIT_GAP = 4.0        # ثانیه بین دو ارسال — جلوی سقف نرخ را می‌گیرد
MAX_TRIES = 4
BACKOFF = [60, 180, 420]   # ثانیه، بعد از تلاش ۱ و ۲ و ۳


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def throttled_submit():
    """مطمئن می‌شود دو درخواست پشت سر هم با فاصله ارسال شوند."""
    with _submit_lock:
        wait = SUBMIT_GAP - (time.monotonic() - _last_submit[0])
        if wait > 0:
            time.sleep(wait)
        _last_submit[0] = time.monotonic()


def build_request(row):
    """پنجرهٔ ۷ روزه را به فهرست سال/ماه/روز تبدیل می‌کند.

    ⚠️ محدودیت شناخته‌شده (تأییدشده 2026-07-29):
        CDS این سه فیلد را **ضرب دکارتی** می‌گیرد، نه فهرست تاریخ. اگر پنجره از
        مرز ماه رد شود، به‌جای ۷ روز، ۱۳ تا ۱۴ روز برمی‌گردد (۳۱۲ یا ۳۳۶ گام
        به‌جای ۱۶۸).

        عمداً تعمیر نشده: تعمیرش یعنی شکستن به دو درخواست، و گلوگاه ما **تعداد
        درخواست** است نه حجم (فایل‌ها ~۱۴۰ کیلوبایت‌اند).
        👉 **`05_build_features.py` باید دقیقاً ۷ روزِ قبل از تاریخ نمونه را برش
        بزند و بقیه را دور بریزد.**

    ⚠️ ERA5-Land فقط خشکی است — کاشی‌های ساحلی NaN دارند (تا ۸٪ دیده شد).
        در `05` از `nanmean`/`nanmax` استفاده شود و درصد NaN ثبت گردد.
    """
    d0 = date.fromisoformat(row["date"])
    days = [d0 - timedelta(days=k) for k in range(1, DAYS_BEFORE + 1)]

    years = sorted({f"{d.year}" for d in days})
    months = sorted({f"{d.month:02d}" for d in days})
    dnums = sorted({f"{d.day:02d}" for d in days})

    north = float(row["lat_max"]) + MARGIN_DEG
    south = float(row["lat_min"]) - MARGIN_DEG
    west = float(row["lon_min"]) - MARGIN_DEG
    east = float(row["lon_max"]) + MARGIN_DEG

    return {
        "variable": VARIABLES,
        "year": years,
        "month": months,
        "day": dnums,
        "time": HOURS,
        "area": [round(north, 3), round(west, 3), round(south, 3), round(east, 3)],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }, days


def out_path(row):
    return OUT_DIR / f"{int(row['sample_id']):04d}__{row['tile']}__{row['date']}.nc"


def fetch(row, total):
    global _done
    p = out_path(row)
    if p.exists() and p.stat().st_size > 0:
        with _print_lock:
            _done += 1
        return {"sample_id": row["sample_id"], "file": p.name, "status": "skipped", "note": "از قبل بود"}

    req, _ = build_request(row)
    last_err = None

    for attempt in range(1, MAX_TRIES + 1):
        try:
            throttled_submit()
            client = cdsapi.Client(quiet=True, progress=False)
            client.retrieve("reanalysis-era5-land", req, str(p))
            kb = p.stat().st_size / 1024
            with _print_lock:
                _done += 1
                extra = f" (تلاش {attempt})" if attempt > 1 else ""
                log(f"  [{_done}/{total}] ✅ {p.name}  {kb:.0f} KB{extra}")
            return {"sample_id": row["sample_id"], "file": p.name,
                    "status": "ok", "note": f"{kb:.0f}KB tries={attempt}"}

        except Exception as e:
            last_err = e
            if p.exists():
                p.unlink()      # فایل ناقص نماند و دفعهٔ بعد اشتباهاً رد نشود
            if attempt < MAX_TRIES:
                wait = BACKOFF[attempt - 1]
                log(f"     ↻ {row['sample_id']} تلاش {attempt} نشد ({type(e).__name__}) — {wait}s صبر")
                time.sleep(wait)
            else:
                with _print_lock:
                    _done += 1
                    log(f"  [{_done}/{total}] ⛔ {row['sample_id']} — بعد از {MAX_TRIES} تلاش: {type(e).__name__}: {e}")

    return {"sample_id": row["sample_id"], "file": p.name, "status": "failed",
            "note": f"{type(last_err).__name__}: {last_err}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="فقط N نمونهٔ اول")
    ap.add_argument("--workers", type=int, default=4,
                    help="سقف CDS روی «تعداد درخواست در صف» است. ۵ کار کرد، ۱۶ همه rejected شد. ۴ = حاشیهٔ امن")
    args = ap.parse_args()

    if not IN_CSV.exists():
        print(f"⛔ {IN_CSV} نیست — اول 03_make_split.py را اجرا کن")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with IN_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    already = sum(1 for r in rows if out_path(r).exists())
    print(f"نمونه: {len(rows)}   از قبل دانلود شده: {already}   مانده: {len(rows)-already}")
    print(f"کارگر هم‌زمان: {args.workers}   فاصلهٔ ارسال: {SUBMIT_GAP}s   تلاش: تا {MAX_TRIES} بار")
    print(f"پنجره: {DAYS_BEFORE} روز قبل   متغیر: {len(VARIABLES)}")
    print(f"خروجی: {OUT_DIR}\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, r, len(rows)): r for r in rows}
        for fu in as_completed(futs):
            results.append(fu.result())

    ok = sum(1 for r in results if r["status"] == "ok")
    sk = sum(1 for r in results if r["status"] == "skipped")
    fa = sum(1 for r in results if r["status"] == "failed")

    with LOG_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "file", "status", "note"])
        w.writeheader()
        w.writerows(sorted(results, key=lambda r: int(r["sample_id"])))

    print("\n" + "=" * 62)
    print(f"موفق: {ok}   رد شد (از قبل بود): {sk}   ناموفق: {fa}")
    print(f"لاگ: {LOG_CSV}")
    if fa:
        print("\n⚠️ برای تلاش دوباره فقط همین اسکریپت را دوباره اجرا کن —")
        print("   موفق‌ها رد می‌شوند و فقط ناموفق‌ها دوباره امتحان می‌شوند.")


if __name__ == "__main__":
    main()

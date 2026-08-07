# -*- coding: utf-8 -*-
"""
90_recon_gpp.py — شناسایی دیتاست GPP (`hls_merra2_gppFlux`)

⚠️ پیشوند ۹۰ عمدی است: این **خط لولهٔ سیل نیست.** یک بررسی جانبی است که کار
   `burn` در صندوق خواست، پیش از دانلود دیتاست سیل.

پنج سؤالی که باید جواب بگیرد:
  ۱ خروجی: یک عدد به ازای chip یا نقشه؟
  ۲ ده اسکالر MERRA-2 چطور ذخیره شده‌اند؟ نام ستون‌ها؟
  ۳ کانفیگ رسمی TerraTorch دارد؟
  ۴ ۵۰×۵۰ بر patch بخش‌پذیر نیست — چه می‌کنند؟
  ۵ split سال‌محور: چند سال، چند نمونه در هر سال؟

الزامات کار طولانی (`AI-RULES-REF` بخش ۹): کَش، timeout، تلاش دوباره با لاگ
دلیل، شمارندهٔ پیشرفت، زمان دیواری، اعلام صریح آنچه نیامد. تک‌نخی، پس RLock لازم نیست.

نوشته: 2026-07-30 · کار `flood`
"""
import io
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(r"C:\Users\aliso\Desktop\big-files\injection-routing-flood\_recon-gpp")
BASE = "https://huggingface.co/datasets/ibm-nasa-geospatial/hls_merra2_gppFlux/raw/main"
FILES = ["data_train_hls_37sites_v0_1.csv", "prep_input.py", "make_chips.py", "fluxconfig.yaml"]
TIMEOUT_S, MAX_TRIES = 60, 4


def fetch(name):
    dest = OUT / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cache] {name} ({dest.stat().st_size:,})")
        return dest
    url = f"{BASE}/{name}"
    for attempt in range(1, MAX_TRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "injection-routing/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                blob = r.read()
            dest.write_bytes(blob)
            print(f"  [ok]    {name} ({len(blob):,})")
            return dest
        except Exception as exc:  # noqa: BLE001
            print(f"  [fail]  {name} {attempt}/{MAX_TRIES} — {type(exc).__name__}: {exc}")
            if attempt < MAX_TRIES:
                time.sleep(2 ** attempt)
    return None


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("شناسایی GPP — پنج سؤالِ کار `burn`")
    print("=" * 78)
    got = {}
    for i, n in enumerate(FILES, 1):
        print(f"[{i}/{len(FILES)}] {n}   ({time.time()-t0:.1f}s)")
        p = fetch(n)
        if p:
            got[n] = p
    missing = [n for n in FILES if n not in got]
    if missing:
        print(f"\n🔴 نیامد: {missing}")

    csv_name = "data_train_hls_37sites_v0_1.csv"
    if csv_name not in got:
        print("🔴 CSV نیامد → سؤال ۲ و ۵ بی‌جواب. سکوت نمی‌کنم.")
        return

    import csv as _csv
    rows = list(_csv.DictReader(io.StringIO(got[csv_name].read_text(encoding="utf-8"))))
    cols = list(rows[0].keys())
    print("\n" + "-" * 78)
    print(f"سؤال ۲ — ستون‌ها ({len(cols)} ستون · {len(rows)} سطر)")
    print("-" * 78)
    for c in cols:
        print(f"   {c:<28} نمونه: {rows[0][c]!r}")

    # سؤال ۵ — split سال‌محور
    print("\n" + "-" * 78)
    print("سؤال ۵ — سال و سایت")
    print("-" * 78)
    ycol = next((c for c in cols if c.lower() in ("year", "yr")), None)
    dcol = next((c for c in cols if "date" in c.lower() or c.lower() == "time"), None)
    scol = next((c for c in cols if "site" in c.lower() or c.lower() == "id"), None)

    def year_of(r):
        if ycol:
            return str(r[ycol])[:4]
        if dcol:
            return str(r[dcol])[:4]
        return "?"

    years = Counter(year_of(r) for r in rows)
    print(f"ستون سال: {ycol!r} · ستون تاریخ: {dcol!r} · ستون سایت: {scol!r}")
    for y, n in sorted(years.items()):
        print(f"   {y}: {n} نمونه")
    print(f"جمع: {sum(years.values())}")
    if scol:
        sites = Counter(r[scol] for r in rows)
        print(f"سایت متمایز: {len(sites)}")
        # سایت در چند سال ظاهر می‌شود؟ (نشت سایت بین foldها)
        pairs = {(r[scol], year_of(r)) for r in rows}
        by_site = Counter(s for s, _ in pairs)
        multi = sum(1 for s, k in by_site.items() if k > 1)
        print(f"سایت‌هایی که در بیش از یک سال هستند: {multi} از {len(by_site)}"
              f"  ← 🔴 نشتِ سایت بین foldهای سال‌محور")
    summary = {"n_rows": len(rows), "columns": cols, "per_year": dict(sorted(years.items())),
               "n_sites": len(set(r[scol] for r in rows)) if scol else None,
               "wall_seconds": round(time.time() - t0, 2)}
    (OUT / "gpp_recon_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nزمان دیواری: {time.time()-t0:.1f}s · {OUT / 'gpp_recon_summary.json'}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
03_download_flood.py — گام ۳ب · دانلود ۴۴۶ chip دست‌برچسب Sen1Floods11
=====================================================================
نسخه: v1 · تاریخ: 2026-07-30 · کار `flood`

فقط دو لایه که لازم داریم — **نه کل دیتاست**:
    S2Hand     تصویر نوری، ۵۱۲×۵۱۲×۱۳
    LabelHand  ماسک دست‌برچسب

نام‌ها از چهار CSV گام ۱ می‌آید (train/valid/test/bolivia)، پس **دقیقاً ۴۴۶×۲ فایل**
دانلود می‌شود و یک بایت اضافه نه. SAR (`S1Hand`) دانلود نمی‌شود — تسک نوری است.

الزامات کار طولانی (`_ai/AI-RULES-REF.md` بخش ۹) — هر هفت مورد:
    ۱ شمارندهٔ پیشرفت با زمان سپری‌شده و تخمین باقی‌مانده
    ۲ timeout صریح روی هر درخواست
    ۳ تلاش دوباره با عقب‌نشینی پلکانی + لاگ دلیل هر شکست
    ۴ کَش قابل‌ازسرگیری — فایل موجود با اندازهٔ درست رد می‌شود
    ۵ 🔴 `RLock` نه `Lock` — درس هشت‌ساعتهٔ نشست `burn`
    ۶ زمان دیواری در پایان چاپ و در JSON ثبت می‌شود
    ۷ هر چیزی که نیامد، صریح فهرست می‌شود

و **کوچک‌ترین تست ممکن اول**: یک فایل، قبل از ۸۹۲ تا. درس فصل ۳.۳ سند
`data-acquisition-story` — اگر آدرس غلط باشد، ۸۹۲ خطا نمی‌گیریم.

⛔ هیچ فایلی داخل `proje\` نمی‌آید.
خروجی: <BIG>/data/flood_events/HandLabeled/{S2Hand,LabelHand}/  +  download_report.json
اجرا:  python 03_download_flood.py [--workers 4]
"""
import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing-flood"
META = BIG / "data" / "meta"
DEST = BIG / "data" / "flood_events" / "HandLabeled"
BASE = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled"
SPLITS = ("train", "valid", "test", "bolivia")
TIMEOUT_S, MAX_TRIES = 120, 4

_lock = threading.RLock()          # ⚠️ RLock — نه Lock
_done = {"n": 0, "bytes": 0}


def log(msg):
    """با RLock — و ⚠️ هرگز از داخل بلوکی که همین قفل را گرفته صدا نمی‌شود."""
    with _lock:
        print(msg, flush=True)


def names():
    """۴۴۶ ریشهٔ نام از چهار CSV گام ۱. ریشه = بخش قبل از `_S1Hand.tif`."""
    stems, per_split = [], {}
    for s in SPLITS:
        f = META / f"flood_{s}_data.csv"
        if not f.exists():
            raise SystemExit(f"⛔ {f} نیست — اول 01_fetch_flood_metadata.py را اجرا کن")
        got = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            first = line.split(",")[0].strip().rsplit("/", 1)[-1]
            got.append(first.replace("_S1Hand.tif", ""))
        per_split[s] = got
        stems += got
    return stems, per_split


def fetch_one(url, dest):
    """→ (ok, bytes, reason)  ·  کَش: فایل موجود و ناتهی دوباره دانلود نمی‌شود."""
    if dest.exists() and dest.stat().st_size > 0:
        return True, dest.stat().st_size, "cache"
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = ""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "injection-routing/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r, tmp.open("wb") as fh:
                n = 0
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    fh.write(buf)
                    n += len(buf)
            if n == 0:
                raise OSError("صفر بایت")
            tmp.replace(dest)                      # اتمیک — فایل نیم‌کاره جا نمی‌ماند
            return True, n, "downloaded"
        except Exception as exc:                   # noqa: BLE001 — دلیل لاگ می‌شود
            last = f"{type(exc).__name__}: {exc}"
            tmp.unlink(missing_ok=True)
            if attempt < MAX_TRIES:
                time.sleep(2 ** attempt)
    return False, 0, last


def job(args):
    layer, stem, total, t0 = args
    url = f"{BASE}/{layer}/{stem}_{layer}.tif"
    dest = DEST / layer / f"{stem}_{layer}.tif"
    ok, n, reason = fetch_one(url, dest)
    with _lock:
        _done["n"] += 1
        _done["bytes"] += n
        i = _done["n"]
        if i % 25 == 0 or i == total:
            el = time.time() - t0
            rate = i / el if el else 0
            log(f"   {i}/{total} · {_done['bytes']/2**20:,.0f} MiB · {el:.0f}s · "
                f"{rate:.1f} فایل/ثانیه · تخمین باقی {(total-i)/rate if rate else 0:.0f}s")
    return {"layer": layer, "stem": stem, "ok": ok, "bytes": n, "reason": reason}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t_all = time.time()
    stems, per_split = names()
    for layer in ("S2Hand", "LabelHand"):
        (DEST / layer).mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"دانلود Sen1Floods11 دست‌برچسب · {len(stems)} chip × ۲ لایه = "
          f"{len(stems)*2} فایل · {args.workers} کارگر")
    print(" · ".join(f"{k}:{len(v)}" for k, v in per_split.items()))
    print("=" * 78)

    # کوچک‌ترین تست ممکن، قبل از ۸۹۲ درخواست
    probe = stems[0]
    print(f"\n[تست تک‌فایلی] {probe}_S2Hand.tif")
    ok, n, reason = fetch_one(f"{BASE}/S2Hand/{probe}_S2Hand.tif",
                              DEST / "S2Hand" / f"{probe}_S2Hand.tif")
    if not ok:
        print(f"⛔ تست شکست خورد — {reason}")
        print("   آدرس یا ساختار باکت عوض شده. ۸۹۲ درخواست فرستاده نشد.")
        return
    print(f"   ✅ {n/2**20:.2f} MiB ({reason}) → پس آدرس درست است، ادامه می‌دهیم\n")

    tasks = [(layer, s, len(stems) * 2, time.time())
             for layer in ("S2Hand", "LabelHand") for s in stems]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(job, tasks))

    failed = [r for r in res if not r["ok"]]
    cached = sum(1 for r in res if r["reason"] == "cache")
    total_b = sum(r["bytes"] for r in res)

    print("\n" + "-" * 78)
    print(f"دانلودشده {len(res)-len(failed)-cached} · از کَش {cached} · "
          f"شکست {len(failed)} · حجم {total_b/2**30:.2f} GiB")

    if failed:                                     # الزام ۷ — سکوت نمی‌کنیم
        print("\n🔴 این‌ها نیامدند:")
        for r in failed[:20]:
            print(f"   {r['stem']}_{r['layer']}.tif — {r['reason']}")
        if len(failed) > 20:
            print(f"   … و {len(failed)-20} مورد دیگر (کاملش در JSON)")

    # تطبیق ۱:۱ — هشدار کار `burn`: تطبیق زیررشته‌ای نشت خاموش می‌سازد
    print("\n" + "-" * 78)
    print("بررسی سلامت — تطبیق ۱:۱ تصویر و ماسک")
    img = sorted(p.stem.replace("_S2Hand", "") for p in (DEST / "S2Hand").glob("*_S2Hand.tif"))
    msk = sorted(p.stem.replace("_LabelHand", "") for p in (DEST / "LabelHand").glob("*_LabelHand.tif"))
    print(f"   تصویر {len(img)} · ماسک {len(msk)} · ناجفت {len(set(img) ^ set(msk))}")
    print(f"   انتظار: {len(stems)} و {len(stems)} و ۰")

    # 🔴 زیررشته‌بودن یک نام در نام دیگر → یک خط split دو فایل می‌گیرد
    subs = [(a, b) for a in img for b in img if a != b and a in b]
    print(f"   نامی که زیررشتهٔ نام دیگری است: {len(subs)}"
          + (f"  ← 🔴 {subs[:5]}" if subs else "  ✅"))

    rep = {"n_chips": len(stems), "per_split": {k: len(v) for k, v in per_split.items()},
           "n_files_expected": len(stems) * 2, "n_failed": len(failed),
           "n_cached": cached, "bytes_total": total_b,
           "images_on_disk": len(img), "masks_on_disk": len(msk),
           "unpaired": sorted(set(img) ^ set(msk)),
           "substring_collisions": subs[:50],
           "failures": [{"file": f"{r['stem']}_{r['layer']}.tif", "reason": r["reason"]}
                        for r in failed],
           "wall_seconds": round(time.time() - t_all, 1)}
    out = BIG / "data" / "meta" / "download_report.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nزمان دیواری: {rep['wall_seconds']}s · گزارش: {out}")


if __name__ == "__main__":
    main()

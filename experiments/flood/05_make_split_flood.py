# -*- coding: utf-8 -*-
"""
05_make_split_flood.py — split رویدادمحور بدون نشت
===================================================
نسخه: v1 · تاریخ: 2026-07-30 · کار `flood`
مبنا: کپی و وصلهٔ `../lab/src/03_make_split.py` (2026-07-28)

چرا split رسمی استفاده نمی‌شود — اندازه‌گیری گام ۱:
    chip مشترک **۰٪** ولی رویداد مشترک **۱۰۰٪**. هر ده رویداد در هر سه split
    حاضرند — همان روز، همان سیل، همان منطقه. بدتر از ۷۳.۵٪ نشت آتش‌سوزی، چون
    اینجا تاریخ هم **دقیقاً** یکی است.

گروه = **رویداد**، نه کاشی. با یازده گروه، این محدودیت پذیرفته شده است و در
`../SYNC.md` بخش ۸ **پیش از دیدن هر نتیجه‌ای** ثبت شده بود.

⚠️ سه چیز که عیناً از نسخهٔ آتش‌سوزی حفظ می‌شود:
    `SEED=0` · یک بار ساخته و **دیگر هرگز عوض نمی‌شود** · تخصیص حریصانهٔ
    گروه‌محور با متوازن نگه داشتن درصد کلاس مثبت (اینجا آب، آنجا سوختگی)

⚠️ و یک تفاوت اجباری: **بولیوی از قبل کنارگذاشته است** (۱۵ نمونه، `val_chip`
   جدا در متادیتای رسمی). آن را به `test` می‌دهیم و در تخصیص حریصانه شرکت نمی‌کند
   — چون دست‌کاری‌اش یعنی تصمیم پس‌نگرانه.

خروجی: <BIG>/data/meta/samples_split.csv  +  split_report.json
اجرا:  python 05_make_split_flood.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing-flood"
META = BIG / "data" / "meta"
IN_CSV = META / "samples.csv"
OUT_CSV = META / "samples_split.csv"
OUT_JSON = META / "split_report.json"

SEED = 0
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}
HELD_OUT_EVENT = "Bolivia"          # از قبل کنارگذاشته در دیتاست رسمی


def main():
    rows = list(csv.DictReader(IN_CSV.open(encoding="utf-8")))
    for r in rows:
        r["pct_water"] = float(r["pct_water"])
    by_event = defaultdict(list)
    for r in rows:
        by_event[r["event"]].append(r)

    print("=" * 78)
    print(f"split رویدادمحور · {len(rows)} نمونه · {len(by_event)} رویداد · SEED={SEED}")
    print("=" * 78)

    assign = {HELD_OUT_EVENT: "test"} if HELD_OUT_EVENT in by_event else {}
    pool = sorted(e for e in by_event if e not in assign)

    # حریصانه: رویداد بزرگ‌تر اول، به splitی که از سهمش عقب‌تر است
    rng = np.random.default_rng(SEED)
    order = sorted(pool, key=lambda e: (-len(by_event[e]), e))
    n_total = len(rows)
    cur = {k: sum(len(by_event[e]) for e, s in assign.items() if s == k)
           for k in TARGET}
    for e in order:
        deficit = {k: TARGET[k] * n_total - cur[k] for k in TARGET}
        best = max(deficit, key=lambda k: (deficit[k], -ord(k[0])))
        assign[e] = best
        cur[best] += len(by_event[e])

    for r in rows:
        r["split"] = assign[r["event"]]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------- بررسی سلامت ----------
    print(f"\n{'split':>6}{'n':>6}{'٪':>8}{'رویداد':>8}   میانهٔ آب   میانگین آب   رویدادها")
    rep = {"seed": SEED, "assignment": assign, "splits": {}}
    for k in ("train", "val", "test"):
        sel = [r for r in rows if r["split"] == k]
        evs = sorted({r["event"] for r in sel})
        w_ = np.array([r["pct_water"] for r in sel])
        print(f"{k:>6}{len(sel):>6}{100*len(sel)/n_total:>7.1f}%{len(evs):>8}"
              f"{np.median(w_):>11.2f}{w_.mean():>13.2f}   {', '.join(evs)}")
        rep["splits"][k] = {"n": len(sel), "events": evs,
                            "water_median": round(float(np.median(w_)), 3),
                            "water_mean": round(float(w_.mean()), 3),
                            "zero_water": int((w_ == 0).sum())}

    ev_sets = {k: {r["event"] for r in rows if r["split"] == k} for k in ("train", "val", "test")}
    overlap = {f"{a}∩{b}": sorted(ev_sets[a] & ev_sets[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    print("\nرویداد مشترک بین splitها: " +
          " · ".join(f"{k}={len(v)}" for k, v in overlap.items()))
    dates = {k: sorted({r["date"] for r in rows if r["split"] == k}) for k in ev_sets}
    print("تاریخ مشترک: " + " · ".join(
        f"{a}∩{b}={len(set(dates[a]) & set(dates[b]))}"
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))))
    rep["event_overlap"] = overlap
    rep["n_dates_per_split"] = {k: len(v) for k, v in dates.items()}
    OUT_JSON.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{OUT_CSV}\n{OUT_JSON}")


if __name__ == "__main__":
    main()

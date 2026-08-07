"""
03_make_split.py — ساخت split جغرافیایی بر اساس کاشی
=====================================================
نسخه: v1 · تاریخ: 2026-07-28

تصمیم (علی، 2026-07-28): **گزینهٔ الف — تفکیک بر اساس کاشی.**
هیچ کاشی MGRS نباید در دو split ظاهر شود.

چرا این و نه تفکیک زمانی:
    تفکیک زمانی (۲۰۲۱ به‌عنوان آزمون) فقط ۹۰ نمونه می‌گذارد. با ۸۰۴ نمونهٔ کل،
    مجموعهٔ آزمونِ کوچک یعنی نوسان زیاد — و آن با «انحراف معیار بین seedها» که
    مهم‌ترین عدد پروژه است بد ترکیب می‌شود.

چه چیزی را درست می‌کند:
    split آمادهٔ HuggingFace تصادفی است و اندازه‌گیری شد: **۷۳.۵٪** نمونه‌های val
    کاشی‌شان در train هم بود، و **۳۱.۸٪** در فاصلهٔ ۹۰ روز از یک نمونهٔ train.
    برای رد آتش‌سوزی که ماه‌ها باقی می‌ماند، این یعنی مدل ممکن است همان اثر را
    از قبل دیده باشد.

روش:
    کاشی‌ها را به‌عنوان گروه می‌گیریم و حریصانه به سه سبد می‌ریزیم، طوری‌که
    هم سهم نمونه نزدیک ۷۰/۱۵/۱۵ بماند و هم توزیع «درصد سوختگی» و «سال» متوازن.
    گروه‌های بزرگ اول تقسیم می‌شوند چون کم‌انعطاف‌ترین‌اند.

خروجی:
    data/meta/samples_split.csv  — همان ستون‌های samples.csv + ستون `split`
    (ستون `orig_split` دست‌نخورده می‌ماند تا بشود با ادبیات مقایسه کرد)

⚠️ این فایل یک بار ساخته می‌شود و **دیگر هرگز عوض نمی‌شود**. SEED ثابت است.

اجرا:
    python 03_make_split.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import random
from pathlib import Path
from collections import defaultdict
import numpy as np

SEED = 0
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
META = BIG / "data" / "meta"
IN_CSV = META / "samples.csv"
OUT_CSV = META / "samples_split.csv"


def load():
    with IN_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["pct_burn"] = float(r["pct_burn"])
        r["pct_nodata"] = float(r["pct_nodata"])
        r["year"] = int(r["year"])
    return rows


def main():
    if not IN_CSV.exists():
        print(f"⛔ {IN_CSV} نیست — اول 02_build_meta.py را اجرا کن")
        return

    rows = load()
    n = len(rows)
    print(f"{n} نمونه خوانده شد\n")

    # --- گروه‌بندی بر اساس کاشی ---
    groups = defaultdict(list)
    for r in rows:
        groups[r["tile"]].append(r)
    print(f"کاشی یکتا: {len(groups)}")
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"بزرگ‌ترین کاشی‌ها: {sizes[:8]}   کاشی‌های تک‌نمونه‌ای: {sizes.count(1)}\n")

    # --- تخصیص حریصانه با توازن ---
    # کلید مرتب‌سازی: اول اندازه (نزولی)، بعد میانهٔ سوختگی — تا گروه‌های
    # پرسوخت و کم‌سوخت به‌نوبت بین سبدها پخش شوند نه اینکه یک‌جا جمع شوند.
    rnd = random.Random(SEED)
    keys = list(groups.keys())
    rnd.shuffle(keys)
    keys.sort(key=lambda t: (-len(groups[t]), -np.median([x["pct_burn"] for x in groups[t]])))

    quota = {k: v * n for k, v in RATIOS.items()}
    cur_n = {k: 0 for k in RATIOS}
    cur_burn = {k: [] for k in RATIOS}
    assign = {}

    for t in keys:
        g = groups[t]
        gb = float(np.median([x["pct_burn"] for x in g]))
        best, best_cost = None, None
        for s in RATIOS:
            # کسریِ نمونه نسبت به سهمیه — هرچه منفی‌تر، نیازمندتر
            deficit = (cur_n[s] + len(g)) / quota[s]
            # فاصلهٔ میانگین سوختگی این سبد از میانگین کل، اگر این گروه اضافه شود
            merged = cur_burn[s] + [gb]
            burn_gap = abs(float(np.mean(merged)) - float(np.mean([x["pct_burn"] for x in rows])))
            cost = deficit + 0.02 * burn_gap
            if best_cost is None or cost < best_cost:
                best, best_cost = s, cost
        assign[t] = best
        cur_n[best] += len(g)
        cur_burn[best].append(gb)

    for r in rows:
        r["split"] = assign[r["tile"]]

    # --- بررسی سلامت ---
    print("=" * 62)
    tiles_by_split = defaultdict(set)
    for r in rows:
        tiles_by_split[r["split"]].add(r["tile"])
    leak = 0
    for a in RATIOS:
        for b in RATIOS:
            if a < b:
                leak += len(tiles_by_split[a] & tiles_by_split[b])
    print(f"🔍 کاشی مشترک بین split ها: {leak}   " + ("✅ صفر — نشت مکانی بسته شد" if leak == 0 else "⛔ نشت دارد"))

    print(f"\n{'split':<8}{'نمونه':>7}{'سهم':>8}{'کاشی':>8}{'میانهٔ سوختگی':>16}{'میانگین':>10}")
    all_burn = np.array([r["pct_burn"] for r in rows])
    for s in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == s]
        b = np.array([r["pct_burn"] for r in sub])
        print(f"{s:<8}{len(sub):>7}{100*len(sub)/n:>7.1f}%{len(tiles_by_split[s]):>8}"
              f"{np.median(b):>15.2f}%{np.mean(b):>9.2f}%")
    print(f"{'کل':<8}{n:>7}{100.0:>7.1f}%{len(groups):>8}"
          f"{np.median(all_burn):>15.2f}%{np.mean(all_burn):>9.2f}%")

    print(f"\n{'split':<8}" + "".join(f"{y:>8}" for y in (2018, 2019, 2020, 2021)))
    for s in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == s]
        cnt = {y: sum(1 for r in sub if r["year"] == y) for y in (2018, 2019, 2020, 2021)}
        print(f"{s:<8}" + "".join(f"{cnt[y]:>8}" for y in (2018, 2019, 2020, 2021)))

    hi = [r for r in rows if r["pct_nodata"] > 5]
    print(f"\nنمونه با بیش از ۵٪ بی‌داده: {len(hi)} — "
          + ", ".join(f"{s}:{sum(1 for r in hi if r['split']==s)}" for s in ("train", "val", "test")))
    print("   (حذف نشدند. تصمیمش جداست — ستون pct_nodata برای فیلترکردن هست.)")

    # --- نوشتن ---
    fields = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n✅ نوشته شد → {OUT_CSV}")
    print(f"   SEED={SEED} · این split دیگر عوض نمی‌شود.")


if __name__ == "__main__":
    main()

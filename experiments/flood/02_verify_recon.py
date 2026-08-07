# -*- coding: utf-8 -*-
"""
02_verify_recon.py — بازآزمایی مستقل اعداد گام ۱

هدف: اعداد `recon_summary.json` را از **مسیر کد دیگری** بازتولید کن. اگر یکی
نشدند، یکی از دو مسیر باگ دارد و هیچ‌کدام قابل اعتماد نیست.

سه بررسی:
  ۱ جمع‌های مستقل از فایل‌های خام (بدون خواندن recon_summary)
  ۲ نگاشت رویداد→تاریخ: آیا هر ۱۱ رویداد دست‌برچسب تاریخ می‌گیرند؟
  ۳ تلهٔ باگ TerraTorch: چند chip با نگاشت خامِ نام، تاریخ نمی‌گیرند؟

نوشته: 2026-07-29 · نشست A
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
META = Path(r"C:\Users\aliso\Desktop\big-files\injection-routing-flood\data\meta")
SPLITS = ("train", "valid", "test", "bolivia")
# نگاشت دستیِ ما — به تاریخِ TerraTorch اتکا نمی‌کنیم
NAME_FIX = {"Mekong": "Cambodia"}

fails = []


def check(label, got, want):
    ok = got == want
    print(f"{'✅' if ok else '❌'} {label}: {got}" + ("" if ok else f"  (انتظار {want})"))
    if not ok:
        fails.append(label)


print("=" * 74)
print("بازآزمایی مستقل — اعداد گزارش‌شده در RESULTS.md و SYNC.md")
print("=" * 74)

# ---- ۱ شمارش مستقل ----
per_split, chips_by_split = {}, {}
for s in SPLITS:
    lines = [l for l in (META / f"flood_{s}_data.csv").read_text(encoding="utf-8").splitlines() if l.strip()]
    per_split[s] = len(lines)
    chips_by_split[s] = {l.split(",")[0].strip() for l in lines}

check("train", per_split["train"], 252)
check("valid", per_split["valid"], 89)
check("test", per_split["test"], 90)
check("bolivia", per_split["bolivia"], 15)
check("جمع کل", sum(per_split.values()), 446)

# هیچ فایلی تکراری نباشد — نه درون یک split نه بین splitها
all_files = [f for s in SPLITS for f in chips_by_split[s]]
check("فایل تکراری بین/درون splitها", len(all_files) - len(set(all_files)), 0)

events_by_split = {s: {f.split("_")[0] for f in chips_by_split[s]} for s in SPLITS}
check("رویداد در train", len(events_by_split["train"]), 10)
shared = events_by_split["train"] & events_by_split["valid"] & events_by_split["test"]
check("رویداد مشترک در هر سه split", len(shared), 10)
all_events = set().union(*events_by_split.values())
check("رویداد دست‌برچسب (با بولیوی)", len(all_events), 11)

# نشت رویدادمحور به درصد — همان عددی که در SYNC نوشتم
leak_val = sum(1 for f in chips_by_split["valid"] if f.split("_")[0] in events_by_split["train"])
leak_test = sum(1 for f in chips_by_split["test"] if f.split("_")[0] in events_by_split["train"])
check("٪ نمونهٔ valid با رویدادش در train", round(100 * leak_val / per_split["valid"]), 100)
check("٪ نمونهٔ test با رویدادش در train", round(100 * leak_test / per_split["test"]), 100)


# ---- ۲ نگاشت رویداد → تاریخ ----
gj = json.loads((META / "Sen1Floods11_Metadata.geojson").read_text(encoding="utf-8"))
by_loc = {f["properties"]["location"]: f["properties"] for f in gj["features"]}
check("feature در متادیتا", len(gj["features"]), 12)
check("رویداد متادیتا بدون split دست‌برچسب", len(set(by_loc) - {NAME_FIX.get(e, e) for e in all_events}), 1)

resolved, unresolved = {}, []
for e in sorted(all_events):
    key = NAME_FIX.get(e, e)
    if key in by_loc and by_loc[key].get("s2_date"):
        resolved[e] = by_loc[key]["s2_date"]
    else:
        unresolved.append(e)
check("رویداد با تاریخ حل‌شده (با نگاشت ما)", len(resolved), 11)
check("رویداد بی‌تاریخ", len(unresolved), 0)
check("تاریخ متمایز", len(set(resolved.values())), 11)

print("\nرویداد → s2_date:")
for e, d in sorted(resolved.items(), key=lambda kv: kv[1]):
    print(f"   {e:<12} {d}")

# ---- ۳ تلهٔ باگ TerraTorch: بدون نگاشت چند chip تاریخ جعلی می‌گیرد؟ ----
bad = Counter()
for s in SPLITS:
    for f in chips_by_split[s]:
        ev = f.split("_")[0]
        if ev not in by_loc:                     # همان شرط TerraTorch
            bad[ev] += 1
print(f"\n🐛 بدون نگاشت، chipهایی که تاریخ جعلی 1998-10-13 می‌گیرند: "
      f"{sum(bad.values())} → {dict(bad)}")
check("chip آسیب‌دیده از باگ TerraTorch", sum(bad.values()), 30)
check("درصد آسیب‌دیده", round(100 * sum(bad.values()) / 446, 1), 6.7)

print("\n" + "=" * 74)
print("همه گذشت ✅" if not fails else f"❌ {len(fails)} بررسی نگذشت: {fails}")
print("=" * 74)
sys.exit(1 if fails else 0)

# -*- coding: utf-8 -*-
"""
01_fetch_flood_metadata.py  —  گام ۱ و ۲ · شناسایی متادیتای Sen1Floods11

هدف: بدون دانلود دیتاست (چند گیگ)، این چهار سؤال را با عدد جواب بده:
  ۱. 🔴 مسیر بحرانی — تاریخ هر chip در دسترس است؟ با چه دانه‌بندی؟
  ۲. تعداد نمونه و تعداد رویداد در هر split رسمی
  ۳. گروه split واقعاً «رویداد» است یا چیز دیگری؟
  ۴. نام رویداد در فایل‌های split با فیلد location متادیتا جور است؟

مبنا: هیچ. این اسکریپت معادلی در lab/src/ ندارد — چون آتش‌سوزی تاریخ را از
      نام فایل می‌خواند (01_scan_dataset.py) ولی سیل متادیتای جدا دارد.

الزامات کار طولانی (`_ai/AI-RULES-REF.md` بخش ۹؛ پیش‌تر `SYNC.md` بخش ۷ج بود و
2026-07-30 به قواعد عمومی منتقل شد) که در این فایل رعایت شده:
  ۱ شمارندهٔ پیشرفت · ۲ timeout صریح · ۳ تلاش دوباره با عقب‌نشینی + لاگ دلیل
  ۴ کَش قابل‌ازسرگیری (فایل موجود دوباره دانلود نمی‌شود) · ۶ زمان دیواری
  ۷ اعلام صریح هر چیزی که نیامد
  (۵ RLock لازم نیست — تک‌نخی است، پنج فایل کوچک)

نوشته: 2026-07-29 · نشست A (سیل)
"""
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(r"C:\Users\aliso\Desktop\big-files\injection-routing-flood\data\meta")
TIMEOUT_S = 60
MAX_TRIES = 4
BUCKET = "https://storage.googleapis.com/sen1floods11/v1.1"
GH_RAW = "https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master"

# نام فایل → فهرست URLهای کاندید، به ترتیب اولویت
TARGETS = {
    "flood_train_data.csv": [f"{BUCKET}/splits/flood_handlabeled/flood_train_data.csv"],
    "flood_valid_data.csv": [f"{BUCKET}/splits/flood_handlabeled/flood_valid_data.csv"],
    "flood_test_data.csv": [f"{BUCKET}/splits/flood_handlabeled/flood_test_data.csv"],
    "flood_bolivia_data.csv": [f"{BUCKET}/splits/flood_handlabeled/flood_bolivia_data.csv"],
    "Sen1Floods11_Metadata.geojson": [
        f"{BUCKET}/Sen1Floods11_Metadata.geojson",
        f"{GH_RAW}/Sen1Floods11_Metadata.geojson",
    ],
}


def fetch(name, urls):
    """کَش قابل‌ازسرگیری + تلاش دوباره با عقب‌نشینی پلکانی + لاگ دلیل هر شکست."""
    dest = OUT / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cache] {name} ({dest.stat().st_size:,} بایت) — دانلود نشد")
        return dest, "cache", []
    failures = []
    for url in urls:
        for attempt in range(1, MAX_TRIES + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "injection-routing/1.0"})
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                    blob = r.read()
                dest.write_bytes(blob)
                print(f"  [ok]    {name} ({len(blob):,} بایت) از {url}")
                return dest, url, failures
            except Exception as exc:  # noqa: BLE001 — دلیل شکست باید لاگ شود
                reason = f"{type(exc).__name__}: {exc}"
                failures.append({"url": url, "attempt": attempt, "reason": reason})
                print(f"  [fail]  {name} تلاش {attempt}/{MAX_TRIES} — {reason}")
                if attempt < MAX_TRIES:
                    time.sleep(2 ** attempt)
    return None, None, failures


def read_split(path):
    """هر سطر: <chip>_S1Hand.tif,<chip>_LabelHand.tif — رویداد = قبل از اولین _"""
    rows = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    events, chips = Counter(), []
    for ln in rows:
        first = ln.split(",")[0].strip()
        stem = first.rsplit("/", 1)[-1]
        event = stem.split("_")[0]
        chip_id = stem.split("_")[1] if "_" in stem else ""
        events[event] += 1
        chips.append((event, chip_id, stem))
    return rows, events, chips


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("گام ۱ — شناسایی متادیتای Sen1Floods11 (بدون دانلود دیتاست)")
    print("=" * 78)

    got, missing, all_failures = {}, [], []
    for i, (name, urls) in enumerate(TARGETS.items(), 1):
        print(f"[{i}/{len(TARGETS)}] {name}   (سپری‌شده {time.time()-t0:.1f}s)")
        path, src, failures = fetch(name, urls)
        all_failures += failures
        if path is None:
            missing.append(name)          # الزام ۷ — سکوت نمی‌کنیم
        else:
            got[name] = (path, src)

    if missing:
        print("\n🔴 نیامد (الزام ۷ — صریح اعلام می‌شود):")
        for m in missing:
            print(f"   - {m}")

    summary = {"missing": missing, "failures": all_failures, "splits": {}, "metadata": {}}

    # ---------- بخش ۱: splitها ----------
    print("\n" + "-" * 78)
    print("split های رسمی — تعداد نمونه و رویداد")
    print("-" * 78)
    total = 0
    all_chip_keys = {}
    for name in ("flood_train_data.csv", "flood_valid_data.csv",
                 "flood_test_data.csv", "flood_bolivia_data.csv"):
        if name not in got:
            continue
        rows, events, chips = read_split(got[name][0])
        total += len(rows)
        split = name.replace("flood_", "").replace("_data.csv", "")
        all_chip_keys[split] = {f"{e}_{c}" for e, c, _ in chips}
        summary["splits"][split] = {"n": len(rows), "events": dict(sorted(events.items()))}
        print(f"\n{split:>8}: n={len(rows):>4}  رویداد={len(events)}")
        print("          " + " · ".join(f"{k}:{v}" for k, v in sorted(events.items())))
    print(f"\nجمع کل chipهای دست‌برچسب: {total}")
    summary["total_handlabeled"] = total

    # نشت بین splitها — chip مشترک؟ (معادل اندازه‌گیری ۷۳.۵٪ آتش‌سوزی)
    keys = list(all_chip_keys)
    overlaps = {}
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ka, kb = keys[a], keys[b]
            inter = all_chip_keys[ka] & all_chip_keys[kb]
            overlaps[f"{ka}∩{kb}"] = len(inter)
    summary["chip_overlap_between_splits"] = overlaps
    print("\nهم‌پوشانی chip بین splitها: " + " · ".join(f"{k}={v}" for k, v in overlaps.items()))

    # ---------- بخش ۲: 🔴 مسیر بحرانی — تاریخ ----------
    print("\n" + "-" * 78)
    print("🔴 مسیر بحرانی — تاریخ")
    print("-" * 78)
    gj_name = "Sen1Floods11_Metadata.geojson"
    if gj_name not in got:
        print("🔴 متادیتا نیامد → سؤال تاریخ بی‌جواب ماند. جلو نرو.")
    else:
        gj = json.loads(got[gj_name][0].read_text(encoding="utf-8"))
        feats = gj.get("features", [])
        print(f"تعداد feature: {len(feats)}")
        if feats:
            print(f"فیلدها: {sorted(feats[0].get('properties', {}).keys())}")
            print(f"نوع هندسه: {feats[0].get('geometry', {}).get('type')}")
        rows = []
        for f in feats:
            p = f.get("properties", {})
            rows.append({k: p.get(k) for k in
                         ("ID", "location", "ISO_CC", "s1_date", "s2_date",
                          "train_chip", "val_chip")})
        summary["metadata"]["n_features"] = len(feats)
        summary["metadata"]["fields"] = sorted(feats[0].get("properties", {}).keys()) if feats else []
        summary["metadata"]["events"] = rows
        print(f"\n{'ID':>4} {'location':<14} {'s1_date':<12} {'s2_date':<12} {'train':>6} {'val':>5}")
        for r in rows:
            print(f"{str(r['ID']):>4} {str(r['location']):<14} {str(r['s1_date']):<12} "
                  f"{str(r['s2_date']):<12} {str(r['train_chip']):>6} {str(r['val_chip']):>5}")
        n_dates = sum(1 for r in rows if r["s2_date"])
        print(f"\nرویداد با s2_date موجود: {n_dates} از {len(rows)}")
        summary["metadata"]["events_with_s2_date"] = n_dates

        # ---------- بخش ۳: تلهٔ نام — split vs metadata ----------
        print("\n" + "-" * 78)
        print("تلهٔ نام — رویدادهای split در برابر فیلد location متادیتا")
        print("-" * 78)
        split_events = set()
        for s in summary["splits"].values():
            split_events |= set(s["events"])
        meta_locs = {str(r["location"]) for r in rows}
        only_split = sorted(split_events - meta_locs)
        only_meta = sorted(meta_locs - split_events)
        print(f"در split ولی نه در متادیتا: {only_split or '— هیچ'}")
        print(f"در متادیتا ولی نه در split: {only_meta or '— هیچ'}")
        summary["name_mismatch"] = {"only_in_split": only_split, "only_in_metadata": only_meta}

    wall = time.time() - t0
    summary["wall_seconds"] = round(wall, 2)
    (OUT / "recon_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print(f"زمان دیواری: {wall:.1f} ثانیه  ·  خلاصه: {OUT / 'recon_summary.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()

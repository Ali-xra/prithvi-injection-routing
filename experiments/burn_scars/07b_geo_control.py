# -*- coding: utf-8 -*-
"""
07b_geo_control.py — آیا آب‌وهوا واقعاً چیزی به جغرافیا اضافه می‌کند؟
======================================================================
نسخه: v1 · تاریخ: 2026-07-29

چرا این فایل وجود دارد — یک نقص در طراحی `07`:
    `07` نتیجه گرفت «سیگنال فراتر از جغرافیاست»، بر اساس این عدد:
        AUC(W+G) − AUC(G) = 0.7162 − 0.6946 = **+0.0216**
    و آستانهٔ من `0.02` بود. یعنی با اختلاف ۰.۰۰۱۶ رد شد.

    آن `0.02` **از هیچ‌جا نیامده بود.** برای هر مجموعه به‌تنهایی توزیع پوچ ساختم،
    ولی برای خودِ **افزوده** نساختم. اگر پرسیده شود «این ۰.۰۲۱ از نویز جدا
    هست؟»، جوابی نداریم. این فایل آن جواب را می‌سازد.

آزمون درست — جایگشت بلوکی فقط روی ستون‌های جوّی:
    ستون‌های جغرافیا **دست‌نخورده** می‌مانند. شش ستون جوّی به‌صورت **یک بلوک**
    بین نمونه‌ها جابه‌جا می‌شوند (یک جایگشت واحد برای هر شش ستون).

    ⚠️ چرا بلوکی و نه ستون‌به‌ستون: جایگشت مستقلِ هر ستون، همبستگی بین خودِ
    ویژگی‌ها را هم نابود می‌کند (مثلاً بارش بالا با دمای پایین می‌آید). آن‌وقت
    توزیع پوچ «خیلی پوچ‌تر» از واقعیت می‌شود و آزمون به‌ناحق آسان.
    جایگشت بلوکی فقط پیوند **آب‌وهوا ↔ هدف** و **آب‌وهوا ↔ جغرافیا** را می‌شکند
    و ساختار داخلی آب‌وهوا را نگه می‌دارد.

    در train و val هر دو جایگشت می‌شود (با جایگشت‌های مستقل) — یعنی سناریوی
    «اعداد جوّی صرفاً اعداد تصادفی با همان توزیع‌اند».

فرضیهٔ صفر: آب‌وهوا هیچ اطلاعاتی فراتر از مکان و فصل ندارد.
    اگر p < 0.05 → فرضیهٔ صفر رد می‌شود، ادعا واقعی است.
    اگر p ≥ 0.05 → آن +0.0216 از نویز جدا نیست. **قبل از GPU باید بدانیم.**

⚠️ این نسخهٔ ارزان و بدون GPUِ همان بازوی P4 (شافل) است. اگر اینجا رد شود،
   انتظار زیادی از بازوی adaLN نداشته باش.

test باز نمی‌شود.

خروجی: <BIG>/data/meta/geo_control.json
اجرا:  python 07b_geo_control.py [--perms 300]
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
import time
import argparse
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("⛔ scikit-learn نصب نیست:  pip install scikit-learn")
    sys.exit(1)

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
META = BIG / "data" / "meta"
IN_CSV = META / "wind_features.csv"
OUT_JSON = META / "geo_control.json"

SEED = 0
W_FEATS = ["mean_speed_z", "max_speed_z", "dir_sin_z", "dir_cos_z",
           "precip_7d_log_z", "mean_temp_z"]


def load():
    rows = list(csv.DictReader(IN_CSV.open(encoding="utf-8")))
    for r in rows:
        for k in W_FEATS:
            r[k] = float(r[k])
        r["pct_burn"] = float(r["pct_burn"])
        r["lat_center"] = float(r["lat_center"])
        r["lon_center"] = float(r["lon_center"])
        r["doy"] = int(r["doy"])
    return rows


def make_blocks(rows):
    """W و G را جدا می‌سازد تا بتوان فقط W را جایگشت داد."""
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]

    def geo_raw(r):
        a = 2 * np.pi * r["doy"] / 365.25
        return [r["lat_center"], r["lon_center"], np.sin(a), np.cos(a)]

    Gtr_raw = np.array([geo_raw(r) for r in tr], dtype=float)
    mu, sd = Gtr_raw.mean(axis=0), Gtr_raw.std(axis=0)
    sd[sd < 1e-9] = 1.0

    Gtr = (Gtr_raw - mu) / sd
    Gva = (np.array([geo_raw(r) for r in va], dtype=float) - mu) / sd
    Wtr = np.array([[r[k] for k in W_FEATS] for r in tr], dtype=float)
    Wva = np.array([[r[k] for k in W_FEATS] for r in va], dtype=float)

    thr = float(np.median([r["pct_burn"] for r in tr]))
    ytr = np.array([int(r["pct_burn"] > thr) for r in tr])
    yva = np.array([int(r["pct_burn"] > thr) for r in va])
    return Wtr, Gtr, ytr, Wva, Gva, yva, thr


def model(kind, seed):
    if kind == "linear":
        return LogisticRegression(max_iter=2000, random_state=seed)
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                  random_state=seed, n_jobs=-1)


def auc_of(Xtr, ytr, Xva, yva, kind, seed):
    m = model(kind, seed)
    m.fit(Xtr, ytr)
    return float(roc_auc_score(yva, m.predict_proba(Xva)[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=300)
    args = ap.parse_args()

    if not IN_CSV.exists():
        print(f"⛔ {IN_CSV} نیست — اول 05b را اجرا کن")
        return

    rows = load()
    Wtr, Gtr, ytr, Wva, Gva, yva, thr = make_blocks(rows)
    print(f"train {len(ytr)} · val {len(yva)} · آستانه {thr:.3f}٪")
    print(f"جایگشت: {args.perms} بار · فقط بلوک جوّی، جغرافیا دست‌نخورده\n")

    rng = np.random.default_rng(SEED)
    results = {}

    for kind, label in (("linear", "خطی"), ("forest", "جنگل")):
        t0 = time.time()
        a_g = auc_of(Gtr, ytr, Gva, yva, kind, SEED)
        a_wg = auc_of(np.hstack([Wtr, Gtr]), ytr,
                      np.hstack([Wva, Gva]), yva, kind, SEED)
        delta = a_wg - a_g

        null = []
        for i in range(args.perms):
            ptr = rng.permutation(len(ytr))
            pva = rng.permutation(len(yva))
            d = auc_of(np.hstack([Wtr[ptr], Gtr]), ytr,
                       np.hstack([Wva[pva], Gva]), yva, kind, SEED) - a_g
            null.append(d)
            if (i + 1) % 50 == 0:
                print(f"   {label}: {i+1}/{args.perms}  ({time.time()-t0:.0f}s)",
                      flush=True)
        null = np.array(null)

        p = float((np.sum(null >= delta) + 1) / (len(null) + 1))
        results[kind] = {
            "auc_geo_only": a_g, "auc_weather_plus_geo": a_wg,
            "delta_observed": delta,
            "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "null_q95": float(np.quantile(null, 0.95)),
            "p_value": p, "significant": bool(p < 0.05),
            "seconds": round(time.time() - t0, 1),
        }
        print()

    print("=" * 84)
    print(f"{'مدل':<10}{'AUC(G)':>10}{'AUC(W+G)':>11}{'افزوده':>10}"
          f"{'میانگین پوچ':>13}{'صدک۹۵':>10}{'p':>8}")
    print("-" * 84)
    for kind, label in (("linear", "خطی"), ("forest", "جنگل")):
        r = results[kind]
        mark = "✅" if r["significant"] else "❌"
        print(f"{label:<10}{r['auc_geo_only']:>10.4f}{r['auc_weather_plus_geo']:>11.4f}"
              f"{r['delta_observed']:>+10.4f}{r['null_mean']:>+13.4f}"
              f"{r['null_q95']:>+10.4f}{r['p_value']:>8.4f}  {mark}")
    print("=" * 84)

    any_sig = any(results[k]["significant"] for k in results)
    best = max(results, key=lambda k: results[k]["delta_observed"])
    rb = results[best]

    print("\nحکم")
    print("-" * 84)
    if any_sig:
        print("  🟢 **آب‌وهوا واقعاً چیزی فراتر از جغرافیا و فصل می‌داند.**")
        print(f"     بهترین: {best} · افزوده {rb['delta_observed']:+.4f} · p={rb['p_value']:.4f}")
        print("     حالا ادعای اسلاید یک آزمون آماری پشتش دارد، نه یک آستانهٔ دلبخواهی.")
        print("     👉 آمادهٔ رفتن به فاز GPU.")
    else:
        print("  🔴 **افزودهٔ آب‌وهوا از نویز جدا نیست.**")
        print(f"     افزودهٔ مشاهده‌شده {rb['delta_observed']:+.4f} در برابر صدک۹۵ پوچ "
              f"{rb['null_q95']:+.4f} · p={rb['p_value']:.4f}")
        print()
        print("     این ایده را **باطل نمی‌کند** ولی سه چیز را عوض می‌کند:")
        print("     ۱) در ارائه بگو «اثر کوچک است و از جغرافیا جدا نشد» — قبل از اینکه بپرسند")
        print("     ۲) ادعای اصلی می‌شود «**کجا** تزریق شود»، نه «چقدر کمک می‌کند»")
        print("     ۳) بازوی شافل‌شده (P4) از همه مهم‌تر می‌شود — چون احتمالاً")
        print("        هر بهبودی که در ماتریس ببینیم، با شافل هم می‌ماند")
        print()
        print("     🔴 و مهم‌ترین نتیجه: **هنوز نرو سراغ GPU.** اول تصمیم بگیر که")
        print("        آیا هدفِ آزمایش را عوض می‌کنی یا با همین انتظارِ کم جلو می‌روی.")

    results["_meta"] = {
        "n_train": int(len(ytr)), "n_val": int(len(yva)),
        "threshold_pct_burn": thr, "n_permutations": args.perms,
        "permutation": "block over 6 weather columns; geography untouched",
        "test_untouched": True,
    }
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n💾 {OUT_JSON}")


if __name__ == "__main__":
    main()

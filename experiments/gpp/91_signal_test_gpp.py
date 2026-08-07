# -*- coding: utf-8 -*-
"""
91_signal_test_gpp.py — دروازهٔ سیگنال GPP · بدون آموزش هیچ مدلی
================================================================
نسخه: v1 · تاریخ: 2026-07-30 · کار `flood`
مبنا: `../lab/src/07b_geo_control.py` (2026-07-29) — کپی و وصله، همان منطق
      جایگشت بلوکی. ⚠️ خط لولهٔ سیل نیست؛ دروازهٔ انتخاب تسک است.

سؤال: ده اسکالر MERRA-2 چیزی می‌گویند که **سایت، فصل، و خودِ تصویر** نمی‌گویند؟

سه بلوک کنترل — از ضعیف به قوی:
    C1  فصل            doy_sin/cos
    C2  + جغرافیای بوم  PFT (شش نوع) + Köppen (پنج) + فصل
    C3  + پروکسی تصویر  شش میانگین باند + هفت شاخص گیاهی + C2   ← 🔴 مهم‌ترین
    C3 نزدیک‌ترین چیز به سؤال واقعی پژوهش است: آیا دادهٔ غیرتصویری چیزی به
    مدلی که **تصویر را می‌بیند** اضافه می‌کند؟ اگر نه، تزریق بی‌فایده است.

دو split — هر دو گزارش می‌شوند:
    year  سال‌محور رسمی (test = ۲۰۲۱) — قابل مقایسه با مقاله، ولی ۷۹٪ نشت سایت
    site  سایت‌محور (seed 0، ۷۰/۳۰ روی SITE_ID) — بدون نشت، عدد صادقانه

🔒 آستانه‌ها — پیش از دیدن هر نتیجه‌ای نوشته شدند (پیش‌ثبت):
    قبول = delta_AUC ≥ 0.02  **و**  p < 0.05
    دروازهٔ اصلی: بلوک C3، split سایت‌محور، هر دو مدل هم‌جهت
    اگر C3 رد شود ولی C2 قبول شود → یعنی آب‌وهوا فقط چیزی را می‌گوید که تصویر
    از قبل می‌داند؛ برای «تزریق» شاهد ضعیفی است و باید صریح گزارش شود.

فرضیهٔ صفر: آب‌وهوا هیچ اطلاعاتی فراتر از بلوک کنترل ندارد.
جایگشت **بلوکی** روی ده ستون جوّی؛ بلوک کنترل دست‌نخورده (دلیلش در سربرگ 07b).
نرمال‌سازی z فقط از train. `test` سال‌محور جدا نمی‌شود چون این آزمون val-محور است.

خروجی: <BIG>/_recon-gpp/gpp_signal_test.json
اجرا:  python 91_signal_test_gpp.py [--perms 200]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("⛔ scikit-learn نصب نیست:  pip install scikit-learn")
    sys.exit(1)

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing-flood" / "_recon-gpp"
IN_CSV = BIG / "data_train_hls_37sites_v0_1.csv"
OUT_JSON = BIG / "gpp_signal_test.json"

SEED = 0
GATE_DELTA, GATE_P = 0.02, 0.05

W_FEATS = ["T2MIN", "T2MAX", "T2MEAN", "TSMDEWMEAN", "GWETROOT",
           "LHLAND", "SHLAND", "SWLAND", "PARDFLAND", "PRECTOTLAND"]
IMG_FEATS = ["b2", "b3", "b4", "b5", "b6", "b7",
             "NDVI", "EVI", "GCI", "NDWI", "NIRv", "kNDVI"]
PFTS = ["CRO", "DBF", "ENF", "GRA", "SH-SA", "WET"]
KOPPENS = ["Arid", "Cold", "Polar", "Temperate", "Tropical"]


def load():
    rows = list(csv.DictReader(IN_CSV.open(encoding="utf-8")))
    good = []
    for r in rows:
        try:
            for k in W_FEATS + IMG_FEATS + ["GPP"]:
                r[k] = float(r[k])
            r["doy"] = int(r["doy"])
        except (ValueError, TypeError, KeyError):
            continue                      # سطر ناقص — شمرده و اعلام می‌شود
        good.append(r)
    dropped = len(rows) - len(good)
    if dropped:
        print(f"⚠️ {dropped} سطر با مقدار ناقص کنار گذاشته شد (از {len(rows)})")
    return good, dropped


def blocks(r):
    a = 2 * np.pi * r["doy"] / 365.25
    season = [np.sin(a), np.cos(a)]
    eco = [1.0 if r["PFT"] == p else 0.0 for p in PFTS] + \
          [1.0 if r["koppen"] == k else 0.0 for k in KOPPENS]
    img = [r[k] for k in IMG_FEATS]
    return season, eco, img


def build(rows, tr_idx, va_idx, control):
    """control: 'C1' فصل · 'C2' +بوم · 'C3' +پروکسی تصویر"""
    S, E, I = zip(*[blocks(r) for r in rows])
    S, E, I = np.array(S), np.array(E), np.array(I)
    if control == "C1":
        G = S
    elif control == "C2":
        G = np.hstack([S, E])
    else:
        G = np.hstack([S, E, I])
    W = np.array([[r[k] for k in W_FEATS] for r in rows], dtype=float)

    def zfit(X, idx):
        mu, sd = X[idx].mean(0), X[idx].std(0)
        sd[sd < 1e-9] = 1.0
        return (X - mu) / sd

    W, G = zfit(W, tr_idx), zfit(G, tr_idx)          # z فقط از train
    y_raw = np.array([r["GPP"] for r in rows])
    thr = float(np.median(y_raw[tr_idx]))            # آستانه فقط از train
    y = (y_raw > thr).astype(int)
    return W[tr_idx], G[tr_idx], y[tr_idx], W[va_idx], G[va_idx], y[va_idx], thr


def split_year(rows, test_year="2021"):
    tr = [i for i, r in enumerate(rows) if r["year"] != test_year]
    va = [i for i, r in enumerate(rows) if r["year"] == test_year]
    return np.array(tr), np.array(va)


def split_site(rows, frac_val=0.30):
    """سایت‌محور، seed 0. انتخاب **مستقل از هدف** — هیچ سایتی بر اساس GPP جدا نمی‌شود."""
    sites = sorted({r["SITE_ID"] for r in rows})
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(sites))
    n_val = max(1, int(round(frac_val * len(sites))))
    val_sites = {sites[i] for i in order[:n_val]}
    tr = [i for i, r in enumerate(rows) if r["SITE_ID"] not in val_sites]
    va = [i for i, r in enumerate(rows) if r["SITE_ID"] in val_sites]
    return np.array(tr), np.array(va), sorted(val_sites)


def model(kind, seed):
    if kind == "linear":
        return LogisticRegression(max_iter=3000, random_state=seed)
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                  random_state=seed, n_jobs=-1)


def auc_of(Xtr, ytr, Xva, yva, kind, seed):
    m = model(kind, seed)
    m.fit(Xtr, ytr)
    return float(roc_auc_score(yva, m.predict_proba(Xva)[:, 1]))


def run(rows, tr, va, control, kind, perms, rng, label):
    Wtr, Gtr, ytr, Wva, Gva, yva, thr = build(rows, tr, va, control)
    if len(set(ytr)) < 2 or len(set(yva)) < 2:
        return {"error": "یک کلاس خالی — آزمون ممکن نیست"}
    a_g = auc_of(Gtr, ytr, Gva, yva, kind, SEED)
    a_wg = auc_of(np.hstack([Wtr, Gtr]), ytr, np.hstack([Wva, Gva]), yva, kind, SEED)
    delta = a_wg - a_g
    t0, null = time.time(), []
    for i in range(perms):
        ptr, pva = rng.permutation(len(ytr)), rng.permutation(len(yva))
        null.append(auc_of(np.hstack([Wtr[ptr], Gtr]), ytr,
                           np.hstack([Wva[pva], Gva]), yva, kind, SEED) - a_g)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"      {label}: {i+1}/{perms} · {el:.0f}s · "
                  f"تخمین باقی {el/(i+1)*(perms-i-1):.0f}s", flush=True)
    null = np.array(null)
    p = float((np.sum(null >= delta) + 1) / (len(null) + 1))
    return {"n_train": int(len(ytr)), "n_val": int(len(yva)), "gpp_median_thr": thr,
            "auc_control": a_g, "auc_control_plus_weather": a_wg, "delta": delta,
            "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "p_value": p, "passes": bool(delta >= GATE_DELTA and p < GATE_P),
            "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    args = ap.parse_args()
    if not IN_CSV.exists():
        print(f"⛔ {IN_CSV} نیست — اول 90_recon_gpp.py را اجرا کن")
        return
    t_all = time.time()
    rows, dropped = load()
    print("=" * 78)
    print(f"دروازهٔ سیگنال GPP · {len(rows)} نمونه · {args.perms} جایگشت")
    print(f"🔒 آستانه‌های پیش‌ثبت‌شده: delta ≥ {GATE_DELTA} و p < {GATE_P}")
    print("=" * 78)

    tr_y, va_y = split_year(rows)
    tr_s, va_s, val_sites = split_site(rows)
    splits = {"year": (tr_y, va_y), "site": (tr_s, va_s)}
    print(f"split سال‌محور: train {len(tr_y)} · val {len(va_y)} (test_year=2021)")
    print(f"split سایت‌محور: train {len(tr_s)} · val {len(va_s)} · "
          f"{len(val_sites)} سایت در val: {', '.join(val_sites)}")

    out = {"n_rows": len(rows), "rows_dropped": dropped, "perms": args.perms,
           "gates": {"delta": GATE_DELTA, "p": GATE_P}, "seed": SEED,
           "val_sites": val_sites, "results": {}}
    rng = np.random.default_rng(SEED)

    for sname, (tr, va) in splits.items():
        for control in ("C1", "C2", "C3"):
            for kind in ("linear", "forest"):
                key = f"{sname}|{control}|{kind}"
                print(f"\n▶ {key}")
                res = run(rows, tr, va, control, kind, args.perms, rng, key)
                out["results"][key] = res
                if "error" in res:
                    print(f"   ⛔ {res['error']}")
                    continue
                print(f"   AUC کنترل {res['auc_control']:.4f} → با آب‌وهوا "
                      f"{res['auc_control_plus_weather']:.4f} · delta "
                      f"{res['delta']:+.4f} · صدک۹۵ پوچ {res['null_p95']:+.4f} · "
                      f"p={res['p_value']:.4f} · {'✅ قبول' if res['passes'] else '❌ رد'}")

    print("\n" + "=" * 78)
    print("خلاصه — دروازهٔ اصلی: C3 روی split سایت‌محور، هر دو مدل")
    print("=" * 78)
    print(f"{'کلید':<24}{'AUC کنترل':>11}{'+آب‌وهوا':>11}{'delta':>9}{'p':>9}  حکم")
    for k, r in out["results"].items():
        if "error" in r:
            print(f"{k:<24}{'—':>11}{'—':>11}{'—':>9}{'—':>9}  ⛔")
            continue
        print(f"{k:<24}{r['auc_control']:>11.4f}{r['auc_control_plus_weather']:>11.4f}"
              f"{r['delta']:>+9.4f}{r['p_value']:>9.4f}  "
              f"{'✅' if r['passes'] else '❌'}")

    gate = [out["results"].get(f"site|C3|{m}", {}) for m in ("linear", "forest")]
    out["primary_gate_passes"] = all(g.get("passes") for g in gate)
    out["wall_seconds_total"] = round(time.time() - t_all, 1)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n🚦 دروازهٔ اصلی: {'✅ قبول' if out['primary_gate_passes'] else '❌ رد'}"
          f"  ·  کل زمان دیواری {out['wall_seconds_total']}s")
    print(f"خروجی: {OUT_JSON}")


if __name__ == "__main__":
    main()

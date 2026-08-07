# -*- coding: utf-8 -*-
"""
10_signal_test_flood.py — 🚦 دروازهٔ سیگنال سیل · عدد سوم
=========================================================
نسخه: v1 · تاریخ: 2026-07-30 · کار `flood`
مبنا: `../lab/src/07b_geo_control.py` + درس‌های دروازهٔ GPP (`91_signal_test_gpp.py`)

سؤال: شش عدد جوّی چیزی می‌گویند که **رویداد، مکان، و خودِ تصویر** نمی‌گویند؟

## دو طرح — و تفاوتشان بنیادی است

**D1 «اطلاعات درون‌رویدادی»** — پیشنهاد `SYNC.md` بخش ۸.
    کنترل شامل **one-hot رویداد** است، یعنی قوی‌ترین کنترل ممکن برای مکان و زمان.
    split تصادفی **طبقه‌بندی‌شده بر اساس رویداد** (۷۰/۳۰) تا هر رویداد در دو طرف
    باشد. جایگشت **درون رویداد** انجام می‌شود.
    ⚠️ این طرح **تعمیم را نمی‌سنجد** — محتوای اطلاعاتی را می‌سنجد. با ۱۱ تاریخ،
    این تنها راهی است که هم‌خطی «رویداد ↔ آب‌وهوا» را برمی‌دارد.

**D2 «تعمیم»** — split قفل‌شدهٔ رویدادمحور (۳۱۵/۶۸).
    one-hot رویداد **قابل استفاده نیست** (رویدادهای train و val مجزا هستند).
    جایگشت سراسری. این طرح همان چیزی است که مدل واقعی با آن روبروست.

## سه بلوک کنترل — از ضعیف به قوی

    B1  رویداد (D1) یا فصل+مکان (D2)
    B2  + lat/lon
    B3  + **پروکسی تصویر**: شش میانگین باند + NDVI/NDWI/MNDWI/EVI + pct_nodata
        🔴 B3 دروازهٔ واقعی است. در GPP همین بلوک آب‌وهوا را صفر کرد.
        `pct_nodata` عمداً داخل کنترل است: ۴۲٪ نمونه‌ها بیش از ۵٪ بی‌داده دارند و
        اگر ابر با بارش همبسته باشد، «بی‌داده بودن» مسیر نشت می‌سازد.

## دو هدف
    T1  presence  `pct_water > 0`            ← ۵۴ نمونهٔ منفی خالص. آتش‌سوزی نداشت
    T2  extent    `pct_water > میانهٔ train`  ← معادل `pct_burn` آتش‌سوزی

🔒 **آستانه‌ها پیش از دیدن هر نتیجه‌ای:** قبول = `delta_AUC ≥ 0.02` **و** `p < 0.05`.
   **دروازهٔ اصلی: D1 · T2 · B3 · هر دو مدل.** بقیه گزارش می‌شوند.
   و یک واریانت تشخیصی: افزودن `precip_30d_log_z` به بلوک جوّی (r با ۷ روزه = ۰.۳۲).

`test` باز نمی‌شود. جایگشت بلوکی است نه ستون‌به‌ستون (دلیلش در سربرگ `07b`).

خروجی: <BIG>/data/meta/flood_signal_test.json
اجرا:  python 10_signal_test_flood.py [--perms 200]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from sklearn.ensemble import RandomForestClassifier          # noqa: E402
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.metrics import roc_auc_score                    # noqa: E402

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing-flood"
META = BIG / "data" / "meta"
COND = META / "conditioning_v1.csv"
PROXY = META / "image_proxy.csv"
FEATS = META / "wind_features.csv"
OUT_JSON = META / "flood_signal_test.json"

SEED = 0
GATE_DELTA, GATE_P = 0.02, 0.05
W6 = ["mean_speed_z", "max_speed_z", "dir_sin_z", "dir_cos_z",
      "precip_7d_log_z", "mean_temp_z"]
IMG = ["blue_mean", "green_mean", "red_mean", "nir_mean", "swir1_mean", "swir2_mean",
       "NDVI", "NDWI", "MNDWI", "EVI"]


def load():
    cond = {r["filename"].replace("_S2Hand.tif", ""): r
            for r in csv.DictReader(COND.open(encoding="utf-8"))}
    proxy = {r["stem"]: r for r in csv.DictReader(PROXY.open(encoding="utf-8"))}
    extra = {r["filename"].replace("_S2Hand.tif", ""): r
             for r in csv.DictReader(FEATS.open(encoding="utf-8"))}
    rows, dropped = [], 0
    for stem, c in cond.items():
        if stem not in proxy:                    # پنج chip بی‌پیکسل برچسب‌دار
            dropped += 1
            continue
        r = {**c, **proxy[stem]}
        r["precip_30d_log"] = float(extra[stem]["precip_30d_log"])
        for k in W6 + IMG:
            r[k] = float(r[k])
        r["pct_water"] = float(r["pct_water"])
        r["pct_nodata"] = float(r["pct_nodata"])
        r["lat_z"], r["lon_z"] = float(r["lat_z"]), float(r["lon_z"])
        r["doy_sin_z"], r["doy_cos_z"] = float(r["doy_sin_z"]), float(r["doy_cos_z"])
        rows.append(r)
    return rows, dropped


def z_from(X, idx):
    mu, sd = X[idx].mean(0), X[idx].std(0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd


def build(rows, tr, va, design, block, add30):
    events = sorted({r["event"] for r in rows})
    onehot = np.array([[1.0 if r["event"] == e else 0.0 for e in events] for r in rows])
    geo = np.array([[r["lat_z"], r["lon_z"]] for r in rows])
    season = np.array([[r["doy_sin_z"], r["doy_cos_z"]] for r in rows])
    img = z_from(np.array([[r[k] for k in IMG] + [r["pct_nodata"]] for r in rows]), tr)

    base = onehot if design == "D1" else season
    G = {"B1": base,
         "B2": np.hstack([base, geo]),
         "B3": np.hstack([base, geo, img])}[block]

    wcols = W6 + (["precip_30d_log"] if add30 else [])
    W = z_from(np.array([[r[k] for k in wcols] for r in rows]), tr)
    return W, G


def targets(rows, tr):
    w = np.array([r["pct_water"] for r in rows])
    thr = float(np.median(w[tr]))
    return {"T1_presence": (w > 0).astype(int),
            "T2_extent": (w > thr).astype(int)}, thr


def model(kind):
    if kind == "linear":
        return LogisticRegression(max_iter=4000, random_state=SEED)
    return RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
                                  random_state=SEED, n_jobs=-1)


def auc_of(Xtr, ytr, Xva, yva, kind):
    m = model(kind)
    m.fit(Xtr, ytr)
    return float(roc_auc_score(yva, m.predict_proba(Xva)[:, 1]))


def perm_within_event(idx, events_of, rng):
    """جایگشت **درون رویداد** — ساختار سطح رویداد دست‌نخورده می‌ماند."""
    out = np.array(idx, copy=True)
    by = {}
    for pos, i in enumerate(idx):
        by.setdefault(events_of[i], []).append(pos)
    for _, poss in by.items():
        p = rng.permutation(len(poss))
        out[poss] = np.array(idx)[np.array(poss)[p]]
    return out


def split_D1(rows, rng):
    """۷۰/۳۰ طبقه‌بندی‌شده بر اساس رویداد — هر رویداد در دو طرف حاضر است."""
    tr, va = [], []
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["event"], []).append(i)
    for _, ids in sorted(by.items()):
        ids = np.array(ids)[rng.permutation(len(ids))]
        k = max(1, int(round(0.30 * len(ids))))
        va += list(ids[:k])
        tr += list(ids[k:])
    return np.array(sorted(tr)), np.array(sorted(va))


def split_D2(rows):
    tr = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    va = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])
    return tr, va


def run(rows, tr, va, design, block, kind, y, perms, rng, events_of, add30, label):
    W, G = build(rows, tr, va, design, block, add30)
    ytr, yva = y[tr], y[va]
    if len(set(ytr)) < 2 or len(set(yva)) < 2:
        return {"error": "یک کلاس خالی"}
    a_g = auc_of(G[tr], ytr, G[va], yva, kind)
    a_wg = auc_of(np.hstack([W, G])[tr], ytr, np.hstack([W, G])[va], yva, kind)
    delta = a_wg - a_g
    t0, null = time.time(), []
    for i in range(perms):
        if design == "D1":
            ptr = perm_within_event(tr, events_of, rng)
            pva = perm_within_event(va, events_of, rng)
        else:
            ptr = np.array(tr)[rng.permutation(len(tr))]
            pva = np.array(va)[rng.permutation(len(va))]
        Xtr = np.hstack([W[ptr], G[tr]])
        Xva = np.hstack([W[pva], G[va]])
        null.append(auc_of(Xtr, ytr, Xva, yva, kind) - a_g)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"      {label}: {i+1}/{perms} · {el:.0f}s · "
                  f"باقی ~{el/(i+1)*(perms-i-1):.0f}s", flush=True)
    null = np.array(null)
    p = float((np.sum(null >= delta) + 1) / (len(null) + 1))
    return {"n_train": int(len(ytr)), "n_val": int(len(yva)),
            "pos_rate_val": round(float(yva.mean()), 3),
            "auc_control": a_g, "auc_control_plus_weather": a_wg, "delta": delta,
            "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "p_value": p, "passes": bool(delta >= GATE_DELTA and p < GATE_P),
            "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    args = ap.parse_args()
    t_all = time.time()
    rows, dropped = load()
    events_of = [r["event"] for r in rows]
    rng = np.random.default_rng(SEED)

    print("=" * 86)
    print(f"🚦 دروازهٔ سیگنال سیل · {len(rows)} نمونه (۵ chip بی‌پیکسل برچسب‌دار کنار گذاشته شد)")
    print(f"🔒 آستانه‌های پیش‌ثبت‌شده: delta ≥ {GATE_DELTA} و p < {GATE_P}")
    print(f"   دروازهٔ اصلی: D1 · T2_extent · B3 · هر دو مدل")
    print("=" * 86)

    tr1, va1 = split_D1(rows, np.random.default_rng(SEED))
    tr2, va2 = split_D2(rows)
    y1, thr1 = targets(rows, tr1)
    y2, thr2 = targets(rows, tr2)
    print(f"D1 درون‌رویدادی: train {len(tr1)} · val {len(va1)} · آستانهٔ آب {thr1:.2f}٪")
    print(f"D2 تعمیم       : train {len(tr2)} · val {len(va2)} · آستانهٔ آب {thr2:.2f}٪")

    out = {"n_rows": len(rows), "dropped_no_label_px": dropped, "perms": args.perms,
           "gates": {"delta": GATE_DELTA, "p": GATE_P}, "seed": SEED,
           "primary": "D1|T2_extent|B3", "results": {}}

    plan = [("D1", tr1, va1, y1), ("D2", tr2, va2, y2)]
    for design, tr, va, ys in plan:
        for tname, y in ys.items():
            for block in ("B1", "B2", "B3"):
                for kind in ("linear", "forest"):
                    key = f"{design}|{tname}|{block}|{kind}"
                    print(f"\n▶ {key}")
                    r = run(rows, tr, va, design, block, kind, y,
                            args.perms, rng, events_of, False, key)
                    out["results"][key] = r
                    if "error" in r:
                        print(f"   ⛔ {r['error']}")
                        continue
                    print(f"   AUC {r['auc_control']:.4f} → {r['auc_control_plus_weather']:.4f}"
                          f" · delta {r['delta']:+.4f} · پوچ۹۵ {r['null_p95']:+.4f}"
                          f" · p={r['p_value']:.4f} · {'✅' if r['passes'] else '❌'}")

    # واریانت تشخیصی: افزودن پنجرهٔ ۳۰ روزه به بلوک جوّی
    print("\n" + "-" * 86)
    print("واریانت تشخیصی — افزودن precip_30d_log به بلوک جوّی (بیرون از بردار قفل‌شده)")
    print("-" * 86)
    for design, tr, va, ys in plan:
        for kind in ("linear", "forest"):
            key = f"{design}|T2_extent|B3|{kind}|+30d"
            print(f"\n▶ {key}")
            r = run(rows, tr, va, design, "B3", kind, ys["T2_extent"],
                    args.perms, rng, events_of, True, key)
            out["results"][key] = r
            if "error" not in r:
                print(f"   delta {r['delta']:+.4f} · p={r['p_value']:.4f}"
                      f" · {'✅' if r['passes'] else '❌'}")

    print("\n" + "=" * 86)
    print(f"{'کلید':<34}{'AUC کنترل':>11}{'+جوّی':>10}{'delta':>10}{'p':>9}  حکم")
    print("=" * 86)
    for k, r in out["results"].items():
        if "error" in r:
            print(f"{k:<34}{'—':>11}{'—':>10}{'—':>10}{'—':>9}  ⛔")
            continue
        print(f"{k:<34}{r['auc_control']:>11.4f}{r['auc_control_plus_weather']:>10.4f}"
              f"{r['delta']:>+10.4f}{r['p_value']:>9.4f}  {'✅' if r['passes'] else '❌'}")

    g = [out["results"].get(f"D1|T2_extent|B3|{m}", {}) for m in ("linear", "forest")]
    out["primary_gate_passes"] = all(x.get("passes") for x in g)
    out["wall_seconds_total"] = round(time.time() - t_all, 1)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n🚦 دروازهٔ اصلی: {'✅ قبول' if out['primary_gate_passes'] else '❌ رد'}"
          f"  ·  کل {out['wall_seconds_total']}s\n{OUT_JSON}")


if __name__ == "__main__":
    main()

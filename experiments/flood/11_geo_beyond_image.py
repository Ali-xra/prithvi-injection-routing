# -*- coding: utf-8 -*-
"""
11_geo_beyond_image.py — دو خانهٔ ❓ جدول: «مکان و تاریخ مطلق، فراتر از تصویر»
=============================================================================
نسخه: v1 · تاریخ: 2026-07-30 · کار `flood`
مبنا: `10_signal_test_flood.py` و `91_signal_test_gpp.py` + درخواست کار `burn`

## ایرادی که کار `burn` گرفت و وارد است

من در هر سه تسک **«آب‌وهوا فراتر از تصویر»** را سنجیدم و نتیجه هر سه بار ❌ شد.
ولی **«مکان و تاریخ مطلق فراتر از تصویر»** را جدا نسنجیده بودم. `burn` سنجید و روی
آتش‌سوزی **قبول** شد (`+0.0287`، p=۰.۰۱۰). پس جدول من دربارهٔ **آب‌وهوا** بود، نه
دربارهٔ «دادهٔ غیرتصویری».

قاعدهٔ دوطرفه‌ای که `burn` پیشنهاد کرد و این اسکریپت آزمونش می‌کند:
    دادهٔ کمکی که تصویر ضمنی دارد (باران → سبزی → NDVI) → **افزونه**
    دادهٔ کمکی که از پیکسل بازیابی نمی‌شود (مکان مطلق، تاریخ مطلق) → **مفید**

## چهار ترکیب، روی هر دیتاست

    A  تصویر            ← + مکان/تاریخ      «آیا مکان از تصویر بازیابی نمی‌شود؟»
    B  تصویر            ← + جوّی            (تکرار برای مقایسهٔ مستقیم)
    C  تصویر            ← + هر دو
    D  تصویر + مکان     ← + جوّی            🔴 قاطع‌ترین: جوّی بعد از جغرافیا

## ⚠️ محدودیت ساختاری سیل — پیش از دیدن نتیجه ثبت می‌شود

با **۱۱ رویداد**، `lat/lon` تقریباً **هویت رویداد** است. پس:
  · در split درون‌رویدادی (D1) مدل می‌تواند مکان را **حفظ** کند → ✅ ممکن است مصنوع باشد
  · در split رویدادمحور (D2) مقادیر `lat/lon` در val **دیده‌نشده**‌اند → ❌ اجباری
هر دو طرف منحرف‌اند. پس این اسکریپت **هم‌خطی را اندازه می‌گیرد** (AUC پیش‌بینی
رویداد از `lat/lon`) و نتیجه را کنار عدد می‌گذارد. اگر هم‌خطی کامل بود، جوابِ درست
«قابل تأیید نیست» است، نه ✅ و نه ❌.

## GPP
`lat/lon` در CSV دیتاست **نیست** (فقط `SITE_ID`). پس برای GPP فقط **تاریخ مطلق**
(`doy` sin/cos) سنجیده می‌شود و این صریح گزارش می‌شود. تاریخ در GPP به ازای chip
است و ۶۳۶ مقدار متمایز دارد، پس آزمونش معنادار است.

🔒 آستانه‌ها پیش از اجرا: `delta_AUC ≥ 0.02` و `p < 0.05`.
خروجی: <BIG>/data/meta/geo_beyond_image.json
اجرا:  python 11_geo_beyond_image.py [--perms 200]
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

HOME = Path.home() / "Desktop" / "big-files"
FBIG = HOME / "injection-routing-flood"
FMETA = FBIG / "data" / "meta"
GPP = FBIG / "_recon-gpp"
OUT_JSON = FMETA / "geo_beyond_image.json"

SEED = 0
GATE_DELTA, GATE_P = 0.02, 0.05
IMG_F = ["blue_mean", "green_mean", "red_mean", "nir_mean", "swir1_mean", "swir2_mean",
         "NDVI", "NDWI", "MNDWI", "EVI"]
W6 = ["mean_speed_z", "max_speed_z", "dir_sin_z", "dir_cos_z",
      "precip_7d_log_z", "mean_temp_z"]
G4 = ["lat_z", "lon_z", "doy_sin_z", "doy_cos_z"]
GPP_W = ["T2MIN", "T2MAX", "T2MEAN", "TSMDEWMEAN", "GWETROOT",
         "LHLAND", "SHLAND", "SWLAND", "PARDFLAND", "PRECTOTLAND"]
GPP_IMG = ["b2", "b3", "b4", "b5", "b6", "b7", "NDVI", "EVI", "GCI", "NDWI", "NIRv", "kNDVI"]


def z(X, idx):
    mu, sd = X[idx].mean(0), X[idx].std(0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd


def model(kind):
    if kind == "linear":
        return LogisticRegression(max_iter=4000, random_state=SEED)
    return RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
                                  random_state=SEED, n_jobs=-1)


def auc(Xtr, ytr, Xva, yva, kind):
    m = model(kind)
    m.fit(Xtr, ytr)
    return float(roc_auc_score(yva, m.predict_proba(Xva)[:, 1]))


def test(add, ctrl, y, tr, va, kind, perms, rng, label):
    """افزودهٔ `add` روی کنترل `ctrl` + توزیع پوچ با جایگشت بلوکی سراسری."""
    ytr, yva = y[tr], y[va]
    if len(set(ytr)) < 2 or len(set(yva)) < 2:
        return {"error": "یک کلاس خالی"}
    a_c = auc(ctrl[tr], ytr, ctrl[va], yva, kind)
    a_ca = auc(np.hstack([add, ctrl])[tr], ytr, np.hstack([add, ctrl])[va], yva, kind)
    d = a_ca - a_c
    t0, null = time.time(), []
    for i in range(perms):
        ptr, pva = np.array(tr)[rng.permutation(len(tr))], np.array(va)[rng.permutation(len(va))]
        null.append(auc(np.hstack([add[ptr], ctrl[tr]]), ytr,
                        np.hstack([add[pva], ctrl[va]]), yva, kind) - a_c)
        if (i + 1) % 100 == 0:
            print(f"      {label}: {i+1}/{perms} · {time.time()-t0:.0f}s", flush=True)
    null = np.array(null)
    p = float((np.sum(null >= d) + 1) / (len(null) + 1))
    return {"auc_control": a_c, "auc_control_plus_add": a_ca, "delta": d,
            "null_p95": float(np.percentile(null, 95)), "p_value": p,
            "passes": bool(d >= GATE_DELTA and p < GATE_P),
            "n_train": int(len(ytr)), "n_val": int(len(yva)),
            "wall_s": round(time.time() - t0, 1)}


def collinearity_event(latlon, events, tr, va):
    """AUC پیش‌بینی «کدام رویداد» از lat/lon — یک‌به‌یک، میانگین.
    نزدیک ۱ یعنی lat/lon عملاً برچسب رویداد است و آزمون جغرافیا بی‌معنا می‌شود."""
    uniq = sorted(set(events))
    aucs = []
    for e in uniq:
        yb = np.array([1 if x == e else 0 for x in events])
        if len(set(yb[tr])) < 2 or len(set(yb[va])) < 2:
            continue
        aucs.append(auc(latlon[tr], yb[tr], latlon[va], yb[va], "forest"))
    return (round(float(np.mean(aucs)), 4), len(aucs)) if aucs else (None, 0)


def run_flood(perms, out):
    cond = {r["filename"].replace("_S2Hand.tif", ""): r
            for r in csv.DictReader((FMETA / "conditioning_v1.csv").open(encoding="utf-8"))}
    prox = {r["stem"]: r for r in csv.DictReader((FMETA / "image_proxy.csv").open(encoding="utf-8"))}
    rows = [{**c, **prox[s]} for s, c in cond.items() if s in prox]
    print(f"\n{'='*86}\n🌊 سیل · {len(rows)} نمونه\n{'='*86}")

    events = [r["event"] for r in rows]
    IMG = np.array([[float(r[k]) for k in IMG_F] + [float(r["pct_nodata"])] for r in rows])
    G = np.array([[float(r[k]) for k in G4] for r in rows])
    W = np.array([[float(r[k]) for k in W6] for r in rows])
    water = np.array([float(r["pct_water"]) for r in rows])

    # D1 درون‌رویدادی (۷۰/۳۰ طبقه‌بندی‌شده) · D2 رویدادمحور قفل‌شده
    rng0 = np.random.default_rng(SEED)
    by = {}
    for i, e in enumerate(events):
        by.setdefault(e, []).append(i)
    tr1, va1 = [], []
    for _, ids in sorted(by.items()):
        ids = np.array(ids)[rng0.permutation(len(ids))]
        k = max(1, int(round(0.30 * len(ids))))
        va1 += list(ids[:k]); tr1 += list(ids[k:])
    tr1, va1 = np.array(sorted(tr1)), np.array(sorted(va1))
    tr2 = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    va2 = np.array([i for i, r in enumerate(rows) if r["split"] == "val"])

    for dname, tr, va in (("D1_within_event", tr1, va1), ("D2_event_split", tr2, va2)):
        col, k = collinearity_event(z(G[:, :2], tr), events, tr, va)
        print(f"\n▶ {dname} · train {len(tr)} · val {len(va)}")
        print(f"   هم‌خطی lat/lon با رویداد: AUC میانگین {col} روی {k} رویداد"
              f"  {'← 🔴 عملاً برچسب رویداد' if col and col > 0.9 else ''}")
        out["flood_collinearity"] = out.get("flood_collinearity", {})
        out["flood_collinearity"][dname] = {"mean_auc_event_from_latlon": col, "n_events": k}

        thr = float(np.median(water[tr]))
        y = (water > thr).astype(int)
        Iz, Gz, Wz = z(IMG, tr), z(G, tr), z(W, tr)
        combos = [("A_geo_over_img", Gz, Iz),
                  ("B_wx_over_img", Wz, Iz),
                  ("C_both_over_img", np.hstack([Gz, Wz]), Iz),
                  ("D_wx_over_img_geo", Wz, np.hstack([Iz, Gz]))]
        rng = np.random.default_rng(SEED)
        for cname, add, ctrl in combos:
            for kind in ("linear", "forest"):
                key = f"flood|{dname}|{cname}|{kind}"
                r = test(add, ctrl, y, tr, va, kind, perms, rng, key)
                out["results"][key] = r
                if "error" in r:
                    print(f"   {key:<44} ⛔"); continue
                print(f"   {cname:<20}{kind:<8}{r['auc_control']:.4f}→{r['auc_control_plus_add']:.4f}"
                      f"  Δ{r['delta']:+.4f}  p={r['p_value']:.4f}  {'✅' if r['passes'] else '❌'}")


def run_gpp(perms, out):
    rows = []
    for r in csv.DictReader((GPP / "data_train_hls_37sites_v0_1.csv").open(encoding="utf-8")):
        try:
            for k in GPP_W + GPP_IMG + ["GPP"]:
                r[k] = float(r[k])
            r["doy"] = int(r["doy"])
        except (ValueError, TypeError, KeyError):
            continue
        rows.append(r)
    print(f"\n{'='*86}\n🌱 GPP · {len(rows)} نمونه"
          f"\n⚠️ lat/lon در دیتاست نیست → فقط **تاریخ مطلق** سنجیده می‌شود\n{'='*86}")

    IMG = np.array([[r[k] for k in GPP_IMG] for r in rows])
    W = np.array([[r[k] for k in GPP_W] for r in rows])
    a = 2 * np.pi * np.array([r["doy"] for r in rows]) / 365.25
    D = np.stack([np.sin(a), np.cos(a)], axis=1)          # تاریخ مطلق، دو بُعد
    gpp = np.array([r["GPP"] for r in rows])
    sites = [r["SITE_ID"] for r in rows]

    # split سایت‌محور تمیز (همان seed و منطق 91)
    uniq = sorted(set(sites))
    order = np.random.default_rng(SEED).permutation(len(uniq))
    val_s = {uniq[i] for i in order[:max(1, int(round(0.30 * len(uniq))))]}
    tr = np.array([i for i, s in enumerate(sites) if s not in val_s])
    va = np.array([i for i, s in enumerate(sites) if s in val_s])
    print(f"\n▶ site_split · train {len(tr)} · val {len(va)} · {len(val_s)} سایت در val")

    thr = float(np.median(gpp[tr]))
    y = (gpp > thr).astype(int)
    Iz, Dz, Wz = z(IMG, tr), z(D, tr), z(W, tr)
    combos = [("A_date_over_img", Dz, Iz),
              ("B_wx_over_img", Wz, Iz),
              ("C_both_over_img", np.hstack([Dz, Wz]), Iz),
              ("D_wx_over_img_date", Wz, np.hstack([Iz, Dz]))]
    rng = np.random.default_rng(SEED)
    for cname, add, ctrl in combos:
        for kind in ("linear", "forest"):
            key = f"gpp|site_split|{cname}|{kind}"
            r = test(add, ctrl, y, tr, va, kind, perms, rng, key)
            out["results"][key] = r
            if "error" in r:
                print(f"   {key:<44} ⛔"); continue
            print(f"   {cname:<22}{kind:<8}{r['auc_control']:.4f}→{r['auc_control_plus_add']:.4f}"
                  f"  Δ{r['delta']:+.4f}  p={r['p_value']:.4f}  {'✅' if r['passes'] else '❌'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()
    out = {"perms": args.perms, "gates": {"delta": GATE_DELTA, "p": GATE_P},
           "seed": SEED, "results": {}}
    print("=" * 86)
    print("«مکان و تاریخ مطلق، فراتر از تصویر» — دو خانهٔ ❓ جدول سه‌دیتاستی")
    print(f"🔒 آستانه‌های پیش‌ثبت‌شده: delta ≥ {GATE_DELTA} و p < {GATE_P}")
    print("=" * 86)
    run_flood(args.perms, out)
    run_gpp(args.perms, out)

    print("\n" + "=" * 86)
    print(f"{'کلید':<46}{'کنترل':>9}{'+افزوده':>10}{'Δ':>10}{'p':>9}  حکم")
    print("=" * 86)
    for k, r in out["results"].items():
        if "error" in r:
            print(f"{k:<46}{'—':>9}{'—':>10}{'—':>10}{'—':>9}  ⛔"); continue
        print(f"{k:<46}{r['auc_control']:>9.4f}{r['auc_control_plus_add']:>10.4f}"
              f"{r['delta']:>+10.4f}{r['p_value']:>9.4f}  {'✅' if r['passes'] else '❌'}")
    out["wall_seconds_total"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nکل {out['wall_seconds_total']}s · {OUT_JSON}")


if __name__ == "__main__":
    main()

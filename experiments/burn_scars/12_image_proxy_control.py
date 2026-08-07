# -*- coding: utf-8 -*-
"""
12_image_proxy_control.py — آیا بردار شرط چیزی دارد که **تصویر** ندارد؟
=========================================================================
نسخه: v1 · تاریخ: 2026-07-30 · دروازهٔ C3 برای تسک آتش‌سوزی

چرا این فایل هست — ایرادی که نشست `flood` گرفت و **وارد است**:

    من گفتم «بردار شرط سیگنال دارد: AUC=0.695، و مدل آن را ندارد».
    ولی آن ۰.۶۹۵ از یک **طبقه‌بند جدولی روی pct_burn** آمده بود، نه از مدل
    قطعه‌بندی. مدل **تصویر** را می‌بیند — و تصویر ممکن است جغرافیا و آب‌وهوا را
    از قبل لو بدهد.

    روی GPP دقیقاً همین اتفاق افتاد: کنترل فصل ✅، کنترل بوم ✅، ولی به‌محض
    اضافه شدن پروکسی تصویر → **صفر**. آب‌وهوا آنجا پروکسیِ سبزینگی بود.

    این پنجمین بار است که عددی از منبع غیرمستقیم، اندازه‌گیری مستقیم را جواب نداد.

سؤال دقیق:
    آیا بردار شرط ۱۰ بعدی، **فراتر از آنچه از خودِ پیکسل‌ها درمی‌آید**، چیزی
    دربارهٔ درصد سوختگی می‌گوید؟

پروکسی تصویر (۱۳ ویژگی، همه از خودِ chip):
    میانگین شش باند + هفت شاخص طیفی (NDVI · NBR · NDWI · NDMI · SAVI ·
    نسبت SWIR · روشنایی)

    ⚠️ این پروکسی **ضعیف‌تر** از چیزی است که Prithvi می‌بیند (میانگین سراسری در
    برابر امبدینگ مکانی). پس آزمون **سخت‌گیرانه‌تر** نیست، **ملایم‌تر** است:
    اگر بردار شرط حتی از این پروکسی سادهٔ ۱۳ عددی جلو نزند، از Prithvi قطعاً نمی‌زند.
    اگر جلو زد، **نتیجه قطعی نیست** و باید با امبدینگ واقعی تکرار شود.

آستانه‌ها — پیش از اجرا تثبیت شده: `delta_AUC ≥ 0.02` **و** `p < 0.05`
جایگشت: بلوکی، ۲۰۰ بار، فقط روی بلوک آزمون؛ کنترل دست‌نخورده.
`test` باز نمی‌شود — فقط train و val.

خروجی: <BIG>/data/meta/image_proxy_control.json
اجرا:  <venv>\Scripts\python.exe 12_image_proxy_control.py [--perms 200]
"""

import sys, os, csv, json, time, argparse
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from pathlib import Path

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
DS = BIG / "data" / "burn_scars"
CSV = BIG / "data" / "meta" / "conditioning_v1.csv"
CACHE = BIG / "data" / "meta" / "image_proxy_features.npz"
OUT = BIG / "data" / "meta" / "image_proxy_control.json"

SEED = 0
W = ["mean_speed_z", "max_speed_z", "dir_sin_z", "dir_cos_z",
     "precip_7d_log_z", "mean_temp_z"]
G = ["lat_z", "lon_z", "doy_sin_z", "doy_cos_z"]
IMG_NAMES = ["b_blue", "b_green", "b_red", "b_nir", "b_swir1", "b_swir2",
             "ndvi", "nbr", "ndwi", "ndmi", "savi", "swir_ratio", "bright"]


def chip_features(path):
    """میانگین شش باند + هفت شاخص، فقط روی پیکسل‌های معتبر."""
    with rasterio.open(path) as src:
        a = src.read().astype(np.float32)          # (6, H, W)
    valid = ~np.all(a == 0, axis=0)                # nodata = همهٔ باندها صفر
    if valid.sum() < 100:
        valid = np.ones(a.shape[1:], bool)
    b = np.stack([a[i][valid].mean() for i in range(a.shape[0])])
    blue, green, red, nir, sw1, sw2 = b
    e = 1e-6
    return np.array([
        *b,
        (nir - red) / (nir + red + e),             # NDVI
        (nir - sw2) / (nir + sw2 + e),             # NBR
        (green - nir) / (green + nir + e),         # NDWI
        (nir - sw1) / (nir + sw1 + e),             # NDMI
        1.5 * (nir - red) / (nir + red + 0.5 + e), # SAVI
        sw1 / (sw2 + e),                           # نسبت SWIR
        b.mean(),                                  # روشنایی
    ], dtype=np.float64)


def build_cache(rows):
    """۸۰۴ chip → ماتریس ۸۰۴×۱۳. با شمارندهٔ پیشرفت و کَش."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        print(f"   کَش پیدا شد: {z['X'].shape}")
        return {int(s): x for s, x in zip(z["sid"], z["X"])}

    idx = {}
    for sub in ("training", "validation"):
        for p in (DS / sub).glob("*_merged.tif"):
            idx[p.name] = p

    sids, feats = [], []
    t0 = time.time()
    n = len(rows)
    for i, r in enumerate(rows, 1):
        p = idx.get(r["filename"])
        if p is None:
            raise FileNotFoundError(r["filename"])
        feats.append(chip_features(p))
        sids.append(int(r["sample_id"]))
        if i % 100 == 0 or i == n:
            el = time.time() - t0
            print(f"   [{i}/{n}] {el:6.1f}s گذشته · تخمین باقی {el/i*(n-i):6.1f}s",
                  flush=True)

    X = np.vstack(feats)
    np.savez_compressed(CACHE, sid=np.array(sids), X=X, names=np.array(IMG_NAMES))
    print(f"   💾 کَش شد: {CACHE.name}")
    return {s: x for s, x in zip(sids, X)}


def auc(Xtr, ytr, Xva, yva, kind, seed=SEED):
    m = (LogisticRegression(max_iter=3000, random_state=seed) if kind == "linear"
         else RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     random_state=seed, n_jobs=-1))
    m.fit(Xtr, ytr)
    return float(roc_auc_score(yva, m.predict_proba(Xva)[:, 1]))


def gate(name, Ctr, Cva, Ttr, Tva, ytr, yva, kind, n_perm, rng):
    """AUC(control+test) − AUC(control)، با جایگشت بلوکیِ فقط بلوک آزمون."""
    a_c = auc(Ctr, ytr, Cva, yva, kind)
    a_ct = auc(np.hstack([Ctr, Ttr]), ytr, np.hstack([Cva, Tva]), yva, kind)
    d = a_ct - a_c
    null = []
    for _ in range(n_perm):
        null.append(auc(np.hstack([Ctr, Ttr[rng.permutation(len(ytr))]]), ytr,
                        np.hstack([Cva, Tva[rng.permutation(len(yva))]]), yva, kind) - a_c)
    null = np.array(null)
    p = float((np.sum(null >= d) + 1) / (len(null) + 1))
    return dict(control_auc=a_c, full_auc=a_ct, delta=d,
                null_q95=float(np.quantile(null, .95)), p_value=p,
                passes=bool(d >= 0.02 and p < 0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    a = ap.parse_args()

    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    print("=" * 82)
    print("۱. استخراج پروکسی تصویر از ۸۰۴ chip")
    print("=" * 82)
    fmap = build_cache(rows)

    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    thr = float(np.median([float(r["pct_burn"]) for r in tr]))
    yb = lambda rr: np.array([int(float(r["pct_burn"]) > thr) for r in rr])
    ytr, yva = yb(tr), yb(va)

    # نرمال‌سازی پروکسی تصویر — فقط از train
    Itr_raw = np.vstack([fmap[int(r["sample_id"])] for r in tr])
    Iva_raw = np.vstack([fmap[int(r["sample_id"])] for r in va])
    mu, sd = Itr_raw.mean(0), Itr_raw.std(0)
    sd[sd < 1e-9] = 1.0
    IMG_TR, IMG_VA = (Itr_raw - mu) / sd, (Iva_raw - mu) / sd

    blk = lambda rr, cols: np.array([[float(r[c]) for c in cols] for r in rr])
    W_TR, W_VA = blk(tr, W), blk(va, W)
    G_TR, G_VA = blk(tr, G), blk(va, G)
    C10_TR, C10_VA = np.hstack([W_TR, G_TR]), np.hstack([W_VA, G_VA])
    SEA = ["doy_sin_z", "doy_cos_z"]
    S_TR, S_VA = blk(tr, SEA), blk(va, SEA)

    print(f"\ntrain {len(tr)} · val {len(va)} · test {len(rows)-len(tr)-len(va)} (باز نشد)")
    print(f"آستانه: میانهٔ pct_burn در train = {thr:.3f}٪ · نسبت مثبت val = {yva.mean():.3f}")
    print(f"جایگشت: {a.perms} بلوکی · آستانهٔ قبولی: delta ≥ 0.02 و p < 0.05\n")

    tests = [
        ("C1 فصل        ← بردار ۱۰", S_TR, S_VA, C10_TR, C10_VA),
        ("C2 فصل+جغرافیا ← ۶ جوّی", np.hstack([S_TR, G_TR]), np.hstack([S_VA, G_VA]), W_TR, W_VA),
        ("C3 پروکسی تصویر ← بردار ۱۰", IMG_TR, IMG_VA, C10_TR, C10_VA),
        ("C3 پروکسی تصویر ← ۴ جغرافیا", IMG_TR, IMG_VA, G_TR, G_VA),
        ("C3 پروکسی تصویر ← ۶ جوّی", IMG_TR, IMG_VA, W_TR, W_VA),
        ("C4 تصویر+جغرافیا ← ۶ جوّی", np.hstack([IMG_TR, G_TR]), np.hstack([IMG_VA, G_VA]), W_TR, W_VA),
    ]

    rng = np.random.default_rng(SEED)
    res, t0 = {}, time.time()
    print("=" * 82)
    print(f"{'آزمون':<30}{'مدل':<8}{'AUC کنترل':>11}{'AUC کامل':>10}{'افزوده':>10}{'p':>8}")
    print("-" * 82)
    for name, Ctr, Cva, Ttr, Tva in tests:
        res[name] = {}
        for kind, lab in (("linear", "خطی"), ("forest", "جنگل")):
            r = gate(name, Ctr, Cva, Ttr, Tva, ytr, yva, kind, a.perms, rng)
            res[name][kind] = r
            print(f"{name:<30}{lab:<8}{r['control_auc']:>11.4f}{r['full_auc']:>10.4f}"
                  f"{r['delta']:>+10.4f}{r['p_value']:>8.4f}  {'✅' if r['passes'] else '❌'}",
                  flush=True)
    print("=" * 82)

    key = "C3 پروکسی تصویر ← بردار ۱۰"
    passed = any(res[key][k]["passes"] for k in ("linear", "forest"))
    best = max(res[key].values(), key=lambda r: r["delta"])

    print("\nحکم — دروازهٔ C3")
    print("-" * 82)
    print(f"  AUC فقط پروکسی تصویر : {best['control_auc']:.4f}")
    print(f"  + بردار شرط ۱۰ بعدی  : {best['full_auc']:.4f}")
    print(f"  افزوده               : {best['delta']:+.4f}   (صدک۹۵ پوچ {best['null_q95']:+.4f} · p={best['p_value']:.4f})")
    print()
    if passed:
        print("  🟢 **بردار شرط چیزی دارد که تصویر ندارد.**")
        print("     ⚠️ ولی قطعی نیست: پروکسی من میانگین سراسری است، Prithvi امبدینگ")
        print("        مکانی می‌بیند. باید با امبدینگ واقعی تکرار شود.")
        print("     👉 آتش‌سوزی نامزد معتبر تسک اصلی است.")
    else:
        print("  🔴 **بردار شرط چیزی فراتر از تصویر ندارد.**")
        print("     همان الگوی GPP: کنترل ضعیف قبول، پروکسی تصویر رد.")
        print("     👉 `AUC=0.695` من سراب بود — تصویر از قبل داشتش.")
        print("     👉 ادعا باید بشود «**کجا** تزریق شود»، نه «چقدر کمک می‌کند».")

    res["_summary"] = {
        "gate_C3_passed": passed, "threshold": {"delta": 0.02, "p": 0.05},
        "n_train": len(tr), "n_val": len(va), "n_perm": a.perms,
        "pct_burn_threshold": thr, "image_proxy_features": IMG_NAMES,
        "caveat": "پروکسی = میانگین سراسری، ضعیف‌تر از امبدینگ Prithvi → آزمون ملایم‌تر",
        "test_untouched": True, "seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 {OUT}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

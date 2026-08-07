# -*- coding: utf-8 -*-
"""
07_signal_test.py — آیا آن شش عدد اصلاً چیزی می‌دانند؟
=======================================================
نسخه: v1 · تاریخ: 2026-07-29 · **دروازهٔ ۲، نیمهٔ اول**

سؤال:
    قبل از اینکه یک دلار برای GPU خرج کنیم: آیا شش ویژگی جوّی هیچ اطلاعاتی
    دربارهٔ درصد سوختگی دارند، یا صرفاً شش عدد تصادفی کنار داده‌اند؟

سه چیز که این فایل را از یک «AUC گرفتن» ساده جدا می‌کند:

    ۱) 🔴 **کنترل جغرافیا و فصل.**
       آب‌وهوا با مکان و فصل همبسته است. اگر فقط «۶ ویژگی → سوختگی» را بسنجیم و
       AUC=0.65 بگیریم، هیچ نمی‌دانیم که آیا باد چیزی گفته یا مدل فقط فهمیده
       «کالیفرنیا در سپتامبر». پس سه مجموعهٔ پیش‌بین را کنار هم می‌سنجیم:
           W  = شش ویژگی جوّی
           G  = جغرافیا و فصل (lat, lon, sin/cos روزِ سال)
           W+G= هر دو
       اگر W بهتر از G نباشد، یا W+G بهتر از G نباشد → **آب‌وهوا چیزی اضافه نکرده.**
       این دقیقاً همان سؤالی است که یک داور می‌پرسد.

    ۲) 🔴 **توزیع پوچ با جایگشت.**
       با ۱۲۱ نمونهٔ val، AUC=0.58 ممکن است کاملاً تصادفی باشد. برچسب‌ها را
       ۲۰۰ بار به‌هم می‌ریزیم، هر بار مدل را دوباره آموزش می‌دهیم، و توزیع
       AUCهای «بی‌معنا» را می‌سازیم. عدد واقعی باید از صدک ۹۵ آن رد شود.
       بدون این، هر AUC بالای ۰.۵ فریبنده است.

    ۳) 🔴 **مجموعهٔ test اصلاً باز نمی‌شود.**
       فقط train (۵۶۳) و val (۱۲۱). test تا روز تحلیل نهایی دست‌نخورده می‌ماند.

⚠️ تفسیر — قبل از دیدن نتیجه بخوان:
    AUC پایین **ایده را باطل نمی‌کند.** از قبل می‌دانیم هیچ نمونهٔ منفی خالصی در
    دیتاست نیست (۰ از ۸۰۴)، پس تسک «کجا سوخته» است نه «آیا سوخته» — و یک اسکالر
    سراسری ذاتاً نمی‌تواند مکان‌یابی کند. AUC پایین یعنی **«اثر کوچک است»**،
    نه «چارچوب غلط است». ولی باید در ارائه گفته شود.

خروجی:
    <BIG>/data/meta/signal_test.json

اجرا:
    python 07_signal_test.py
    python 07_signal_test.py --perms 500     (جایگشت بیشتر، کندتر)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
import argparse
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("⛔ scikit-learn نصب نیست:\n    pip install scikit-learn")
    sys.exit(1)

from scipy.stats import spearmanr

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
META = BIG / "data" / "meta"
IN_CSV = META / "wind_features.csv"
OUT_JSON = META / "signal_test.json"

SEED = 0

# مجموعه‌های پیش‌بین — نسخهٔ z شده، چون نرمال‌سازی فقط از train انجام شده
W_FEATS = ["mean_speed_z", "max_speed_z", "dir_sin_z", "dir_cos_z",
           "precip_7d_log_z", "mean_temp_z"]
SETS = {}   # در main پر می‌شود


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


def build_matrices(rows):
    """سه مجموعهٔ پیش‌بین. جغرافیا و فصل از train نرمال می‌شوند."""
    tr = [r for r in rows if r["split"] == "train"]

    # روزِ سال دوّار است: ۳۶۵ و ۱ همسایه‌اند → sin/cos، نه عدد خام
    def geo(r):
        a = 2 * np.pi * r["doy"] / 365.25
        return [r["lat_center"], r["lon_center"], np.sin(a), np.cos(a)]

    G_tr = np.array([geo(r) for r in tr], dtype=float)
    mu, sd = G_tr.mean(axis=0), G_tr.std(axis=0)
    sd[sd < 1e-9] = 1.0

    def G(r):
        return list((np.array(geo(r)) - mu) / sd)

    def W(r):
        return [r[k] for k in W_FEATS]

    return {
        "W  — فقط جوّی":        (W,  W_FEATS),
        "G  — جغرافیا و فصل":   (G,  ["lat", "lon", "doy_sin", "doy_cos"]),
        "W+G — هر دو":          (lambda r: W(r) + G(r), W_FEATS + ["lat", "lon", "doy_sin", "doy_cos"]),
    }


def split_xy(rows, fn, y_bin, y_cont):
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    Xtr = np.array([fn(r) for r in tr], dtype=float)
    Xva = np.array([fn(r) for r in va], dtype=float)
    return (Xtr, Xva,
            np.array([y_bin(r) for r in tr]), np.array([y_bin(r) for r in va]),
            np.array([y_cont(r) for r in tr]), np.array([y_cont(r) for r in va]))


def fit_auc(Xtr, ytr, Xva, yva, kind, seed=SEED):
    if len(np.unique(ytr)) < 2:
        return np.nan
    if kind == "linear":
        m = LogisticRegression(max_iter=2000, random_state=seed)
    else:
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                   random_state=seed, n_jobs=-1)
    m.fit(Xtr, ytr)
    return float(roc_auc_score(yva, m.predict_proba(Xva)[:, 1]))


def fit_rho(Xtr, ytr, Xva, yva, kind, seed=SEED):
    """همبستگی رتبه‌ای پیش‌بینی با واقعیت. به مقیاس حساس نیست."""
    if kind == "linear":
        m = Ridge(alpha=1.0, random_state=None)
    else:
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                  random_state=seed, n_jobs=-1)
    m.fit(Xtr, ytr)
    p = m.predict(Xva)
    if np.std(p) < 1e-12:
        return 0.0
    return float(spearmanr(p, yva).statistic)


def permutation_null(Xtr, ytr, Xva, yva, kind, n_perm, rng):
    """
    برچسب‌های train را به‌هم می‌ریزد و AUC روی val را دوباره حساب می‌کند.
    نتیجه: توزیع AUCهایی که از «هیچ» می‌آیند. عدد واقعی باید از صدک ۹۵ رد شود.
    """
    out = []
    for i in range(n_perm):
        yp = rng.permutation(ytr)
        a = fit_auc(Xtr, yp, Xva, yva, kind, seed=SEED + i)
        if np.isfinite(a):
            out.append(a)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    args = ap.parse_args()

    if not IN_CSV.exists():
        print(f"⛔ {IN_CSV} نیست — اول 05b_build_features_openmeteo.py را اجرا کن")
        return

    rows = load()
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    n_test = sum(1 for r in rows if r["split"] == "test")

    thr = float(np.median([r["pct_burn"] for r in tr]))   # آستانه فقط از train
    y_bin = lambda r: int(r["pct_burn"] > thr)
    y_cont = lambda r: float(np.log1p(r["pct_burn"]))

    print(f"train {len(tr)} · val {len(va)} · test {n_test} (**باز نمی‌شود**)")
    print(f"آستانهٔ دودویی = میانهٔ درصد سوختگی در train = {thr:.3f}٪")
    print(f"نسبت مثبت در val: {np.mean([y_bin(r) for r in va]):.3f}")
    print(f"جایگشت: {args.perms} بار\n")

    sets = build_matrices(rows)
    rng = np.random.default_rng(SEED)
    results = {}

    print("=" * 86)
    print(f"{'مجموعهٔ پیش‌بین':<22}{'مدل':<10}{'AUC':>8}{'صدک۹۵ پوچ':>12}{'p':>8}{'rho':>9}")
    print("-" * 86)

    for name, (fn, cols) in sets.items():
        Xtr, Xva, ytr_b, yva_b, ytr_c, yva_c = split_xy(rows, fn, y_bin, y_cont)
        results[name] = {"n_features": Xtr.shape[1], "columns": cols}
        for kind, label in (("linear", "خطی"), ("forest", "جنگل")):
            auc = fit_auc(Xtr, ytr_b, Xva, yva_b, kind)
            rho = fit_rho(Xtr, ytr_c, Xva, yva_c, kind)
            null = permutation_null(Xtr, ytr_b, Xva, yva_b, kind, args.perms, rng)
            q95 = float(np.quantile(null, 0.95))
            # p یک‌طرفه: چند درصد جایگشت‌ها به عدد واقعی رسیدند یا از آن گذشتند
            p = float((np.sum(null >= auc) + 1) / (len(null) + 1))
            results[name][kind] = {"auc": auc, "spearman": rho,
                                   "null_q95": q95, "null_mean": float(null.mean()),
                                   "p_value": p, "beats_null": bool(auc > q95)}
            mark = "✅" if auc > q95 else "❌"
            print(f"{name:<22}{label:<10}{auc:>8.4f}{q95:>12.4f}{p:>8.3f}{rho:>9.4f}  {mark}")

    print("=" * 86)

    # --- همبستگی تک‌متغیره: کدام ویژگی به‌تنهایی چیزی می‌گوید؟ ---
    print("\nهمبستگی رتبه‌ای تک‌متغیره با درصد سوختگی (روی train):")
    print(f"{'ویژگی':<20}{'rho':>9}{'p':>10}")
    print("-" * 40)
    burn = np.array([r["pct_burn"] for r in tr])
    uni = {}
    for k in W_FEATS:
        x = np.array([r[k] for r in tr])
        s = spearmanr(x, burn)
        uni[k] = {"rho": float(s.statistic), "p": float(s.pvalue)}
        star = "*" if s.pvalue < 0.05 else " "
        print(f"{k:<20}{s.statistic:>9.4f}{s.pvalue:>10.4f} {star}")
    results["univariate_train"] = uni

    # --- حکم ---
    print("\n" + "=" * 86)
    print("حکم")
    print("=" * 86)

    def best(name):
        return max(results[name][k]["auc"] for k in ("linear", "forest"))

    W, G, WG = "W  — فقط جوّی", "G  — جغرافیا و فصل", "W+G — هر دو"
    aW, aG, aWG = best(W), best(G), best(WG)
    passes_null = any(results[W][k]["beats_null"] for k in ("linear", "forest"))
    adds_over_geo = (aWG - aG) > 0.02

    print(f"  بهترین AUC — فقط جوّی        : {aW:.4f}")
    print(f"  بهترین AUC — جغرافیا و فصل   : {aG:.4f}")
    print(f"  بهترین AUC — هر دو           : {aWG:.4f}")
    print(f"  افزودهٔ آب‌وهوا بر جغرافیا    : {aWG - aG:+.4f}")

    print()
    if passes_null and adds_over_geo:
        verdict = "signal"
        print("  🟢 **سیگنال هست، و فراتر از جغرافیا.**")
        print("     آب‌وهوا چیزی می‌داند که مکان و فصل نمی‌گویند. طبق برنامه جلو برو.")
    elif passes_null and not adds_over_geo:
        verdict = "confounded"
        print("  🟡 **سیگنال هست ولی از جغرافیا جدا نمی‌شود.**")
        print("     ویژگی‌های جوّی از توزیع پوچ رد می‌شوند، ولی چیزی به مکان و فصل")
        print("     اضافه نمی‌کنند. یعنی ممکن است مدل فقط «کجا و کِی» را یاد بگیرد.")
        print("     👉 آزمایش ادامه می‌یابد، ولی **این را در ارائه بگو** و آزمون")
        print("        شافل‌شده (P4) از همیشه مهم‌تر می‌شود.")
    else:
        verdict = "no_signal"
        print("  🔴 **از توزیع پوچ رد نشد.**")
        print("     شش عدد چیزی دربارهٔ درصد سوختگی نمی‌گویند.")
        print("     👉 این ایده را **باطل نمی‌کند** — از قبل می‌دانستیم سقف شرط")
        print("        سراسری پایین است (۰ نمونهٔ منفی خالص از ۸۰۴). ولی یعنی:")
        print("        ۱) انتظار بهبود بزرگ در ماتریس نداشته باش")
        print("        ۲) ادعای اسلاید باید «کجا تزریق شود» باشد، نه «چقدر کمک می‌کند»")
        print("        ۳) بازوی شافل‌شده (P4) حالا مهم‌ترین بازوی آزمایش است")

    results["_summary"] = {
        "verdict": verdict, "auc_weather": aW, "auc_geo": aG, "auc_both": aWG,
        "weather_adds_over_geo": aWG - aG, "threshold_pct_burn": thr,
        "n_train": len(tr), "n_val": len(va), "n_permutations": args.perms,
        "test_untouched": True,
    }
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n💾 {OUT_JSON}")


if __name__ == "__main__":
    main()

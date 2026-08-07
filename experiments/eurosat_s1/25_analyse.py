# -*- coding: utf-8 -*-
"""
25_analyse.py - the table, judged by the rule locked in PREREG-eurosat-injection.md

    s  = pooled seed std across arms (ddof=1)
    SE = s * sqrt(2/n)
    a difference counts only if |delta| >= 2*SE AND Welch p < 0.05
"""
import sys, json, math
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy import stats

R = Path(r"C:\Users\aliso\Desktop\big-files\loc\runs")
ARMS = ("none", "shuffle", "adaln", "add", "gate", "token")
INJ = {"none": 0, "add": 1, "gate": 9313, "token": 37056,
       "adaln": 889344, "shuffle": 889344}

# 🔴 خطا — اندازه‌گیری‌شده ۱ اوت پس از اتمام اجرای ۹۰ epoch:
#    این اسکریپت همهٔ فایل‌های JSON را می‌خواند، پس اجراهای ۳۰ و ۹۰ epoch را
#    در یک ظرف می‌ریخت. نتیجه: n=8 برای بعضی بازوها (۵ تای ۳۰ep + ۳ تای ۹۰ep)
#    و n=5 برای `gate` که اصلاً در سوییپ ۹۰ep نبود.
#
#    اثرش: انحراف مجتمع از 0.3241 به 0.5073 پرید — نه چون نویز بیشتر شد،
#    بلکه چون اختلاف *بین دو بودجه* را به‌عنوان نویز *درون بازو* شمرد.
#    و مقایسهٔ `gate` با بقیه، ۳۰ep را با مخلوط ۳۰/۹۰ep می‌سنجید.
#
#    درس: وقتی یک بُعد آزمایشی جدید اضافه می‌کنی (اینجا بودجهٔ آموزش)،
#    هر تحلیلی که از قبل نوشته‌ای باید صریح دربارهٔ آن بُعد تصمیم بگیرد.
BUDGET = "90" if "--e90" in sys.argv else "30"

acc, scales, gates = {}, [], {}
for f in sorted(R.glob("*.json")):
    is_e90 = f.stem.endswith("_e90")
    if is_e90 != (BUDGET == "90"):
        continue
    # 🔴 6 Aug 2026. This script predates the geo split. It filtered only on the
    # epoch budget, so re-running it TODAY silently pooled official-split runs
    # with geo-split runs (and, since this morning, the depth sweep as well).
    # The two splits differ by about 8 accuracy points, so the pooled seed std
    # came out at 5.98 — a number that measures the split, not seed noise, and
    # that makes every threshold in this table meaningless.
    # This is the same class of bug as the 33/37 fixes: a glob that outgrew its
    # assumptions. The recorded official-split numbers were taken before any geo
    # run existed and are unaffected; only re-runs were wrong.
    if "_geo" in f.stem:
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    acc.setdefault(d["arm"], []).append((d["seed"], d["best_val_acc"] * 100))
    if d.get("learned_scale") is not None:
        scales.append((d["seed"], d["learned_scale"]))
    if d.get("gate_stats"):
        gates[d["seed"]] = d["gate_stats"]

vals = {a: np.array([v for _, v in sorted(acc.get(a, []))]) for a in ARMS}
ns = {a: len(vals[a]) for a in ARMS}


# ---- noise floor, computed exactly as the PREREG specifies ----
pool = [v for a in ARMS if ns[a] > 1 for v in (vals[a] - vals[a].mean())]
dof = sum(ns[a] - 1 for a in ARMS if ns[a] > 1)
s = math.sqrt(sum(x * x for x in pool) / dof)
n_min = min(n for n in ns.values() if n)
SE = s * math.sqrt(2 / n_min)
THR = 2 * SE

print(f"=== budget: {BUDGET} epochs ===")
print(f"seeds per arm: " + "  ".join(f"{a}={ns[a]}" for a in ARMS))
print(f"\npooled seed std   s  = {s:.4f}   ({dof} dof)")
print(f"SE of difference     = {SE:.4f}")
print(f"THRESHOLD  2*SE      = {THR:.4f} accuracy points")
if n_min < 5:
    print(f"  provisional - {n_min} seeds, the locked rule uses 5")

print(f"\n{'arm':9s} {'n':>2s} {'mean':>7s} {'std':>6s} {'inj params':>11s}   seeds")
print("-" * 76)
for a in ARMS:
    v = vals[a]
    if not len(v):
        print(f"{a:9s}  -"); continue
    sd = v.std(ddof=1) if len(v) > 1 else float("nan")
    print(f"{a:9s} {len(v):2d} {v.mean():7.2f} {sd:6.3f} {INJ[a]:11,d}   "
          + " ".join(f"{x:.2f}" for x in v))

def cmp(x, y):
    a, b = vals[x], vals[y]
    if len(a) < 2 or len(b) < 2:
        return None
    d = a.mean() - b.mean()
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    return d, p, abs(d) >= THR and p < 0.05


PAIRS = [
    ("add", "none", "H1  payload reaches the network"),
    ("adaln", "shuffle", "H4  control: same params, real vs scrambled coords"),
    ("shuffle", "none", "H4  control: scrambled coords should be worthless"),
    ("token", "add", "H2  TerraMind-style vs Prithvi-style"),
    ("adaln", "add", "H2  per-block modulation vs one scalar"),
    ("token", "adaln", "H2  best vs worst injection point"),
    ("gate", "add", "H3  per-sample gate vs global scalar (ours)"),
    ("token", "gate", "H2  token vs our gate"),
]
print(f"\n{'comparison':44s} {'delta':>7s} {'p':>8s}  verdict")
print("-" * 76)
for x, y, label in PAIRS:
    r = cmp(x, y)
    if r is None:
        print(f"{label:44s}    (not enough seeds)"); continue
    d, p, ok = r
    print(f"{label:44s} {d:+7.2f} {p:8.4f}  {'CROSSES' if ok else 'below threshold'}")

if scales:
    v = np.array([s_ for _, s_ in sorted(scales)])
    print(f"\nlearned scalar of `add`   init 0.1  ->  "
          + " ".join(f"{x:.4f}" for x in v)
          + f"   mean {v.mean():.4f}")
    print(f"   Prithvi-EO-2.0-300M-TL released value: 0.05815186")
    print(f"   same mechanism, same init, opposite direction")

if gates:
    print(f"\ngate distribution (per-sample scalar, val split)")
    for sd_, g in sorted(gates.items()):
        print(f"   seed {sd_}: mean {g['mean']:+.4f}  std {g['std']:.4f}  "
              f"range [{g['min']:+.3f}, {g['max']:+.3f}]")

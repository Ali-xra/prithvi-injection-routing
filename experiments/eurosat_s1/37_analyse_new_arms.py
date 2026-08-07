# -*- coding: utf-8 -*-
"""
37_analyse_new_arms.py - the arms added AFTER the pre-registration.

🔴 6 Aug 2026. This file is deliberately separate from 33_analyse_geo.py.
33 is the frozen pre-registered six-arm table and must never change its
numbers. Everything here is exploratory / post-hoc and is labelled as such.

Same threshold rule as PREREG, with unequal seed counts handled honestly:
  s   = pooled within-arm standard deviation over every arm in the table
  SE  = s * sqrt(1/n_a + 1/n_b)
  a difference counts only if BOTH |delta| >= 2*SE AND Welch p < 0.05
"""
import sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy import stats

RUNS = Path(__file__).resolve().parent / "runs"

OLD = ("none", "add", "token", "gate", "adaln", "shuffle")
NEW = ("add_mid", "gate_late", "gate_std", "gate_max", "gate_coord", "film")

acc, ctl = {}, {}
for f in sorted(RUNS.glob("*_geo*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    # 🔴 6 Aug 2026: stage 9 sweeps the injection depth. Those runs must NOT be
    # folded into the depth-3 arms this table reports - that would silently
    # average five different architectures into one row. Depth sweep lives in
    # 38_analyse_depth.py. Runs written before mid_at existed have no key and
    # were all depth 3.
    if d.get("mid_at", 3) != 3:
        continue
    tgt = ctl if d.get("shuffle_coords") or d["arm"] == "shuffle" else acc
    tgt.setdefault(d["arm"], []).append((d["seed"], d["best_val_acc"] * 100))

vals = {a: np.array([v for _, v in sorted(x)]) for a, x in acc.items()}
ctls = {a: np.array([v for _, v in sorted(x)]) for a, x in ctl.items()}

# pooled s over every arm that has at least 2 seeds
s = np.sqrt(np.mean([v.var(ddof=1) for v in vals.values() if len(v) > 1]))
print(f"pooled s over all arms = {s:.4f}")


def line(name, v):
    print(f"{name:<12} {len(v):>2}  {v.mean():7.2f} "
          f"{(v.std(ddof=1) if len(v) > 1 else 0):6.2f}   "
          + " ".join(f"{x:.2f}" for x in v))


print(f"\n{'arm':<12} {'n':>2}  {'mean':>7} {'std':>6}   seeds")
print("-- pre-registered (frozen, see 33_analyse_geo.py) --")
for a in OLD:
    if a in vals:
        line(a, vals[a])
print("-- added 6 Aug 2026 (post-hoc, exploratory) --")
for a in NEW:
    if a in vals:
        line(a, vals[a])
print("-- shuffled-coordinate controls --")
for a in sorted(ctls):
    line(a + "+shuf", ctls[a])


def contrast(a, b, note=""):
    if a not in vals or b not in vals:
        print(f"{a+' - '+b:<24} MISSING")
        return
    va, vb = vals[a], vals[b]
    se = s * np.sqrt(1 / len(va) + 1 / len(vb))
    d = va.mean() - vb.mean()
    p = stats.ttest_ind(va, vb, equal_var=False).pvalue
    ok = abs(d) >= 2 * se and p < 0.05
    print(f"{a+' - '+b:<24} {d:+7.2f}  2SE={2*se:5.2f}  p={p:6.4f}  "
          + ("SIGNIFICANT" if ok else "in the noise") + ("   " + note if note else ""))


print("\nthe contrasts that carry the argument")
print("-" * 78)
contrast("gate_late", "add",       "does reading later beat the 1-parameter baseline?")
contrast("gate_late", "gate",      "does moving the gate later help at all?")
contrast("gate_late", "add_mid",   "THE key one: reading later vs merely injecting later")
contrast("add_mid",   "add",       "is injecting later alone enough?")
contrast("gate_std",  "gate",      "richer summary at the OLD position")
contrast("gate_max",  "gate",      "different pooling at the OLD position")
contrast("gate_coord", "gate",     "gate that sees only coordinates")
contrast("film",      "add",       "multiply-and-shift at the front")
contrast("film",      "adaln",     "cheap FiLM vs full adaLN")

print("\ncontrols (each must land on none = "
      f"{vals['none'].mean():.2f})")
print("-" * 78)
for a in sorted(ctls):
    v = ctls[a]
    se = s * np.sqrt(1 / len(v) + 1 / len(vals["none"]))
    d = v.mean() - vals["none"].mean()
    print(f"{a+'+shuf - none':<24} {d:+7.2f}  2SE={2*se:5.2f}  "
          + ("LEAK - investigate" if abs(d) >= 2 * se else "clean"))

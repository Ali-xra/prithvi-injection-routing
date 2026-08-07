# -*- coding: utf-8 -*-
"""
33_analyse_geo.py - the six-arm table on the location-disjoint split.

Threshold rule, taken verbatim from PREREG-eurosat-injection.md:
  1. s = pooled seed standard deviation across the six arms (ddof=1)
  2. SE of a difference between two arm means = s * sqrt(2/n)
  3. a difference counts only if BOTH |delta| >= 2*SE AND Welch p < 0.05

PREREG defines n = 5. This stage used n = 3, so the same rule is applied with
sqrt(2/3). The derivation is unchanged; only n differs. Stated, not hidden.
"""
import sys, json, itertools
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy import stats

RUNS = Path(__file__).resolve().parent / "runs"
ARMS = ("none", "add", "token", "gate", "adaln", "shuffle")

acc = {a: [] for a in ARMS}
for f in sorted(RUNS.glob("*_geo.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    # 🔴 6 Aug 2026: arms added after the pre-registration (add_mid, gate_late,
    # gate_std, gate_max, gate_coord, film) live in the same runs/ folder. This
    # table is the FROZEN pre-registered six-arm table, so anything outside
    # ARMS is skipped here and analysed separately in 37_analyse_new_arms.py.
    if d["arm"] not in acc:
        continue
    if d.get("shuffle_coords"):
        continue
    # 🔴 6 Aug 2026: PREREG fixed n = 5, seeds 0-4. Stage 7 adds seeds 5-9 to
    # `add` for a post-hoc contrast. Without this guard those seeds would walk
    # straight into the frozen pre-registered table and change its numbers.
    if d["seed"] > 4:
        continue
    acc[d["arm"]].append((d["seed"], d["best_val_acc"] * 100))

vals = {a: np.array([v for _, v in sorted(acc[a])]) for a in ARMS}
n = {a: len(vals[a]) for a in ARMS}
if len(set(n.values())) != 1:
    print(f"WARNING unequal seed counts: {n}")
N = min(n.values())

print(f"\n{'arm':<9} {'n':>2}  {'mean':>7} {'std':>6}   seeds")
for a in ARMS:
    v = vals[a]
    print(f"{a:<9} {len(v):>2}  {v.mean():7.2f} {v.std(ddof=1):6.2f}   "
          + " ".join(f"{x:.2f}" for x in v))

s = np.sqrt(np.mean([vals[a].var(ddof=1) for a in ARMS]))
se = s * np.sqrt(2 / N)
thr = 2 * se
print(f"\npooled s = {s:.4f}   SE(diff) = s*sqrt(2/{N}) = {se:.4f}   threshold 2*SE = {thr:.4f}")

print(f"\n{'pair':<20} {'delta':>7} {'|d|>=2SE':>9} {'welch p':>9}  verdict")
print("-" * 62)
for a, b in itertools.combinations(ARMS, 2):
    d = vals[a].mean() - vals[b].mean()
    p = stats.ttest_ind(vals[a], vals[b], equal_var=False).pvalue
    sig = abs(d) >= thr and p < 0.05
    print(f"{a+' - '+b:<20} {d:+7.2f} {str(abs(d)>=thr):>9} {p:9.4f}  "
          + ("SIGNIFICANT" if sig else "in the noise"))

print("\nhypotheses from PREREG:")
inj = ("add", "token", "gate", "adaln")
d_add = vals["add"].mean() - vals["none"].mean()
print(f"  H1 payload reaches the network   add - none = {d_add:+.2f}  "
      + ("HOLDS" if abs(d_add) >= thr else "REFUTED"))
best = max(inj, key=lambda a: abs(vals[a].mean() - vals["add"].mean()))
d_h2 = vals[best].mean() - vals["add"].mean()
print(f"  H2 routing matters               max |arm - add| = {abs(d_h2):.2f} ({best})  "
      + ("HOLDS" if abs(d_h2) >= thr else "REFUTED"))
d_h3 = vals["gate"].mean() - vals["add"].mean()
print(f"  H3 gate beats add                gate - add = {d_h3:+.2f}  "
      + ("HOLDS" if d_h3 >= thr else "REFUTED"))
d_h4a = vals["shuffle"].mean() - vals["none"].mean()
d_h4b = vals["adaln"].mean() - vals["shuffle"].mean()
print(f"  H4 control behaves               shuffle - none = {d_h4a:+.2f}   "
      f"adaln - shuffle = {d_h4b:+.2f}  "
      + ("HOLDS" if abs(d_h4a) < thr and d_h4b >= thr else "CHECK"))

print(f"\nwhat 5 seeds would have given: 2*SE = {2*s*np.sqrt(2/5):.4f}")

# -*- coding: utf-8 -*-
"""
38_analyse_depth.py - the injection-depth profile (stage 9).

🔴 Read docs/PREREG-stage9-depth.md first. This is a SHAPE question, not a
"which depth wins" question. Depth 3 stays the headline whatever this prints.
Nothing here is pre-registered; it is all exploratory and labelled so.

The model has 6 blocks. Depth d means d blocks run before the injection, so
d = 6 injects after the last block, immediately before norm + head.
"""
import sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy import stats

RUNS = Path(__file__).resolve().parent / "runs"
DEPTHS = (1, 2, 3, 4, 6)

acc = {}
ref = {}
for f in sorted(RUNS.glob("*_geo*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    if d.get("shuffle_coords"):
        continue
    if d["arm"] in ("add", "none", "gate"):
        ref.setdefault(d["arm"], []).append(d["best_val_acc"] * 100)
    if d["arm"] not in ("gate_late", "add_mid"):
        continue
    acc.setdefault((d["arm"], d.get("mid_at", 3)), []).append(
        (d["seed"], d["best_val_acc"] * 100))

vals = {k: np.array([v for _, v in sorted(x)]) for k, x in acc.items()}
refs = {k: np.array(v) for k, v in ref.items()}

pool = [v.var(ddof=1) for v in list(vals.values()) + list(refs.values()) if len(v) > 1]
s = np.sqrt(np.mean(pool))
print(f"pooled s = {s:.4f}   (model has 6 blocks; depth d = d blocks before injection)")
for k in ("none", "add", "gate"):
    if k in refs:
        print(f"reference  {k:<6} n={len(refs[k]):<2} {refs[k].mean():7.2f}")

print(f"\n{'arm':<11} {'depth':>5} {'n':>2}  {'mean':>7} {'std':>6}   seeds")
for arm in ("gate_late", "add_mid"):
    for d in DEPTHS:
        v = vals.get((arm, d))
        if v is None:
            continue
        print(f"{arm:<11} {d:>5} {len(v):>2}  {v.mean():7.2f} "
              f"{(v.std(ddof=1) if len(v) > 1 else 0):6.2f}   "
              + " ".join(f"{x:.2f}" for x in v))

print("\nprofile: gate_late(d) - add_mid(3) = 77.23, the same payload at depth 3 with no gate")
base = vals.get(("add_mid", 3))
for d in DEPTHS:
    v = vals.get(("gate_late", d))
    if v is None or base is None:
        continue
    se = s * np.sqrt(1 / len(v) + 1 / len(base))
    delta = v.mean() - base.mean()
    p = stats.ttest_ind(v, base, equal_var=False).pvalue
    mark = "SIGNIFICANT" if abs(delta) >= 2 * se and p < 0.05 else "in the noise"
    bar = "#" * max(0, int(round(delta * 20)))
    print(f"  depth {d}:  {delta:+6.2f}  2SE={2*se:5.2f}  p={p:7.4f}  {mark:<12} {bar}")

# 🔴 6 Aug, added after seeing the first table. The block above compares every
# gate_late depth against add_mid at depth 3, which is only a clean gate-vs-no-gate
# contrast AT depth 3. Everywhere else it mixes the gate with a depth change. The
# matched contrast below is the honest one and it is only available where add_mid
# was actually run (depths 1, 3, 6 - fixed in the pre-declaration).
print("\nMATCHED contrast: gate_late(d) - add_mid(d), same depth, same payload, gate or no gate")
for d in DEPTHS:
    a, b = vals.get(("gate_late", d)), vals.get(("add_mid", d))
    if a is None or b is None:
        print(f"  depth {d}:  add_mid not run at this depth - no clean contrast")
        continue
    se = s * np.sqrt(1 / len(a) + 1 / len(b))
    delta = a.mean() - b.mean()
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    mark = "SIGNIFICANT" if abs(delta) >= 2 * se and p < 0.05 else "in the noise"
    print(f"  depth {d}:  {delta:+6.2f}  2SE={2*se:5.2f}  p={p:7.4f}  {mark}")

print("\nthe same for add_mid, to separate 'reading later' from 'injecting later' at each end")
for d in DEPTHS:
    v = vals.get(("add_mid", d))
    if v is None or "add" not in refs:
        continue
    se = s * np.sqrt(1 / len(v) + 1 / len(refs["add"]))
    delta = v.mean() - refs["add"].mean()
    p = stats.ttest_ind(v, refs["add"], equal_var=False).pvalue
    mark = "SIGNIFICANT" if abs(delta) >= 2 * se and p < 0.05 else "in the noise"
    print(f"  add_mid depth {d} - add:  {delta:+6.2f}  2SE={2*se:5.2f}  p={p:7.4f}  {mark}")

print("\nshape test: is the depth-1 point already at the depth-3 level?")
a1, a3 = vals.get(("gate_late", 1)), vals.get(("gate_late", 3))
if a1 is not None and a3 is not None:
    se = s * np.sqrt(1 / len(a1) + 1 / len(a3))
    d13 = a3.mean() - a1.mean()
    print(f"  gate_late depth3 - depth1 = {d13:+.2f}  2SE={2*se:.2f}  "
          + ("STEP is above depth 1" if abs(d13) >= 2 * se
             else "depth 1 already there -> one attention block is enough"))
print("\nreminder: exploratory. Depth 3 stays the headline. See PREREG-stage9-depth.md.")

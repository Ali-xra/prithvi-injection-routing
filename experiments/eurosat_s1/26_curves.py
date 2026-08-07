# -*- coding: utf-8 -*-
"""
26_curves.py - is adaLN worse, or just slower?

A real confound to rule out before claiming adaLN is a bad injection point:

  adaLN is ZERO-INITIALISED, so at step 0 it passes NO location at all and has
  to learn to open its own gate. `add` starts at scale=0.1 and `gate` at
  bias=0.1 - both have location flowing from the first step.

  DiT's adaLN-Zero works over hundreds of thousands of steps. Here we train
  30 epochs x 64 steps = ~1900 steps total. adaLN may simply be undertrained.

If adaLN is still climbing at the last epoch while `add` has plateaued, the
comparison at this budget is unfair and the claim must be softened.
"""
import sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

R = Path(r"C:\Users\aliso\Desktop\big-files\loc\runs")
ARMS = ("none", "add", "token", "adaln", "gate", "shuffle")

hist = {}
for f in sorted(R.glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    hist.setdefault(d["arm"], []).append(
        [h["val_acc"] * 100 for h in d["history"]])


print(f"{'arm':9s} " + "".join(f"ep{e:<5d}" for e in (0, 4, 9, 14, 19, 24, 29))
      + "   last5-mid5   still climbing?")
print("-" * 92)
for a in ARMS:
    if a not in hist:
        continue
    m = np.array(hist[a]).mean(axis=0)          # mean curve over seeds
    row = "".join(f"{m[e]:7.2f}" for e in (0, 4, 9, 14, 19, 24, 29))
    last5 = m[-5:].mean()
    mid5 = m[12:17].mean()                       # epochs 12-16
    gain = last5 - mid5
    print(f"{a:9s} {row}   {gain:+9.2f}   {'YES' if gain > 0.5 else 'plateaued'}")

print("\nper-arm: epoch of best accuracy (mean over seeds)")
for a in ARMS:
    if a not in hist:
        continue
    bests = [int(np.argmax(c)) for c in hist[a]]
    print(f"   {a:9s} best at epoch {np.mean(bests):5.1f}  "
          f"(seeds: {bests})   of 29")

print("\nlast-epoch value vs best value  (a big gap means late instability,")
print("a zero gap means the run ended at its peak = likely undertrained)")
for a in ARMS:
    if a not in hist:
        continue
    gaps = [max(c) - c[-1] for c in hist[a]]
    print(f"   {a:9s} best-minus-final {np.mean(gaps):+.2f}")

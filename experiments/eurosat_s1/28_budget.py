# -*- coding: utf-8 -*-
"""
28_budget.py - 30 vs 90 epochs, side by side.

The open question: at 30 epochs adaLN was still climbing hardest and ended
exactly at its own peak. Is it a bad injection point, or just a slow one?
"""
import sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

R = Path(r"C:\Users\aliso\Desktop\big-files\loc\runs")
ARMS = ("none", "add", "token", "adaln")

acc30, acc90 = {}, {}
for f in sorted(R.glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    (acc90 if f.stem.endswith("_e90") else acc30).setdefault(
        d["arm"], []).append(d["best_val_acc"] * 100)

print(f"{'arm':8s} {'30ep':>16s} {'90ep':>16s} {'gain':>8s}")
print("-" * 54)
base30 = base90 = None
for a in ARMS:
    v30 = np.array(acc30.get(a, []))
    v90 = np.array(acc90.get(a, []))
    s30 = f"{v30.mean():6.2f} (n={len(v30)})" if len(v30) else "     -"
    s90 = f"{v90.mean():6.2f} (n={len(v90)})" if len(v90) else "     -"
    g = f"{v90.mean()-v30.mean():+7.2f}" if len(v30) and len(v90) else "      -"
    print(f"{a:8s} {s30:>16s} {s90:>16s} {g:>8s}")
    if a == "none":
        base30 = v30.mean() if len(v30) else None
        base90 = v90.mean() if len(v90) else None

print("\ngain from location (arm minus none), per budget:")
for a in ("add", "token", "adaln"):
    v30, v90 = np.array(acc30.get(a, [])), np.array(acc90.get(a, []))
    p30 = f"{v30.mean()-base30:+6.2f}" if len(v30) and base30 else "    -"
    p90 = f"{v90.mean()-base90:+6.2f}" if len(v90) and base90 else "    -"
    print(f"   {a:8s} 30ep {p30}   90ep {p90}")

print("\nadaLN gap to the single scalar `add`:")
for lbl, d in (("30 epochs", acc30), ("90 epochs", acc90)):
    if "adaln" in d and "add" in d and d["adaln"] and d["add"]:
        gap = np.mean(d["adaln"]) - np.mean(d["add"])
        print(f"   {lbl}: {gap:+.2f}"
              + ("   still behind" if gap < 0 else "   CAUGHT UP"))

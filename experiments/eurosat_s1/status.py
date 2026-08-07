import json, time
from pathlib import Path
import numpy as np

H = Path(r"C:\Users\aliso\Desktop\big-files\loc")
R = H / "runs"
ARMS = ("none", "add", "token", "adaln", "gate", "shuffle")

rows = {}
for f in sorted(R.glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    rows.setdefault(d["arm"], []).append(d)

print(f"{'arm':10s} {'n':>2s}  {'mean':>7s} {'std':>7s}   seeds")
print("-" * 62)
for a in ARMS:
    rs = sorted(rows.get(a, []), key=lambda r: r["seed"])
    if not rs:
        print(f"{a:10s}  -")
        continue
    v = np.array([r["best_val_acc"] * 100 for r in rs])
    sd = v.std(ddof=1) if len(v) > 1 else float("nan")
    seeds = " ".join(f"{x:.2f}" for x in v)
    print(f"{a:10s} {len(v):2d}  {v.mean():7.2f} {sd:7.3f}   {seeds}")

    if a == "add":
        sc = [r.get("learned_scale") for r in rs if r.get("learned_scale") is not None]
        if sc:
            print(f"{'':10s}     learned scale: " + " ".join(f"{s:.4f}" for s in sc)
                  + f"   (Prithvi released: 0.05815)")
    if a == "gate":
        gs = [r.get("gate_stats") for r in rs if r.get("gate_stats")]
        for r, g in zip(rs, gs):
            print(f"{'':10s}     s{r['seed']} gate mean {g['mean']:+.4f} "
                  f"std {g['std']:.4f} range [{g['min']:+.3f},{g['max']:+.3f}]")

done = sum(len(v) for v in rows.values())
print(f"\n{done} of 30 runs done")

live = sorted(H.glob("_run_*.log"), key=lambda p: p.stat().st_mtime)
if live:
    p = live[-1]
    age = (time.time() - p.stat().st_mtime) / 60
    tail = [l for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    print(f"\ncurrent: {p.stem}   last update {age:.1f} min ago")
    for l in tail[-3:]:
        print("   " + l)

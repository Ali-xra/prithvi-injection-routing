# -*- coding: utf-8 -*-
"""
40_depth12.py - does the "window, not direction" hold with real depth resolution?

The window claim (gate_late beats add_mid only at a mid depth, dies early and
late) was measured on a 6-block model that has only 5 interior positions. This
re-runs it on a 12-block model so the window has resolution.

Design (EXPLORATORY map, 3 seeds; a 5-seed confirm at the peak is queued after):
  depth = 12, otherwise identical recipe to 24_train (geo split, 30 epochs).
  configs: none, add  (baselines: did the payload reach a 12-block from-scratch ViT?)
           add_mid @ mid, gate_late @ mid   for mid in {3, 6, 9}
  headline window = max_mid ( mean gate_late@mid - mean add_mid@mid )

VALIDITY GATE (checked first): add - none must be ~+4. If a 12-block from-scratch
ViT fails to learn the payload (add-none < 1), the window test is uninformative
and the runner should halt for a rethink (more epochs / fewer blocks), not report.

Writes depth12_result.json with the grid, the window, and add-none.
"""
import sys, json, time, importlib.util
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CACHE = HERE / "eurosat_s1_cache.npz"
OUT = HERE / "depth12_result.json"
DEPTH = 12
SEEDS = (0, 1, 2)
MIDS = (3, 6, 9)
EPOCHS = 30

_s = importlib.util.spec_from_file_location("m", HERE / "22_model.py")
M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M)
dev = "cuda" if torch.cuda.is_available() else "cpu"

z = np.load(CACHE, allow_pickle=True)
img, crd, lab = z["images"], z["coords"], z["labels"]
sp = np.load(HERE / "geo_split.npz", allow_pickle=True)["splits"]
tr, va = sp == "train", sp == "val"
mu = img[tr].mean(axis=(0, 2, 3), keepdims=True)
sd = img[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-6
imgn = (img - mu) / sd
Xtr = torch.from_numpy(imgn[tr]).float().to(dev)
Ctr = torch.from_numpy(crd[tr]).float().to(dev)
ytr = torch.from_numpy(lab[tr]).long().to(dev)
Xva = torch.from_numpy(imgn[va]).float().to(dev)
Cva = torch.from_numpy(crd[va]).float().to(dev)
yva = lab[va]
n_cls = int(lab.max()) + 1
print(f"geo split: train {tr.sum()}  val {va.sum()}  depth {DEPTH}", flush=True)


def train(arm, seed, mid):
    torch.manual_seed(seed); np.random.seed(seed)
    net = M.LocViT(arm, depth=DEPTH, n_classes=n_cls, mid_at=mid).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    n, bs = len(ytr), 256
    steps = (n + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3,
                                                total_steps=EPOCHS * steps, pct_start=0.15)
    use_c = arm != "none"
    best = 0.0
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            xb, yb = Xtr[idx], ytr[idx]
            cb = Ctr[idx] if use_c else None
            if torch.rand(1).item() < 0.5: xb = torch.flip(xb, dims=[3])
            if torch.rand(1).item() < 0.5: xb = torch.flip(xb, dims=[2])
            loss = F.cross_entropy(net(xb, cb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(Xva[i:i+512], None if not use_c else Cva[i:i+512]).argmax(1).cpu()
                              for i in range(0, len(Xva), 512)]).numpy()
        best = max(best, (pred == yva).mean())
    return float(best) * 100

CACHE_F = HERE / "depth12_cache.json"


def run_config(arm, mid=None):
    mm = mid if mid is not None else M.MID_AT
    key = f"{arm}@{mm}"
    cache = json.loads(CACHE_F.read_text()) if CACHE_F.exists() else {}
    if key in cache:                                    # resumable: reuse a finished config
        c = cache[key]
        print(f"   {arm:10s} mid={str(mid):>4}  {c['mean']:6.2f}  (cached)", flush=True)
        return c["mean"], c["accs"]
    accs = [train(arm, s, mm) for s in SEEDS]
    m = float(np.mean(accs))
    cache[key] = {"mean": m, "accs": accs}
    CACHE_F.write_text(json.dumps(cache, indent=2))
    print(f"   {arm:10s} mid={str(mid):>4}  {m:6.2f}  seeds={[round(a,2) for a in accs]}", flush=True)
    return m, accs


t0 = time.time()
res = {"none": run_config("none")[0], "add": run_config("add")[0]}
add_none = res["add"] - res["none"]

grid = {}
for mid in MIDS:
    am, _ = run_config("add_mid", mid)
    gl, _ = run_config("gate_late", mid)
    grid[mid] = {"add_mid": am, "gate_late": gl, "gate_late_minus_add_mid": gl - am}

window = max(v["gate_late_minus_add_mid"] for v in grid.values())
peak_mid = max(grid, key=lambda k: grid[k]["gate_late_minus_add_mid"])

# validity gate + verdict
if add_none < 1.0:
    verdict = "INVALID-12BLOCK-DID-NOT-LEARN-PAYLOAD"
elif window >= 0.4:
    verdict = "WINDOW-PERSISTS"
elif window < 0.2:
    verdict = "WINDOW-VANISHED"
else:
    verdict = "WEAK"

print("\n" + "=" * 66)
print(f"   add - none (payload reaches 12-block):  {add_none:+.2f}")
print(f"   window = max(gate_late - add_mid):      {window:+.2f}  at mid={peak_mid}")
print(f"   VERDICT: {verdict}   (6-block reference: window +0.82 at mid 3)")
print("=" * 66)

OUT.write_text(json.dumps({
    "design": "12-block depth window, 3 seeds exploratory, geo split",
    "seeds": list(SEEDS), "mids": list(MIDS), "depth": DEPTH,
    "none": res["none"], "add": res["add"], "add_minus_none": add_none,
    "grid": {str(k): v for k, v in grid.items()},
    "window": window, "peak_mid": peak_mid, "verdict": verdict,
    "minutes": round((time.time() - t0) / 60, 1),
}, indent=2), encoding="utf-8")
print(f"-> {OUT}  ({(time.time()-t0)/60:.1f} min)")

# -*- coding: utf-8 -*-
"""
39_logit_prior.py - does the class-prior argument hold on the ViT's OWN output?

35_prior_test.py showed a location prior recovers ~86% of `add`'s gain, but its
image-only model was a HistGradientBoosting on 12 hand features (val ~63 on the
geo split). Does that still hold when the image model is the actual 2.7M ViT?

RUN 1 (2026-08-07) was confounded: it averaged the five seeds' softmax first,
making the baseline a 5-model ENSEMBLE (76.07) rather than the single-model
`none` (73.21) the trained gain was defined on. See SPRINT-2026-08-GATE.md.

RUN 2 (this file): recovery is computed PER SEED against each single model, then
averaged - matched to how none/add were measured. Per-seed val logits are saved
to runs/ so re-analysis never needs retraining again.

  per seed:  p_img = softmax(single ViT logits)
             recovered = acc(p_img * p_loc^1) - acc(p_img)       (points)
  report mean +/- std over seeds; fraction of (add-none = +3.96) explained.
CONTROL: same prior from DERANGED val coords, per seed, must collapse to ~0.
"""
import sys, json, time, importlib.util
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
CACHE = HERE / "eurosat_s1_cache.npz"
RUNS = HERE / "runs"; RUNS.mkdir(exist_ok=True)
OUT = HERE / "logit_prior_result.json"
SEEDS = (0, 1, 2, 3, 4)
K = 200
SMOOTH = 1.0
ALPHA = 1.0
EPOCHS = 30
TRAINED_NONE, TRAINED_ADD = 73.21, 77.17

_s = importlib.util.spec_from_file_location("m", HERE / "22_model.py")
M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- data (geo split, train-only normalisation: same as 24_train) ----------
z = np.load(CACHE, allow_pickle=True)
img, crd, lab = z["images"], z["coords"], z["labels"]
sp = np.load(HERE / "geo_split.npz", allow_pickle=True)["splits"]
tr, va = sp == "train", sp == "val"
mu = img[tr].mean(axis=(0, 2, 3), keepdims=True)
sd = img[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-6
imgn = (img - mu) / sd
Xtr = torch.from_numpy(imgn[tr]).float().to(dev)
ytr = torch.from_numpy(lab[tr]).long().to(dev)
Xva = torch.from_numpy(imgn[va]).float().to(dev)
yva = lab[va]
n_cls = int(lab.max()) + 1
print(f"geo split: train {tr.sum()}  val {va.sum()}  classes {n_cls}", flush=True)


def train_none(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = M.LocViT("none", n_classes=n_cls).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    n, bs = len(ytr), 256
    steps = (n + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3,
                                                total_steps=EPOCHS * steps, pct_start=0.15)
    best_acc, best_logits = 0.0, None
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            xb, yb = Xtr[idx], ytr[idx]
            if torch.rand(1).item() < 0.5: xb = torch.flip(xb, dims=[3])
            if torch.rand(1).item() < 0.5: xb = torch.flip(xb, dims=[2])
            loss = F.cross_entropy(net(xb, None), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            logits = torch.cat([net(Xva[i:i+512], None).cpu() for i in range(0, len(Xva), 512)])
        acc = (logits.argmax(1).numpy() == yva).mean()
        if acc > best_acc:
            best_acc, best_logits = acc, logits.clone()
    return float(best_acc), best_logits.numpy()


# ---------- the regional location prior (same construction as 35) ----------
def prior_from(coords_val, coords_tr, labels_tr):
    nn = NearestNeighbors(n_neighbors=K).fit(coords_tr)
    _, idx = nn.kneighbors(coords_val)
    pl = np.eye(n_cls)[labels_tr][idx].sum(axis=1) + SMOOTH
    return pl / pl.sum(axis=1, keepdims=True)


ctr, cva, ltr = crd[tr], crd[va], lab[tr]
p_loc = prior_from(cva, ctr, ltr)
rng = np.random.default_rng(0)
perm = rng.permutation(len(cva))
while np.any(perm == np.arange(len(cva))):
    perm = rng.permutation(len(cva))
p_loc_shuf = prior_from(cva[perm], ctr, ltr)

eps = 1e-12
def acc_of(p):        return float((p.argmax(1) == yva).mean())
def comb(p_img, p_l, a): return np.log(p_img + eps) + a * np.log(p_l + eps)

# ---------- per-seed: train, save logits, measure recovery on the SINGLE model ----------
rec, rec_shuf, img_acc, ens_probs = [], [], [], []
for s in SEEDS:
    t0 = time.time()
    acc, logits = train_none(s)
    np.save(RUNS / f"none_geo_logits_s{s}.npy", logits)      # so we never retrain again
    p_img = F.softmax(torch.from_numpy(logits), dim=1).numpy()
    ens_probs.append(p_img)
    a_img = acc_of(p_img)
    a_comb = acc_of(comb(p_img, p_loc, ALPHA))
    a_shuf = acc_of(comb(p_img, p_loc_shuf, ALPHA))
    img_acc.append(a_img); rec.append((a_comb - a_img) * 100); rec_shuf.append((a_shuf - a_img) * 100)
    print(f"   seed {s}: img {a_img*100:6.2f}  +prior {a_comb*100:6.2f}"
          f"  recovered {(a_comb-a_img)*100:+.2f}  (control {(a_shuf-a_img)*100:+.2f})"
          f"  [{(time.time()-t0)/60:.1f} min]", flush=True)

rec = np.array(rec); rec_shuf = np.array(rec_shuf)
trained_gain = TRAINED_ADD - TRAINED_NONE
mean_img = float(np.mean(img_acc) * 100)
mean_rec, sd_rec = float(rec.mean()), float(rec.std(ddof=1))
frac = mean_rec / trained_gain * 100

# ensemble reference (what Run 1 wrongly used as the headline)
p_ens = np.mean(ens_probs, axis=0)
ens_img = acc_of(p_ens) * 100
ens_rec = (acc_of(comb(p_ens, p_loc, ALPHA)) - acc_of(p_ens)) * 100

print("\n" + "=" * 72)
print(f"PER-SEED single-model baseline (matched to none/add):")
print(f"   image-only mean            {mean_img:6.2f}")
print(f"   prior recovered (alpha=1)  {mean_rec:+.2f} +/- {sd_rec:.2f}  points")
print(f"   deranged-coord control     {rec_shuf.mean():+.2f}  (must be ~0 or negative)")
print(f"   trained gain (add-none)    {trained_gain:+.2f}")
print(f"   fraction of trained gain a prior explains on the real ViT: {frac:5.1f}%")
print(f"   (35's hand-feature probe gave 86%; ensemble-baseline Run 1 gave ~0)")
print(f"   [reference] 5-seed ensemble img {ens_img:.2f}  recovered {ens_rec:+.2f}")
print("=" * 72)

# alpha sweep on the per-seed mean (exploratory; alpha fixed at 1 in advance)
grid = {}
for a in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
    r = np.mean([acc_of(comb(p, p_loc, a)) - acc_of(p) for p in ens_probs]) * 100
    grid[a] = r
    print(f"      alpha {a:4.2f}   recovered {r:+.2f}")

verdict = ("PRIOR" if mean_rec >= 0.75 * trained_gain else
           "PARTLY" if mean_rec >= 0.4 * trained_gain else "NOT-MAINLY-PRIOR")
print(f"\nVERDICT (per-seed, single model): {verdict}")

OUT.write_text(json.dumps({
    "design": "per-seed single-model baseline; alpha=1 pre-registered; geo split",
    "seeds": list(SEEDS), "k": K,
    "image_only_mean": mean_img, "recovered_mean": mean_rec, "recovered_std": sd_rec,
    "recovered_per_seed": rec.tolist(), "control_recovered_mean": float(rec_shuf.mean()),
    "trained_gain": trained_gain, "fraction_explained": frac,
    "ensemble_ref": {"img": ens_img, "recovered": ens_rec},
    "alpha_grid": {str(k): v for k, v in grid.items()},
    "verdict": verdict,
}, indent=2), encoding="utf-8")
print(f"-> {OUT}")

# -*- coding: utf-8 -*-
"""
24_train.py - train one arm, one seed, on EuroSAT-S1.

Everything except the injection point is held fixed: same backbone, same
schedule, same augmentation, same seed handling. Fixed epoch budget, no early
stopping (burn-scars lesson: early stopping partly measures WHEN you stopped).

Run: python 24_train.py --arm add --seed 0 [--epochs 30]
"""
import sys, json, time, argparse, importlib.util
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CACHE = HERE / "eurosat_s1_cache.npz"
RUNS = HERE / "runs"

_s = importlib.util.spec_from_file_location("m", HERE / "22_model.py")
M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M)


def load(device, arm, shuffle_seed, split="official", shuffle_coords=False):
    z = np.load(CACHE, allow_pickle=True)
    img, crd, lab, sp = z["images"], z["coords"], z["labels"], z["splits"]

    # 🔴 additive only. `official` reproduces every locked result byte for byte.
    #    `geo` swaps in the location-disjoint split built by 30_make_geo_split.py,
    #    because the official split is random and the median val chip sits 1.13 km
    #    from a train chip (29_leakage_probe.py).
    if split == "geo":
        g = np.load(HERE / "geo_split.npz", allow_pickle=True)["splits"]
        if len(g) != len(sp):
            raise RuntimeError("geo split length does not match cache")
        sp = g
        print(f"   split: GEO (location-disjoint)")

    tr, va = sp == "train", sp == "val"
    # 🔴 normalisation statistics from the TRAINING split only.
    mu = img[tr].mean(axis=(0, 2, 3), keepdims=True)
    sd = img[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    img = (img - mu) / sd

    d = {}
    for name, mask in (("train", tr), ("val", va)):
        d[name] = [torch.from_numpy(img[mask]).float(),
                   torch.from_numpy(crd[mask]).float(),
                   torch.from_numpy(lab[mask]).long()]

    # 🔴 additive: `shuffle_coords` applies the exact same derangement to ANY arm.
    #    The `shuffle` arm alone only controls adaLN's capacity. It cannot test
    #    whether `token` gains from the 66th slot regardless of content, nor what
    #    our own `gate` does with meaningless coordinates. Same code path, so the
    #    control is byte-identical to the one already used.
    if arm == "shuffle" or shuffle_coords:
        # 🔴 destroy the image/location correspondence on the TRAIN split only,
        #    with zero fixed points. Val keeps its true coordinates, so the task
        #    itself is unchanged - only what the model could learn from.
        c = d["train"][1]
        g = torch.Generator().manual_seed(shuffle_seed)
        perm = torch.randperm(len(c), generator=g)
        idx = torch.arange(len(c))

        # 🔴 خطا — اندازه‌گیری‌شده ۱ اوت، seed 3:
        #        RuntimeError: shuffle left 2 fixed points
        #    نسخهٔ قبلی یک بار `torch.roll` می‌زد و امیدوار بود. چرخاندن یک
        #    جایگشت هیچ تضمینی برای حذف نقاط ثابت نمی‌دهد. assertion گرفتش —
        #    وگرنه دو نمونه مختصات درست خودشان را نگه می‌داشتند و کنترل
        #    بی‌صدا آلوده می‌شد.
        #
        #    اصلاح: نقاط ثابت را پیدا کن و **بین خودشان** بچرخان. اگر
        #    perm[f] == f برای همهٔ fها، بعد از چرخش perm[f_i] = f_{i-1} != f_i.
        #    برای تک نقطهٔ ثابت، با همسایه جابه‌جا کن.
        for _ in range(64):
            fixed = (perm == idx).nonzero().flatten()
            if len(fixed) == 0:
                break
            if len(fixed) == 1:
                f = fixed[0].item()
                j = (f + 1) % len(c)
                perm[[f, j]] = perm[[j, f]]
            else:
                perm[fixed] = perm[torch.roll(fixed, 1)]
        n_fixed = (perm == idx).sum().item()
        if n_fixed:
            raise RuntimeError(f"shuffle left {n_fixed} fixed points")
        if len(torch.unique(perm)) != len(c):
            raise RuntimeError("shuffle permutation is not a bijection")
        d["train"][1] = c[perm]
        print(f"   shuffle: {len(c)} coords permuted, 0 fixed points")

    for k in d:
        d[k] = [t.to(device) for t in d[k]]
    return d, int(z["labels"].max()) + 1


@torch.no_grad()
def evaluate(net, data, bs=512):
    net.eval()
    x, c, y = data
    correct, gates = 0, []
    for i in range(0, len(y), bs):
        xb, cb = x[i:i+bs], c[i:i+bs]
        if net.arm == "gate":
            out, g = net(xb, cb, return_gate=True)
            gates.append(g.squeeze(1).cpu())
        else:
            out = net(xb, None if net.arm == "none" else cb)
        correct += (out.argmax(1) == y[i:i+bs]).sum().item()
    g = torch.cat(gates) if gates else None
    return correct / len(y), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=M.ARMS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split", choices=("official", "geo"), default="official")
    ap.add_argument("--shuffle-coords", action="store_true",
                    help="apply the derangement control to any arm, not just `shuffle`")
    ap.add_argument("--mid-at", type=int, default=None,
                    help="blocks to run before a *_mid / *_late injection (default 3)")
    a = ap.parse_args()
    if a.shuffle_coords and a.arm in ("none", "shuffle"):
        raise SystemExit(f"--shuffle-coords is meaningless for arm {a.arm!r}")
    if a.mid_at is not None and a.arm not in ("add_mid", "gate_late"):
        raise SystemExit(f"--mid-at does nothing for arm {a.arm!r}")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    # 🔴 بودجهٔ آموزشی در تگ می‌آید. اگر نیاید، اجرای ۹۰ epoch روی نتیجهٔ
    #    ۳۰ epoch می‌نویسد و دو بودجه بی‌صدا قاتی می‌شوند — همان اشتباهی که
    #    با split در فاز آتش‌سوزی کردیم.
    tag = (f"{a.arm}_s{a.seed}"
           + ("" if a.epochs == 30 else f"_e{a.epochs}")
           + ("" if a.split == "official" else f"_{a.split}")
           + ("_shufcoord" if a.shuffle_coords else "")
           # 🔴 depth in the tag. Same reason as epochs: a depth-5 run must never
           #    silently overwrite the depth-3 result the whole claim rests on.
           + ("" if a.mid_at in (None, M.MID_AT) else f"_d{a.mid_at}"))
    print("=" * 70)
    print(f"arm {a.arm}   seed {a.seed}   {a.epochs} epochs   device {a.device}")
    print("=" * 70)

    data, n_cls = load(a.device, a.arm, shuffle_seed=1000 + a.seed, split=a.split,
                       shuffle_coords=a.shuffle_coords)
    net = M.LocViT(a.arm, n_classes=n_cls, mid_at=a.mid_at).to(a.device)
    n_inj = M.count_injection_params(net)
    print(f"   params {sum(p.numel() for p in net.parameters())/1e6:.3f} M"
          f"   injection {n_inj:,}")
    print(f"   train {len(data['train'][2])}   val {len(data['val'][2])}")

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.05)
    x, c, y = data["train"]
    n = len(y)
    steps = (n + a.batch_size - 1) // a.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * steps, pct_start=0.15)

    best, best_ep, hist = 0.0, -1, []
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(n, device=x.device)
        tot = 0.0
        for i in range(0, n, a.batch_size):
            idx = perm[i:i+a.batch_size]
            xb, cb, yb = x[idx], c[idx], y[idx]
            if torch.rand(1).item() < 0.5:                 # flips: label-safe
                xb = torch.flip(xb, dims=[3])
            if torch.rand(1).item() < 0.5:
                xb = torch.flip(xb, dims=[2])
            out = net(xb, None if a.arm == "none" else cb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(yb)

        acc, gate = evaluate(net, data["val"])
        hist.append({"epoch": ep, "loss": tot / n, "val_acc": acc})
        if acc > best:
            best, best_ep = acc, ep
        extra = ""
        if a.arm == "add":
            extra = f"  scale={net.scale.item():+.5f}"
        elif a.arm == "gate" and gate is not None:
            extra = (f"  gate mean={gate.mean():+.4f} std={gate.std():.4f}"
                     f" range=[{gate.min():+.3f},{gate.max():+.3f}]")
        print(f"   ep {ep:3d}  loss {tot/n:.4f}  val {acc*100:6.2f}"
              f"  best {best*100:6.2f}{extra}", flush=True)

    mins = (time.time() - t0) / 60
    _, gate = evaluate(net, data["val"])
    rec = {
        "arm": a.arm, "seed": a.seed, "epochs": a.epochs,
        "shuffle_coords": bool(a.shuffle_coords),
        "mid_at": net.mid_at,
        "best_val_acc": best, "best_epoch": best_ep,
        "final_val_acc": hist[-1]["val_acc"],
        "injection_params": n_inj, "minutes": round(mins, 2),
        "lr": a.lr, "batch_size": a.batch_size,
        "history": hist,
    }
    if a.arm == "add":
        rec["learned_scale"] = net.scale.item()
    if a.arm == "gate" and gate is not None:
        rec["gate_stats"] = {"mean": float(gate.mean()), "std": float(gate.std()),
                             "min": float(gate.min()), "max": float(gate.max())}
        np.save(RUNS / f"{tag}_gates.npy", gate.numpy())

    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{tag}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n   best val {best*100:.2f} at epoch {best_ep}   {mins:.1f} min")
    print(f"   -> runs/{tag}.json")


if __name__ == "__main__":
    main()

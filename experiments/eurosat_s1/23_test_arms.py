# -*- coding: utf-8 -*-
"""
23_test_arms.py - prove each arm actually uses location before training anything.

Written because of the single most expensive bug of the burn-scars phase: an
equivalence test that only checked "these should be equal" passed while the
injection was silently disconnected. Every check here has a MUST-DIFFER twin.

  T1  none   ignores coords entirely                       (must be equal)
  T2  each injecting arm changes output when coords change (MUST DIFFER)
  T3  adaln at init == none                                (zero-init identity)
  T4  every injection parameter receives gradient          (MUST be non-zero)
  T5  gate depends on the image, not on the coordinates    (must be equal)

Run: python 23_test_arms.py
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import torch
import importlib.util

_s = importlib.util.spec_from_file_location("m", Path(__file__).parent / "22_model.py")
m = importlib.util.module_from_spec(_s); _s.loader.exec_module(m)

torch.manual_seed(0)
B = 8
img = torch.randn(B, 2, 64, 64)
c1 = torch.stack([torch.rand(B) * 40 - 10, torch.rand(B) * 25 + 35], dim=1)
c2 = torch.stack([torch.rand(B) * 40 - 10, torch.rand(B) * 25 + 35], dim=1)

fails = []


def check(name, cond, detail=""):
    print(f"   {'PASS' if cond else 'FAIL'}  {name}   {detail}")
    if not cond:
        fails.append(name)


print("\nT1  none ignores coords")
net = m.LocViT("none").eval()
with torch.no_grad():
    a, b = net(img), net(img)
check("none deterministic", torch.equal(a, b))

print("\nT2  each injecting arm responds to a change of coordinates  (MUST DIFFER)")
INJECTING = ("add", "token", "adaln", "gate", "shuffle",
             "add_mid", "gate_late", "gate_std", "gate_max", "gate_coord", "film")
for arm in INJECTING:
    net = m.LocViT(arm).eval()
    # zero-initialised arms would pass this trivially by being the identity,
    # so randomise their modulators first.
    if arm in ("adaln", "shuffle"):
        for mod in net.to_mod:
            torch.nn.init.normal_(mod[1].weight, std=0.02)
            torch.nn.init.normal_(mod[1].bias, std=0.02)
    if arm == "film":
        torch.nn.init.normal_(net.to_film[1].weight, std=0.02)
        torch.nn.init.normal_(net.to_film[1].bias, std=0.02)
    with torch.no_grad():
        o1, o2 = net(img, c1), net(img, c2)
    d = (o1 - o2).abs().max().item()
    check(f"{arm}: output changes with coords", d > 1e-6, f"max|delta|={d:.3e}")

print("\nT3  adaln at initialisation is identical to none  (zero-init identity)")
torch.manual_seed(0); base = m.LocViT("none").eval()
torch.manual_seed(0); ada = m.LocViT("adaln").eval()
ada.load_state_dict(base.state_dict(), strict=False)
with torch.no_grad():
    d = (base(img) - ada(img, c1)).abs().max().item()
check("adaln zero-init == none", d < 1e-5, f"max|delta|={d:.3e}")

torch.manual_seed(0); fil = m.LocViT("film").eval()
fil.load_state_dict(base.state_dict(), strict=False)
with torch.no_grad():
    d = (base(img) - fil(img, c1)).abs().max().item()
check("film zero-init == none", d < 1e-5, f"max|delta|={d:.3e}")


print("\nT4  every injection parameter receives gradient  (MUST be non-zero)")
y = torch.randint(0, 10, (B,))
for arm in INJECTING:
    net = m.LocViT(arm).train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    keys = ("scale", "loc_proj", "to_mod", "gate_net", "scale_mid", "to_film")
    inj = [(n, p) for n, p in net.named_parameters() if any(k in n for k in keys)]
    check(f"{arm}: injection params exist", len(inj) > 0, f"{len(inj)} tensors")

    # 🔴 adaLN-zero has exactly zero gradient at step 1 by construction
    #    (dL/dloc-encoder = W^T g = 0 while W is zero). Measured on burn scars:
    #    0.000e+00 at step 1, 1.38e-01 at step 2. So we look at the LAST step.
    seen = 0.0
    for step in range(3):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(net(img, c1), y).backward()
        seen = sum(p.grad.abs().sum().item() for _, p in inj if p.grad is not None)
        opt.step()
    check(f"{arm}: injection params get gradient", seen > 0, f"|grad|={seen:.3e}")

    ids = {id(p) for p in net.parameters()}
    check(f"{arm}: injection params in optimiser",
          all(id(p) in ids for _, p in inj))

print("\nT5  image-fed gates read the image, not the coordinates  (must be equal)")
for arm in ("gate", "gate_late", "gate_std", "gate_max"):
    net = m.LocViT(arm).eval()
    with torch.no_grad():
        _, g1 = net(img, c1, return_gate=True)
        _, g2 = net(img, c2, return_gate=True)
    check(f"{arm} independent of coords", torch.allclose(g1, g2),
          f"max|delta|={(g1-g2).abs().max().item():.3e}")

print("\nT6  the coordinate-fed gate MUST depend on coords  (must-differ twin of T5)")
net = m.LocViT("gate_coord").eval()
# zero-init makes gate_net output the constant 0.1 for every input, so this
# would fail trivially. Randomise the head first - same treatment as T7.
torch.nn.init.normal_(net.gate_net[2].weight, std=0.05)
with torch.no_grad():
    _, g1 = net(img, c1, return_gate=True)
    _, g2 = net(img, c2, return_gate=True)
d = (g1 - g2).abs().max().item()
check("gate_coord depends on coords", d > 1e-6, f"max|delta|={d:.3e}")

print("\nT7  image-fed gates MUST depend on the image  (must-differ)")
img2 = torch.randn(B, 2, 64, 64)
for arm in ("gate", "gate_late", "gate_std", "gate_max"):
    net = m.LocViT(arm).eval()
    # zero-init makes gate_net output constant, so randomise the head first
    torch.nn.init.normal_(net.gate_net[2].weight, std=0.05)
    with torch.no_grad():
        _, ga = net(img, c1, return_gate=True)
        _, gb = net(img2, c1, return_gate=True)
    d = (ga - gb).abs().max().item()
    check(f"{arm} depends on the image", d > 1e-6, f"max|delta|={d:.3e}")

print("\nT8  mid_at is actually honoured  (must-differ across depths)")
# 🔴 6 Aug 2026. MID_AT used to be a module constant. If --mid-at were silently
#    ignored, a whole depth sweep would produce five identical arms and look like
#    a beautiful flat profile. This test is the thing that would catch that.
check("default mid_at is still 3", m.LocViT("gate_late").mid_at == 3)
for arm in ("gate_late", "add_mid"):
    outs = []
    for d in (1, 6):
        torch.manual_seed(0)                      # identical weights, only depth differs
        net = m.LocViT(arm, mid_at=d).eval()
        if arm == "gate_late":
            torch.nn.init.normal_(net.gate_net[2].weight, std=0.05)
        with torch.no_grad():
            outs.append(net(img, c1))
    delta = (outs[0] - outs[1]).abs().max().item()
    check(f"{arm}: depth 1 differs from depth 6", delta > 1e-6, f"max|delta|={delta:.3e}")
try:
    m.LocViT("gate_late", mid_at=7)
    check("mid_at beyond depth is rejected", False)
except AssertionError:
    check("mid_at beyond depth is rejected", True)

print("\ninjection parameter counts")
for arm in m.ARMS:
    net = m.LocViT(arm)
    tot = sum(p.numel() for p in net.parameters())
    print(f"   {arm:9s} total {tot/1e6:6.3f} M   injection {m.count_injection_params(net):,}")

print("\n" + ("ALL PASS" if not fails else f"FAILED: {fails}"))
sys.exit(1 if fails else 0)

# -*- coding: utf-8 -*-
"""
22_model.py - one small ViT, five ways to inject location.

The point of the whole study: the PAYLOAD is identical in every arm (the same
lon/lat for the same chip). Only the ENTRY POINT changes. Anything that differs
between arms is therefore attributable to routing, not to information.

Arms
    none      no location at all
    add       Prithvi-EO-2.0 style: fixed sinusoidal location embedding, scaled by
              ONE learned global scalar, added to every patch token.
              (Prithvi's released 300M-TL has this scalar at 0.05815186.)
    token     TerraMind style: location becomes an extra token in the sequence.
    adaln     per-block modulation: every block's two norms get a shift and scale
              predicted from location.
    gate      ours: like `add`, but the scalar is predicted PER SAMPLE from the
              image itself instead of being one global constant.
    shuffle   control: `adaln` with each chip given another chip's coordinates.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

ARMS = ("none", "add", "token", "adaln", "gate", "shuffle",
        # 🔴 additive, 6 Aug 2026. The six arms above are frozen and reproduce
        #    every locked number. These six are the follow-up from Measurement 19
        #    (the gate tracks mean backscatter at r = -0.90 because it reads the
        #    mean patch embedding BEFORE any attention block).
        "add_mid",     # control for gate_late: same mid injection, constant scalar
        "gate_late",   # gate read from CLS after MID_AT blocks, injected there
        "gate_std",    # gate input = [mean, std] of patch tokens, same entry point
        "gate_max",    # gate input = max over patch tokens, same entry point
        "gate_coord",  # gate fed by the COORDINATES instead of the image
        "film")        # per-dimension shift+scale from location, applied once

MID_AT = 3             # how many blocks run before a `*_mid` / `*_late` injection


def sincos_location(coords: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Fixed sinusoidal encoding of (lon, lat) - no learned parameters.

    Deliberately parameter-free so that `add`, `token`, `adaln` and `gate` all
    start from the SAME representation of location. If the encoder itself were
    learned, arms would differ in how well they can read the coordinates, not
    only in where they inject them, and the comparison would be confounded.

    coords: (B, 2) raw lon/lat in degrees.
    """
    lon = coords[:, 0] / 180.0 * math.pi
    lat = coords[:, 1] / 90.0 * math.pi
    half = dim // 4
    freqs = torch.exp(torch.arange(half, device=coords.device, dtype=torch.float32)
                      * (-math.log(10000.0) / max(half - 1, 1)))
    out = []
    for ang in (lon, lat):
        a = ang[:, None] * freqs[None, :] * 10.0
        out += [torch.sin(a), torch.cos(a)]
    e = torch.cat(out, dim=1)                      # (B, 4*half)
    if e.shape[1] < dim:                           # pad if dim not divisible by 4
        e = torch.cat([e, torch.zeros(e.shape[0], dim - e.shape[1],
                                      device=e.device, dtype=e.dtype)], dim=1)
    return e


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x, mod=None):
        if mod is None:
            a = self.n1(x)
            x = x + self.attn(a, a, a, need_weights=False)[0]
            return x + self.mlp(self.n2(x))
        s1, g1, s2, g2 = mod
        a = self.n1(x) * (1 + g1.unsqueeze(1)) + s1.unsqueeze(1)
        x = x + self.attn(a, a, a, need_weights=False)[0]
        b = self.n2(x) * (1 + g2.unsqueeze(1)) + s2.unsqueeze(1)
        return x + self.mlp(b)


class LocViT(nn.Module):
    def __init__(self, arm="none", dim=192, depth=6, heads=3,
                 patch=8, in_ch=2, img=64, n_classes=10, mid_at=None):
        super().__init__()
        assert arm in ARMS, arm
        self.arm = arm
        self.dim = dim
        # 🔴 6 Aug 2026: MID_AT was a module constant, so every `*_mid` / `*_late`
        #    number ever recorded used 3 and nothing else. Made per-instance so the
        #    depth can be swept. Default is still 3 — no existing result moves.
        self.mid_at = MID_AT if mid_at is None else int(mid_at)
        assert 1 <= self.mid_at <= depth, f"mid_at {self.mid_at} outside 1..{depth}"
        n_patch = (img // patch) ** 2

        self.stem = nn.Conv2d(in_ch, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patch + 1, dim) * 0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

        # ---- the only thing that differs between arms ----
        if arm == "add":
            # Prithvi: ONE global scalar gating a fixed location embedding.
            # Init 0.1 to match terratorch's prithvi_mae default.
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        elif arm == "token":
            # TerraMind: location as its own token in the sequence.
            self.loc_proj = nn.Linear(dim, dim)
        elif arm in ("adaln", "shuffle"):
            self.to_mod = nn.ModuleList([
                nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim)) for _ in range(depth)])
            for m in self.to_mod:                 # zero-init -> starts as identity
                nn.init.zeros_(m[1].weight); nn.init.zeros_(m[1].bias)
        elif arm == "gate":
            # ours: same additive path as `add`, but the scalar is predicted
            # per sample from the image, so the model can decide chip by chip
            # how much to trust location.
            self.gate_net = nn.Sequential(nn.Linear(dim, dim // 4), nn.SiLU(),
                                          nn.Linear(dim // 4, 1))
            nn.init.zeros_(self.gate_net[2].weight)
            nn.init.constant_(self.gate_net[2].bias, 0.1)   # same start as `add`

        # ---- follow-up arms (additive, 6 Aug 2026) ----
        elif arm == "add_mid":
            # control for `gate_late`: identical mid injection, constant scalar.
            # Without this we could not tell whether a gain came from reading the
            # gate later or from injecting later.
            self.scale_mid = nn.Parameter(torch.full((1,), 0.1))
        elif arm in ("gate_late", "gate_std", "gate_max", "gate_coord"):
            # same two-layer shape and same zero/0.1 initialisation as `gate`,
            # so every one of these starts bit-identical to `add`.
            in_dim = 2 * dim if arm == "gate_std" else dim
            self.gate_net = nn.Sequential(nn.Linear(in_dim, dim // 4), nn.SiLU(),
                                          nn.Linear(dim // 4, 1))
            nn.init.zeros_(self.gate_net[2].weight)
            nn.init.constant_(self.gate_net[2].bias, 0.1)
        elif arm == "film":
            # the missing rung between `gate` (one scalar) and `adaln`
            # (per-dimension, per-block): per-dimension, ONE point.
            self.to_film = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
            nn.init.zeros_(self.to_film[1].weight)
            nn.init.zeros_(self.to_film[1].bias)            # starts as identity

    def forward(self, img, coords=None, return_gate=False):
        B = img.shape[0]
        x = self.stem(img).flatten(2).transpose(1, 2)          # (B, P, D)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        x = x + self.pos

        if self.arm == "none":
            loc = None
        else:
            if coords is None:
                raise ValueError(f"arm {self.arm!r} needs coords")
            loc = sincos_location(coords, self.dim)             # (B, D)

        gate_val = None
        if self.arm == "add":
            x = x + self.scale * loc.unsqueeze(1)
        elif self.arm == "gate":
            # 🔴 gate reads the IMAGE (mean patch token), never the location -
            # otherwise it could smuggle location in through a second path and
            # the comparison with `add` would no longer isolate the gate.
            g = self.gate_net(x[:, 1:].mean(dim=1))             # (B, 1)
            gate_val = g
            x = x + g.unsqueeze(1) * loc.unsqueeze(1)
        elif self.arm == "token":
            x = torch.cat([x, self.loc_proj(loc).unsqueeze(1)], dim=1)
        elif self.arm == "gate_std":
            p = x[:, 1:]
            g = self.gate_net(torch.cat([p.mean(dim=1), p.std(dim=1)], dim=1))
            gate_val = g
            x = x + g.unsqueeze(1) * loc.unsqueeze(1)
        elif self.arm == "gate_max":
            g = self.gate_net(x[:, 1:].amax(dim=1))
            gate_val = g
            x = x + g.unsqueeze(1) * loc.unsqueeze(1)
        elif self.arm == "gate_coord":
            # 🔴 exploratory arm: location now reaches the model by TWO paths -
            #    once as `loc`, once inside `g`. Not a clean rival to `add`.
            g = self.gate_net(loc)
            gate_val = g
            x = x + g.unsqueeze(1) * loc.unsqueeze(1)
        elif self.arm == "film":
            shift, scale = self.to_film(loc).chunk(2, dim=1)
            x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        for i, blk in enumerate(self.blocks):
            if self.arm in ("adaln", "shuffle"):
                x = blk(x, self.to_mod[i](loc).chunk(4, dim=1))
            else:
                x = blk(x)
            # 🔴 mid-stack injection, after MID_AT blocks have run
            if i == self.mid_at - 1:
                if self.arm == "add_mid":
                    x = x + self.scale_mid * loc.unsqueeze(1)
                elif self.arm == "gate_late":
                    # read the CLS token, which by now has seen the whole image
                    # through three rounds of attention
                    g = self.gate_net(x[:, 0])
                    gate_val = g
                    x = x + g.unsqueeze(1) * loc.unsqueeze(1)

        out = self.head(self.norm(x)[:, 0])
        return (out, gate_val) if return_gate else out


def count_injection_params(model):
    """Parameters that exist only because of the injection path."""
    keys = ("scale", "loc_proj", "to_mod", "gate_net", "scale_mid", "to_film")
    return sum(p.numel() for n, p in model.named_parameters()
               if any(k in n for k in keys))

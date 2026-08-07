# -*- coding: utf-8 -*-
"""
adaln.py — تزریق اسکالر سراسری از مسیر نرمال‌سازی هر بلوک
============================================================
نسخه: v1 · تاریخ: 2026-07-30 · **مشترک بین هر دو تسک** — بازنویسی نکن، import کن

بردار شرط: **۴ بُعد** — `lat_z, lon_z, doy_sin_z, doy_cos_z`
(از ۱۰ بُعد کم شد؛ دلیلش اندازه‌گیری `12_image_proxy_control.py`:
 شش بُعد جوّی فراتر از تصویر چیزی ندادند و چهار بُعد مفید را رقیق می‌کردند.)

مکانیزم — همان adaLN-Zero مقالهٔ DiT، با **یک تفاوت حیاتی**:

    DiT سه چیز تولید می‌کند: shift · scale · **gate**، و هر سه را صفر می‌دهد.
    ما **gate را حذف می‌کنیم.**

    🔴 چرا: در DiT مدل **از صفر** آموزش می‌بیند، پس gate صفر یعنی «بلوک فعلاً
    کاری نکند» و بعد یاد می‌گیرد. اینجا مدل **از پیش آموزش‌دیده** است — gate صفر
    یعنی خروجی هر ۲۴ بلوک Prithvi ضربدر صفر، و مدل **کور** می‌شود.
    فقط shift و scale صفر مقداردهی می‌شوند.

با مقداردهی صفر، در گام صفرِ آموزش:
    modulate(y, shift=0, scale=0) = y * (1 + 0) + 0 = y
یعنی خروجی **بیت‌به‌بیت** برابر خط پایه است. `test_equivalence.py` همین را می‌سنجد.

ساختار بلوک Prithvi (اندازه‌گیری‌شده در `lab/src/13_inspect_backbone.py`):
    norm1 → attn → ls1 → drop_path1   ·   norm2 → mlp → ls2 → drop_path2
    ۲۴ بلوک · embed_dim=1024 · هر دو LayerNorm با affine=True
"""
from __future__ import annotations

import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x شکل (B, N, D) · shift و scale شکل (B, D) → پخش روی محور توکن."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ScalarEncoder(nn.Module):
    """بردار شرط کم‌بعد → یک امبدینگ مشترک که همهٔ بلوک‌ها از آن تغذیه می‌کنند."""

    def __init__(self, in_dim: int = 4, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.out_dim = hidden

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


class AdaLNBlockWrapper(nn.Module):
    """
    یک بلوک ترانسفورمر Prithvi را می‌پوشاند و LayerNormهایش را با بردار شرط
    تنظیم می‌کند.

    ⚠️ زیرماژول‌های بلوک اصلی **دست‌نخورده** استفاده می‌شوند (`inner.attn`,
    `inner.mlp`, ...) — پس وزن‌های از پیش آموزش‌دیده عیناً سر جایشان می‌مانند.
    فقط ترتیبِ فراخوانی بازنویسی می‌شود تا `modulate` وسطش جا شود.

    شرط از یک متغیر ماژول‌سطحی خوانده می‌شود (`self.cond`)، نه از آرگومان forward،
    چون امضای forward بلوک را نمی‌توانیم در زنجیرهٔ TerraTorch عوض کنیم.
    """

    def __init__(self, inner: nn.Module, cond_dim: int, embed_dim: int):
        super().__init__()
        self.inner = inner
        # چهار خروجی: shift1, scale1, shift2, scale2 — **بدون gate**
        self.to_mod = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 4 * embed_dim))
        nn.init.zeros_(self.to_mod[1].weight)
        nn.init.zeros_(self.to_mod[1].bias)
        self.embed_dim = embed_dim
        self.cond: torch.Tensor | None = None      # هر گام از بیرون ست می‌شود

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        b = self.inner
        if self.cond is None:                       # حالت خط پایه
            return b(x, *args, **kwargs)

        s1, c1, s2, c2 = self.to_mod(self.cond).chunk(4, dim=-1)
        h = b.attn(modulate(b.norm1(x), s1, c1))
        h = b.ls1(h) if hasattr(b, "ls1") else h
        h = b.drop_path1(h) if hasattr(b, "drop_path1") else h
        x = x + h

        h = b.mlp(modulate(b.norm2(x), s2, c2))
        h = b.ls2(h) if hasattr(b, "ls2") else h
        h = b.drop_path2(h) if hasattr(b, "drop_path2") else h
        return x + h


class AdaLNInjector(nn.Module):
    """
    ۲۴ بلوک بک‌بون را می‌پوشاند و شرط را قبل از هر forward پخش می‌کند.

    استفاده:
        inj = AdaLNInjector(model, cond_dim=4)
        inj.set_condition(c)      # c شکل (B, 4)
        out = model(x)
    """

    def __init__(self, model: nn.Module, cond_dim: int = 4, hidden: int = 256):
        super().__init__()
        bb = self._find_backbone(model)
        blocks = bb.blocks
        embed_dim = blocks[0].norm1.normalized_shape[0]
        self.encoder = ScalarEncoder(cond_dim, hidden)
        # 🔴 حیاتی — اندازه‌گیری‌شده 2026-07-30 در `14_smoke_cpu.py`:
        #    `to_mod` داخل بلوک‌هاست، پس در `model.parameters()` می‌آید. ولی
        #    `encoder` فرزندِ **injector** است نه مدل. `configure_optimizers` تسک
        #    فقط `model.parameters()` را می‌بیند → انکودر **هرگز آموزش نمی‌دید** و
        #    یک تصویرسازی تصادفیِ ثابت می‌ماند. هیچ خطایی هم نمی‌داد.
        #    ثبتش می‌کنیم تا در بهینه‌ساز بیاید.
        model.add_module("cond_encoder", self.encoder)
        self.wrappers = nn.ModuleList()
        for i in range(len(blocks)):
            w = AdaLNBlockWrapper(blocks[i], hidden, embed_dim)
            blocks[i] = w
            self.wrappers.append(w)
        self.n_blocks, self.embed_dim = len(blocks), embed_dim

    @staticmethod
    def _find_backbone(model: nn.Module) -> nn.Module:
        for name, mod in model.named_children():
            if hasattr(mod, "blocks"):
                return mod
        raise AttributeError("بک‌بونی با صفت `blocks` پیدا نشد")

    def set_condition(self, c: torch.Tensor | None) -> None:
        """c شکل (B, cond_dim) — یا None برای برگشت دقیق به حالت خط پایه."""
        e = None if c is None else self.encoder(c)
        for w in self.wrappers:
            w.cond = e

    def extra_parameters(self):
        """پارامترهای تازه — برای گزارش «چند پارامتر اضافه شد»."""
        yield from self.encoder.parameters()
        for w in self.wrappers:
            yield from w.to_mod.parameters()

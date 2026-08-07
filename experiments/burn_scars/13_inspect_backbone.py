# -*- coding: utf-8 -*-
"""
13_inspect_backbone.py — ساختار واقعی بلوک‌های Prithvi، برای وصل کردن adaLN
=============================================================================
نسخه: v1 · تاریخ: 2026-07-30

چرا این اول از همه: ریسکی‌ترین بخش پروژه، وصل کردن adaLN به یک ViT **از پیش
آموزش‌دیده** است. اگر بلوک‌ها قابل بسته‌بندی نباشند یا LayerNorm جای غیرمنتظره‌ای
باشد، آن بازو می‌میرد — و بهتر است امروز بفهمیم نه سه روز دیگر.

چه چیزی را می‌خواهیم بدانیم:
    ۱) بلوک‌های ترانسفورمر کجا هستند و اسمشان چیست؟ چندتا؟
    ۲) داخل هر بلوک، LayerNormها کدام‌اند؟ (`norm1` قبل از attention، `norm2` قبل از MLP)
    ۳) بعد ویژگی (`embed_dim`) چند است؟
    ۴) آیا مدل مسیر متادیتای خودش را دارد؟ (`temporal_embed`, `location_embed`,
       `coords_encoding`, یا هر نام دیگری) — این **بازوی صفر** ماست
    ۵) امضای `forward` بک‌بون چیست؟ آیا آرگومان اضافه برای زمان و مکان می‌گیرد؟

هیچ چیزی عوض نمی‌شود. فقط گزارش.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import inspect
import torch
import torch.nn as nn
from terratorch.models import EncoderDecoderFactory

ARGS = dict(
    backbone="prithvi_eo_v2_300", backbone_pretrained=False,
    backbone_bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
    necks=[{"name": "SelectIndices", "indices": [5, 11, 17, 23]},
           {"name": "ReshapeTokensToImage"},
           {"name": "LearnedInterpolateToPyramidal"}],
    decoder="UNetDecoder", decoder_channels=[512, 256, 128, 64], num_classes=2,
)

print("ساخت مدل (بدون وزن، فقط ساختار)...", flush=True)
model = EncoderDecoderFactory().build_model(task="segmentation", **ARGS)

# --- پیدا کردن بک‌بون ---
bb = None
for name, mod in model.named_children():
    print(f"  فرزند سطح اول: {name:<12} {type(mod).__name__}")
    if "encoder" in name or "backbone" in name:
        bb = mod
if bb is None:
    bb = dict(model.named_children()).get("encoder")

print("\n" + "=" * 78)
print("۱. بک‌بون")
print("=" * 78)
print(f"   کلاس: {type(bb).__name__}")
print(f"   ماژول‌های سطح اول: {[n for n, _ in bb.named_children()]}")

# --- بلوک‌ها ---
print("\n" + "=" * 78)
print("۲. بلوک‌های ترانسفورمر")
print("=" * 78)
blocks = None
for cand in ("blocks", "layers", "encoder_layers", "transformer"):
    if hasattr(bb, cand):
        blocks = getattr(bb, cand)
        print(f"   ✅ پیدا شد زیر نام: `{cand}`   تعداد: {len(blocks)}")
        break
if blocks is None:
    print("   ⛔ پیدا نشد. فهرست کامل ماژول‌ها:")
    for n, _ in list(bb.named_modules())[:40]:
        print("      ", n)
    sys.exit(1)

b0 = blocks[0]
print(f"\n   ساختار بلوک ۰ ({type(b0).__name__}):")
for n, m in b0.named_children():
    extra = ""
    if isinstance(m, nn.LayerNorm):
        extra = f"  ← LayerNorm  shape={tuple(m.normalized_shape)}  affine={m.elementwise_affine}"
    print(f"      {n:<12} {type(m).__name__}{extra}")

norms = [n for n, m in b0.named_children() if isinstance(m, nn.LayerNorm)]
print(f"\n   🔴 LayerNormهای قابل تنظیم در هر بلوک: {norms}")
dim = getattr(bb, "embed_dim", None) or blocks[0].norm1.normalized_shape[0]
print(f"   بعد ویژگی (embed_dim): {dim}")


# --- مسیر متادیتای خود مدل: بازوی صفر ما ---
print("\n" + "=" * 78)
print("۳. 🔴 مسیر متادیتای خودِ Prithvi — بازوی صفر")
print("=" * 78)
KEYS = ("temporal", "location", "coord", "meta", "time", "date", "geo")
hits = [n for n, _ in bb.named_modules() if any(k in n.lower() for k in KEYS)]
attrs = [n for n in dir(bb) if not n.startswith("_") and any(k in n.lower() for k in KEYS)]
print(f"   ماژول‌های مرتبط : {hits if hits else '— هیچ —'}")
print(f"   صفت‌های مرتبط   : {attrs if attrs else '— هیچ —'}")
params = [n for n, _ in bb.named_parameters() if any(k in n.lower() for k in KEYS)]
print(f"   پارامترهای مرتبط: {params[:10] if params else '— هیچ —'}")

print("\n   امضای forward بک‌بون:")
try:
    sig = inspect.signature(bb.forward)
    for p in sig.parameters.values():
        print(f"      {p.name:<20} default={p.default}")
except Exception as e:
    print("      ", e)

# --- آیا کلاس بک‌بون گزینهٔ متادیتا در __init__ دارد؟ ---
print("\n   پارامترهای __init__ که به متادیتا مربوط‌اند:")
try:
    sig = inspect.signature(type(bb).__init__)
    found = [f"{p.name}={p.default}" for p in sig.parameters.values()
             if any(k in p.name.lower() for k in KEYS)]
    print(f"      {found if found else '— هیچ —'}")
except Exception as e:
    print("      ", e)

# --- آزمون بسته‌بندی: آیا می‌شود یک بلوک را جایگزین کرد؟ ---
print("\n" + "=" * 78)
print("۴. آزمون بسته‌بندی — آیا جایگزینی بلوک ممکن است؟")
print("=" * 78)


class Probe(nn.Module):
    """بلوک را می‌پوشاند و فقط می‌شمارد که صدا زده شد."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, *a, **kw):
        self.calls += 1
        return self.inner(*a, **kw)


blocks[0] = Probe(blocks[0])
blocks[12] = Probe(blocks[12])
x = torch.randn(1, 6, 224, 224)
model.eval()
try:
    with torch.no_grad():
        out = model(x)
    y = out.output if hasattr(out, "output") else out
    print(f"   ✅ forward با بلوک‌های پوشانده‌شده کار کرد → خروجی {tuple(y.shape)}")
    print(f"   بلوک ۰ صدا زده شد: {blocks[0].calls} بار · بلوک ۱۲: {blocks[12].calls} بار")
    print("\n   🟢 **جایگزینی بلوک ممکن است.** مسیر adaLN باز است.")
except Exception as e:
    print(f"   ⛔ شکست: {type(e).__name__}: {e}")
    print("   🔴 مسیر adaLN با این روش بسته است — روش دیگری لازم است (hook یا monkey-patch)")

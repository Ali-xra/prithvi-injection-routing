# -*- coding: utf-8 -*-
"""
15_arm0_official.py — بازوی صفر: مسیر رسمی خودِ Prithvi
=========================================================
نسخه: v1 · تاریخ: 2026-07-30

این بازو **مالِ ما نیست، مالِ آن‌هاست** — و هستهٔ ادعای پروژه است:

    Prithvi مکان و زمان را با یک **جمع وزن‌دار** وارد می‌کند (دو وزن یادگرفتنی).
    هیچ ابلیشنی از این نقطهٔ ورود منتشر نشده. ما همان چهار عدد را از adaLN وارد
    می‌کنیم و مقایسه می‌کنیم.

پس این بازو باید **دقیقاً مسیر رسمی** باشد، نه بازسازی ما. چیزی که باید بفهمیم:

    ۱) آیا `coords_encoding` از طریق کارخانهٔ TerraTorch قابل روشن کردن است؟
    ۲) با روشن شدنش چه پارامترهایی اضافه می‌شود و چند تا؟
    ۳) forward چه آرگومان‌های تازه‌ای می‌خواهد و با چه **شکلی**؟
    ۴) آیا واقعاً روی خروجی اثر دارد؟ (همان درس تست هم‌ارزی: «اجرا شد» کافی نیست)
    ۵) 🔴 آیا مختصات **غلط** خروجی متفاوت می‌دهد؟ اگر نه، مسیر رسمی وصل نیست و
       مقایسهٔ ما بی‌معناست.

هیچ آموزشی اینجا نیست. فقط راستی‌آزمایی ساختار — رایگان، روی CPU.
"""
import sys, inspect, itertools
sys.stdout.reconfigure(encoding="utf-8")

import torch
from terratorch.models import EncoderDecoderFactory

BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
BASE = dict(
    backbone="prithvi_eo_v2_300", backbone_pretrained=False, backbone_bands=BANDS,
    necks=[{"name": "SelectIndices", "indices": [5, 11, 17, 23]},
           {"name": "ReshapeTokensToImage"},
           {"name": "LearnedInterpolateToPyramidal"}],
    decoder="UNetDecoder", decoder_channels=[512, 256, 128, 64], num_classes=2,
)
torch.manual_seed(0)


def build(**extra):
    return EncoderDecoderFactory().build_model(task="segmentation", **{**BASE, **extra})


print("=" * 78)
print("۱) آیا `coords_encoding` از کارخانه قابل روشن کردن است؟")
print("=" * 78)
plain = build()
n_plain = sum(p.numel() for p in plain.parameters())
print(f"   بدون متادیتا: {n_plain/1e6:.2f} M پارامتر")

variants = [["time", "location"], ["location"], ["time"]]
built = {}
for v in variants:
    try:
        m = build(backbone_coords_encoding=v, backbone_coords_scale_learn=True)
        n = sum(p.numel() for p in m.parameters())
        built[tuple(v)] = m
        print(f"   ✅ coords_encoding={v}  →  {n/1e6:.2f} M  (+{(n-n_plain)/1e3:.1f} K)")
    except Exception as e:
        print(f"   ⛔ coords_encoding={v}  →  {type(e).__name__}: {str(e)[:90]}")


if not built:
    print("\n🔴 هیچ گونه‌ای ساخته نشد — بازوی صفر از این مسیر ممکن نیست.")
    sys.exit(1)

key = ("time", "location") if ("time", "location") in built else next(iter(built))
model = built[key]
bb = dict(model.named_children())["encoder"]

print("\n" + "=" * 78)
print(f"۲) چه چیزی اضافه شد؟  (گونهٔ {list(key)})")
print("=" * 78)
new = [(n, tuple(p.shape), p.numel()) for n, p in bb.named_parameters()
       if any(k in n.lower() for k in ("coord", "temporal", "location", "time"))]
for n, s, c in new:
    print(f"   {n:<45} {str(s):<16} {c:,}")
print(f"   جمع: {sum(c for _,_,c in new):,} پارامتر")
mods = [n for n, _ in bb.named_modules()
        if any(k in n.lower() for k in ("coord", "temporal", "location"))]
print(f"   ماژول‌ها: {mods}")

print("\n" + "=" * 78)
print("۳) forward چه می‌خواهد؟")
print("=" * 78)
sig = inspect.signature(bb.forward)
print(f"   امضای بک‌بون: {list(sig.parameters)}")
src = inspect.getsource(type(bb).forward)
for ln in src.splitlines():
    if any(k in ln for k in ("temporal_coords", "location_coords", "def forward")):
        print(f"      | {ln.strip()[:96]}")

print("\n" + "=" * 78)
print("۴ و ۵) آیا واقعاً اثر دارد؟ و آیا مختصات **غلط** فرق می‌کند؟")
print("=" * 78)
model.eval()
x = torch.randn(2, 6, 224, 224)
# lat/lon واقعی از دیتاست آتش‌سوزی · temporal = (سال، روزِ سال)
loc_a = torch.tensor([[37.5, -122.3], [45.1, -110.8]])
loc_b = torch.tensor([[-33.9, 151.2], [12.4, 77.6]])       # نیم‌کرهٔ دیگر
tim_a = torch.tensor([[[2020., 248.]], [[2020., 200.]]])
tim_b = torch.tensor([[[2018., 12.]], [[2019., 305.]]])


def fwd(**kw):
    with torch.no_grad():
        o = model(x, **kw)
    return (o.output if hasattr(o, "output") else o).clone()


try:
    ya = fwd(temporal_coords=tim_a, location_coords=loc_a)
    yb = fwd(temporal_coords=tim_b, location_coords=loc_b)
    d = (ya - yb).abs().max().item()
    print(f"   ✅ forward با مختصات کار کرد → {tuple(ya.shape)}")
    print(f"   اختلاف بین دو مجموعه مختصات: {d:.6e}")
    eff = d > 1e-6
    print(f"   {'🟢 مسیر رسمی واقعاً وصل است' if eff else '🔴 مختصات هیچ اثری ندارد'}")
except Exception as e:
    print(f"   ⛔ {type(e).__name__}: {str(e)[:200]}")
    eff = False

print("\n" + "=" * 78)
print("🟢 بازوی صفر آماده است." if eff else "🔴 بازوی صفر هنوز آماده نیست.")
print("=" * 78)

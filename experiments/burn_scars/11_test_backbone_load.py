# -*- coding: utf-8 -*-
"""
11_test_backbone_load.py — آیا مدل واقعاً ساخته و بارگذاری می‌شود؟
====================================================================
نسخه: v1 · تاریخ: 2026-07-30

سؤال‌هایی که این فایل جواب می‌دهد، **قبل از اینکه یک دلار GPU خرج شود**:

    ۱) آیا `EncoderDecoderFactory` با پارامترهای کانفیگ رسمی مدل می‌سازد؟
    ۲) وزن ۱.۲ گیگی از HuggingFace می‌آید؟ کجا کَش می‌شود؟
    ۳) 🔴 **`missing_keys` و `unexpected_keys` چه می‌گویند؟**
       اگر بخش بزرگی از وزن‌ها لود نشده باشد، مدل عملاً تصادفی است و
       «خط پایه»ی ما بی‌معنا. این خاموش‌ترین شکست ممکن است.
    ۴) یک forward pass با تنسور تصادفی جواب می‌دهد؟ شکل خروجی درست است؟
    ۵) 🔴 `SelectIndices [5,11,17,23]` با بک‌بون ۲۴ لایه سازگار است؟
    ۶) چند پارامتر؟ (برای برآورد حافظهٔ GPU)

⚠️ چرا اندازهٔ ۴۴۸ نه ۵۱۲: نقل مستقیم مقاله —
   «The 512 × 512 images were resized to 448 × 448 as 512 × 512 is not
   divisible by the patch size.» patch=۱۶ → ۴۴۸/۱۶ = ۲۸ توکن در هر بعد.

هیچ آموزشی انجام نمی‌شود. فقط ساخت، بارگذاری، و یک forward.

اجرا: <venv>\Scripts\python.exe 11_test_backbone_load.py
"""
import sys, time, io, contextlib
sys.stdout.reconfigure(encoding="utf-8")

t0 = time.time()
import torch
from terratorch.models import EncoderDecoderFactory

IMG = 448
BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

ARGS = dict(
    backbone="prithvi_eo_v2_300",
    backbone_pretrained=True,
    backbone_bands=BANDS,
    necks=[
        {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
        {"name": "ReshapeTokensToImage"},
        {"name": "LearnedInterpolateToPyramidal"},
    ],
    decoder="UNetDecoder",
    decoder_channels=[512, 256, 128, 64],
    num_classes=2,
)

print("=" * 74)
print("۱. ساخت مدل — وزن از HuggingFace می‌آید (~۱.۲ گیگ، بار اول)")
print("=" * 74)
for k, v in ARGS.items():
    print(f"   {k:<22} {v}")
print(flush=True)

buf = io.StringIO()
try:
    with contextlib.redirect_stderr(buf):
        model = EncoderDecoderFactory().build_model(task="segmentation", **ARGS)
except Exception as e:
    print(f"⛔ ساخت مدل شکست خورد: {type(e).__name__}: {e}")
    tail = buf.getvalue().strip().splitlines()[-15:]
    print("\n".join(tail))
    sys.exit(1)

print(f"✅ مدل ساخته شد   ({time.time()-t0:.1f}s)")

# --- هشدارهای بارگذاری وزن: مهم‌ترین بخش ---
warn = buf.getvalue()
keylines = [l for l in warn.splitlines()
            if any(w in l.lower() for w in ("missing", "unexpected", "mismatch", "not initialized"))]
print("\n" + "=" * 74)
print("۲. 🔴 هشدارهای بارگذاری وزن")
print("=" * 74)
if keylines:
    for l in keylines[:20]:
        print("   " + l.strip())
else:
    print("   (هیچ هشدار missing/unexpected در stderr نبود)")

n_all = sum(p.numel() for p in model.parameters())
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n   پارامتر کل: {n_all/1e6:,.1f} M   ·   آموزش‌پذیر: {n_tr/1e6:,.1f} M")

print("\n" + "=" * 74)
print("۳. forward pass با تنسور تصادفی")
print("=" * 74)
x = torch.randn(1, len(BANDS), IMG, IMG)
print(f"   ورودی : {tuple(x.shape)}")
model.eval()
t1 = time.time()
with torch.no_grad():
    out = model(x)
y = out.output if hasattr(out, "output") else out
print(f"   خروجی : {tuple(y.shape)}   ({time.time()-t1:.1f}s روی CPU)")

ok_shape = tuple(y.shape) == (1, ARGS["num_classes"], IMG, IMG)
print(f"   {'✅ شکل درست است' if ok_shape else '⛔ شکل غلط — انتظار (1, 2, %d, %d)' % (IMG, IMG)}")
print(f"   بازهٔ لاجیت‌ها: [{y.min():.3f}, {y.max():.3f}]   NaN: {bool(torch.isnan(y).any())}")

print("\n" + "=" * 74)
print(f"🟢 مدل ساخته می‌شود و forward می‌دهد." if ok_shape else "🔴 مشکل دارد.")
print(f"زمان کل: {time.time()-t0:.1f}s")
print("=" * 74)

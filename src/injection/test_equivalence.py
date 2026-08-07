# -*- coding: utf-8 -*-
"""
test_equivalence.py — 🔴 تست هم‌ارزی بیت‌به‌بیت
===============================================
نسخه: v1 · تاریخ: 2026-07-30 · قفل `SYNC.md` بخش ۳ سطر ۱۱

**چرا این مهم‌ترین تست پروژه است:** هر بازوی تزریق باید حالتی داشته باشد که در آن
ریاضیاتش **دقیقاً** به خط پایه فرو بریزد. اگر نریزد، باگ داری — و بدون این تست،
آن باگ را به‌عنوان «نتیجهٔ علمی» گزارش می‌کنی.

سه چیز سنجیده می‌شود:

    A) بلوک پوشانده‌شده با `cond=None`  →  **بیت‌به‌بیت** برابر بلوک اصلی
    B) بلوک پوشانده‌شده با شرط و وزن‌های صفر (مقداردهی اولیه)  →  **بیت‌به‌بیت** برابر
       چون modulate(y, 0, 0) = y * (1+0) + 0 = y
    C) با وزن‌های تصادفی (نه صفر)  →  خروجی **باید فرق کند**، وگرنه شرط اصلاً وصل نیست

    ⚠️ آزمون C به‌اندازهٔ A و B مهم است: تستی که فقط برابری را چک کند، با یک
    ماژول کاملاً بی‌اثر هم پاس می‌شود.

اجرا: <venv>\\Scripts\\python.exe test_equivalence.py
"""
import sys, copy
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from terratorch.models import EncoderDecoderFactory
from injection.adaln import AdaLNInjector

torch.manual_seed(0)
IMG, B, COND = 224, 2, 4

ARGS = dict(
    backbone="prithvi_eo_v2_300", backbone_pretrained=False,
    backbone_bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
    necks=[{"name": "SelectIndices", "indices": [5, 11, 17, 23]},
           {"name": "ReshapeTokensToImage"},
           {"name": "LearnedInterpolateToPyramidal"}],
    decoder="UNetDecoder", decoder_channels=[512, 256, 128, 64], num_classes=2,
)


def run(m, x):
    m.eval()
    with torch.no_grad():
        o = m(x)
    return (o.output if hasattr(o, "output") else o).clone()


# ⚠️ عمداً **یک** مدل، نه دو تا با deepcopy.
#    اندازه‌گیری‌شده 2026-07-30: با `copy.deepcopy` روی مدل TerraTorch، تزریق در
#    نسخهٔ کپی‌شده روی خروجی اثر نمی‌گذارد — حتی وقتی wrapper سر جایش است و
#    shift/scale تولید می‌شود. علتش را نکاویدم چون لازم نبود؛ تک‌مدل هم دقیق‌تر
#    است: خط پایه از **همان** شیء گرفته می‌شود، پیش از تزریق.
print("ساخت مدل...", flush=True)
test = EncoderDecoderFactory().build_model(task="segmentation", **ARGS)

x = torch.randn(B, 6, IMG, IMG)
c = torch.randn(B, COND)
y_base = run(test, x)          # خط پایه: پیش از هر تزریقی
print(f"خروجی خط پایه: {tuple(y_base.shape)}\n")

n_base = sum(p.numel() for p in test.parameters())
inj = AdaLNInjector(test, cond_dim=COND)
n_extra = sum(p.numel() for p in inj.extra_parameters())
print(f"بلوک‌های پوشانده‌شده: {inj.n_blocks} · embed_dim={inj.embed_dim}")
print(f"پارامتر اضافه: {n_extra/1e6:.2f} M  ({100*n_extra/n_base:.2f}٪ خط پایه)\n")


def report(name, a, b, expect_equal):
    d = (a - b).abs().max().item()
    exact = torch.equal(a, b)
    ok = exact if expect_equal else (d > 1e-6)
    verdict = "✅" if ok else "⛔"
    want = "باید برابر باشد" if expect_equal else "باید فرق کند"
    print(f"{verdict} {name}")
    print(f"     بیشینهٔ اختلاف مطلق: {d:.3e}   ·   برابری دقیق: {exact}   ·   {want}")
    return ok


print("=" * 78)
print("A) شرط = None  →  باید بیت‌به‌بیت برابر خط پایه باشد")
print("=" * 78)
inj.set_condition(None)
okA = report("cond=None", run(test, x), y_base, True)

print("\n" + "=" * 78)
print("B) شرط فعال، وزن‌های to_mod صفر (مقداردهی اولیه)  →  باز هم بیت‌به‌بیت برابر")
print("=" * 78)
inj.set_condition(c)
okB = report("cond فعال · وزن صفر", run(test, x), y_base, True)

print("\n" + "=" * 78)
print("C) وزن‌های to_mod تصادفی  →  خروجی **باید** فرق کند")
print("=" * 78)
with torch.no_grad():
    for w in inj.wrappers:
        w.to_mod[1].weight.normal_(0, 0.05)
        w.to_mod[1].bias.normal_(0, 0.05)
inj.set_condition(c)
# --- خوددیاگنوستیک: اگر C رد شد، از این سه عدد می‌فهمیم کجا ---
w0 = inj.wrappers[0]
bb0 = AdaLNInjector._find_backbone(test).blocks[0]
print(f"     [dbg] |to_mod.weight| بلوک۰ = {w0.to_mod[1].weight.abs().sum():.1f}")
print(f"     [dbg] |shift/scale| تولیدشده = {w0.to_mod(w0.cond).abs().sum():.1f}")
print(f"     [dbg] blocks[0] همان wrappers[0] است؟ {bb0 is w0}")
y_rand = run(test, x)
okC = report("وزن تصادفی", y_rand, y_base, False)

print("\n" + "=" * 78)
print("D) شرط متفاوت → خروجی متفاوت (یعنی واقعاً به شرط وابسته است)")
print("=" * 78)
inj.set_condition(torch.randn(B, COND) * 3)
okD = report("شرط دیگر", run(test, x), y_rand, False)

print("\n" + "=" * 78)
allok = okA and okB and okC and okD
if allok:
    print("🟢 هر چهار آزمون گذشت.")
    print("   بازوی adaLN در گام صفر **دقیقاً** خط پایه است، و به شرط واقعاً وابسته.")
else:
    print("🔴 حداقل یک آزمون رد شد — قبل از هر اجرایی باید حل شود.")
    print("   اگر A رد شد: بسته‌بندی مسیر forward بلوک را عوض کرده.")
    print("   اگر B رد شد: مقداردهی صفر درست نیست یا modulate جای غلطی است.")
    print("   اگر C یا D رد شد: شرط اصلاً به مدل وصل نیست — تست برابری بی‌معنا بوده.")
print("=" * 78)
sys.exit(0 if allok else 1)

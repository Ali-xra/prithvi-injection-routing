# -*- coding: utf-8 -*-
"""
17_gpu_probe.py — چه پیکربندی‌ای در ۸ گیگ جا می‌شود؟ اندازه‌گیری، نه تخمین
=============================================================================
نسخه: v1 · تاریخ: 2026-07-30 · GTX 1070 · 8 GB · sm_61 (Pascal)

چرا این اسکریپت پیش از هر اجرایی می‌آید:

    تخمین سرانگشتی من این بود که «۳۲۴ M پارامتر با AdamW حدود ۵ گیگ می‌شود پیش از
    فعال‌سازی‌ها». آن یک **حدس** است. اگر بر پایه‌اش batch را انتخاب کنم و وسط
    اجرای پنجم OOM بخورم، ساعت‌ها از دست رفته.

    پس هر ترکیب را سه گام واقعی جلو می‌بریم و **اوج حافظه** و **زمان هر گام** را
    می‌خوانیم. OOM هم یک نتیجه است، نه یک خطا.

⚠️ نکتهٔ Pascal: `torch.cuda.is_bf16_supported()` روی این کارت `True` برمی‌گرداند،
ولی هستهٔ bf16 وجود ندارد و نرم‌افزاری تقلید می‌شود. پس `bf16-mixed` آزموده
نمی‌شود؛ فقط `fp16` و `fp32`.

⚠️ بازوی adaLN حدود ۲۵ M پارامتر بیشتر دارد → حالت‌های Adam بیشتر → حافظهٔ بیشتر.
پس هر دو بازو سنجیده می‌شوند، وگرنه ممکن است خط پایه جا شود و adaLN نشود.

خروجی: <BIG>/data/meta/gpu_probe.json
اجرا:  <venv-gpu>\Scripts\python.exe 17_gpu_probe.py
"""
import sys, json, time, gc
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\aliso\Desktop\proje\uni\ideas\injection-routing\lab-shared\src")

import torch
from terratorch.models import EncoderDecoderFactory
from injection.adaln import AdaLNInjector

OUT = Path.home() / "Desktop" / "big-files" / "injection-routing" / "data" / "meta" / "gpu_probe.json"
BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

BASE = dict(
    backbone="prithvi_eo_v2_300", backbone_pretrained=False, backbone_bands=BANDS,
    necks=[{"name": "SelectIndices", "indices": [5, 11, 17, 23]},
           {"name": "ReshapeTokensToImage"},
           {"name": "LearnedInterpolateToPyramidal"}],
    decoder="UNetDecoder", decoder_channels=[512, 256, 128, 64], num_classes=2,
)

# (تصویر, batch, دقت, بازو)
CONFIGS = [
    (224, 8, "fp16", "baseline"),
    (224, 4, "fp16", "baseline"),
    (224, 4, "fp16", "adaln"),
    (224, 2, "fp16", "adaln"),
    (224, 4, "fp32", "adaln"),
    (512, 2, "fp16", "adaln"),
    (512, 1, "fp16", "adaln"),
]


def probe(img, bs, prec, arm, steps=3):
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = EncoderDecoderFactory().build_model(task="segmentation", **BASE).cuda()
        inj = AdaLNInjector(model, cond_dim=4).cuda() if arm == "adaln" else None
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=(prec == "fp16"))

        x = torch.randn(bs, 6, img, img, device="cuda")
        y = torch.randint(0, 2, (bs, img, img), device="cuda")
        c = torch.randn(bs, 4, device="cuda")

        model.train()
        t0 = None
        for i in range(steps):
            opt.zero_grad(set_to_none=True)
            if inj is not None:
                inj.set_condition(c)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(prec == "fp16")):
                out = model(x)
                logits = out.output if hasattr(out, "output") else out
                loss = torch.nn.functional.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            if i == 0:                     # گام اول شامل تخصیص اولیه است، کنار گذاشته می‌شود
                t0 = time.time()
        dt = (time.time() - t0) / max(steps - 1, 1)

        peak = torch.cuda.max_memory_allocated() / 1024**3
        res = dict(ok=True, peak_gb=round(peak, 2), sec_per_step=round(dt, 3))
    except torch.cuda.OutOfMemoryError:
        res = dict(ok=False, error="OOM")
    except Exception as e:
        res = dict(ok=False, error=f"{type(e).__name__}: {str(e)[:80]}")
    finally:
        for n in ("model", "inj", "opt", "x", "y", "c"):
            if n in dir(): pass
        try:
            del model, opt, x, y, c
            if inj is not None: del inj
        except Exception:
            pass
        torch.cuda.empty_cache(); gc.collect()
    return res


print("=" * 78)
print(f"کارت: {torch.cuda.get_device_name(0)}  ·  "
      f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB  ·  "
      f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
print("=" * 78)
print(f"{'تصویر':>7}{'batch':>7}{'دقت':>7}{'بازو':>10}{'اوج حافظه':>12}{'ثانیه/گام':>12}")
print("-" * 78)

results = {}
for img, bs, prec, arm in CONFIGS:
    r = probe(img, bs, prec, arm)
    key = f"{img}_{bs}_{prec}_{arm}"
    results[key] = dict(img=img, batch=bs, precision=prec, arm=arm, **r)
    if r["ok"]:
        print(f"{img:>7}{bs:>7}{prec:>7}{arm:>10}{r['peak_gb']:>10.2f} GB"
              f"{r['sec_per_step']:>11.2f}s   ✅", flush=True)
    else:
        print(f"{img:>7}{bs:>7}{prec:>7}{arm:>10}{'—':>12}{'—':>12}   ⛔ {r['error']}",
              flush=True)
print("=" * 78)

# --- انتخاب پیکربندی: بزرگ‌ترین batch که **بازوی adaLN** هم در آن جا شود ---
ok_adaln = [v for v in results.values() if v["ok"] and v["arm"] == "adaln"]
print("\nانتخاب پیکربندی")
print("-" * 78)
if not ok_adaln:
    print("  🔴 هیچ ترکیبی برای بازوی adaLN جا نشد.")
    print("     گزینه‌های بعدی: gradient checkpointing · بهینه‌ساز SGD به‌جای AdamW")
    print("     (AdamW دو حالت به ازای هر پارامتر نگه می‌دارد؛ SGD صفر یا یکی.)")
    chosen = None
else:
    # 🔴 معیار انتخاب: adaLN سنگین‌ترین بازوست. اگر پیکربندی را با خط پایه
    #    انتخاب کنیم، اجرای هفتم OOM می‌خورد و ساعت‌ها هدر می‌رود.
    chosen = max(ok_adaln, key=lambda v: (v["img"], v["batch"]))
    eff = 8
    accum = max(1, eff // chosen["batch"])
    steps_per_epoch = 563 / (chosen["batch"] * accum)
    epoch_min = steps_per_epoch * accum * chosen["sec_per_step"] / 60
    print(f"  تصویر {chosen['img']} · batch {chosen['batch']} · {chosen['precision']}")
    print(f"  اوج حافظه: {chosen['peak_gb']:.2f} GB از 8.00 GB")
    print(f"  انباشت گرادیان: {accum}×  →  batch مؤثر {chosen['batch']*accum}")
    print(f"  تخمین هر epoch: {epoch_min:.1f} دقیقه")
    print(f"  تخمین ۶۰ epoch با توقف زودهنگام (~۳۵): {epoch_min*35/60:.1f} ساعت هر اجرا")
    print(f"  🔴 ده اجرا: **{epoch_min*35*10/60:.0f} ساعت**")
    if epoch_min * 35 * 10 / 60 > 48:
        print("     ⚠️ بیش از دو شبانه‌روز. گزینه‌ها: کاهش epoch، کاهش seedها از ۳ به ۲،")
        print("        یا فریز کردن بخشی از بک‌بون. تصمیمش با کاربر است.")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(
    {"device": torch.cuda.get_device_name(0), "results": results, "chosen": chosen},
    ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n💾 {OUT}")

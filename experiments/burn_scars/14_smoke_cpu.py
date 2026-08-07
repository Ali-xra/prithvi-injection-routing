# -*- coding: utf-8 -*-
"""
14_smoke_cpu.py — دودتست کامل روی CPU، بدون GPU و بدون هزینه
==============================================================
نسخه: v1 · تاریخ: 2026-07-30

هدف: **پیش از اجارهٔ GPU** ثابت کنیم کل زنجیره کار می‌کند. هر خطایی که اینجا پیدا
شود، رایگان است؛ همان خطا روی GPU اجاره‌ای پول و ساعت می‌برد.

پنج چیز سنجیده می‌شود:
    ۱) دیتاماژول ساخته می‌شود و batch کلید `cond` با شکل (B,4) دارد
    ۲) مقدار `cond` با CSV **برای همان فایل** می‌خواند — نه فقط شکل درست
    ۳) بازوی شافل: بردارها واقعاً جابه‌جا شده‌اند و **هیچ نقطهٔ ثابتی** ندارند
    ۴) دو گام آموزش واقعی روی CPU اجرا می‌شود و loss عدد سالمی است
    ۵) 🔴 وزن‌های `to_mod` **گرادیان می‌گیرند** — یعنی شرط واقعاً در گراف است

    آزمون ۵ مهم‌ترین است: بدون آن، ممکن است همه‌چیز اجرا شود و بازوی adaLN در عمل
    هرگز آموزش نبیند — و ما «تزریق کمکی نکرد» گزارش کنیم.

اجرا: <venv>\\Scripts\\python.exe 14_smoke_cpu.py
"""
import sys, csv
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\aliso\Desktop\proje\uni\ideas\injection-routing\lab-shared\src")

import torch
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from injection.adaln import AdaLNInjector
from injection.conditioned_data import wrap_datamodule, load_conditioning, COND_COLS

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
DS = BIG / "data" / "burn_scars"
CSV_PATH = BIG / "data" / "meta" / "conditioning_v1.csv"
BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
ok = {}


def make_dm(shuffle_seed=None):
    dm = GenericNonGeoSegmentationDataModule(
        batch_size=2, num_workers=0, num_classes=2,
        train_data_root=DS / "training", train_label_data_root=DS / "training",
        val_data_root=DS / "validation", val_label_data_root=DS / "validation",
        test_data_root=DS / "validation", test_label_data_root=DS / "validation",
        img_grep="*_merged.tif", label_grep="*.mask.tif",
        means=[0.033, 0.056, 0.055, 0.223, 0.171, 0.106],
        stds=[0.025, 0.028, 0.041, 0.078, 0.070, 0.058],
        dataset_bands=BANDS, output_bands=BANDS,
        no_data_replace=0, no_label_replace=-1,
    )
    return wrap_datamodule(dm, CSV_PATH, shuffle_seed=shuffle_seed)


print("=" * 78)
print("۱ و ۲) دیتاماژول شرط‌دار — وجود، شکل، و **درستی مقدار**")
print("=" * 78)
dm = make_dm()
dm.setup("fit")
print(f"   {dm._cond_info}")
dl = dm.train_dataloader()
b = next(iter(dl))
print(f"   کلیدهای batch: {sorted(b.keys())}")
ok["1_key"] = "cond" in b and tuple(b["cond"].shape) == (2, 4)
print(f"   شکل cond: {tuple(b['cond'].shape)}   {'✅' if ok['1_key'] else '⛔'}")

# درستی مقدار: نمونهٔ ۰ دیتاست train باید بردار فایل خودش را داشته باشه
cmap = load_conditioning(CSV_PATH)
ds = dm.train_dataset
k0 = ds.keys[0]
got = ds[0]["cond"]
exp = cmap[k0]
ok["2_value"] = torch.allclose(got, exp)
print(f"   فایل نمونهٔ ۰: {k0}")
print(f"   cond گرفته‌شده: {[round(v,4) for v in got.tolist()]}")
print(f"   cond مورد انتظار: {[round(v,4) for v in exp.tolist()]}   "
      f"{'✅ منطبق' if ok['2_value'] else '⛔ ناهماهنگ'}")

print("\n" + "=" * 78)
print("۳) بازوی شافل — جابه‌جایی واقعی و صفر نقطهٔ ثابت")
print("=" * 78)
dm_s = make_dm(shuffle_seed=0)
dm_s.setup("fit")
ds_s = dm_s.train_dataset
n_same = sum(torch.allclose(ds_s[i]["cond"], cmap[ds_s.keys[i]]) for i in range(30))
print(f"   نقاط ثابت جایگشت (کل): {ds_s.n_fixed_points}")
print(f"   از ۳۰ نمونهٔ اول، چندتا بردار خودشان را گرفتند: {n_same}")
ok["3_shuffle"] = ds_s.n_fixed_points == 0 and n_same == 0
print(f"   {'✅ شافل سالم' if ok['3_shuffle'] else '⛔ شافل ناقص'}")
# و val نباید شافل شده باشد
val_same = torch.allclose(dm_s.val_dataset[0]["cond"], cmap[dm_s.val_dataset.keys[0]])
ok["3b_val"] = val_same
print(f"   val شافل **نشده** است؟ {'✅ بله' if val_same else '⛔ خیر — معیار خراب می‌شود'}")


print("\n" + "=" * 78)
print("۴ و ۵) دو گام آموزش واقعی روی CPU · و **آیا to_mod گرادیان می‌گیرد؟**")
print("=" * 78)
from injection.conditioned_task import ConditionedSegmentationTask

task = ConditionedSegmentationTask(
    model_factory="EncoderDecoderFactory",
    model_args=dict(
        backbone="prithvi_eo_v2_300", backbone_pretrained=False,
        backbone_bands=BANDS,
        necks=[{"name": "SelectIndices", "indices": [5, 11, 17, 23]},
               {"name": "ReshapeTokensToImage"},
               {"name": "LearnedInterpolateToPyramidal"}],
        decoder="UNetDecoder", decoder_channels=[512, 256, 128, 64], num_classes=2),
    loss="ce", optimizer="AdamW", lr=1e-4, ignore_index=-1,
)
inj = AdaLNInjector(task.model, cond_dim=4)
task.set_injector(inj)
print(f"   بلوک پوشانده: {inj.n_blocks} · پارامتر اضافه: "
      f"{sum(p.numel() for p in inj.extra_parameters())/1e6:.2f} M")

# 🔴 آزمون واقعی: آیا انکودر در پارامترهای **خودِ تسک** هست؟
#    اگر نباشد، `configure_optimizers` نمی‌بیندش و در آموزش واقعی یخ می‌زند.
task_ids = {id(p) for p in task.parameters()}
enc_in = all(id(p) in task_ids for p in inj.encoder.parameters())
mod_in = all(id(p) in task_ids for p in inj.wrappers[0].to_mod.parameters())
ok["0_optim"] = enc_in and mod_in
print(f"   انکودر در task.parameters()؟ {'✅' if enc_in else '⛔ یخ می‌زند'}"
      f"   ·   to_mod؟ {'✅' if mod_in else '⛔'}")

opt = torch.optim.AdamW(list(task.parameters()), lr=1e-4)   # بدون تکرار
task.train()
losses, gnorms = [], []
it = iter(dl)
for step in range(2):
    batch = next(it)
    opt.zero_grad()
    cond = batch.pop("cond")
    inj.set_condition(cond)
    out = task.model(batch["image"])
    logits = out.output if hasattr(out, "output") else out
    loss = torch.nn.functional.cross_entropy(
        logits, batch["mask"].long(), ignore_index=-1)
    loss.backward()
    g = inj.wrappers[0].to_mod[1].weight.grad
    gn = 0.0 if g is None else g.abs().sum().item()
    ge = sum(p.grad.abs().sum().item() for p in inj.encoder.parameters()
             if p.grad is not None)
    opt.step()
    losses.append(loss.item()); gnorms.append((gn, ge))
    print(f"   گام {step+1}: loss={loss.item():.4f} · "
          f"|grad to_mod[0]|={gn:.3e} · |grad encoder|={ge:.3e}", flush=True)

import math
ok["4_loss"] = all(math.isfinite(l) for l in losses)
ok["5_grad"] = all(a > 0 for a, _ in gnorms)
# ⚠️ در گام ۱ گرادیانِ انکودر **باید** صفر باشد و این باگ نیست:
#    to_mod با وزن صفر شروع می‌شود، پس d(loss)/d(encoder) = Wᵀ·g = 0.
#    از گام ۲ که to_mod از صفر جدا شد، انکودر شروع به یادگیری می‌کند.
#    پس شرط درست «همهٔ گام‌ها» نیست، «گام آخر» است.
ok["5b_enc"] = gnorms[-1][1] > 0

print("\n" + "=" * 78)
labels = {"0_optim": "🔴 انکودر در بهینه‌ساز تسک هست",
          "1_key": "کلید cond در batch", "2_value": "مقدار cond درست است",
          "3_shuffle": "شافل بدون نقطهٔ ثابت", "3b_val": "val شافل نشده",
          "4_loss": "loss متناهی", "5_grad": "🔴 to_mod گرادیان می‌گیرد",
          "5b_enc": "🔴 encoder گرادیان می‌گیرد"}
for k, v in ok.items():
    print(f"   {'✅' if v else '⛔'} {labels[k]}")
allok = all(ok.values())
print("\n" + ("🟢 زنجیره کامل است. آمادهٔ GPU." if allok
              else "🔴 حداقل یکی رد شد — قبل از اجارهٔ GPU باید حل شود."))
print("=" * 78)
sys.exit(0 if allok else 1)

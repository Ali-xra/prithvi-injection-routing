# -*- coding: utf-8 -*-
"""
16_run_arm.py — اجرای یک بازو · همان اسکریپت برای هر چهار بازو
================================================================
نسخه: v1 · تاریخ: 2026-07-30

🔴 **چرا یک اسکریپت برای همهٔ بازوها:** اگر هر بازو اسکریپت خودش را داشته باشد،
هر تفاوت ناخواسته‌ای (نرخ یادگیری، seed، augmentation) به‌عنوان «اثر تزریق» گزارش
می‌شود. اینجا تنها چیزی که بین بازوها فرق می‌کند، `--arm` است.

    baseline  خط پایه — هیچ متادیتایی. کانفیگ منتشرشده.
    official  مسیر خودِ Prithvi — `coords_encoding`، **۲ پارامتر**
    adaln     مسیر ما — adaLN روی هر ۲۴ بلوک، **۲۵.۳۳ M پارامتر**
    shuffle   کنترل — همان adaLN، ولی مختصات **جابه‌جاشده**

`--smoke` دو گام روی CPU می‌زند و خارج می‌شود؛ برای راستی‌آزمایی رایگان پیش از GPU.

اجرا:
    python 16_run_arm.py --arm baseline --seed 0
    python 16_run_arm.py --arm adaln --seed 0 --smoke
"""
import sys, os, json, time, argparse
from contextlib import contextmanager
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\aliso\Desktop\proje\uni\ideas\injection-routing\lab-shared\src")

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from terratorch.datamodules import GenericNonGeoSegmentationDataModule

from injection.adaln import AdaLNInjector
from injection.conditioned_data import wrap_datamodule
from injection.conditioned_task import ConditionedSegmentationTask

BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
DS = BIG / "data" / "burn_scars"
CSV_PATH = BIG / "data" / "meta" / "conditioning_v1.csv"
OUT = BIG / "runs"
BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
MEANS = [0.033, 0.056, 0.055, 0.223, 0.171, 0.106]
STDS = [0.025, 0.028, 0.041, 0.078, 0.070, 0.058]
ARMS = ("baseline", "official", "adaln", "shuffle", "tl_on", "tl_off")


def build_transforms(img_size):
    """
    برش ۲۲۴ به‌جای chip کامل ۵۱۲.

    ⚠️ این یک **انحراف از کانفیگ منتشرشده** است و باید در گزارش بیاید. دلیلش
    اندازه‌گیری `17_gpu_probe.py` روی GTX 1070 است:

        224 · batch 4  →  0.285 ثانیه به ازای هر نمونه
        512 · batch 1  →  1.46  ثانیه به ازای هر نمونه   (۵ برابر)
        512 · batch 2  →  5.08  ثانیه به ازای هر نمونه   (ترشینگ در ۷ از ۸ گیگ)

    ده اجرا در ۲۲۴ حدود ۱۶ ساعت است و در ۵۱۲ حدود ۲۷۸ ساعت.

    ✅ توجیه علمی هم دارد: ۲۲۴ **اندازهٔ بومی پیش‌آموزش خودِ Prithvi** است.
    ⚠️ ولی چون هر چهار بازو یکسان برش می‌خورند، **مقایسهٔ بین بازوها** دست‌نخورده
       می‌ماند؛ چیزی که قابل مقایسه با اعداد مقاله نیست، **قدر مطلق** IoU است.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    train = [A.RandomCrop(img_size, img_size), A.HorizontalFlip(p=0.5),
             A.VerticalFlip(p=0.5), ToTensorV2()]
    val = [A.CenterCrop(img_size, img_size), ToTensorV2()]
    return train, val


# 🔴🔴 خطای ۱۷ — کشف‌شده ۱ اوت، بعد از ۱۱ اجرای GPU.
#
# تا این لحظه این تابع مستقیم به `burn_scars/training` و `burn_scars/validation`
# وصل بود — یعنی **split منتشرشده**، نه split جدا-به-tile خودمان.
#
#   tileهای مشترک train/val در split منتشرشده : ۱۲۴
#   chipهای val روی tile مشترک                : ۱۹۴ از ۲۶۴  (۷۳٪)
#
# و `12_image_proxy_control.py` — همان اسکریپتی که کل پروژه را توجیه کرد —
# از ستون `split` یعنی نسخهٔ **تمیز** استفاده می‌کند.
#
# پس فرض را روی split تمیز تأیید کردیم و آزمایش را روی split نشتی اجرا کردیم.
#
# و طبق یافتهٔ F5 خودمان، نشتی دقیقاً در جهتی اثر می‌گذارد که نتیجه را خراب
# می‌کند: «نشتی دادهٔ کمکی را کم‌فایده‌تر نشان می‌دهد» (+0.0052 نشتی در برابر
# +0.0288 تمیز). وقتی مدل همان tile را دیده، ظاهرش را حفظ می‌کند و به مختصات
# نیازی ندارد — یعنی همان مکانیزمی که نتیجهٔ صفر ما را می‌سازد.
#
# `20_build_split_dirs.py` پوشه‌های جدا-به-tile را ساخت و صفر tile مشترک را
# تأیید کرد. `DS_SPLIT` پیش‌فرض است؛ `--orig-split` برای بازتولید دور اول.
DS_TILEDISJOINT = BIG / "data" / "burn_scars_tiledisjoint"
USE_ORIG_SPLIT = False


def _roots():
    if USE_ORIG_SPLIT:
        return DS, "training", "validation", "validation"
    return DS_TILEDISJOINT, "training", "validation", "test"


def build_datamodule(arm, seed, batch_size, workers, img_size=224):
    tr_t, va_t = build_transforms(img_size)
    root, d_tr, d_va, d_te = _roots()
    dm = GenericNonGeoSegmentationDataModule(
        batch_size=batch_size, num_workers=workers, num_classes=2,
        train_transform=tr_t, val_transform=va_t, test_transform=va_t,
        train_data_root=root / d_tr, train_label_data_root=root / d_tr,
        val_data_root=root / d_va, val_label_data_root=root / d_va,
        test_data_root=root / d_te, test_label_data_root=root / d_te,
        img_grep="*_merged.tif", label_grep="*.mask.tif",
        means=MEANS, stds=STDS, dataset_bands=BANDS, output_bands=BANDS,
        no_data_replace=0, no_label_replace=-1,
    )
    if arm in ("baseline", "tl_off"):
        return dm                                  # هیچ شرطی
    if arm in ("official", "tl_on"):
        return wrap_datamodule(dm, CSV_PATH, mode="official")
    if arm == "adaln":
        return wrap_datamodule(dm, CSV_PATH, mode="adaln")
    if arm == "shuffle":
        # 🔴 seed شافل از seed اجرا جداست تا با تغییر seed اجرا،
        #    الگوی شافل هم عوض نشود و کنترل بین seedها یکسان بماند.
        return wrap_datamodule(dm, CSV_PATH, mode="adaln", shuffle_seed=1000 + seed)
    raise ValueError(arm)


# 🔴🔴 خطای ۱۵ — کشف‌شده ۱ اوت، بعد از تمام‌شدن هر ده اجرا.
#
# کاربر پرسید: «ما آموزش ندادیم، فاینتیون کردیم — شاید در آموزش فرق کند.»
# رفتیم چک کنیم و چیز بدتری پیدا شد: **ما ابلیشن را روی چک‌پوینت اشتباه اجرا کردیم.**
#
# IBM دو وزنِ جدا منتشر کرده:
#   Prithvi-EO-2.0-300M      → coords_encoding = []                  ← ما این را گرفتیم
#   Prithvi-EO-2.0-300M-TL   → coords_encoding = ["time","location"] ← این یکی درست بود
#
# اندازه‌گیری مستقیم روی هر دو فایل (`_check_tl.py`):
#   300M    : ۳۹۸ کلید، هیچ کلید مربوط به coords ندارد
#   300M-TL : ۴۰۲ کلید، شاملِ
#              encoder.location_embed_enc.scale = 0.0582
#              encoder.temporal_embed_enc.scale = 0.00000128
#
# یعنی آن `Missing key(s)` که ۳۱ ژوئیه گرفتیم و «دورش زدیم»، **پیام خطا نبود —
# علامت این بود که چک‌پوینت اشتباه را برداشته‌ایم.** ما یک نشانه را به‌جای مانع
# گرفتیم و از رویش رد شدیم. این بدترین نوع خطای این پروژه است، چون خودِ سیستم
# داشت درست هشدار می‌داد.
#
# در نتیجه بازوی `official` ما این را نسنجید که «مسیر متادیتای Prithvi چقدر
# کمک می‌کند»، بلکه این را سنجید که «اگر مکانیزم متادیتا را به بدنه‌ای بچسبانیم
# که هرگز با آن pretrain نشده و فقط ۵۶۳ نمونه برای یادگرفتنش داریم، چه می‌شود».
# اینها دو سؤال کاملاً متفاوت‌اند.
#
# دو بازوی جدید، ابلیشن واقعی روی وزن‌های درست:
#   tl_on   : چک‌پوینت TL با coords روشن  (حالت بومی خودشان)
#   tl_off  : چک‌پوینت TL با coords خاموش (ابلیشنی که هیچ‌کس منتشر نکرده)
# هر دو دقیقاً یک مجموعه وزنِ pretrain‌شده دارند؛ تنها تفاوت، مسیر متادیتاست.

TL_BACKBONE = "prithvi_eo_v2_300_tl"


def build_task(arm, lr, pretrained, backbone="prithvi_eo_v2_300"):
    if arm in ("tl_on", "tl_off"):
        backbone = TL_BACKBONE
    margs = dict(
        backbone=backbone, backbone_pretrained=pretrained,
        backbone_bands=BANDS,
        necks=[{"name": "SelectIndices", "indices": [5, 11, 17, 23]},
               {"name": "ReshapeTokensToImage"},
               {"name": "LearnedInterpolateToPyramidal"}],
        decoder="UNetDecoder", decoder_channels=[512, 256, 128, 64], num_classes=2,
    )
    # 🔴 بازوی رسمی نمی‌تواند وزن‌های از پیش آموزش‌دیده را از مسیر عادی بگیرد.
    #    اندازه‌گیری‌شده 2026-07-31:
    #
    #      RuntimeError: Missing key(s) in state_dict for PrithviViT:
    #                    "temporal_embed_enc.scale", "location_embed_enc.scale"
    #
    #    با روشن شدن `coords_encoding` مدل آن دو اسکالر را می‌سازد، ولی چک‌پوینت
    #    منتشرشده آن‌ها را ندارد و TerraTorch با `strict=True` بارگذاری می‌کند.
    #    یعنی **مسیر رسمی متادیتا با وزن‌های از پیش آموزش‌دیده اصلاً بالا نمی‌آید.**
    #
    #    این احتمالاً یکی از دلایل واقعیِ اینکه هیچ کانفیگ رسمی‌ای آن را روشن نکرده
    #    و هیچ ابلیشنی از آن منتشر نشده.
    #
    #    راه‌حل: مدل را بدون وزن بساز، بعد چک‌پوینت را دستی با `strict=False`
    #    بارگذاری کن — و **تأیید کن که دقیقاً همان دو کلید غایب‌اند و صفر کلید
    #    اضافه**. بدون آن تأیید، این کار می‌تواند یک بارگذاری نیم‌بند را پنهان کند.
    relax = False
    if arm == "official":
        margs["backbone_coords_encoding"] = ["time", "location"]
        margs["backbone_coords_scale_learn"] = True
        relax = pretrained
    elif arm == "tl_on":
        # حالت بومی چک‌پوینت TL. کانفیگ خودِ TerraTorch همین است، ولی صریح
        # می‌نویسیم تا از پیش‌فرض‌ها مستقل باشد و در دیف قابل دیدن.
        margs["backbone_coords_encoding"] = ["time", "location"]
        margs["backbone_coords_scale_learn"] = True
        # relax لازم نیست: چک‌پوینت TL هر دو اسکالر را دارد.
    elif arm == "tl_off":
        # 🔴 ابلیشن واقعی: همان وزن‌ها، مسیر متادیتا خاموش.
        #    اینجا وضعیت **برعکس** بازوی official است — مدل آن دو اسکالر را
        #    نمی‌سازد ولی چک‌پوینت داردشان، پس ۰ کلید غایب و ۲ کلید اضافه.
        margs["backbone_coords_encoding"] = []
        relax = pretrained

    def _build():
        return ConditionedSegmentationTask(
            model_factory="EncoderDecoderFactory", model_args=margs,
            loss="ce", optimizer="AdamW", lr=lr, ignore_index=-1,
            plot_on_val=False,
        )

    if relax:
        # ⚠️ بارگذاری دستی را امتحان کردیم و **غلط بود**: چک‌پوینت `pos_embed`
        #    چهارفریمی دارد (785 موقعیت) و مدل ما تک‌فریم است (197). خودِ
        #    TerraTorch این تبدیل را انجام می‌دهد؛ اگر دور بزنیم، از دستش می‌دهیم.
        #    پس مسیر بارگذاری خودشان دست‌نخورده می‌ماند و فقط `strict` شل می‌شود.
        with _relaxed_loading() as cap:
            task = _build()
        _verify_official_load(cap, arm)
    else:
        task = _build()

    # 🔴 هر اجرا هویت چک‌پوینت خودش را تأیید می‌کند — نه فقط بازوهای TL.
    #    اگر روز اول این را داشتیم، خطای ۱۵ هرگز اتفاق نمی‌افتاد.
    if pretrained:
        _verify_backbone_identity(task, arm)

    inj = None
    if arm in ("adaln", "shuffle"):
        inj = AdaLNInjector(task.model, cond_dim=4)
    task.set_injector(inj)
    return task, inj


EXPECTED_MISSING = {"temporal_embed_enc.scale", "location_embed_enc.scale"}


@contextmanager
def _relaxed_loading():
    """`load_state_dict` را موقتاً غیرسخت‌گیر می‌کند و گزارشش را برمی‌دارد."""
    orig = torch.nn.Module.load_state_dict
    cap = {}

    def patched(self, state_dict, strict=True, assign=False):
        res = orig(self, state_dict, strict=False, assign=assign)
        if type(self).__name__ == "PrithviViT":
            cap["missing"] = list(res.missing_keys)
            cap["unexpected"] = list(res.unexpected_keys)
        return res

    torch.nn.Module.load_state_dict = patched
    try:
        yield cap
    finally:
        torch.nn.Module.load_state_dict = orig


def _verify_official_load(cap, arm="official"):
    """
    🔴 بدون این تأیید، شل کردن `strict` می‌تواند یک بارگذاری نیم‌بند را پنهان کند
    و ما یک بازوی نیمه‌تصادفی را با خط پایهٔ کاملاً از پیش آموزش‌دیده مقایسه کنیم.

    دو حالت، دقیقاً برعکس هم:
      official → مدل دو اسکالر را می‌سازد، چک‌پوینت 300M ندارد   → ۲ غایب، ۰ اضافه
      tl_off   → مدل نمی‌سازد، چک‌پوینت TL دارد                 → ۰ غایب، ۲ اضافه
    """
    if "missing" not in cap:
        raise RuntimeError("🔴 هیچ بارگذاری‌ای روی PrithviViT رخ نداد.")
    missing = set(cap["missing"])
    unexpected = set(cap["unexpected"])
    print(f"   بارگذاری بازوی {arm}:")
    print(f"     غایب: {sorted(missing)}")
    print(f"     اضافه: {sorted(unexpected)[:5]}")

    if arm == "tl_off":
        exp_missing, exp_unexpected = set(), EXPECTED_MISSING
    else:
        exp_missing, exp_unexpected = EXPECTED_MISSING, set()

    bad_missing = missing - exp_missing
    bad_unexpected = unexpected - exp_unexpected
    if bad_missing:
        raise RuntimeError(
            f"🔴 کلیدهای غایبِ غیرمنتظره: {sorted(bad_missing)[:10]} — بخشی از وزن‌های "
            f"از پیش آموزش‌دیده بارگذاری نشده و این بازو با خط پایه قابل مقایسه نیست.")
    if bad_unexpected:
        raise RuntimeError(
            f"🔴 کلیدهای اضافهٔ غیرمنتظره: {sorted(bad_unexpected)[:10]} — ساختار مدل با "
            f"چک‌پوینت نمی‌خواند و مقایسه بی‌معنی است.")
    print("     ✅ بارگذاری همان‌طور که انتظار می‌رفت.")


# 🔴 خطای ۱۶ — اولین تلاش برای تشخیص چک‌پوینت TL غلط بود.
#
# فرض کردم اگر چک‌پوینت TL دو اسکالری داشته باشد که مدل ندارد، آن‌ها به‌عنوان
# `unexpected_keys` گزارش می‌شوند. **نشدند.** اندازه‌گیری‌شده ۱ اوت: صفر غایب،
# صفر اضافه — چون TerraTorch پیش از `load_state_dict` کلیدهای ناسازگار را
# بی‌صدا دور می‌ریزد. یعنی `unexpected_keys` اصلاً نمی‌تواند بگوید روی کدام
# چک‌پوینت هستیم.
#
# اثر انگشت واقعی، از یک تانسور که **هر دو** چک‌پوینت دارند ولی مقدارش فرق دارد
# (اندازه‌گیری‌شده در `_fingerprint.py`):
#
#     blocks.0.attn.qkv.weight  →  300M:  sum = -50.225471
#                                  300M-TL: sum = +118.693604
#
# فاصله‌شان ۱۶۸ واحد است؛ هیچ ابهامی ندارد.
FINGERPRINT_KEY = "blocks.0.attn.qkv.weight"
FINGERPRINT_TL = 118.693604
FINGERPRINT_PLAIN = -50.225471


def _verify_backbone_identity(task, arm):
    """
    تأیید می‌کند که وزن‌های TL واقعاً نشسته‌اند. بدون این، بازوی `tl_off`
    می‌توانست بی‌صدا روی چک‌پوینت غیر-TL اجرا شود و ما دقیقاً همان اشتباهی را
    تکرار کنیم که این دو بازو برای جبرانش ساخته شدند.
    """
    hits = [p for n, p in task.named_parameters() if n.endswith(FINGERPRINT_KEY)]
    if len(hits) != 1:
        raise RuntimeError(f"🔴 {len(hits)} تانسور با نام {FINGERPRINT_KEY} پیدا شد، انتظار ۱.")
    got = hits[0].sum().item()
    want = FINGERPRINT_TL if arm in ("tl_on", "tl_off") else FINGERPRINT_PLAIN
    other = FINGERPRINT_PLAIN if arm in ("tl_on", "tl_off") else FINGERPRINT_TL
    print(f"   اثر انگشت بدنه: sum({FINGERPRINT_KEY}) = {got:+.6f}")
    if abs(got - want) > 0.5:
        which = "غیر-TL" if abs(got - other) <= 0.5 else "ناشناخته"
        raise RuntimeError(
            f"🔴 بازوی {arm} انتظار چک‌پوینت {'TL' if want == FINGERPRINT_TL else '300M'} "
            f"را داشت (sum={want:+.4f}) ولی مقدار {got:+.4f} دیده شد — چک‌پوینت {which} "
            f"بارگذاری شده. اجرا متوقف شد.")
    print(f"     ✅ چک‌پوینت درست است ({'TL' if want == FINGERPRINT_TL else '300M'}).")


def smoke(arm, task, inj, dm):
    """دو گام روی CPU — بدون Trainer، تا خطاها خام و خوانا بمانند."""
    dm.setup("fit")
    dl = dm.train_dataloader()
    task.train()
    opt = torch.optim.AdamW(task.parameters(), lr=1e-4)
    it = iter(dl)
    for step in range(2):
        batch = next(it)
        opt.zero_grad()
        extra = {k: v for k, v in batch.items()
                 if k not in ("image", "mask", "filename", "cond")}
        if inj is not None:
            inj.set_condition(batch["cond"])
        out = task.model(batch["image"], **extra)
        logits = out.output if hasattr(out, "output") else out
        loss = torch.nn.functional.cross_entropy(
            logits, batch["mask"].long(), ignore_index=-1)
        loss.backward()
        opt.step()
        g = (inj.wrappers[0].to_mod[1].weight.grad.abs().sum().item()
             if inj is not None else float("nan"))
        print(f"      گام {step+1}: loss={loss.item():.4f} · کلیدهای اضافه="
              f"{sorted(extra)} · |grad adaLN|={g:.3e}", flush=True)
    # 🔴 برای بازوی رسمی: آیا آن ۲ پارامتر واقعاً گرادیان گرفتند؟
    if arm == "official":
        sc = [(n, p.grad.abs().item() if p.grad is not None else None)
              for n, p in task.model.named_parameters() if n.endswith("_enc.scale")]
        print(f"      🔴 گرادیان دو اسکالر رسمی: {sc}")
        return all(v is not None and v > 0 for _, v in sc)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4)     # از 17_gpu_probe
    ap.add_argument("--accum", type=int, default=2)          # batch مؤثر = ۸
    ap.add_argument("--img-size", type=int, default=224)
    # ⚠️ Pascal هستهٔ bf16 ندارد؛ `is_bf16_supported()` گمراه‌کننده True می‌دهد.
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--orig-split", action="store_true",
                    help="split منتشرشدهٔ نشتی — فقط برای بازتولید دور اول")
    # بازیابی: اگر آموزش تمام شده ولی مرحلهٔ آخر شکسته، فقط اعتبارسنجی را از
    # روی `best.ckpt` اجرا کن — بدون آموزش دوباره.
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()

    # 🔴 split در تگ می‌آید. اگر خطای ۱۷ روز اول این را داشتیم، اجراهای نشتی و
    #    تمیز نمی‌توانستند در یک پوشه قاتی شوند و اشتباه دیده می‌شد.
    global USE_ORIG_SPLIT
    USE_ORIG_SPLIT = a.orig_split
    pl.seed_everything(a.seed, workers=True)
    tag = f"{a.arm}_s{a.seed}" + ("_origsplit" if a.orig_split else "")
    _root, _dtr, _dva, _dte = _roots()
    print(f"   split: {'منتشرشده (نشتی)' if a.orig_split else 'جدا-به-tile'}"
          f"   ریشه: {_root.name}/{{{_dtr},{_dva},{_dte}}}")
    print("=" * 78)
    print(f"بازو: {a.arm}   seed: {a.seed}   "
          f"{'دودتست CPU' if a.smoke else f'{a.epochs} epoch'}")
    print("=" * 78)

    dm = build_datamodule(a.arm, a.seed, 2 if a.smoke else a.batch_size,
                          0 if a.smoke else a.workers, img_size=a.img_size)
    task, inj = build_task(a.arm, a.lr, pretrained=not a.no_pretrained)
    n_tot = sum(p.numel() for p in task.parameters())
    n_inj = sum(p.numel() for p in inj.extra_parameters()) if inj else (
        2 if a.arm in ("official", "tl_on") else 0)
    print(f"   پارامتر کل: {n_tot/1e6:.2f} M   ·   پارامتر تزریق: {n_inj:,}")

    if a.smoke:
        okk = smoke(a.arm, task, inj, dm)
        print(f"\n{'🟢 دودتست گذشت' if okk else '🔴 دودتست رد شد'}: {a.arm}")
        sys.exit(0 if okk else 1)

    OUT.mkdir(parents=True, exist_ok=True)
    # 🔴 معیار انتخاب: `val/mIoU`. اندازه‌گیری‌شده — TerraTorch این نام‌ها را
    #    می‌دهد: val/mIoU · val/IoU_0 · val/IoU_1 · val/F1_Score · val/Boundary_mIoU
    #    (نامی مثل `Multiclass_Jaccard_Index` وجود ندارد و اجرا را در پایان
    #     epoch اول می‌شکند — یعنی بعد از سوختن یک epoch کامل.)
    #    انتخاب روی mIoU است چون کم‌نوسان‌تر است؛ ولی `val/IoU_1` (کلاس سوخته)
    #    هم ثبت می‌شود، چون همان چیزی است که مقاله گزارش می‌کند.
    MONITOR = "val/mIoU"
    ckpt = ModelCheckpoint(dirpath=OUT / tag, filename="best",
                           monitor=MONITOR, mode="max", save_top_k=1)

    # 🔴 توقف زودهنگام **حذف شد** — اندازه‌گیری‌شده 2026-07-30 روی دو seed خط پایه:
    #
    #     seed 0 → بهترین در epoch ۳۶ ، ۴۸ epoch اجرا شد ، mIoU 0.8677
    #     seed 1 → بهترین در epoch  ۷ ، در آستانهٔ توقف   ، mIoU 0.8501
    #
    #     نوسان mIoU بین epochهای متوالی حدود ۰.۰۵ است. با این نویز، توقف
    #     زودهنگام در نقطه‌ای **تصادفی** شلیک می‌کند: یک جهش شانسی زودهنگام،
    #     اجرا را قبل از همگرایی خاموش می‌کند.
    #
    #     نتیجه‌اش این است که اختلاف بین بازوها بخشی‌اش «کِی توقف فعال شد» را
    #     می‌سنجد، نه «کدام نقطهٔ تزریق بهتر است» — یعنی یک متغیر مزاحم که
    #     مستقیماً روی همان چیزی می‌نشیند که قرار است اندازه بگیریم.
    #
    # بودجهٔ ثابت: هر اجرا دقیقاً `--epochs` می‌رود و بهترین چک‌پوینت از میان
    # همهٔ آن‌ها انتخاب می‌شود. همهٔ بازوها بودجهٔ یکسان می‌گیرند.
    trainer = pl.Trainer(
        max_epochs=a.epochs, accelerator="auto", devices=1,
        precision=a.precision, callbacks=[ckpt],
        accumulate_grad_batches=a.accum,
        default_root_dir=OUT / tag, log_every_n_steps=10,
        deterministic=False,          # ⚠️ عمداً: قطعیت کامل روی GPU کند است و
    )                                 #    ما تغییرپذیری seed را **اندازه می‌گیریم**
    print(f"   تصویر {a.img_size} · batch {a.batch_size} × انباشت {a.accum} "
          f"= مؤثر {a.batch_size*a.accum} · {a.precision}")
    t0 = time.time()
    if a.validate_only:
        ck = OUT / tag / "best.ckpt"
        if not ck.exists():
            raise FileNotFoundError(f"چک‌پوینتی نیست: {ck}")
        print(f"   بازیابی از {ck.name} — بدون آموزش دوباره")
        sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = task.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"🔴 چک‌پوینت جور درنیامد — غایب {list(missing)[:5]} "
                               f"اضافه {list(unexpected)[:5]}")
        res = trainer.validate(task, datamodule=dm)[0]
    else:
        trainer.fit(task, datamodule=dm)
        res = trainer.validate(task, datamodule=dm, ckpt_path="best")[0]

    rec = {"arm": a.arm, "seed": a.seed, "epochs_run": trainer.current_epoch,
           "n_params_total": n_tot, "n_params_injection": n_inj,
           "minutes": round((time.time() - t0) / 60, 1),
           "best_ckpt": str(ckpt.best_model_path), "val": res}
    (OUT / f"{tag}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ {tag}   mIoU={res.get('val/mIoU'):.4f}   "
          f"IoU_burn={res.get('val/IoU_1'):.4f}   {rec['minutes']} دقیقه")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
18_eval_per_chip.py — ماتریس درهم‌ریختگی به ازای هر chip، برای هر اجرا
=======================================================================
نسخه: v1 · تاریخ: 2026-08-01

چرا: طرح ثبت‌شده در `docs/PREREG-stratified.md` می‌پرسد آیا متادیتا **آنجا که
تصویر مبهم است** کمک می‌کند. برای جواب‌دادن باید بتوانیم mIoU را روی هر
زیرمجموعه‌ای از chipها دوباره حساب کنیم — پس شمارش‌های پیکسلی هر chip را جدا
ذخیره می‌کنیم و تحلیل را از اندازه‌گیری جدا نگه می‌داریم.

🔴 دروازهٔ درستی: mIoU کلِ بازسازی‌شده از این شمارش‌ها باید با mIoU ثبت‌شده در
   JSON همان اجرا تا ۰.۰۰۱ بخواند. اگر نخواند، خط لولهٔ استنتاج با اعتبارسنجی
   زمان آموزش اختلاف دارد و **هیچ عددی از این فایل قابل استفاده نیست**.

خروجی: <BIG>/runs/_perchip/<tag>.npz
        filenames (N,) · conf (N,2,2) با conf[i,t,p] = تعداد پیکسل با برچسب t و پیش‌بینی p

اجرا:  python 18_eval_per_chip.py --tag baseline_s0 [--device cpu]
       python 18_eval_per_chip.py --all
"""
import sys, json, argparse, importlib.util
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BIG = Path.home() / "Desktop" / "big-files" / "injection-routing"
RUNS = BIG / "runs"
OUT = RUNS / "_perchip"

# `16_run_arm.py` با رقم شروع می‌شود و import عادی نمی‌شود.
_spec = importlib.util.spec_from_file_location("run_arm", HERE / "16_run_arm.py")
run_arm = importlib.util.module_from_spec(_spec)
sys.modules["run_arm"] = run_arm
_spec.loader.exec_module(run_arm)

# کلیدهایی که به مدل داده نمی‌شوند.
NOT_MODEL_KWARGS = {"image", "mask", "filename", "cond"}


def arm_of(tag):
    """`tl_on_s2` → `tl_on`. جدا کردن از سمت راست چون نام بازوها زیرخط دارند."""
    arm, _, seed = tag.rpartition("_s")
    if arm not in run_arm.ARMS:
        raise ValueError(f"بازوی ناشناخته در تگ {tag!r}: {arm!r}")
    return arm, int(seed)


def pick_checkpoint(tag):
    """
    🔴 بعضی پوشه‌ها هم `best.ckpt` دارند هم `best-v1.ckpt` — یادگار اجرای
    `--validate-only` که ModelCheckpoint را دوباره ساخت. کدام درست است را
    حدس نمی‌زنیم؛ هر دو را برمی‌گردانیم تا انتخاب با اندازه‌گیری باشد
    (هرکدام mIoU ثبت‌شده را بازتولید کرد).
    """
    d = RUNS / tag
    cands = sorted(d.glob("*.ckpt"))
    if not cands:
        raise FileNotFoundError(f"هیچ چک‌پوینتی در {d} نیست.")
    return cands


def val_filenames(dm):
    """نام فایل هر نمونهٔ val، به همان ترتیبی که dataloader می‌دهد."""
    ds = dm.val_dataset
    for _ in range(4):                       # ممکن است ConditionedDataset دورش باشد
        for attr in ("image_files", "images", "image_list", "files"):
            v = getattr(ds, attr, None)
            if v:
                return [Path(str(p)).name for p in v]
        ds = getattr(ds, "inner", None) or getattr(ds, "dataset", None)
        if ds is None:
            break
    raise RuntimeError("نام فایل‌های val پیدا نشد.")


@torch.no_grad()
def evaluate(tag, ckpt_path, device):
    arm, seed = arm_of(tag)
    dm = run_arm.build_datamodule(arm, seed, batch_size=1, workers=0, img_size=224)
    task, inj = run_arm.build_task(arm, lr=1e-4, pretrained=True)
    task.set_injector(inj)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    missing, unexpected = task.load_state_dict(sd, strict=False)
    # 🔴 چک‌پوینت باید کل مدل را پر کند. کلید غایب یعنی بخشی از وزن‌ها
    #    از مقداردهی اولیه مانده و عدد بی‌معنی می‌شود.
    if missing:
        raise RuntimeError(f"{tag}: {len(missing)} کلید غایب هنگام بارگذاری چک‌پوینت، "
                           f"مثلاً {missing[:5]}")

    task.eval().to(device)
    dm.setup("fit")
    names = val_filenames(dm)
    dl = dm.val_dataloader()

    confs = []
    for i, batch in enumerate(dl):
        batch = task._bind_condition(batch)          # `cond` را می‌کَند و می‌بندد
        img = batch["image"].to(device)
        y = batch["mask"].to(device)
        kw = {k: (v.to(device) if torch.is_tensor(v) else v)
              for k, v in batch.items() if k not in NOT_MODEL_KWARGS}

        out = task.model(img, **kw)
        logits = getattr(out, "output", out)
        pred = logits.argmax(dim=1)

        valid = y != -1                              # ignore_index
        yt = y[valid].reshape(-1).long()
        yp = pred[valid].reshape(-1).long()
        c = torch.zeros(2, 2, dtype=torch.long, device=device)
        idx = yt * 2 + yp
        c.view(-1).scatter_add_(0, idx, torch.ones_like(idx))
        confs.append(c.cpu().numpy())

        if (i + 1) % 20 == 0:
            print(f"   [{i+1}/{len(dl)}]", flush=True)

    task._clear()
    conf = np.stack(confs)                            # (N, 2, 2)
    if len(names) != len(conf):
        raise RuntimeError(f"{tag}: {len(names)} نام در برابر {len(conf)} نمونه.")
    return names, conf


def miou_from_conf(conf):
    """conf با شکل (...,2,2) → (mIoU, IoU_0, IoU_1) روی مجموع همهٔ chipها."""
    t = conf.sum(axis=0) if conf.ndim == 3 else conf
    ious = []
    for c in (0, 1):
        tp = t[c, c]
        fn = t[c].sum() - tp
        fp = t[:, c].sum() - tp
        ious.append(tp / (tp + fp + fn) if (tp + fp + fn) else float("nan"))
    return float(np.mean(ious)), float(ious[0]), float(ious[1])


def recorded_miou(tag):
    p = RUNS / f"{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["val"]["val/mIoU"]


TOL = 0.001


def run_one(tag, device):
    print("=" * 70)
    print(f"{tag}   device={device}")
    want = recorded_miou(tag)
    print(f"   mIoU ثبت‌شده: {want if want is None else f'{want:.4f}'}")

    best = None
    for ck in pick_checkpoint(tag):
        names, conf = evaluate(tag, ck, device)
        got, i0, i1 = miou_from_conf(conf)
        gap = None if want is None else abs(got - want)
        print(f"   {ck.name}: mIoU={got:.4f}  IoU_0={i0:.4f}  IoU_1={i1:.4f}"
              + ("" if gap is None else f"   |اختلاف|={gap:.4f}"))
        if gap is not None and gap <= TOL:
            best = (ck, names, conf, got)
            break
        if best is None:
            best = (ck, names, conf, got)

    ck, names, conf, got = best
    if want is not None and abs(got - want) > TOL:
        # 🔴 صریح شکست می‌خوریم. یک خط لولهٔ استنتاج که mIoU زمان آموزش را
        #    بازتولید نمی‌کند، حق ندارد عدد تفکیک‌شده تولید کند.
        raise RuntimeError(
            f"🔴 {tag}: خط لوله {got:.4f} داد ولی ثبت‌شده {want:.4f} است "
            f"(اختلاف {abs(got-want):.4f} > {TOL}). هیچ چک‌پوینتی نخواند. "
            f"تا رفع این، تحلیل تفکیک‌شده معتبر نیست.")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{tag}.npz", filenames=np.array(names),
                        conf=conf, checkpoint=str(ck.name), miou=got)
    print(f"   ✅ ذخیره شد → _perchip/{tag}.npz   ({ck.name})")
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    tags = ([p.stem for p in sorted(RUNS.glob("*.json"))] if a.all else [a.tag])
    if not tags or tags == [None]:
        ap.error("یکی از --tag یا --all لازم است.")

    failed = []
    for t in tags:
        try:
            run_one(t, a.device)
        except Exception as e:
            print(f"   🔴 {t}: {e}")
            failed.append(t)
    print("\n" + "=" * 70)
    print(f"{len(tags) - len(failed)} از {len(tags)} موفق."
          + (f"  ناموفق: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

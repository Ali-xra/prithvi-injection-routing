# -*- coding: utf-8 -*-
"""
conditioned_data.py — رساندن بردار شرط ۴ بعدی به هر نمونه
==========================================================
نسخه: v1 · تاریخ: 2026-07-30 · **مشترک بین دو تسک**

مسئله: `GenericNonGeoSegmentationDataModule` هیچ پارامتری برای متادیتا ندارد
(اندازه‌گیری‌شده). پس حتی برای اجرای **بازوی خودِ Prithvi** هم باید مختصات و تاریخ
را خودمان همراه هر نمونه بفرستیم.

راه‌حل: دیتاست را **می‌پوشانیم**، نه اینکه بازنویسی کنیم.

    ⚠️ `cond` **بعد از** ترنسفورم‌ها اضافه می‌شود. اگر قبلش اضافه شود، albumentations
    با کلید ناشناس یا خطا می‌دهد یا آن را می‌اندازد — هر دو خاموش‌اند.

بردار شرط (به همین ترتیب، قفل):
    `lat_z, lon_z, doy_sin_z, doy_cos_z`
از `conditioning_v1.csv` که 🔒 قفل است. کلید اتصال: `filename`.

⚠️ آن CSV ده ستون دارد؛ ما فقط **چهار** تای بالا را می‌خوانیم. دلیلش اندازه‌گیری
`12_image_proxy_control.py` است: شش بُعد جوّی فراتر از تصویر چیزی ندادند
(`+0.0008`, `p=0.31`) و چهار بُعد مفید را رقیق می‌کردند.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

COND_COLS = ["lat_z", "lon_z", "doy_sin_z", "doy_cos_z"]


def load_conditioning(csv_path: str | Path) -> dict[str, torch.Tensor]:
    """`filename` → تنسور شکل (4,). خطای صریح اگر ستونی نبود."""
    csv_path = Path(csv_path)
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV خالی است: {csv_path}")
    missing = [c for c in COND_COLS + ["filename"] if c not in rows[0]]
    if missing:
        raise KeyError(f"ستون‌های غایب در {csv_path.name}: {missing}")
    out = {}
    for r in rows:
        out[r["filename"]] = torch.tensor(
            [float(r[c]) for c in COND_COLS], dtype=torch.float32)
    return out


def load_official(csv_path: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    """
    ورودی **مسیر رسمی خودِ Prithvi** — نه بردار ما.

    forward بک‌بون این دو را می‌خواهد (اندازه‌گیری‌شده در `15_arm0_official.py`):
        `location_coords`  شکل (2,)    → lat, lon  **خام، نه z-شده**
        `temporal_coords`  شکل (1, 2)  → year, doy **خام**

    ⚠️ خام بودن عمدی است: مسیر رسمی خودش نرمال‌سازی درونی دارد. اگر z-شده بدهیم،
    داریم مسیر آن‌ها را عوض می‌کنیم و دیگر «بازوی خودشان» نیست.
    """
    with Path(csv_path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    need = ["lat", "lon", "year", "doy", "filename"]
    missing = [c for c in need if c not in (rows[0] if rows else {})]
    if missing:
        raise KeyError(f"ستون‌های غایب: {missing}")
    return {
        r["filename"]: {
            "location_coords": torch.tensor(
                [float(r["lat"]), float(r["lon"])], dtype=torch.float32),
            "temporal_coords": torch.tensor(
                [[float(r["year"]), float(r["doy"])]], dtype=torch.float32),
        }
        for r in rows
    }


class ConditionedDataset(Dataset):
    """
    یک دیتاست TerraTorch را می‌پوشاند و کلید `cond` را به هر نمونه اضافه می‌کند.

    حالت شافل (`shuffle_seed`): بردارها بین نمونه‌ها **جابه‌جا** می‌شوند.
    هر نمونه یک بردار **معتبر ولی متعلق به نمونهٔ دیگر** می‌گیرد.

        🔴 چرا شافل و نه صفر یا نویز: بازوی شافل باید **همان تعداد پارامتر و همان
        توزیع ورودی** را داشته باشد و فقط **تناظر** را بشکند. اگر به‌جایش صفر بدهیم،
        داریم «با شرط» را با «بدون شرط» مقایسه می‌کنیم — که همان خط پایه است و
        سؤالِ «آیا بهبود از پارامتر اضافه است؟» بی‌جواب می‌ماند.
    """

    def __init__(self, inner: Dataset, cond_map: dict[str, torch.Tensor],
                 shuffle_seed: int | None = None):
        self.inner = inner
        self.cond_map = cond_map
        self.keys = self._extract_keys(inner)
        n_hit = sum(k in cond_map for k in self.keys)
        if n_hit != len(self.keys):
            miss = [k for k in self.keys if k not in cond_map][:5]
            raise KeyError(
                f"{len(self.keys)-n_hit} از {len(self.keys)} نمونه در CSV نیستند. "
                f"نمونهٔ غایب: {miss}")

        self.perm = None
        if shuffle_seed is not None:
            g = torch.Generator().manual_seed(shuffle_seed)
            p = torch.randperm(len(self.keys), generator=g)
            # 🔴 تضمین بی‌ثباتی: هیچ نمونه‌ای نباید بردار خودش را بگیرد
            fixed = (p == torch.arange(len(p))).sum().item()
            if fixed and len(p) > 1:
                p = torch.roll(p, 1)
            self.perm = p
            self.n_fixed_points = int((p == torch.arange(len(p))).sum())

    @staticmethod
    def _extract_keys(ds) -> list[str]:
        """نام فایل هر نمونه — TerraTorch آن را در `image_files` نگه می‌دارد."""
        for attr in ("image_files", "images", "image_list", "files"):
            v = getattr(ds, attr, None)
            if v is not None and len(v) == len(ds):
                return [Path(str(p)).name for p in v]
        raise AttributeError(
            f"نام فایل‌ها از {type(ds).__name__} استخراج نشد؛ "
            f"صفت‌های موجود: {[a for a in dir(ds) if 'file' in a or 'image' in a][:10]}")

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        sample = self.inner[i]          # ⚠️ اول ترنسفورم‌ها، بعد شرط
        j = i if self.perm is None else int(self.perm[i])
        v = self.cond_map[self.keys[j]]
        # مقدار یا یک تنسور است (بازوی adaLN) یا دیکشنری چند کلیدی (بازوی رسمی)
        if isinstance(v, dict):
            sample.update(v)
        else:
            sample["cond"] = v
        return sample


def wrap_datamodule(dm, csv_path, shuffle_seed: int | None = None,
                    shuffle_splits=("train",), mode: str = "adaln"):
    """
    `setup()` دیتاماژول را می‌پوشاند تا هر سه دیتاست شرط‌دار شوند.

    `mode="adaln"`    → کلید `cond` شکل (4,)، z-شده
    `mode="official"` → کلیدهای `location_coords` و `temporal_coords`، خام

    ⚠️ شافل **فقط روی train** پیش‌فرض است. اگر val هم شافل شود، معیار انتخاب
    بهترین چک‌پوینت هم خراب می‌شود و مقایسه بی‌معنا خواهد بود.
    """
    if mode not in ("adaln", "official"):
        raise ValueError(f"mode ناشناخته: {mode}")
    cond_map = (load_conditioning(csv_path) if mode == "adaln"
                else load_official(csv_path))
    orig_setup = dm.setup
    info = {}

    def setup(stage=None):
        orig_setup(stage)
        for split in ("train", "val", "test"):
            attr = f"{split}_dataset"
            ds = getattr(dm, attr, None)
            if ds is None:
                continue
            # 🔴 idempotent — اندازه‌گیری‌شده 2026-07-31 بعد از ۱۷۴ دقیقه آموزش:
            #    `trainer.validate()` بعد از `fit()` دوباره `setup` را صدا می‌زند.
            #    بدون این بررسی، دیتاستِ **قبلاً پوشانده‌شده** یک بار دیگر پوشانده
            #    می‌شود و لایهٔ دوم نام فایل‌ها را پیدا نمی‌کند:
            #        AttributeError: نام فایل‌ها از ConditionedDataset استخراج نشد
            #    خطا در **آخرین قدم** رخ می‌دهد، یعنی بعد از سوختن کل آموزش.
            if isinstance(ds, ConditionedDataset):
                continue
            seed = shuffle_seed if (shuffle_seed is not None
                                    and split in shuffle_splits) else None
            wrapped = ConditionedDataset(ds, cond_map, shuffle_seed=seed)
            setattr(dm, attr, wrapped)
            info[split] = {"n": len(wrapped), "shuffled": seed is not None}

    dm.setup = setup
    dm._cond_info = info
    return dm

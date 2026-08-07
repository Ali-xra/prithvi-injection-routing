# -*- coding: utf-8 -*-
"""
conditioned_task.py — وصل کردن بردار شرط به حلقهٔ آموزش Lightning
===================================================================
نسخه: v1 · تاریخ: 2026-07-30 · **مشترک بین دو تسک**

مسئله: `AdaLNBlockWrapper` شرط را از یک صفت ماژول می‌خواند (`self.cond`)، چون امضای
forward بلوک را در زنجیرهٔ TerraTorch نمی‌توان عوض کرد. پس یک نفر باید **قبل از هر
forward** آن صفت را از روی همان batch ست کند. آن یک نفر، این کلاس است.

    ⚠️ `cond` از batch **بیرون کشیده می‌شود** پیش از فراخوانی `super()`.
    دلیل: تسک‌های TerraTorch کلیدهای اضافهٔ batch را با `**kwargs` به forward مدل
    می‌دهند. اگر `cond` بماند، مدل با آرگومان ناشناس خطا می‌دهد — و آن خطا در
    میانهٔ آموزش رخ می‌دهد، نه در دودتست.

    ⚠️ در `on_*_epoch_end` و هر مسیری که forward بدون batch اجرا شود، شرط باید
    پاک شود، وگرنه بردارِ آخرین batch نشت می‌کند. `_clear` همین کار را می‌کند.
"""
from __future__ import annotations

import torch
from terratorch.tasks import SemanticSegmentationTask


class ConditionedSegmentationTask(SemanticSegmentationTask):
    """`SemanticSegmentationTask` + ست کردن شرط پیش از هر گام."""

    def set_injector(self, injector) -> None:
        """`AdaLNInjector` — یا None برای بازوی خط پایه."""
        self._injector = injector

    # ---- درونی ----
    # 🔴 اسم این متد **نباید** `_apply` باشد. اندازه‌گیری‌شده 2026-07-30:
    #    `nn.Module._apply(fn)` یک متد هستهٔ PyTorch است و `.to()` · `.cuda()` ·
    #    `.float()` همگی از آن برای پیمایش و جابه‌جایی پارامترها استفاده می‌کنند.
    #    با بازنویسی‌اش، `model.to("cuda")` بی‌صدا کاری نمی‌کرد و مدل روی CPU
    #    می‌ماند. خطایی که بیرون می‌زد ربطی به علت نداشت:
    #        RuntimeError: Input type (torch.cuda.FloatTensor) and
    #                      weight type (torch.FloatTensor) should be the same
    #    و در مسیر دیگری: 'function' object has no attribute 'to'
    #    هر دو ما را دنبال «مشکل دقت عددی fp16» فرستادند که اصلاً وجود نداشت.
    def _bind_condition(self, batch):
        inj = getattr(self, "_injector", None)
        cond = batch.pop("cond", None) if isinstance(batch, dict) else None
        if inj is not None:
            if cond is None:
                raise KeyError(
                    "بازوی adaLN فعال است ولی batch کلید `cond` ندارد — "
                    "یعنی `wrap_datamodule` صدا زده نشده یا `setup` جای دیگری اجرا شده.")
            inj.set_condition(cond.to(self.device))
        return batch

    def _clear(self):
        inj = getattr(self, "_injector", None)
        if inj is not None:
            inj.set_condition(None)

    # ---- گام‌ها ----
    def training_step(self, batch, *a, **kw):
        return super().training_step(self._bind_condition(batch), *a, **kw)

    def validation_step(self, batch, *a, **kw):
        return super().validation_step(self._bind_condition(batch), *a, **kw)

    def test_step(self, batch, *a, **kw):
        return super().test_step(self._bind_condition(batch), *a, **kw)

    def predict_step(self, batch, *a, **kw):
        return super().predict_step(self._bind_condition(batch), *a, **kw)

    def on_train_epoch_end(self):
        self._clear()
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        self._clear()
        return super().on_validation_epoch_end()

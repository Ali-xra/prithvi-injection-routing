# Sprint log — gate design-space, Aug 2026

Ten-day sprint on an idle GTX 1070. Goal: touch more cells of the gate design
space (readout x depth x form x payload) on the small model, so a shortlist of
well-controlled candidates can be handed to heavy compute later.

Governed by `conditioning-landscape/BRIEF-gate-continue__v1.md`. Every arm keeps
the frozen six intact; new work is additive and named so no locked number moves.

## Environment (verified 2026-08-07)

- GPU: NVIDIA GeForce GTX 1070, CUDA available, torch 2.13.0+cu126.
- Data cache `eurosat_s1_cache.npz`: images (27000, 2, 64, 64) VV/VH, coords,
  labels, splits, names, classes. Geo split in `geo_split.npz`.
- Working dir: `C:\Users\aliso\Desktop\big-files\loc`; venv `venv-gpu`.

## Fork resolved — incidence-angle payload is NOT available

Candidate #3 (a non-prior physical payload: Sentinel-1 incidence angle / orbit
direction, injected without changing dataset) was the highest-leverage cheap
idea IF the metadata survived. It did not: the cache carries only 2 bands
(VV/VH) + coords + labels, and the raw `all_imgs` tree is gone. Recovering
incidence angle would need re-deriving geometry from the original S1 GRD scenes
— a heavy pipeline, out of scope for this sprint. **Dropped.**

## Queue (guaranteed on the small model, per the design-space doc)

1. **Logit prior on the ViT's own output** (candidate #5) — LAUNCHED.
   `39_logit_prior.py`. 35_prior_test used 12 hand features (val ~63); this asks
   whether the class-prior argument still explains the null on the real 2.7M ViT
   (val ~73). Trains `none` x5 seeds, combines its softmax with the same K=200
   regional prior at alpha=1, plus a deranged-coordinate control that must
   collapse to ~0.
2. **12-block depth model** — give the "window not direction" claim real
   resolution (6 blocks only has 5 interior positions).
3. **Readout zoo at the window** — richer readouts than mean/std/max already
   tested: GeM, second-order/bilinear (texture — the right quantity for SAR),
   attention-pooling. Each with a `--shuffle-coords` twin.

## Results

(appended as each run completes)

### Run 1 — logit prior, first pass (2026-08-07) — CONFOUND CAUGHT

`39_logit_prior.py`, geo split, `none` x5 seeds. The five seeds reproduced the
locked geo `none` numbers byte for byte (73.12 / 74.13 / 72.27 / 73.37 / 73.14),
so the pipeline is verified.

**But the analysis was confounded and its headline is not trustworthy.** I
averaged the five seeds' softmax *before* applying the prior. That makes the
image baseline a 5-model ENSEMBLE (76.07), not the single-model `none` (73.21)
against which the trained gain (`add - none = +3.96`) was defined. At the
pre-registered `alpha = 1` the prior then recovers -0.02 on the ensemble and the
script prints "NOT-PRIOR" — but that compares a prior against a baseline the
gain was never measured on. Numbers:

```
image-only ViT (softmax-avg of 5 seeds)   76.07     <- ENSEMBLE, not single none
location prior alone                        23.97
image x prior (alpha=1)                     76.05    recovered -0.02
CONTROL deranged-coord prior                62.24    recovered -13.83   (collapses, good)
alpha 0.25 -> 78.12 (+2.06)   0.5 -> 77.91 (+1.85)   1.0 -> 76.05 (-0.02)
```

The deranged control behaving (-13.83) shows the prior construction itself is
sound; the flaw is only the ensemble baseline. Signal worth keeping: a weaker
prior (alpha 0.25) still adds +2.06 even on the strong ensemble, so location
contributes something to the real ViT — but far less than the 86% the weak
12-feature probe (35_prior_test.py) suggested. Consistent with: a strong image
model leaves little for a coarse regional prior to fill.

**Fix (Run 2):** compute recovery PER SEED against each single model (baseline
~73.2, matched to how none/add were measured), report the per-seed mean, and
save per-seed logits so re-analysis never needs retraining again.

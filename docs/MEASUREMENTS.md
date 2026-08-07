# Measurements

Every number produced so far. Thresholds were fixed before each run.

**Protocol.** A block of features under test is added to a control block; the difference in
validation AUC is compared against a null built by permuting the test block 200 times.
Both a linear model (logistic regression) and a non-linear one (random forest) are reported,
because a block can carry signal that only one of them can use.

**Pre-registered thresholds:** `delta AUC >= 0.02` **and** `p < 0.05`. Both required.

**Target.** Burn fraction binarised at the training-split median (4.038 %). The published
task is pixel segmentation; this is a tabular proxy for it, which is the central limitation
of everything on this page.

---

## Burn scars — control ladder

Split: tile-disjoint, 563 / 121 / 120. Test split never opened.

| Control block | Test block | Model | Control AUC | Full AUC | Delta | p | |
|---|---|---|---|---|---|---|---|
| season | 10-dim vector | linear | 0.6518 | 0.6513 | -0.0004 | 0.249 | ✗ |
| season | 10-dim vector | forest | 0.5820 | 0.7072 | +0.1252 | 0.005 | ✓ |
| season + geography | 6 weather | linear | 0.6929 | 0.6513 | -0.0416 | 0.891 | ✗ |
| season + geography | 6 weather | forest | 0.7107 | 0.7263 | +0.0156 | 0.105 | ✗ |
| **image proxy** | 10-dim vector | linear | 0.7105 | 0.7080 | -0.0025 | 0.343 | ✗ |
| **image proxy** | 10-dim vector | forest | 0.6787 | 0.6990 | +0.0203 | 0.199 | ✗ |
| **image proxy** | **4 geo/seasonal** | **forest** | **0.6787** | **0.7074** | **+0.0287** | **0.010** | **✓** |
| image proxy | 6 weather | forest | 0.6787 | 0.7028 | +0.0241 | 0.119 | ✗ |
| image proxy + geography | 6 weather | linear | 0.7072 | 0.7080 | +0.0008 | 0.313 | ✗ |

The last row is the decisive one: with location and season already in the control, weather
contributes nothing.

**Image proxy composition (13 features).** Mean of each of the six bands, plus NDVI, NBR,
NDWI, NDMI, SAVI, SWIR ratio, and overall brightness — all computed from the chips
themselves, over valid pixels only.

---

## Cross-dataset summary

| Beyond an image proxy | Burn scars | GPP | Flood |
|---|---|---|---|
| Weather | ✗ `+0.0008` (p=0.31) | ✗ | ✗ `+0.005`/`+0.012` |
| Absolute location + date | ✓ `+0.0287` (p=0.010) | ~ `+0.0097` marginal | ✗ |

Weather fails in all three. Location and date pass on burn scars, are marginal on GPP, and
fail on flood.

**Flood caveats.** 11 events only, so `lat/lon` approximates event identity; and its image
proxy includes MNDWI, a direct water index, which for water segmentation is close to reading
the label. Its negative result is the weakest of the three.

**GPP.** Weather passed a season control and a biome control, then failed once spectral
indices entered: control AUC rose to 0.92–0.96 and weather added nothing. Weather there is a
proxy for greenness, which NDVI reports directly.

---

## Leakage changes the sign of the conclusion

Measured on GPP, same auxiliary block, two splits:

| Split | Auxiliary block contribution |
|---|---|
| leaky (sites shared across splits) | `+0.0052` |
| clean (site-disjoint) | `+0.0288` |

Site identity substitutes for the auxiliary variable when the same site appears on both
sides. Leakage therefore makes auxiliary data appear **less** useful, and published ablations
run on leaky splits have underestimated it.

---

## Model structure (measured, not assumed)

| Property | Value |
|---|---|
| Backbone | `PrithviViT`, 24 blocks |
| `embed_dim` | 1024 |
| Per-block norms | `norm1`, `norm2`, both `elementwise_affine=True` |
| Checkpoint load | zero missing keys |
| Block replacement | verified working end-to-end |
| Total parameters | 324.20 M |

### Injection capacity by arm

| Arm | Injection parameters | Share of baseline |
|---|---|---|
| `baseline` | 0 | — |
| `official` (`coords_encoding`) | **2** | 0.0000006 % |
| `adaln` | **25,331,200** | 7.81 % |
| `shuffle` | 25,331,200 | 7.81 % |

The official path is two learned scalars: `temporal_embed_enc.scale` and
`location_embed_enc.scale`. Both confirmed to receive gradient (`0.0137`, `0.0158`), so the
arm is active rather than an inert flag.

> 🔴 **Correction, 6 August.** Source: `terratorch/models/backbones/prithvi_mae.py`, via the
> papers chat's portfolio sweep. Two things we had wrong about Prithvi's mechanism:
>
> 1. **There are four scalars in the full architecture, not two.** The MAE decoder carries
>    its own `temporal_embed_dec` / `location_embed_dec` weights. Only the two *encoder*
>    scalars survive into fine-tuning, so every number in this file is unaffected — but
>    "Prithvi gives metadata a two-scalar channel" is only true of the fine-tuning path and
>    must be said that way.
> 2. **Time is not one vector per sample.** `TemporalEncoder` builds a vector **per frame**
>    and `repeat_interleave`s it across that frame's tokens only. Location *is* a single
>    vector for the whole sample. Our `add` arm is single-frame, so it reproduces the
>    location path exactly; it does not reproduce the temporal path, and no claim in this
>    file depends on it. Anywhere the docs imply "location and time enter identically",
>    that is wrong.

---

## Equivalence test — adaLN arm

| Check | Expected | Max abs difference | |
|---|---|---|---|
| A · condition `None` | equal to baseline | `0.000e+00`, exact | ✓ |
| B · condition active, zero weights | equal to baseline | `0.000e+00`, exact | ✓ |
| C · randomised modulation weights | **must differ** | `4.654e-01` | ✓ |
| D · different condition vector | **must differ** | `9.820e-01` | ✓ |

C and D matter as much as A and B: without them, an inert module passes.

---

## CPU smoke test — all four arms

| Arm | Total params | Injection params | Extra batch keys | loss step 1 → 2 | |
|---|---|---|---|---|---|
| `baseline` | 324.20 M | 0 | — | 0.7324 → 0.6728 | ✓ |
| `official` | 324.20 M | 2 | `location_coords`, `temporal_coords` | 0.7279 → 0.6640 | ✓ |
| `adaln` | 349.54 M | 25,331,200 | — (via `cond`) | 0.7345 → 0.6915 | ✓ |
| `shuffle` | 349.54 M | 25,331,200 | — (shuffled) | 0.7345 → 0.6899 | ✓ |

Gradient checks: adaLN `to_mod` `1.59e+02`; conditioning encoder `0.000e+00` at step 1 —
correct, see [`FAILURES.md`](FAILURES.md#f3) — and `1.38e-01` at step 2.

---

## Hardware

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1070, 8 GB, driver 560.94 |
| Compute capability | `sm_61` (Pascal) |
| PyTorch | 2.13.0+cu126 |
| `sm_61` in `get_arch_list()` | yes |
| Real matmul + conv backward on device | passed |

`torch.cuda.is_bf16_supported()` returns `True` on this card, but Pascal has no bf16 tensor
cores; mixed bf16 is emulated. Training precision must be `16-mixed` or `32`.

With 8 GB, a 324 M model under AdamW needs roughly 5 GB for parameters, gradients, and
optimiser state before activations. Batch size, crop size, and gradient accumulation will be
set from a measured run, not from this estimate.

---

## Baseline: three seeds, and the detection threshold

Fixed budget of 50 epochs, no early stopping, best checkpoint by `val/mIoU`.

| Run | mIoU | IoU (burned) | minutes |
|---|---|---|---|
| `baseline_s0` | 0.8663 | 0.8135 | 158.0 |
| `baseline_s1` | 0.8705 | 0.8190 | 160.3 |
| `baseline_s2` | 0.8735 | 0.8222 | 154.3 |
| **mean** | **0.8701** | **0.8182** | |
| **std (n−1)** | **0.0036** | **0.0044** | |

The pre-registered concern was that seed noise might exceed 0.03 and make the comparison
impossible. It does not: the spread is roughly ten times smaller.

**Detection threshold, fixed before any injection arm was run.** With 3 seeds per arm, the
standard error of a mean is `0.0036/√3 = 0.0021`, and of a difference between two arms
`0.0021×√2 = 0.0030`. At two sigma:

> A difference of **≥ 0.006 mIoU** between arms is detectable. Anything smaller is noise
> and will be reported as such.

### The noise came from the protocol, not the data

| | gap between `s0` and `s1` |
|---|---|
| early stopping, patience 12 | **0.0176** |
| fixed 50-epoch budget | **0.0042** |

A fourfold reduction with no extra data. With ~0.05 epoch-to-epoch variation in `val/mIoU`,
early stopping fires at an essentially arbitrary point: seed 1 hit a lucky spike at epoch 7
and was terminated at 0.8501; the same seed under a fixed budget reached 0.8705.

> Any ablation that uses early stopping on a noisy validation metric is partly measuring
> *when training stopped* rather than the effect of the intervention.

Caveats: `n=3` is small, so the standard deviation is itself an uncertain estimate; the
monotone `s0 < s1 < s2` ordering is coincidence; and 224 crops mean absolute values are not
comparable to published numbers, though between-arm differences are unaffected because every
arm is cropped identically.

## 🔴 Result 1 — the official injection point has no measurable effect

Three seeds, fixed 50-epoch budget, settings identical to baseline in every respect
except `coords_encoding`.

| Run | mIoU | IoU (burned) |
|---|---|---|
| `official_s0` | 0.8734 | 0.8224 |
| `official_s1` | 0.8680 | 0.8154 |
| `official_s2` | 0.8677 | 0.8142 |
| **mean** | **0.8697** | **0.8173** |
| **std (n−1)** | **0.0032** | **0.0044** |

| Arm | mIoU | IoU (burned) |
|---|---|---|
| `baseline` | 0.8701 ± 0.0036 | 0.8182 ± 0.0044 |
| `official` | 0.8697 ± 0.0032 | 0.8173 ± 0.0044 |
| **difference** | **−0.0004** | **−0.0009** |

Standard error of the difference: `√(0.0036²/3 + 0.0032²/3) = 0.0028`.
Difference over standard error: **−0.14**.

The detection threshold, fixed before any injection arm ran, was **0.006**. The observed
difference is one fifteenth of it.

> **Feeding location and acquisition date through Prithvi's own metadata path produces no
> measurable change in segmentation quality on HLS Burn Scars.** Not better, not worse —
> below the noise floor of an instrument we calibrated first.

### Why a null result is worth reporting here

1. **This ablation has never been published.** The mechanism exists in the code, is off in
   every downstream config, and nobody had measured it.
2. **There is an engineering reason it was never measured:** with `strict=True` checkpoint
   loading, enabling `coords_encoding` fails outright against the released weights. The
   path was not merely unablated — it was not runnable without a workaround.
3. The claim rests on three seeds, a fixed budget, a pre-registered threshold, and a noise
   floor measured before the comparison.

### What this does not say

- Not that location and date are useless — that they do nothing **through this path**.
- Not that it generalises: only burn scars was tested.
- Our tabular probe measured those four numbers carrying `+0.0287` AUC beyond an image
  proxy. That signal exists; it simply does not reach IoU through this entry point.

### This makes the remaining question sharper, not smaller

If the official path does nothing, the question becomes whether adaLN can extract something
from the *same four numbers* that a single additive injection cannot. A yes would be direct
evidence that the entry point matters. A no would say the four numbers are weak for this
task regardless of route — consistent with how small `+0.0287` was.

Against the pre-registered prediction (25 % adaLN wins / 25 % worse / 50 % lost in noise):
for the **official** arm, the "lost in noise" branch is what happened.

## Measurement 8 — adaLN arm: 25 M parameters, still nothing

Same four conditioning dimensions, entering instead through adaLN modulation on all 24
transformer blocks (gate deliberately dropped — see DECISIONS). Three seeds, same fixed
50-epoch budget, same split, same augmentation.

| Run | mIoU | IoU (burned) | minutes |
|---|---|---|---|
| `adaln_s0` | 0.8671 | 0.8128 | 218.7 |
| `adaln_s1` | 0.8647 | 0.8078 | 222.1 |
| `adaln_s2` | 0.8643 | 0.8102 | 214.7 |
| **mean** | **0.8654** | **0.8103** | |
| **std (n−1)** | **0.0015** | **0.0025** | |

```
adaln − baseline  Δ = −0.0047   SE = 0.00226   Δ/SE = −2.09   Welch p ≈ 0.14
```

Below the pre-registered 0.006 threshold. adaLN is lower but not significantly so.

**A tension recorded honestly:** the 0.006 threshold was computed assuming both arms carry
std 0.0036. adaLN's actual std came out at 0.0015 — far tighter. On the observed variances
the same gap sits at 2.09 sigma. **The threshold was not moved.** It was locked before the
data and it stays where it was; moving it after seeing the result is precisely what this
project is built against. Both numbers are reported and the reader can judge.

**Side finding:** seed-to-seed spread halved (0.0036 → 0.0015) while the mean fell.
adaLN made the model *more stable and slightly worse* — a pattern consistent with
over-regularisation or overfitting on 563 training samples.

## Measurement 9 — the shuffle control: the number that settles it

`shuffle` is architecturally identical to `adaln` — the same 25,331,200 trainable
parameters — but each sample's conditioning vector is swapped with another sample's
(zero fixed points, train split only). If location and date were carrying signal, breaking
the correspondence should cost something.

| Run | mIoU | IoU (burned) | minutes |
|---|---|---|---|
| `shuffle_s0` | 0.8653 | 0.8121 | 214.9 |

```
shuffle_s0             0.8653
adaln mean (3 seeds)   0.8654
                       ────────
difference             0.0001
```

One ten-thousandth. The seed-to-seed std *within* the adaLN arm is 0.0015 — fifteen times
larger than this gap.

Of the two branches written down before `shuffle` ran, the first is what happened:

> ✅ **If `shuffle` drops as much as `adaln` → the drop is capacity, not information.**
> ❌ If `shuffle` does not drop → the drop comes from the information itself.

**Destroying the correspondence between conditioning and image costs nothing. Therefore
the correct correspondence was buying nothing.** The −0.0047 adaLN deficit is the price of
adding 25 M trainable parameters on 563 samples; the information contributes no part of it.

## Final table — four arms

| Arm | mIoU | IoU (burned) | seeds | injection params | Δ vs baseline |
|---|---|---|---|---|---|
| `baseline` | **0.8701 ± 0.0036** | 0.8182 ± 0.0044 | 3 | 0 | — |
| `official` | **0.8697 ± 0.0032** | 0.8173 ± 0.0044 | 3 | 2 | −0.0004 |
| `adaln` | **0.8654 ± 0.0015** | 0.8103 ± 0.0025 | 3 | 25,331,200 | −0.0047 |
| `shuffle` | **0.8653** | 0.8121 | 1 | 25,331,200 | −0.0048 |

Raw runs:

```
baseline   0.8663  0.8705  0.8735
official   0.8734  0.8680  0.8677
adaln      0.8671  0.8647  0.8643
shuffle    0.8653
```

Every comparison against the locked 0.006 threshold:

```
official − baseline   Δ = −0.0004   SE = 0.00279   Δ/SE = −0.14   → in the noise
adaln    − baseline   Δ = −0.0047   SE = 0.00226   Δ/SE = −2.09   → below threshold
adaln    − official   Δ = −0.0043   SE = 0.00205   Δ/SE = −2.12   → below threshold
shuffle  − adaln      Δ = −0.0001                                 → zero
```

**Nothing crosses 0.006.**

Against the pre-registered prediction (initially 25 % adaLN wins / 25 % worse / 50 % lost
in noise; revised to 15/20/65 after `adaln_s0`): the "lost in noise" branch — the one that
carried the most weight — is what happened.

## What was shown, and what was not

**Shown** (HLS Burn Scars, Prithvi-EO-2.0-300M, tile-disjoint split, fixed 50 epochs):

1. Prithvi's own metadata path (`coords_encoding`, 2 learned scalars) makes no detectable
   difference against baseline: Δ = −0.0004.
2. Neither does adaLN with 25 M parameters — and the `shuffle` control shows its deficit
   has **nothing to do with the information**, only with capacity.
3. So for this task the question "where should auxiliary data enter?" collapses before it
   can be answered: there is nothing to route.
4. This is consistent with the tabular probe: the image proxy alone reaches AUC 0.7105 and
   the four geographic dimensions add only +0.0287.

**Not shown, and not to be claimed:**

- That this generalises to other tasks. Flood and GPP directories exist; only burn scars ran.
- That adaLN is useless in general. It was tested here on 563 training samples.
- That location and date are unimportant for EO. They were unimportant for *separating
  burned from unburned*, because the spectral bands nearly settle that question alone.
- Anything beyond the single binary question `shuffle` was built to answer — it is one seed.

## Not yet measured

- Whether a fine-tuned Prithvi recovers absolute location from imagery on its own.
- Whether the conditioning helps specifically where the image is ambiguous — cloud, haze,
  look-alike surfaces — rather than on average.

## Measurement 10 — the payload is empty on all three of the paper's HLS tasks

Prompted by the right question on 1 August: if the metadata does nothing in the paper
itself, why test *where* to inject it?

So we went back to the tabular probes, which were run before any GPU time and which ask one
thing: **does location/date add anything beyond what the imagery already tells you?**
Pre-registered gate: `delta ≥ 0.02` **and** `p < 0.05`.

| Task | split | geo/date beyond image | gate |
|---|---|---|---|
| **Burn scars** | tile-disjoint | **+0.0287** (p = 0.010) | ✅ passes |
| **Flood** | event-split (honest) | **−0.0052** | ❌ negative |
| **Flood** | within-event | +0.0194 | ❌ and leaky — see below |
| **GPP** | site-split | +0.0097 / +0.0057 | ❌ image alone already 0.951 |

Burn scars — the task we happened to pick — is the **only** one of the three that passes,
and it passes at 0.0287, which then failed to survive into IoU.

### The flood leakage number is worth its own line

```
mean_auc_event_from_latlon = 1.0   (within-event split, 11 events)
```

On a within-event split, latitude and longitude identify the flood event **perfectly**. Any
"geographic signal" measured there is event memorisation. Split by event instead and the
contribution goes negative. An ablation run on the naive split would report that location
helps.

### The GPP number is a ceiling effect, not a signal

The image proxy alone reaches AUC 0.951–0.961. There is almost nothing left for any
auxiliary variable to explain, so the small positive deltas say more about the headroom than
about the metadata.

## Measurement 11 — what the paper's own evidence for TL actually is

Read from the paper (arXiv:2412.02732v3):

- 🔴 **Corrected 2026-08-05 after independent verification.** The earlier wording here —
  "there is no per-dataset table" — was **wrong** and is retracted rather than quietly
  edited. In the **main text** the GEO-Bench comparison is an aggregated figure (Figure 6)
  with no error bars, and that part stands. But the **appendix, Tables A1–A4**, does give
  per-dataset results with mean / std / max / min for all thirteen models including all
  three TL variants.
- The correct statement is therefore: **there is no dedicated TL-vs-non-TL ablation table;
  the per-dataset numbers exist only in the appendix.** The "Δ is smaller than the noise"
  argument survives unchanged — the standard deviations are in those same appendix tables
  and run 0.2–1.0 mIoU points.
- The one concrete pair visible in the disaster-response tables (Table IV, flood) —
  ✅ verified with the spread printed in the table itself:
  **`600M-TL` = 90.3 (0.3) vs `600M` = 89.9 (0.6) → +0.4 mIoU points.**
- GEO-Bench benchmarking does use 10 seeds, and Figure 7 shows the spread — but the
  TL-vs-non-TL comparison itself is not reported with a spread.

Placed next to what we measured on the same kind of task:

```
seed-to-seed noise, baseline, 3 seeds   =  ± 0.36 mIoU points
our detection threshold (locked)        =    0.60 mIoU points
the paper's TL gain on flood            =    0.40 mIoU points
```

**The published improvement is smaller than the noise floor we measured.** That does not
make it wrong — a real effect can hide under a noise floor, and their setup is not ours.
It does mean the claim, as published, is not separable from seed variation without a
reported spread.

### What this does to the project

The original question — *where* should auxiliary data enter? — cannot be answered on these
tasks, because the payload is empty. That is the user's objection and it is correct.

The question that **can** be answered, and that this repository now answers, is narrower and
better posed:

> Prithvi-EO-2.0 ships a metadata pathway and a full family of TL checkpoints. Does that
> pathway change anything on the paper's own downstream tasks, measured against a
> pre-registered threshold with the seed noise established first?

`tl_on` vs `tl_off` — the same TL weights with the metadata path enabled and disabled — is
that measurement, and it appears nowhere in the literature.

## Measurement 12 — the gate varies, but not for the reason H3 required

Run 2026-08-04, `27_gate_mechanism.py`, zero GPU. Analysis only; no arm was retrained.

**Background.** H3 predicted `gate > add`: if the value of location varies by sample, a
scalar conditioned on image content should capture more than one global constant. The
observed result was `gate − add = +0.25`, below the locked threshold of `0.3735` — H3
refuted. This measurement asks *why*.

**Does the gate vary at all? Yes, substantially.**

| run | mean | std | min |
|---|---|---|---|
| `gate_s0` | 0.3849 | 0.1340 | 0.2515 |
| `gate_s1` | 0.3304 | 0.0406 | 0.2792 |
| `gate_s2` | 0.4355 | 0.1944 | 0.2223 |
| `gate_s3` | 0.4626 | 0.1909 | 0.2986 |
| `gate_s4` | 0.3911 | 0.0867 | 0.3286 |

Range across seeds `[0.22, 2.52]` — a tenfold spread. So the "it collapsed to a constant"
explanation is wrong: the network is deciding chip by chip.

> 🔴 **Pointer added 6 August.** The `2.52` here is real but comes from the 142 all-black
> chips found in Measurement 21. On clean chips the maximum is **1.30**, roughly threefold
> on the median seed. The number is left as originally recorded; do not quote it without
> this caveat. The conclusion of this paragraph — chip-by-chip variation, not a constant —
> is unaffected.

**But it is not deciding what the hypothesis required.**

The hypothesis written in the script before running it: *the gate opens where the image is
uninformative*, which predicts a **negative** correlation between an image-only
classifier's confidence and the gate the network chose.

| run | pearson r | p | gate when image UNSURE | gate when image SURE |
|---|---|---|---|---|
| `gate_s0` | **+0.0676** | 6.5e-07 | 0.3854 | 0.4183 |
| `gate_s1` | **+0.0721** | 1.1e-07 | 0.3300 | 0.3403 |
| `gate_s2` | **+0.0927** | 8.9e-12 | 0.4311 | 0.4911 |
| `gate_s3` | **+0.1526** | 1.7e-29 | 0.4439 | 0.5298 |
| `gate_s4` | **+0.2032** | 2.0e-51 | 0.3781 | 0.4277 |
| **mean** | **+0.1177** | | | |

Positive in all five seeds. In all five, the gate is *smaller* where the image classifier is
unsure and *smaller* where it is wrong. The pre-registered direction is refuted, consistently.

> **The gate varies tenfold, but it does not track image uncertainty. The mechanism H3
> depended on did not form — which is why `gate ≈ add`.**

**Caveats, stated rather than smoothed over.**

- The effect is **weak**: mean `r = 0.118`, roughly 1.4 % of variance. The *direction* is
  consistent across seeds; the *magnitude* is not (`+0.068` to `+0.203`).
- Spearman is unstable across seeds (`−0.031`, `+0.054`, `−0.007`, `+0.394`, `+0.365`), so
  there is a weak consistent linear relationship but no stable monotone one.
- "Image confidence" here is the confidence of a 12-feature spectral classifier, **not of
  the ViT itself**. The network may find different chips hard.
- No mechanism for the positive direction is claimed. A post-hoc story (e.g. "the gate
  tracks backscatter intensity rather than ambiguity") would be a guess, not a measurement.

**Pipeline consistency check:** the image-only classifier reached `70.50` on val — identical
to the locked tabular probe figure. The two paths agree.

**Follow-up that would settle the mechanism** (not run): correlate the gate against the
trained ViT's own predictive entropy rather than a proxy classifier's, and against simple
image statistics (mean backscatter, texture energy) to see whether the gate is tracking
image content rather than image difficulty.

## 🔴 Measurement 13 — the EuroSAT location payload is proximity leakage

Run 2026-08-04, `29_leakage_probe.py`, zero GPU. Prompted by the observation that
`21_cache.py:29` loads the **official** EuroSAT split, which is random rather than
geographically disjoint — the exact thing the burn-scars phase rejected and this phase
did not check.

**Validation first.** The script reproduces three of the four locked probe figures
exactly on the official split — `image only 70.50`, `location only 69.94`,
`image + location 87.02` — so the feature alignment and the pipeline are correct.
(The fourth, `image + shuffled loc`, differs: this script shuffles the **validation**
coordinates only while training on the true correspondence, which is a stricter variant
than the locked `69.98`. Not comparable; reported separately.)

### A · proximity upper bound — no learning, no image

For every val chip, copy the label of its nearest **train** chip in coordinate space.

| | official split | location-disjoint split |
|---|---|---|
| median distance to nearest train chip | **1.13 km** | 149.50 km |
| p10 distance | 0.63 km | 71.78 km |
| copy-the-neighbour accuracy | **77.30** | 33.10 |

> On the official split the median validation chip has a training chip **1.1 km away**,
> and copying that neighbour's label alone scores **77.30** — higher than the image-only
> probe (70.50) and higher than the location-only probe (69.94).

### B · the four-way probe under both splits

Location-disjoint split: 60 k-means clusters on coordinates, whole clusters assigned to
one side. 15 859 train / 5 741 val.

| | official split | location-disjoint split |
|---|---|---|
| image only | 70.50 | 63.16 |
| location only | **69.94** | **24.06** |
| image + location | **87.02** | **62.85** |
| **gain of location over image alone** | **+16.52** | **−0.31** |

### Verdict

> **The `+16.52` that justified choosing EuroSAT-S1 does not survive a geographically
> disjoint split. Under a clean split, location adds nothing on top of the imagery —
> it is very slightly negative.**

The failure mode is identical to the burn-scars phase: the payload is empty. The
difference is that in burn scars it was detected before the conclusion was drawn, and
here it was not.

### What this does and does not establish

- **Does:** the official-split figure `+16.52` is an upper bound and most of it is
  nearest-neighbour lookup, not geographic knowledge. Copy-the-neighbour at 77.30 settles
  that on its own.
- **Does not:** prove location is worth exactly zero. On a geographically disjoint split,
  validation coordinates fall outside the ranges seen in training, so a tree model cannot
  extrapolate — the same structural problem as the flood event-split. `−0.31` is therefore
  a **lower** bound, deflated by design, just as `+16.52` is an upper bound inflated by
  leakage. The truth lies between, and the size of the proximity effect puts it near the
  lower end.
- `location only` at `24.06` is still well above the 10.00 chance level, so some coarse
  regional structure does generalise across 150 km. It simply does not add anything the
  imagery has not already supplied.

### Consequence for the routing result

The six-arm comparison remains **internally valid** — every arm shares the split, the
payload and the augmentation, so the between-arm differences still measure routing.
But what they measure routing *of* has changed: the arms differ in how well they exploit
a **proximity lookup key**, not a source of geographic knowledge.

> **`shuffle` cannot catch this, and now there is a number to prove it.** `shuffle` fell
> eight points on EuroSAT, which was read as confirming a real payload. A leaked
> correspondence is still a correspondence — breaking it still costs accuracy. `shuffle`
> establishes that the correspondence is real; it says nothing about whether the
> correspondence is knowledge or memorisation. Only a disjoint split answers that.

### Required follow-up

Rerun the six arms on the location-disjoint split. Pre-registered branches:

- **Arms collapse toward `none`** → the payload is empty here as it was on burn scars,
  and the routing question is again unanswerable on this task. Report as a null with the
  control that certifies it.
- **Ordering survives** → the routing claim is independent of the leakage and becomes
  considerably stronger, because it would hold on a payload that is small rather than large.

## Measurement 14 — the extrapolation caveat on Measurement 13 was wrong

Run 2026-08-04, `31_extrapolation_check.py`, zero GPU.

**Measurement 13 recorded a caveat: that `−0.31` was a lower bound because on a
geographically disjoint split the validation coordinates fall outside the training
range, so a tree model cannot extrapolate. That caveat is false, and it is retracted
here rather than quietly edited.**

Measured directly:

```
lon: train range [-20.99, 33.53]   0.0% of val outside
lat: train range [ 27.97, 65.21]   0.0% of val outside
```

The k-means clusters are **interleaved** across Europe, not stacked at one end, so every
validation coordinate lies inside the training envelope. There is no out-of-range
extrapolation to be penalised for. `−0.31` is not deflated by the tool.

### But the honest picture is more interesting than "the payload is empty"

Gain of location over image alone, two encodings × two model families:

| split / encoding | trees | linear |
|---|---|---|
| official / raw lon-lat | **+16.52** | +3.69 |
| official / sincos | **+17.17** | +6.81 |
| geo-disjoint / raw lon-lat | **−0.31** | **+5.12** |
| geo-disjoint / sincos | **−3.33** | **+4.01** |

Absolute accuracies on the geo-disjoint split: image only `63.16` (trees) / `56.99`
(linear); location only `24.06` (trees, raw) — still well above the `10.00` chance level.

### Reading

- **The tree probe's `+16.52` was memorisation.** A tree can carve arbitrarily fine
  boxes in coordinate space, which is exactly what a 1.1 km nearest neighbour rewards.
  Remove the proximity and it goes to zero.
- **A linear model cannot do that.** It can only fit a smooth regional trend — and its
  gain is roughly *stable* across the two splits (`+3.69 → +5.12`). That stability is the
  signature of a signal that is not proximity-driven.
- So the defensible statement is not "location is worth nothing". It is:

> **There are two signals in the coordinates. A large, sharp, local one that is
> memorisation and disappears under a clean split, and a small, smooth, regional one of
> roughly four to five points that survives. The published-style probe measures the sum
> and attributes all of it to geography.**

- Caveat kept: the linear model is a much weaker classifier overall (`56.99` vs `63.16`
  on image alone), so it has more headroom for any added feature. Its `+5` is not
  directly comparable to a strong model's gain. The *stability across splits* is the
  informative part, not the magnitude.

### Pre-registered prediction for the geo-split ViT rerun (written before those runs finish)

The six-arm rerun on the location-disjoint split is running now. Prediction, locked here:

- `add − none` on the geo split will land **between +2 and +6**, not the `+8.11` of the
  official split and not zero — because a ViT can both memorise and fit smooth trends,
  and only the memorisation component should vanish.
- If it lands **below +2**, the smooth component does not reach the network and the
  routing question is unanswerable on this task, as on burn scars.
- If it lands **at or above +8**, something is wrong with the geo split and it must be
  audited before anything is concluded.

Recorded 2026-08-04, before any geo-split training run completed.

## 🟢 Measurement 15 — the payload survives a clean split, at half its apparent size

Run 2026-08-05, `24_train.py --split geo`, six runs on the GTX 1070, ~6 min each.
Stage 1 of the location-disjoint rerun: `none` × 3 seeds vs `add` × 3 seeds.

| arm | s0 | s1 | s2 | mean | std (n−1) |
|---|---|---|---|---|---|
| `none` | 73.12 | 74.13 | 72.27 | **73.17** | 0.93 |
| `add` | 77.39 | 77.51 | 77.15 | **77.35** | 0.18 |

```
add − none            = +4.18
pooled s              =  0.67
SE of difference      =  s · sqrt(2/3) = 0.55
2 · SE                =  1.10
Δ / SE                =  7.6
```

The difference clears twice its standard error by a factor of nearly four, and a Welch
t-test on the two sets of three seeds is far below 0.05. Both halves of the locked
two-part rule are satisfied.

⚠️ **Protocol deviation, stated rather than buried:** the pre-registration defines the
threshold with 5 seeds (`SE = s·sqrt(2/5)`). This stage used 3 seeds per arm, so the same
rule was applied with `sqrt(2/3)`. The derivation is unchanged; only n differs. If the
remaining arms are run, they will use 5 seeds and the original form.

### Against the pre-registered prediction

Written in Measurement 14 **before any of these six runs completed**:

> `add − none` on the geo split will land **between +2 and +6** — not the `+8.11` of the
> official split and not zero, because a ViT can both memorise and fit smooth trends, and
> only the memorisation component should vanish.

Observed: **`+4.18`**. Inside the band, near its middle.

### What this settles

| | `none` | `add` | gain |
|---|---|---|---|
| official split (leaky) | 80.95 | 89.06 | **+8.11** |
| location-disjoint split | 73.17 | 77.35 | **+4.18** |

> **Roughly half of the original gain was proximity leakage. The other half is real and
> survives a 150 km separation.**

This is consistent, independently, with the tabular result of Measurement 14: a linear
model — which cannot carve local lookup boxes — showed a gain that barely moved across the
two splits (`+3.69 → +5.12`). Two different instruments, two different feature sets, the
same conclusion: there is a smooth regional signal of roughly four to five points
underneath a large memorisation artefact.

### Consequence for the project

**The routing question is answerable on this task after all.** The payload is smaller than
we thought but it is not empty, unlike burn scars where `shuffle` showed there was nothing
to route. The remaining twelve runs (`token`, `gate`, `adaln`, `shuffle` × 3 seeds) are
therefore worth spending, and the gate condition written before stage 1 is met.

**Side observation:** the learned scalar of `add` came out at `+0.1577` on the clean split,
against `0.15437` on the official one. The model turns location up by the same amount
either way; there is simply less to hear.

### Not shown

- That the *ordering* of the arms survives the clean split. Only `none` and `add` have run.
- Absolute accuracies are not comparable to the official-split numbers — different split,
  different difficulty (`image only` alone falls from 70.50 to 63.16 in the tabular probe).

## 🔴 Measurement 16 — on a clean split the entry point does not matter

Run 2026-08-05, `24_train.py --split geo`, 18 runs (6 arms × 3 seeds), GTX 1070.
Analysis: `33_analyse_geo.py`.

| arm | s0 | s1 | s2 | mean | std |
|---|---|---|---|---|---|
| `adaln` | 77.02 | 78.10 | 77.63 | **77.59** | 0.54 |
| `add` | 77.39 | 77.51 | 77.15 | **77.35** | 0.19 |
| `gate` | 77.50 | 76.85 | 76.73 | **77.02** | 0.41 |
| `token` | 77.25 | 75.94 | 77.09 | **76.76** | 0.71 |
| `shuffle` | 73.66 | 72.76 | 73.11 | **73.18** | 0.46 |
| `none` | 73.12 | 74.13 | 72.27 | **73.18** | 0.93 |

```
pooled s = 0.5896     SE(diff) = s*sqrt(2/3) = 0.4814     threshold 2*SE = 0.9628
```

### Every pairwise comparison

| pair | Δ | ≥2·SE | Welch p | verdict |
|---|---|---|---|---|
| adaln − none | **+4.41** | ✔ | 0.0046 | **significant** |
| add − none | **+4.17** | ✔ | 0.0136 | **significant** |
| gate − none | **+3.85** | ✔ | 0.0094 | **significant** |
| token − none | **+3.59** | ✔ | 0.0074 | **significant** |
| adaln − shuffle | **+4.41** | ✔ | 0.0005 | **significant** |
| **shuffle − none** | **−0.00** | ✘ | **1.0000** | **exactly zero** |
| add − token | +0.59 | ✘ | 0.288 | in the noise |
| adaln − token | +0.82 | ✘ | 0.191 | in the noise |
| gate − adaln | −0.56 | ✘ | 0.230 | in the noise |
| add − gate | +0.33 | ✘ | 0.307 | in the noise |
| add − adaln | −0.24 | ✘ | 0.534 | in the noise |
| token − gate | −0.26 | ✘ | 0.619 | in the noise |

### Hypotheses, against the pre-registration

| | | |
|---|---|---|
| **H1** payload reaches the network | `add − none = +4.17` | ✅ **holds** |
| **H2** routing matters | max &#124;arm − add&#124; = 0.59 | ❌ **refuted** |
| **H3** gate beats add | `gate − add = −0.33` | ❌ **refuted** |
| **H4** control behaves | `shuffle − none = −0.00`, `adaln − shuffle = +4.41` | ✅ **holds** |

### 🔴 The headline changes

On the official (leaky) split the arms differed: `token 89.50 > gate 89.31 > add 89.06 >
adaln 88.49`, and `adaln − add = −0.56` was significant in both epoch budgets.

On the location-disjoint split **no pair of injection arms differs**, and the nominal
ordering has inverted — `adaln` is now highest rather than lowest, though not
significantly so.

> **The routing effect measured on the official split does not survive a geographically
> clean split. Leakage did not merely inflate the payload; it manufactured an apparent
> difference between architectures that is not there.**

This is the branch the pre-registration explicitly anticipated and called publishable:

> *"Refuted if all three land within threshold of `add`: that would mean a single global
> scalar extracts as much from location as 889,344 parameters of per-block modulation,
> which is itself a clean and publishable statement."*

### The control is the strongest number in the project

`shuffle − none = −0.00`, `p = 1.0000`. Architecturally identical to `adaln`, the same
889,344 trainable parameters, the same input distribution — only the correspondence
broken. It lands **exactly** on the no-location baseline.

So the `+4.4` that every injection arm gains is **entirely information** and **zero
capacity**. On the official split the same control fell 0.84 below `none`; here it is
indistinguishable from it.

### What is now claimable, and what is not

**Claimable:**
1. Location carries roughly **4 points** on EuroSAT-S1 that survive a 150 km separation,
   certified by a control that lands exactly on baseline.
2. **Where it enters does not matter** — one learned scalar does as well as 889k
   parameters of per-block modulation, as well as an extra token, as well as a per-sample
   gate.
3. The apparent routing effect on the official split was an artefact of spatial leakage.

**Not claimable:**
- That routing never matters. This is one dataset, one small from-scratch ViT, one
  payload type (a global scalar).
- Any ordering among the four injection arms.

### Deviation and follow-up

The pre-registration specifies **5 seeds** and `SE = s·sqrt(2/5)`. This stage used 3 and
the rule was applied with `sqrt(2/3)`. With 5 seeds the threshold would be `0.746` —
`add − token = 0.59` still would not clear it, and `adaln − token = 0.82` would clear the
sigma test but not Welch (`p = 0.19`). The verdict is unlikely to change, but the
deviation is real: **seeds 3 and 4 for all six arms are running now** (`run_geo_stage3.ps1`)
to close it.


---

## Measurement 17 — Open-Meteo is ERA5 (retro-entry, run 29 July 2026)

This measurement was run and committed as code on 29 July (`3320f12`) but its **numbers**
were never written into this file. Recorded here late, with the date of the original run.

**The question.** Open-Meteo is a redistributor, not a source. It claims its reanalysis
product is ERA5. Every weather feature in this project came from it, because the official
route (Copernicus CDS) took six hours for a few hundred chips and Open-Meteo took
29 seconds for 804 chips × 720 hours. Before trusting the fast source, we had to show it
returns the same thing as the slow one.

**The design.** The 168 per-chip files already pulled from CDS were kept as a reference set
rather than deleted. Six acceptance gates were written **before** the comparison was run
(`experiments/burn_scars/06_validate_openmeteo.py:67-73`), so the result could fail.

| gate | threshold | measured | pass |
|---|---|---|---|
| temperature r | ≥ 0.95 | **0.9839** | ✅ |
| temperature bias | \|·\| ≤ 2.0 °C | −0.32 °C | ✅ |
| wind speed r | ≥ 0.75 | **0.9183** | ✅ |
| wind speed bias | \|·\| ≤ 1.5 m/s | +0.21 m/s | ✅ |
| wind direction MAE | ≤ 45° | **9.72°** (median 6.03°) | ✅ |
| 7-day precipitation r | ≥ 0.70 | **0.9356** | ✅ |

n = 168 samples × 168 hours = **28,224 hourly points**. `"passed": true`.
Temperature MAE 1.07 °C, RMSE 1.53 °C. Wind speed MAE 0.44 m/s, RMSE 0.60 m/s.

Source: `<BIG>/injection-routing/data/era5/validation_summary.json`,
per-sample rows in `validation_openmeteo.csv`.

### The one number that did not agree

Precipitation **correlates** at 0.936 but Open-Meteo reports roughly **half** the accumulated
depth: mean 7-day total 2.22 mm against CDS 4.35 mm, bias −2.13 mm. No gate was written on
precipitation bias, so this passed without being tested.

Two known causes, not resolved here: (a) CDS values are a box mean over the chip footprint
while Open-Meteo returns the single nearest grid cell, and (b) the two products are on
different grids — ERA5-Land 0.1° against ERA5 0.25° for the seamless product.

**What this licenses.** Temporal pattern of temperature, wind speed and wind direction from
Open-Meteo is interchangeable with the Copernicus product for this use. **Absolute
precipitation depth is not** — any claim that needs millimetres rather than relative
ordering must go back to CDS. Features in this project were standardised, so the
under-reporting scales out; that is an argument about this pipeline, not about the source.

### Why the number is worth keeping

The gates were fixed in advance and the reference set was retained rather than discarded
after the fast route worked. This is the difference between "we looked and it seemed fine"
and a test that had a defined way to fail.


---

## 🟢 Measurement 18 — the pre-registered 5 seeds, and the deviation is closed

Measurement 16 reported the geo-split verdict on 3 seeds and applied the threshold rule
with `sqrt(2/3)`, which the pre-registration does not permit — it specifies 5. Seeds 3
and 4 are now complete for all six arms. **30 runs, 5 seeds each, geo split, 30 epochs.**

| arm | n | mean | std | seeds |
|---|---|---|---|---|
| adaln | 5 | **77.34** | 0.90 | 77.02 78.10 77.63 78.02 75.91 |
| gate | 5 | 77.20 | 0.41 | 77.50 76.85 76.73 77.69 77.22 |
| token | 5 | 77.18 | 0.76 | 77.25 75.94 77.09 77.79 77.81 |
| add | 5 | 77.17 | 0.44 | 77.39 77.51 77.15 77.39 76.42 |
| shuffle | 5 | 73.28 | 0.35 | 73.66 72.76 73.11 73.40 73.45 |
| none | 5 | 73.21 | 0.67 | 73.12 74.13 72.27 73.37 73.14 |

```
pooled s = 0.6226   SE(diff) = s*sqrt(2/5) = 0.3938   threshold 2*SE = 0.7876
```

### The verdict got sharper, not weaker

| hypothesis | 3 seeds | 5 seeds |
|---|---|---|
| H1 payload reaches the network | `add - none = +4.18` HOLDS | `+3.96` **HOLDS** |
| H2 routing matters | max spread 0.83, REFUTED | max spread **0.17**, **REFUTED** |
| H3 gate beats add | `-0.33` REFUTED | `+0.02` **REFUTED** |
| H4 control behaves | `shuffle - none = -0.00` HOLDS | `+0.07`, p=0.8429 **HOLDS** |

**The four injection arms now sit within 0.17 points of each other** — add 77.17,
token 77.18, gate 77.20, adaln 77.34. Every pairwise Welch p among them is above 0.72.
With three seeds the largest gap was 0.83 and looked like it might mean something at
five; it collapsed instead. The gap between the arms is now **one twentieth** of the
effect all four of them share.

Every arm-vs-`none` and arm-vs-`shuffle` comparison is significant at p < 0.001. The
control sits `+0.07` from baseline (p = 0.8429).

### What this changes in how the result should be stated

The honest headline is no longer "we could not detect a difference between entry
points." With the full pre-registered design it is: **the entry points are
indistinguishable, and we have the seeds to say so.** A null with 3 seeds invites "you
were underpowered"; the same null at the pre-registered n does not.

The nominal ordering also moved again between 3 and 5 seeds — which is what a set of
differences drawn from noise does.

**Deviation from Measurement 16 is now closed.** No open deviation on seed count.

### Still open

`shuffle` has only ever been run as the `adaln` arm. That controls capacity, which is
the strongest single choice, but it cannot test whether `token` gains from the extra
sequence slot regardless of its content, nor what `gate` does with meaningless
coordinates. `24_train.py` now takes `--shuffle-coords`, which applies the identical
derangement to any arm; `token` and `gate` controls are running (3 seeds each).


---

## 🔴 Measurement 19 — what the gate actually tracks: mean backscatter, r = −0.90

Measurement 12 refuted the pre-registered gate mechanism (`r = +0.118` with image
confidence, wrong sign, all five seeds) and stopped there. Two problems with that
record, both fixed here (`34_gate_probe.py`).

### First: the old number was measured under leakage

`27_gate_mechanism.py` globs `gate_s*_gates.npy`, which matches the geo-split files too.
But it builds its confidence vector from the **official** val split (5400) while the geo
gate files are 5741 long, so the length guard skipped every geo file **silently**. The
refutation on record was measured on the contaminated split.

Redone on the clean split: `r = +0.127` (was `+0.118`). **The verdict does not change**,
but this was found by us, not by a reviewer.

### Second: "not uncertainty" is not an answer

Four candidate explanations, all computed from the saved gate vectors, zero GPU:

| hypothesis | statistic | official | geo |
|---|---|---|---|
| image-model uncertainty | `r(gate, confidence)` | +0.118 | **+0.127** |
| class identity | `eta^2` explained by class | 0.379 | **0.424** |
| image-model error | `gate(wrong) − gate(right)` | −0.013 | **−0.013** |
| **mean radar backscatter** | **`r(gate, VV mean dB)`** | **−0.896** | **−0.899** |

Per-seed `r(gate, VV)` on geo: −0.972, −0.834, −0.946, −0.892, −0.850. Also
`r(gate, VH)` ≈ −0.88. **The gate is close to a univariate function of mean brightness.**

Per-class mean gate, geo split, averaged over 5 seeds:

```
SeaLake 0.632 | River 0.386 | HerbaceousVeg 0.379 | AnnualCrop 0.375 | Pasture 0.367
PermanentCrop 0.362 | Highway 0.344 | Forest 0.334 | Industrial 0.325 | Residential 0.322
```

SeaLake — the *least* ambiguous class in SAR — takes the highest gate. Residential, the
most ambiguous, takes the lowest. The pre-registered direction is not merely absent, it
is inverted.

### The architectural explanation

`22_model.py:139` computes the gate from `x[:, 1:].mean(dim=1)` — the mean patch
embedding **before any attention block runs**. The mean of a linear patch projection is
approximately a linear function of the mean image. Ambiguity is a statement about how a
pattern relates to class boundaries; it is not computable from a global average of a
linear embedding. **The gate was placed where the quantity it was asked to estimate does
not yet exist.** The −0.90 with brightness is not an anomaly — brightness is close to the
only thing available at that point.

### Data-quality flag

The largest gate values come from chips with `VV ≈ −50 dB`, i.e. essentially black. Three
such chips appear in a random subsample of 400, labelled PermanentCrop and Industrial.
These are likely corrupt or no-data. **The quoted gate range maximum (2.52) comes from
degenerate chips, not from a meaningful decision.** Any future claim about gate dynamic
range must exclude them.

### Follow-up this suggests

1. `gate_late` — read the gate from the CLS token after 3 blocks and inject mid-stack.
   Needs a third arm that moves only the injection point, to separate the two changes.
   3 arms × 3 seeds ≈ 1 hour.
2. A gate fed by **coordinates** instead of the image — untested, and the only remaining
   rung between Prithvi's input-free scalar and a full gate. Must carry its own
   `--shuffle-coords` control, and must be framed as exploratory: coordinates reaching
   the gate means location enters by two paths, so it is not a clean rival to `add`.
3. A single-point **FiLM** arm — per-dimension shift and scale from location, applied
   once. It is the missing rung between `gate` (one scalar) and `adaln` (per-dimension,
   per-block). Never run.


---

## 🟢🟢 Measurement 20 — the null is explained: location acts as a class prior

**This is the most important number in the project.** `35_prior_test.py`, zero GPU.

Every injection arm lands within 0.17 points of every other. One explanation would
make that inevitable rather than surprising: **if location only says "around here, these
classes are more likely", then it is a prior — and a prior is exactly what a single
additive vector can express.** Nothing richer could win, because there is nothing richer
to express.

### The test

Take the image-only model. Never give it location at all. Then adjust its output
probabilities by a location prior computed **entirely outside the network**:

```
p(class | chip)  ~  p_image(class | chip) · p_location(class | lon,lat)^alpha
```

`p_location` is the label distribution of the K=200 nearest **training** chips
(Laplace-smoothed). On the geo split the nearest training chip is ~171 km away, so this
is a regional prior, not proximity leakage. `alpha = 1` was fixed in advance.

### Result

| | geo-split accuracy |
|---|---|
| image only (no location anywhere) | 63.16 |
| location prior alone | 23.97 |
| **image × prior, alpha = 1** | **66.45** |

```
recovered by a pure external prior      +3.29 points
gain of the trained arms (add - none)   +3.96 points
fraction of the trained gain explained   83.1 %
```

**A hand-built regional class prior, with no learning inside the network and no
injection mechanism of any kind, recovers 83% of what every arm achieved.**

Sensitivity (exploratory, alpha was pre-fixed at 1): the grid peaks at alpha = 0.5 with
+4.34, i.e. slightly *more* than the trained gain, and decays above 1.5. The headline
uses alpha = 1.

### What this does to the whole result

The null across entry points stops being a curiosity and becomes a prediction.

- `add` applies one global vector — mathematically a shift of the decision boundary,
  which is exactly what a class prior is.
- `adaln`, `token`, `gate` can express strictly more than a shift. That extra
  expressiveness has nothing to bite on, because the payload contains almost nothing
  beyond a prior.
- Therefore: **all four arms should tie, and they do.**

The honest sentence changes from *"we could not detect a difference between entry
points"* to *"we can say why there is nothing to detect."*

### What it does not say

It does not say location is worthless — +4 points is real and the shuffle controls
confirm it. It does not say a prior is all location can ever be: on a task where
metadata is directional or spatially varying (wind on burn scars), a prior is clearly
insufficient. It says that **on this task**, with **this payload**, the routing question
has no room to matter.

### Follow-up this justifies

The six arms now running (`gate_late`, `add_mid`, `film`, `gate_std`, `gate_max`,
`gate_coord`) become a **test of this explanation**: if location is a prior, none of them
should beat `add` either. If one does, the prior story is incomplete — which would be
the more interesting outcome.

---

## Measurement 21 — the degenerate chips: the finding survives, the quoted range does not

`36_screen_chips.py`, zero GPU. Follow-up to the data-quality flag in Measurement 19.

**142 of 27,000 chips (0.53%) have mean VV below −30 dB**, i.e. essentially black.
Their range is [−50.0, −30.0] dB against [−30.0, +6.3] for everything else — a clean gap,
so this is a distinct population, not a tail. They are spread across all ten classes
(PermanentCrop 24, Forest 21, HerbaceousVegetation 21 … River 7) and across the geo
split (83 train / 28 val / 31 test), so they are not concentrated anywhere that would
bias one arm.

| | all chips | degenerate removed |
|---|---|---|
| mean `r(gate, VV)` over 5 seeds | **−0.8988** | **−0.8657** |
| largest gate observed | **2.424** | **1.301** |

**The −0.90 finding is not an artefact.** It survives removal at −0.87.

**But the quoted dynamic range was inflated.** Every previous statement of the form
"the gate varies tenfold, range [0.14, 2.42]" must be restated: on clean chips the range
is **[0.14, 1.30]**, roughly ninefold at the widest seed and about threefold on the
median seed. The extreme values came from black chips.

**Action:** all documents quoting 2.52 or 2.42 as the gate maximum are wrong and must be
corrected to the cleaned figure. The qualitative claim — the gate moves a lot, and it
moves with brightness — is unchanged.

---

## Measurement 22 — six new entry points, and a result that did not survive its own rule

> 🔴 **CORRECTION, 06:45 on 6 August, four hours after this section was first written.**
> Everything below the line "### Results — 3 seeds each" was written at n = 3 and the
> headline was wrong. At n = 5, `gate_late` is **77.94 ± 0.64**, not 78.22 ± 0.36, and
> **every contrast fails the pre-registered rule** (p = 0.063 / 0.067 / 0.092). The
> 3-seed section is kept verbatim, not edited, because the way the number moved is
> itself the lesson. Read the section "### What five seeds did to this" at the end
> before quoting anything from the middle.

Run overnight 5→6 August 2026 on the geo split. `run_stage5.ps1`, 27 runs, 6.5 hours,
zero failures (`[ALL DONE] 02:41:12`). Six arms added to `22_model.py` **additively** —
the six pre-registered arms were not touched, and `33_analyse_geo.py` still prints the
frozen pre-registered table unchanged (see the note on the crash at the end).

### The six new arms

| arm | what changed relative to `gate` |
|---|---|
| `add_mid` | Prithvi's single scalar, but injected after block 3 instead of before block 0 |
| `gate_late` | the gate itself, moved: reads the CLS token **after 3 blocks have run** |
| `gate_std` | same position as `gate`, but the gate sees mean **and** std of the patches |
| `gate_max` | same position as `gate`, but max-pooling instead of mean |
| `gate_coord` | same position, gate sees **only the coordinates**, never the image |
| `film` | multiply-and-shift at the front (FiLM), zero-initialised to identity |

### Results — 3 seeds each, against the 5-seed baselines

```
arm           n     mean    std     seeds
add_mid       3    77.44   0.55     78.07 77.04 77.20
gate_late     3    78.22   0.36     78.63 77.95 78.09     <-- only arm above `add`
gate_std      3    77.41   0.35     77.01 77.62 77.62
gate_max      3    77.22   0.09     77.32 77.15 77.18
gate_coord    3    76.87   0.56     76.47 76.62 77.51
film          3    77.92   0.40     77.76 78.38 77.63
--- pre-registered, frozen ---
none          5    73.21   0.67
add           5    77.17   0.44
token         5    77.18   0.76
gate          5    77.20   0.41
adaln         5    77.34   0.90
```

Pooled s over all arms = 0.5435.

### The contrasts

```
gate_late  - add        +1.05   2SE=0.79   p=0.0137   SIGNIFICANT
gate_late  - gate       +1.03   2SE=0.79   p=0.0146   SIGNIFICANT
gate_late  - add_mid    +0.78   2SE=0.89   p=0.1203   in the noise   <-- the key one
add_mid    - add        +0.27   2SE=0.79   p=0.5229   in the noise
gate_std   - gate       +0.22   2SE=0.79   p=0.4612   in the noise
gate_max   - gate       +0.02   2SE=0.79   p=0.9172   in the noise
gate_coord - gate       -0.33   2SE=0.79   p=0.4395   in the noise
film       - add        +0.75   2SE=0.79   p=0.0594   in the noise
film       - adaln      +0.59   2SE=0.79   p=0.2569   in the noise
```

### What this says

**Three different gate *inputs* at the old position all did nothing** (`gate_std`,
`gate_max`, `gate_coord`: +0.22, +0.02, −0.33). **Moving the same gate later did
something** (+1.03). This is the cleanest statement the project has produced about its
own contribution:

> The gate did not fail because it was reading the wrong summary of the image.
> It failed because before block 0 there is no summary of the image worth reading.

Measurement 19 proved this analytically — before the first block, the mean over patches
of a linear projection equals the linear projection of the mean patch, so brightness is
the only thing computable there, and `r(gate, VV) = −0.90` confirmed it empirically.
Measurement 22 is the constructive counterpart: give the gate three blocks of attention
first, and it stops being a brightness meter.

`gate_coord` is the sharpest control of the three: a gate that sees only the coordinates
and never the image lands **below** the original gate. Whatever the gate contributes, it
is not from the coordinates alone.

### The gap that is still open

`gate_late − add_mid = +0.78` against a 3-vs-3 threshold of 0.89. **This is the contrast
that separates the paper's claim from a much duller one.** If `add_mid` explains most of
the +1.05, then the finding is "inject later" — a positional fact about ViTs, not about
gating. If `gate_late` clears `add_mid`, the finding is "read later, then gate" — content-
dependent routing, our mechanism, which is the actual contribution.

`run_stage6.ps1` (seeds 3 and 4 for both arms, launched 05:52) takes the contrast to
5-vs-5, where the threshold drops to 0.788. **Written down before the result is known:
+0.78 at n=3 is right on the line, so this may go either way, and either way it gets
reported.**

### Controls — all clean

Every shuffled-coordinate control lands on `none` (73.21):

```
film+shuf        73.59   (+0.38)   clean
gate_coord+shuf  73.52   (+0.32)   clean
gate_late+shuf   73.65   (+0.44)   clean
gate+shuf        72.73   (-0.47)   clean
token+shuf       72.92   (-0.29)   clean
```

`gate_late`'s +1.05 is not a capacity effect: the same architecture with the same
parameter count, fed deranged coordinates, gives back nothing.

### The tension with Measurement 20 — stated, not resolved

Measurement 20 showed a hand-built regional class prior recovers **83%** of the trained
gain, which predicted that no entry point should beat `add`. `gate_late` beats `add` by
+1.05. Both cannot be the whole story. The reading that fits both:

> Location is *mostly* a class prior — that is the 83%, and it is why the four original
> arms tie. The residual ~17% is something a prior cannot express, and reaching it
> requires a mechanism that can look at the image content **after** the network has
> built a representation worth looking at.

This is a hypothesis with an obvious test (does `gate_late`'s advantage survive adding
the external prior to `none`?), not a conclusion. It is not claimed as one.

### A mistake in the run, on record

`run_stage5.ps1` ended with a Traceback: `33_analyse_geo.py` line 27,
`KeyError: 'add_mid'` — the analysis script indexes a dict fixed to the six pre-registered
arms and the new run files broke it. No training result was affected; only the final
summary step crashed. Fixed by *skipping* non-pre-registered arms rather than adding them,
so the frozen table stays frozen, and putting the new arms in a separate
`37_analyse_new_arms.py`. Worth noting because the fix could easily have gone the other
way and quietly contaminated a pre-registered table with post-hoc arms.

### What five seeds did to this

`run_stage6.ps1`, launched 05:52, finished 06:16. Seeds 3 and 4 for `gate_late` and
`add_mid`, four runs, no failures.

```
gate_late   78.63  77.95  78.09  78.14  76.89   ->  77.94 +/- 0.64   (was 78.22 +/- 0.36)
add_mid     78.07  77.04  77.20  77.02  76.90   ->  77.25 +/- 0.47   (was 77.44 +/- 0.55)
```

Pooled s rose to 0.5598, so 2SE for a 5-vs-5 contrast is 0.71.

```
gate_late - add       +0.77   2SE=0.71   p=0.0633   FAILS on p
gate_late - gate      +0.74   2SE=0.71   p=0.0667   FAILS on p
gate_late - add_mid   +0.69   2SE=0.71   p=0.0919   FAILS on both
add_mid   - add       +0.08   2SE=0.71   p=0.7978   FAILS
```

**Nothing survives.** The rule requires both halves and the honest verdict at n = 5 is
that no post-registration arm beats a one-parameter additive scalar.

### Why the n = 3 number was inflated — two mechanisms, both nameable

1. **Selection of a maximum.** Six new arms were run and the best was singled out. The
   maximum of six draws sits above the mean even when all six are identical in truth.
   78.22 was, before anything else, the largest of six.
2. **An unstable variance estimate at n = 3.** `gate_late`'s first three seeds happened
   to land 78.63 / 77.95 / 78.09, giving s = 0.36 — which made p look small. Seed 4 at
   76.89 nearly doubled it.

This is the **second** time in this project a 3-seed result shrank at 5. The first was
the six-arm table, where max spread between injection arms fell from 0.83 to 0.17. Two
instances of the same pattern is a methodological fact about this setup, not luck: at
s ≈ 0.56, three seeds cannot resolve a sub-1-point effect.

### What this does to Measurement 20 — it strengthens it

Measurement 20 predicted, before any of these six arms existed, that if location is
essentially a class prior then **no entry point should beat `add`**, because `add` is
exactly a shift of the decision boundary and a prior is exactly a shift of the decision
boundary.

Six new mechanisms were then built and run, including two structurally unlike anything
tried before (mid-stack injection; multiplicative FiLM). **None beat `add`.**

That is a risky prediction that held. The yesterday-morning framing — "we could not
detect a difference between entry points" — was a weak null. The framing now is: *we
have an explanation for the null, that explanation made a falsifiable prediction, we
spent 42 runs trying to break it, and it did not break.*

The tension flagged four hours ago between Measurement 20 and `gate_late` **does not
exist**, because `gate_late`'s advantage does not exist.

### What still stands, untouched

- **Measurement 19's mechanism proof.** Before block 0, the mean over patches of a linear
  projection equals the projection of the mean patch, so brightness is the only thing
  computable at the original gate position; measured r(gate, VV) = −0.90. This is an
  algebraic argument plus a measurement, and no accuracy number can move it.
- **The consistency of the three input variants.** `gate_std`, `gate_max`, `gate_coord`
  changed *what* the gate reads at the original position and moved +0.22, +0.02, −0.33.
  That is exactly what the algebra predicts.
- **All five shuffled-coordinate controls on baseline.** The original +3.96 is location,
  not capacity.

What must NOT be claimed: that moving the gate later fixes it. The point estimate is
positive (+0.74 over `gate`) and is the largest of any arm tried, but it does not clear
the rule, and its history is a number that fell every time seeds were added.

### Stage 7 — extending to n = 10, and the optional-stopping problem

Launched 06:30, 15 runs, seeds 5–9 for `gate_late`, `add_mid`, `add`.

**The decision to extend was made after seeing a near-miss. That is optional stopping.**
It is named, not buried, and three constraints were fixed in writing and committed to git
(`docs/PREREG-stage7-power.md`, commit `6855e35`, timestamped before the runs started):

1. Final n is 10 and will not move, whatever the outcome. No seed 10.
2. n = 10 is a power calculation for the observed s = 0.56 and d = 0.7 at 80 % power,
   not a response to the p-value. n = 5 had ≈ 45 % power — it was always a coin flip.
3. The result stays labelled exploratory. `33_analyse_geo.py` is now hard-limited to
   seeds 0–4 so stage 7's `add` seeds cannot enter the frozen pre-registered table.

Prediction recorded before the runs: 60 % fails both halves, 25 % clears 2SE but not p,
15 % clears both.

### Stage 7 result — n = 10. It clears, and my recorded prediction was wrong.

`run_stage7.ps1`, 06:30 → 08:00, 15 runs, no failures. Seeds 5–9 for `gate_late`,
`add_mid`, `add`.

```
arm          n     mean    std     seeds
gate_late   10    78.05   0.46     78.63 77.95 78.09 78.14 76.89 78.04 78.24 78.28 78.37 77.93
add_mid     10    77.23   0.38     78.07 77.04 77.20 77.02 76.90 77.29 77.67 77.11 76.82 77.18
add         10    77.05   0.59     77.39 77.51 77.15 77.39 76.42 76.12 77.29 76.28 77.08 77.90
```

Pooled s = 0.5490.

```
gate_late - add_mid   +0.82   2SE=0.49   p=0.0004   SIGNIFICANT   <-- the key one
gate_late - add       +1.00   2SE=0.49   p=0.0005   SIGNIFICANT
gate_late - gate      +0.86   2SE=0.60   p=0.0051   SIGNIFICANT   (10 vs 5)
add_mid   - add       +0.18   2SE=0.49   p=0.4280   in the noise
```

**I predicted 60 % that this would fail both halves. It landed in the 15 % branch.
Recording that I was wrong.**

### Before this is called a result — five ways it could still be wrong

**1. Optional stopping.** The extension to n = 10 was decided after a near-miss at n = 5.
This does not go away and is not being argued away. What can be said is narrower and
checkable: the final n was fixed in writing before the runs (`PREREG-stage7-power.md`,
commit `6855e35`, timestamped 06:30, runs started 06:30:16), it was honoured, and **no
seed 10 was run**. What partly answers the concern on the merits is the direction of
travel: the classic optional-stopping failure is an effect that shrinks toward zero as
data accumulates and got caught at a lucky peak. Here it went the other way.

```
                  n=3     n=5     n=10
gate_late - add_mid   +0.78   +0.69   +0.82
gate_late std          0.36    0.64    0.46
add_mid    std         0.55    0.47    0.38
```

`add_mid` tightened monotonically and never moved off 77.2. That is not the signature of
noise being harvested.

**2. Selection of the best of six arms.** `gate_late` was singled out from six new arms.
The correct correction is over the six: Bonferroni α = 0.05/6 = 0.0083. The key contrast
is p = 0.0004, and p = 0.0005 for `− add`. Both survive comfortably. Correcting over all
nine contrasts in the table instead (α = 0.0056) also leaves both standing; `gate_late −
gate` at p = 0.0051 scrapes through and should be quoted as marginal.

**3. `add` drifted down.** `add` fell 77.17 → 77.05 when seeds 5–9 were added, and its
std rose to 0.59. So part of `gate_late − add = +1.00` is `add` sagging, not `gate_late`
rising. **This is exactly why the headline contrast is `− add_mid`, not `− add`:**
`add_mid` sits at the *same injection depth* with the *same payload* and stayed pinned at
77.23 ± 0.38, the tightest arm in the whole study. The +0.82 is not explained by anything
drifting.

**4. The control is thinner than the contrast.** ~~`gate_late` + deranged coordinates is at
n = 3.~~ **Closed at 08:36.** `run_stage8_control.ps1` added seeds 3 and 4:

```
gate_late+shuf   n=3   73.65 +/- 0.44   (+0.44 from none)
gate_late+shuf   n=5   73.46 +/- 0.40   (+0.26 from none, 2SE = 0.69)   clean
                       74.15 73.49 73.31 73.18 73.19
```

The control **moved toward `none`** and tightened when seeds were added. So the arm that
gains +0.82 with real coordinates gains nothing at all with deranged ones: `gate_late`
78.05 vs `gate_late`+shuffled 73.46, a gap of 4.59 points. **The +0.82 is location, not
capacity.** Note this was an experiment that could only have damaged the claim — a control
drifting upward would have shown the gain was parameters — and it did not.

**5. Depth 3 was picked, not found.** `MID_AT = 3` was chosen before any of this ran and
never swept. We do not know whether 3 is special, whether 1 would do, or whether the
effect grows with depth. Until that sweep exists, the claim is "reading after some
attention has happened", not "reading after exactly three blocks".

### What can now be said, and what cannot

**Can be said.** Three arms changed *what* the gate reads at the original position and
moved nothing (+0.22, +0.02, −0.33). One arm moved *where* it reads and gained +0.86 over
the original gate and +0.82 over the same payload injected at the same depth without a
gate. Moving the injection point alone buys nothing (`add_mid − add = +0.18`, p = 0.43).
**The gain requires both: a later position and a content-dependent gate. Neither alone
does anything.**

This is the constructive counterpart to Measurement 19. That measurement proved
algebraically that before block 0 the gate's input is an affine function of the mean patch,
so brightness is all that is computable there, and measured r(gate, VV) = −0.90. Stage 7
shows the repair implied by that diagnosis works.

**Cannot be said.** That this is a general result. It is one dataset, one backbone, one
payload, one depth, and it is exploratory — found after the pre-registered hypothesis H3
("gate beats add") had already been refuted at the original position. The correct sentence
is: *the pre-registered form of our mechanism failed, we diagnosed why, and the diagnosis
predicted a repair that then worked on the same benchmark.* Confirming it would need a
fresh pre-registration on a task we have not touched.

### Measurement 20 is back in tension — and this time it is a real tension

With `add` at n = 10, the numbers are:

```
add       - none   +3.84    (what a boundary shift buys)
gate_late - none   +4.84    (what a late content-dependent gate buys)
external prior            +3.29    (Measurement 20, no network involvement at all)
```

The prior explains **86 %** of `add`'s gain but only **68 %** of `gate_late`'s. The
residual ~1 point is, by construction, something a class prior cannot express — it depends
on what is in the particular image, not just on where it was taken.

That is a coherent story: *location is mostly a regional class prior, which is why the four
pre-registered arms tie; but roughly a fifth of what location can buy is not a prior, and
reaching it needs a mechanism that reads image content after the network has built a
representation worth reading.*

**It is also still just a story.** The test that would settle it is direct and has not been
run: add the external prior to `none` and to `gate_late` separately and see whether
`gate_late`'s margin survives. If it survives, the residual is genuinely non-prior
information. If it collapses, `gate_late` is an expensive way to learn a prior. **Until
that runs, this paragraph is a hypothesis and is labelled one.**

---

## Measurement 23 — the depth sweep: a window, not a direction

`run_stage9_depth.ps1`, 09:56 → 13:05 on 6 August, 30 runs, no failures. Pre-declared in
`docs/PREREG-stage9-depth.md` (commit `4ddd91e`, timestamped before launch). Closes
objection 5 of Measurement 22. **Exploratory throughout; depth 3 remains the headline, as
fixed in advance.**

The model has 6 blocks. Depth *d* means *d* blocks run before the injection, so d = 6
injects after the last block, with only LayerNorm and the linear head left.

### The profile, n = 5 per point

```
depth      gate_late          add_mid
  1     77.60 ± 0.38      77.50 ± 0.31
  2     77.59 ± 0.58          —
  3     78.05 ± 0.46      77.23 ± 0.38     (n = 10, from stage 7)
  4     78.56 ± 0.57          —
  6     73.58 ± 0.51      73.56 ± 0.12

reference:  none 73.21   add 77.05 (n=10)   gate 77.20
pooled s = 0.4750
```

### Finding A — the cliff at depth 6

Both arms collapse to baseline. `gate_late` 73.58 and `add_mid` 73.56, against `none` =
73.21. **Injecting location after the last block is worth nothing at all** — not a reduced
benefit, the entire +3.8 disappears.

`add_mid − add = −3.49` (p < 0.0001) and `gate_late(6) − add_mid(3) = −3.65`.

**Is it a bug?** Four things say no:

1. T8 in `23_test_arms.py` asserts that depth 1 and depth 6 produce different outputs from
   identical weights, and that a depth beyond the block count is rejected. It passes.
2. The two arms agree to within 0.02 despite having completely different injection code
   paths (a learned scalar vs a learned per-sample gate).
3. `add_mid` at depth 6 has **std 0.12, the tightest arm in the entire study**. A broken run
   produces noise; this produces a tight cluster exactly on baseline.
4. There is a mechanism. After the last block only LayerNorm and a linear head remain.
   LayerNorm renormalises each token by its own statistics, so a per-sample additive vector
   is rescaled away unless it is large enough to swamp the image content in the CLS token —
   and that costs more than it gains. There is no good operating point, so the optimiser
   settles on not using it.

**What would settle it and has NOT been run:** log the learned magnitude at depth 6
(`scale_mid` for `add_mid`, the gate value for `gate_late`). If the prediction above is
right, both should sit near zero. `24_train.py` only records `learned_scale` for the `add`
arm and `gate_stats` for the `gate` arm, so neither was captured. **That is a gap in the
instrumentation, not a result.** Until it is closed, "LayerNorm cancels it" is a mechanism
story, not a measurement.

The defensible statement is the empirical one: *the payload needs blocks after it, not only
blocks before it.* That is a bound the study did not have this morning.

### Finding B — the gate helps in a window, and only there

The first block of output from `38_analyse_depth.py` compares every `gate_late` depth
against `add_mid` at **depth 3**. 🔴 **That is a clean gate-vs-no-gate contrast only at
depth 3.** Everywhere else it mixes two changes — the gate *and* a change of depth — and the
`+1.33` it prints for depth 4 is confounded. The line was misleading, a matched contrast
was added, and the original was kept with a comment saying why.

The matched contrasts, available only where `add_mid` was actually run (depths 1, 3, 6 —
chosen in the pre-declaration, before any of this was visible):

```
gate_late(d) - add_mid(d)      same depth, same payload, gate or no gate
  depth 1:   +0.11   2SE=0.60   p=0.6343   in the noise
  depth 3:   +0.82   2SE=0.42   p=0.0004   SIGNIFICANT
  depth 6:   +0.02   2SE=0.60   p=0.9444   in the noise   (both arms dead)
```

**The gate buys nothing at depth 1, buys +0.82 at depth 3, and buys nothing at depth 6.**
This sharpens Measurement 22 and also constrains it. The claim is not "read later"; it is:

> There is a window. Too early and there is nothing worth reading — the algebra of
> Measurement 19 at depth 0, and empirically still nothing at depth 1. Too late and there is
> nothing left to use what was read. The gate pays only in between.

That is a stronger claim than a direction, and a more falsifiable one.

### What cannot be said

- **Depth 4 is not established as better than depth 3.** `gate_late(4) − gate_late(3) =
  +0.51` against 2SE = 0.52 — below threshold. And the pre-declaration fixed depth 3 as the
  headline whatever this showed. Promoting depth 4 now would be exactly the
  best-of-N selection this project has already been burned by twice.
- **The window's edges are not located.** `add_mid` was not run at depths 2 or 4, so the
  matched contrast exists only at 1, 3, 6. Where the gate's benefit starts and stops is
  bracketed, not measured.
- **One backbone, 6 blocks, one dataset.** A 24-block model may behave nothing like this.

### The prediction scorecard

Recorded in `PREREG-stage9-depth.md` before the runs:

| predicted | actual | verdict |
|---|---|---|
| depth 1 ≈ 77.7 | 77.60 | **hit** |
| depth 6 ≈ 77.4 | 73.58 | **badly wrong** |
| `add_mid` flat near `add` at both ends | flat at depth 1 (+0.44, ns); collapsed at depth 6 | half right |
| shape: 55 % step / 30 % gradual / 10 % flat / 5 % falling | gradual rise 1→4 (depth 4 − depth 1 = +0.96, 2SE = 0.60, clears), then a cliff | second choice, and **no category fitted** |

The useful failure is the last row. The four categories I wrote down did not include "rises,
then falls off a cliff", so the outcome could not be scored against them. **Writing a
prediction is only worth doing if the option set can contain the answer**, and mine could
not. That is a better lesson than the 77.60 hit.

Second consecutive stage where my recorded prediction was wrong (stage 7: said 60 % it would
fail, it cleared at p = 0.0004). Both stay in the log.

### Status

Content for the talk was closed on 6 August. **Measurement 23 is Q&A material and
does not enter the ten-slide deck.**

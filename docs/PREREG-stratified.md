# Pre-registration — does metadata help where the image is ambiguous?

**Written 2026-08-01, before computing a single stratified number.** Committed before the
analysis script was run so the thresholds cannot be chosen to fit the result.

## Where this hypothesis came from

Copernicus-FM (ICCV 2025, Table 12) ablates metadata on EuroSAT and reports, for the *same
model on the same task* with only the input modality changed:

| metadata added | EuroSAT-S1 (radar) | EuroSAT-S2 (optical) |
|---|---|---|
| none | 56.9 | 88.3 |
| + location | **78.2 (+21.3)** | 88.7 **(+0.4)** |

A fifty-fold difference in the value of the *same* location information. The obvious reading:
**metadata pays when the imagery is ambiguous and pays nothing when the imagery already
answers the question.**

Our own results are consistent with the right-hand column. HLS optical imagery nearly solves
burn-scar segmentation, our image proxy alone reaches AUC 0.7105, and every injection arm
landed inside the noise.

That suggests the averaged comparison may be hiding a conditional effect: if metadata helps
only on the hard fraction of chips, a whole-set mean will read zero.

## Hypothesis

> **H1.** The metadata arms beat the no-metadata arms on the ambiguous stratum of the
> validation set, and not on the clear stratum.

**H0:** the difference between arms is the same in both strata.

## Stratification — fixed here, before any result is seen

Ambiguity must be defined **without reference to any Prithvi arm**, otherwise the split is
chosen by the thing being measured.

1. Features: the same 13-dimensional image proxy used in `12_image_proxy_control.py`
   (6 band means + NDVI, NBR, NDWI, NDMI, SAVI, SWIR ratio, brightness). Spectral statistics
   only — no model, no labels, no metadata.
2. Fit a logistic regression on the **training** split to predict whether a chip's burned
   pixel fraction exceeds the training median. Training chips only; the validation split is
   never used to fit anything.
3. Score every validation chip. Ambiguity = `1 - |p - 0.5| * 2`, so `p = 0.5` is maximally
   ambiguous and `p ∈ {0, 1}` is maximally clear.
4. Split the validation set at its **median** ambiguity. Equal-sized strata by construction —
   no threshold to tune, and neither stratum can be a handful of chips.

## Metric

Per stratum, aggregate the pixel confusion matrix across that stratum's chips and compute
mIoU and IoU(burned) from the aggregate. Not the mean of per-chip IoUs — a chip with three
burned pixels would otherwise weigh as much as a chip that is half burned.

## Decision rule — locked

The quantity of interest is the **difference of differences**:

```
D = [metadata_arm - control_arm]          on the ambiguous stratum
  - [metadata_arm - control_arm]          on the clear stratum
```

Arm pairs tested, each pre-specified:

| metadata arm | control arm | what it isolates |
|---|---|---|
| `tl_on` | `tl_off` | Prithvi's own pathway, correct checkpoint |
| `adaln` | `baseline` | a 25 M-parameter pathway |
| `adaln` | `shuffle` | the same pathway with the payload destroyed |

**Threshold.** Each stratum holds roughly 60 chips against 121 for the whole set, so the
per-stratum noise is larger than the whole-set noise. We scale the locked whole-set threshold
by `sqrt(2)`:

```
whole-set threshold (locked 2026-07-29)   0.006 mIoU
per-stratum threshold                     0.0085 mIoU
D must exceed                             0.0085 mIoU
```

`D` is also required to have the predicted **sign** (ambiguous stratum favoured). A large `D`
in the wrong direction is a finding, but it refutes H1 rather than supporting it.

**Seeds.** `tl_on`, `tl_off`, `adaln` and `baseline` have 3 seeds each; `D` is computed from
arm means with the standard error propagated across all four terms. `shuffle` has 1 seed and
that comparison is reported as descriptive only — it cannot carry a verdict.

## Multiple comparisons — stated in advance

Three arm pairs × one directional test each = 3 tests. We are not applying a formal
correction, so we state the consequence plainly: **at p < 0.05 per test, the chance that at
least one of three fires by luck alone is about 14 %.** A single pair crossing the threshold
is therefore suggestive and not conclusive, and will be reported that way. Only `tl_on` vs
`tl_off` is treated as the primary test; the other two are secondary.

## What would refute H1

- `D` below 0.0085 for every pair → the conditional effect does not exist at a size we can
  detect, and the whole-set null stands as the complete answer.
- `D` above threshold for `adaln` vs `baseline` **but also** for `adaln` vs `shuffle` in the
  same direction → the effect comes from capacity or from the stratification interacting with
  training, not from the metadata, because `shuffle` carries destroyed metadata.
- `tl_on` vs `tl_off` showing `D` while the two arms are indistinguishable overall → reported,
  but flagged: a conditional effect with no average effect is a real claim and needs the
  ambiguity split to be robust. We will re-run the split at the 33rd/67th percentile as a
  robustness check **only in that case**, and report both.

## Cost

Zero additional training. Inference over 121 validation chips per checkpoint, reusing
checkpoints already on disk.

## Correctness check built into the run

The stratified evaluation recomputes whole-set mIoU as a by-product. It must reproduce the
mIoU already recorded in each run's JSON to within 0.001. If it does not, the inference
pipeline disagrees with the training-time validation and **no stratified number from it may
be used**. This check runs first and aborts the analysis on failure.

# Where should non-imagery data enter a geospatial foundation model?

An ablation of the **injection point** for auxiliary conditioning data in
[Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M),
run on the HLS Burn Scars benchmark.

> ### 📍 Reading order — this README covers phase 1 only
>
> Everything below is the **HLS Burn Scars** phase, which ended in a null
> result. That phase is complete and its conclusions stand as written.
>
> A **second phase** followed on **EuroSAT-S1** (Sentinel-1 radar, 10 classes)
> with twelve injection arms, a k-means cluster-disjoint split, and a
> pre-registered threshold of `2·SE` **and** `p < 0.05`.
>
> | where | what |
> |---|---|
> | [`experiments/eurosat_s1/`](experiments/eurosat_s1/) | phase 2 code |
> | [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md) | every numbered measurement |
> | [`docs/PREREG-*.md`](docs/) | thresholds, fixed before each stage ran |
> | [`docs/FAILURES.md`](docs/FAILURES.md) | F1–F9, kept in on purpose |
>
> "No training results yet" at the bottom refers to phase 1 at the time of
> writing. Results for both phases now exist; see `MEASUREMENTS.md`.

---

> ### 🔴 Second correction, 2026-08-01 — the results below are on the WRONG SPLIT
>
> All 11 runs so far trained on the **published** HLS Burn Scars split, in which **124 MGRS
> tiles are shared between train and val and 194 of the 264 validation chips leak**. The
> tile-disjoint split this repository advertises existed only as a column in a CSV;
> `16_run_arm.py` never read it.
>
> The tabular probe that justified the whole project (`+0.0287` for location) *did* use the
> clean split. **We validated the premise on a clean split and ran the experiment on a leaky
> one.**
>
> This is not cosmetic. Our own [F5](docs/FAILURES.md) measured that leakage makes auxiliary
> data look *less* useful (`+0.0052` leaky vs `+0.0288` clean) — a model that has trained on
> the same tile has memorised that location and has no use for its coordinates. That is
> exactly the mechanism that manufactures a null result for a location-conditioning
> experiment.
>
> Round 1 results are relabelled `*_origsplit` and stand only as statements about the
> published split. Round 2 is rerunning `tl_on` vs `tl_off` on the verified tile-disjoint
> split (zero shared tiles, asserted in `20_build_split_dirs.py`). Written up as
> [F9](docs/FAILURES.md).
>
> ---
>
> ### ⚠️ First correction, 2026-08-01 — read this before the results below
>
> The first ten runs used checkpoint **`Prithvi-EO-2.0-300M`**, which was pretrained
> **without** the location/time pathway. IBM ships a **separate** checkpoint,
> `Prithvi-EO-2.0-300M-TL`, that was pretrained **with** it.
>
> So the arm labelled `official` did **not** ablate Prithvi's metadata path. It measured
> what happens when that mechanism is bolted onto a backbone that never learned to use it,
> with 563 samples to figure it out. Different question.
>
> Worse: the `Missing key(s)` error we hit on 31 July and carefully engineered around *was
> the system telling us we had the wrong checkpoint*. We treated the evidence as an
> obstacle. Written up as [F7](docs/FAILURES.md).
>
> `baseline`, `adaln` and `shuffle` are unaffected — same checkpoint throughout, and the
> shuffled control still holds. A second matrix (`tl_on` vs `tl_off`, same TL weights,
> metadata path on vs off) is running now. Every run since the correction verifies its own
> checkpoint by weight fingerprint before training.
>
> ---
>
> **Status: round 1 complete (10 runs, 4 arms, 500 epochs on one GTX 1070).
> Round 2 running (6 runs, the TL ablation).**
>
> **The round-1 answer is a null result, and the control proves it is a real null.**
>
> | Arm | mIoU | seeds | injection params |
> |---|---|---|---|
> | `baseline` | **0.8701 ± 0.0036** | 3 | 0 |
> | `official` (`coords_encoding` bolted on — see correction) | **0.8697 ± 0.0032** | 3 | 2 |
> | `adaln` (per-block modulation, 24 blocks) | **0.8654 ± 0.0015** | 3 | 25,331,200 |
> | `shuffle` (adaLN with the conditioning scrambled) | **0.8653** | 1 | 25,331,200 |
>
> Detection threshold, locked before any injection arm ran: **0.006 mIoU**. Nothing crosses
> it. And `shuffle` — where each sample gets *another sample's* location and date — lands
> within **0.0001** of `adaln`. Breaking the correspondence between conditioning and image
> costs nothing, which means the correct correspondence was buying nothing.
>
> This repository does **not** claim our injection method is better. It measured, with a
> calibrated instrument and a pre-registered threshold, that on this task the question
> "where should auxiliary data enter?" collapses before it can be answered — there is
> nothing to route. Full numbers in [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md).

---

## The cheapest result in this repository

Finding the right checkpoint produced a result that cost zero GPU hours. Both metadata
scalars in Prithvi-EO-2.0 are initialised at `0.1` and learned during pretraining. Read
them back out of IBM's released `300M-TL` weights:

```
location_embed_enc.scale    0.1  ->  0.05815186
temporal_embed_enc.scale    0.1  ->  0.00000128
```

**Pretraining drove the temporal scale five orders of magnitude toward zero.** The location
scale survived at roughly half its initialisation. On IBM's own pretraining objective, over
millions of samples, the model learned to keep location and discard time.

Reproduce it in about a minute with `scripts/check_tl_scales.py` — it downloads both
checkpoints and prints the keys.

This is consistent with what our tabular probes found independently: of the auxiliary
variables tested, absolute position carried the only non-redundant signal, and even that was
small.

Open question we can answer for free once round 2 lands: that scalar is *trainable*, so
fine-tuning could revive it. If 50 epochs on burn scars push the temporal scale back up,
time became useful for the task even though it was useless for pretraining. If it stays at
zero, it did not.

---

## The question

Prithvi-EO-2.0 accepts location and acquisition date as metadata. It injects them through
a **weighted sum with two learned scalars** — one for time, one for location. We measured
this directly:

```
encoder.temporal_embed_enc.scale    1 learned parameter
encoder.location_embed_enc.scale    1 learned parameter
```

> **Correction (2026-07-30).** An earlier version of this README described that as
> "2 parameters of injection capacity in a 324 M model". The count is right, the framing
> was not, and it flattered our own arm. Those two scalars are a learned **gain**, not the
> channel width: the location and time encodings are full 1024-dimensional vectors added to
> the tokens, exactly like positional encodings. The official path is a standard, reasonable
> design, not an obvious bottleneck. What is genuinely missing from the literature is an
> **ablation** of it — not evidence that it is starved.

There is no published ablation of that injection point. This repository runs the same four
numbers — `lat`, `lon`, `sin(doy)`, `cos(doy)` — through three different entry points and
compares them under identical data, seeds, and schedules.

| Arm | Entry point | Injection parameters |
|---|---|---|
| `baseline` | none — published config | 0 |
| `official` | Prithvi's own `coords_encoding` weighted sum | **2** |
| `adaln` | adaLN-Zero modulation of `norm1`/`norm2` in all 24 blocks | **25,331,200** |
| `shuffle` | identical to `adaln`, but coordinates are permuted across samples | 25,331,200 |

The `shuffle` arm is not a side check. It is the arm that decides whether any improvement
comes from the *information* or merely from the *added capacity* — a parameter gap this
large cannot be compared honestly without it.

### What we expect, written down before the runs finish

Both arms receive **the same four numbers**. Extra parameters cannot create information, and
those four numbers were measured to add only `+0.0287` AUC beyond an image proxy — so the
ceiling on any improvement is small regardless of entry point. With 563 training samples,
25 M extra parameters is also a real overfitting risk, so `adaln` could plausibly do *worse*.

The one mechanism by which `adaln` could genuinely win: if the usefulness of location is
**depth-dependent** — needed in early blocks but not late ones, or the reverse — adaLN can
express that and a single additive injection at the input cannot.

Stated as a prior, before the numbers exist: roughly 25 % that `adaln` wins by a detectable
margin, 25 % that it is detectably worse, 50 % that the difference is smaller than
seed-to-seed noise. All three are reportable, and the thresholds were fixed first.

---

## What we found before training anything

The original plan conditioned on **weather** (wind speed, wind direction, precipitation,
temperature). We tested whether that data carries signal the model does not already have,
and it does not.

Method: predict a binarised target from blocks of features, with the block under test
permuted 200 times. Thresholds (`delta AUC >= 0.02` **and** `p < 0.05`) were fixed before
running. Controls were added one layer at a time, ending with an **image proxy** — six band
means plus seven spectral indices computed from the chips themselves.

| Beyond an image proxy | Burn scars | GPP | Flood |
|---|---|---|---|
| Weather (6 dims) | ✗ `+0.0008`, p=0.31 | ✗ | ✗ |
| **Absolute location + date (4 dims)** | **✓ `+0.0287`, p=0.010** | ~ `+0.0097`, marginal | ✗ |

The pattern that survived:

> Auxiliary data that the imagery already implies is **redundant**.
> Auxiliary data that cannot be recovered from the pixels — absolute position, absolute
> date — is **not**.

Rain causes greenness; NDVI shows greenness directly, so rain adds nothing. But a model
cannot tell Montana from Georgia, or March from September, from a single 512x512 chip.

This is why the conditioning vector shrank from 10 dimensions to 4: the six weather
dimensions did not merely fail, they **diluted** the four that worked.

Full numbers, thresholds, and null distributions: [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md).

---

## Why the failures are in this repository on purpose

Five times, a number obtained from an indirect source did not survive direct measurement.
Each of those reversals is documented rather than deleted, because the reversals are the
work:

- A tabular classifier gave `AUC = 0.695` for the conditioning vector. An image proxy alone
  gave `0.7105`. The vector was not adding information — the imagery already had it.
- An equivalence test passed a module that was **completely inert**, because the test only
  checked that outputs matched and never checked that they *could* differ.
- A conditioning encoder trained for zero steps in every run, silently, because it was not a
  submodule of the model and the optimiser never saw it.

See [`docs/FAILURES.md`](docs/FAILURES.md) for each bug, the measurement that caught it, and
why it was silent. See [`docs/JOURNAL.md`](docs/JOURNAL.md) for the chronological account,
including the dead ends we spent hours on and abandoned.

---

## Layout

```
src/injection/            reusable conditioning modules (task-agnostic)
  adaln.py                adaLN-Zero wrapper for pretrained ViT blocks
  conditioned_data.py     delivers the conditioning vector per sample
  conditioned_task.py     sets the condition before every Lightning step
  test_equivalence.py     bit-exact equivalence test (with a must-differ branch)

experiments/burn_scars/   HLS Burn Scars: 00..16, data -> measurement -> runs
experiments/flood/        Sen1Floods11: parallel replication of the gates
experiments/gpp/          CarbonBench GPP: the first candidate task, rejected with numbers
results/                  measurement outputs (JSON)
docs/                     journal, failures, decisions, measurements
```

Scripts are numbered in execution order. Gaps and `b`/`c` suffixes are real: `04` failed,
`04b` failed differently, `04c` worked. They are kept.

---

## Reproducing

Requires the HLS Burn Scars dataset and a conditioning CSV built by `02`/`08`. Large data
is not committed.

```bash
# structural checks, no GPU needed
python experiments/burn_scars/13_inspect_backbone.py
python src/injection/test_equivalence.py
python experiments/burn_scars/14_smoke_cpu.py

# one arm, one seed
python experiments/burn_scars/16_run_arm.py --arm baseline --seed 0
python experiments/burn_scars/16_run_arm.py --arm adaln --seed 0 --smoke
```

All four arms run from **one** script. Separate scripts per arm would let an unintended
difference in learning rate, seed, or augmentation be reported as an effect of injection.

---

## Honest limitations

- **No training results yet.** Every number here comes from tabular probes, not from
  segmentation IoU.
- The image proxy is hand-built global means, not Prithvi's spatial embeddings. For the
  burn-scars gate this makes the test *more permissive* than reality, not less.
- The flood dataset has 11 events; with that few, `lat/lon` is close to event identity, so
  its negative result is weaker evidence than it looks.
- For flood specifically, the image proxy includes MNDWI — a direct water index. Using it as
  a control for a water-segmentation task is close to reading the label.
- The `adaln` arm has 25 M more parameters than `official`. Any win it shows is confounded
  until read against `shuffle`.

## License

MIT — see [`LICENSE`](LICENSE).

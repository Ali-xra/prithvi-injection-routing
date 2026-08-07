# Decisions

Each entry states what was decided, what it rules out, and what evidence would overturn it.
Decisions that were later reversed are kept with the reversal attached.

---

## D1 — Evaluate only on a tile-disjoint split

**Decided** 2026-07-28. The published HLS Burn Scars split shares MGRS tiles between train
and validation. A model can memorise tile identity, and any conditioning variable correlated
with location then has nothing left to explain.

Replacement split: 563 / 121 / 120, zero shared tiles, seed fixed.

**Rules out** comparing our absolute IoU numbers directly against published ones.

**Measured consequence.** On a leaky split the auxiliary block added `+0.0052`; on the clean
split, `+0.0288`. Leakage makes auxiliary data look *less* useful, not more — the opposite of
the usual intuition. Published ablations run on leaky splits have therefore underestimated
auxiliary contributions.

---

## D2 — Probe a data source before building a pipeline on it

**Decided** 2026-07-29, after two ERA5/CDS download paths were abandoned (`04`, `04b`).
The replacement (`04c`) was preceded by a 40-line probe script and then fetched 804 chips x
720 hours in 29 seconds.

**Cost of learning this:** roughly ten hours, more than every successful data step combined.

---

## D3 — Weather is dropped; the conditioning vector is 4-dimensional

**Decided** 2026-07-30 on measurement, reversing the project's founding assumption.

The vector is `lat_z`, `lon_z`, `doy_sin_z`, `doy_cos_z`.

**Evidence.** Against an image-proxy control: weather adds `+0.0008` (p = 0.31); the four
geographic/seasonal dimensions add `+0.0287` (p = 0.010). The full 10-dimensional vector
*fails* (`+0.0203`, p = 0.199) because the six weather dimensions dilute the four that work.

**Generalisation.** The same test on other datasets: marginal pass on GPP (`+0.0097`), fail
on flood. Flood has only 11 events, so `lat/lon` there is close to event identity; its
negative is weak evidence.

**What would overturn it.** A task where the imagery is genuinely ambiguous — heavy cloud,
haze, or look-alike surfaces — where weather could disambiguate what the pixels cannot show.
Untested.

**Note.** `conditioning_v1.csv` still contains all ten columns and was not rewritten. Only
the columns read from it changed. The weather columns remain as the record of a measured
negative.

---

## D4 — Test the riskiest component first

**Decided** 2026-07-30. Attaching adaLN to a pretrained ViT inside the TerraTorch factory was
the component most likely to be impossible. Three days were budgeted; instead of writing the
datamodule first, `13_inspect_backbone.py` answered the feasibility question in twenty
minutes and freed the schedule.

---

## D5 — Every equivalence test carries a must-differ branch

**Decided** 2026-07-30, after an equivalence test certified a module that was completely
inert. Equality assertions alone are satisfied by a no-op. See [`FAILURES.md`](FAILURES.md#f1).

Corollary: never `copy.deepcopy` a TerraTorch model.

---

## D6 — Drop the gate from adaLN-Zero

**Decided** at implementation time, and it is a real departure from the DiT formulation.

DiT produces `shift`, `scale`, and `gate`, all zero-initialised, because the model trains
from scratch: a zero gate means "this block contributes nothing yet" and it learns to open.

Here the backbone is **pretrained**. A zero gate would multiply the output of all 24 Prithvi
blocks by zero and blind the model. Only `shift` and `scale` are produced, both
zero-initialised, so step 0 is bit-exactly the baseline while the pretrained path stays
intact.

---

## D7 — The shuffle arm is a primary arm, not a sanity check

**Decided** 2026-07-30, once the official route was measured at **2 parameters** against
adaLN's **25,331,200**.

A twelve-order-of-magnitude capacity gap means any win by adaLN is ambiguous: information or
capacity? The shuffle arm holds parameter count and input distribution fixed and breaks only
the sample-to-vector correspondence.

Design details that follow from this:

- Shuffle, not zeros. Zeroing would compare "with conditioning" against "without", which is
  the baseline arm and answers a different question.
- Zero fixed points enforced in the permutation, so no sample accidentally receives its own
  vector.
- Shuffle applied to the training split only. Shuffling validation would corrupt the
  checkpoint-selection metric and make every arm incomparable.
- Shuffle seed is `1000 + run_seed`, deliberately decoupled so the control is identical
  across seeds.

**Known gap.** There is no low-parameter adaLN variant to sample the middle of the capacity
curve. That belongs in future work, not in this scope.

---

## D8 — One script for all arms

**Decided** 2026-07-30. Separate per-arm scripts allow an unintended difference — learning
rate, seed, augmentation, checkpoint criterion — to be reported as an effect of injection.
`16_run_arm.py` differs between arms only by `--arm`.

---

## D9 — Each arm receives its input in the form its own path expects

The official arm receives **raw** `lat`, `lon`, `year`, `doy`, because Prithvi's
`coords_encoding` normalises internally; feeding it z-scores would modify the mechanism and
it would no longer be their arm. The adaLN arm receives z-scored values.

This asymmetry is deliberate and must be stated in any report of the results.

---

## D10 — `deterministic=False` during training

Full determinism is slow on GPU, and seed-to-seed variability is a quantity we intend to
**measure** rather than suppress. Baseline is run with three seeds first, and its standard
deviation defines the threshold for a meaningful difference between arms.

---

## D11 — Reversed: GPP as the second task

**Originally decided** 2026-07-29: CarbonBench GPP would be the second task, on the argument
that photosynthesis is driven by weather.

**Reversed** 2026-07-30 on measurement. Weather passed season and biome controls and then
failed the image-proxy control: control AUC rose to 0.92–0.96 and weather added nothing.
Weather is a proxy for greenness, which NDVI already reports directly.

---

## D12 — Reversed: flood as the second task

**Originally decided** 2026-07-30. **Reversed** the same day.

Weather showed strong within-event signal (`+0.096`/`+0.186`, p = 0.005) and then failed the
image-proxy control (`+0.005`/`+0.012`). Separately, with 11 events the control block itself
scored **below 0.5** on an event-disjoint split — a control worse than a coin flip cannot
support a test.

Caveat recorded in its favour: the flood image proxy includes MNDWI, a direct water index.
For a water-segmentation task that control is close to reading the label, so this rejection
is not equivalent to the GPP one.

# Journal

A chronological account of the work, including the paths that were abandoned.

**On the commit history.** The work was done between 2026-07-28 03:43 and 2026-07-30 12:45
in a working directory that was not under version control at the time. This repository was
assembled on 2026-07-30 from those files. Commit timestamps are taken from the real
modification times of the files, so the ordering and the dates are accurate, but the commits
themselves were created in one session rather than as the work happened. Stating this is
cheaper than pretending otherwise.

---

## Day 1 — 2026-07-28

### The starting claim, and why it was wrong

The project began with a causal story: wildfire spread is driven by wind, so injecting wind
into a burn-scar segmentation model should help. The story was appealing and it survived
for about 30 hours.

Two things were built first: a research dossier on Prithvi-EO-2.0 and its nine downstream
tasks, and an **injection routing table** — a grid of *what kind of auxiliary data* (global
scalar, pixel field, sequence) against *where it could enter* (input concatenation, token
embedding, normalisation modulation, cross-attention, hypernetwork). Most cells were empty
in the literature. That table survived every later reversal and is still the framing of the
project.

### 00 — the first dead end (16:41)

`00_test_cds.py` probed the Copernicus Climate Data Store for ERA5 reanalysis. It worked as
a connectivity test. It did not survive contact with 804 requests.

### 01–03 — dataset reconnaissance (22:56 – 23:52)

`01_scan_dataset.py` read all 804 HLS Burn Scars chips. Each filename encodes sensor, MGRS
tile, and acquisition date, which is where every later coordinate and date came from.

`02_build_meta.py` extracted per-chip geolocation, CRS, resolution, and burn/unburn/nodata
fractions.

`03_make_split.py` produced the first split — and immediately raised the problem that shaped
the rest of the project. The published train/validation division puts **the same MGRS tile
on both sides**. A model can memorise the tile. Any conditioning variable that correlates
with location then has nothing left to explain.

A tile-disjoint split was built instead: 563 / 121 / 120, zero shared tiles.

> **Decision 1.** Never evaluate on the published split. Every number in this repository uses
> the tile-disjoint split.

---

## Day 2 — 2026-07-29

### 04, 04b — two ERA5 paths abandoned (01:34, 11:26)

`04_download_era5.py` requested per-chip hourly ERA5 from CDS. Queue times made 804 requests
impractical. `04b_download_era5_monthly.py` tried monthly bulk files instead, which failed
differently — request-shape limits and enormous downloads for data that would be discarded.

Roughly ten hours went into these two files. Both are kept in the repository because the
third attempt only makes sense against them.

### 04c — Open-Meteo (13:52, 13:58)

`04c_test_openmeteo.py` was written as a five-minute probe before committing to another
provider. `04c_fetch_era5_openmeteo.py` then pulled 804 chips x 720 hours in **29 seconds**
with zero errors.

> **Decision 2.** Probe a data source with a throwaway script before building a pipeline on
> it. The two failed ERA5 paths cost more than every successful step combined.

### 06, 06b — validating the weather before trusting it (14:14, 14:19)

`06_validate_openmeteo.py` ran sanity checks rather than assuming correctness.
`06b_check_tp_convention.py` existed for one reason: precipitation can be reported as an
hourly rate or as an accumulation, and getting it backwards silently scales every
precipitation feature.

The check that mattered most: median 7-day precipitation before a fire was **2.5 mm**;
before a flood (measured later on Sen1Floods11) it was **55.7 mm**. A factor of 22 in the
expected direction. The data was what it claimed to be.

### 07 — the signal test, and the number that later collapsed (15:18)

`07_signal_test.py` asked whether the weather features predict burn fraction better than
season alone. With pre-registered thresholds and 200 block permutations, the conditioning
vector reached **AUC = 0.695**.

That number drove the next 24 hours of planning. It was wrong — not arithmetically, but as
evidence for the claim it was used to support. See Day 3.

### 07b — the first control that hurt (18:25)

`07b_geo_control.py` added latitude and longitude to the control block. The weather block's
contribution beyond season **and geography** dropped to `p = 0.209` — not significant.

This was the first sign that "weather" was standing in for "where and when", not adding
anything of its own.

### 08 — locking the conditioning vector (19:06)

`08_finalize_conditioning.py` wrote `conditioning_v1.csv` and froze it. Ten z-scored
dimensions: six weather, four geographic/seasonal. The lock mattered later: when the vector
was cut from ten dimensions to four, the file did not change — only the columns read from it
did.

---

## Day 3 — 2026-07-30

### 09, 10 — preparing for TerraTorch (00:28, 00:30)

Reconstructed the on-disk layout expected by the segmentation datamodule and wrote explicit
split files, so that the tile-disjoint split survives into training rather than being
re-derived.

### A parallel session, and a criticism that landed

A second working session took the same pipeline to Sen1Floods11 and to CarbonBench GPP, to
test whether the framing generalised. It produced two results and one objection.

The results: on GPP and on flood, weather passed a season control and a geography control,
then **failed** once spectral indices from the imagery were in the control block.

The objection, directed at this session's `AUC = 0.695`:

> That number comes from a tabular classifier on burn fraction, not from the segmentation
> model. The model sees the image. The image may already encode geography.

It was correct, and it was the fifth time a number from an indirect source had failed to
survive direct measurement.

### 11 — the model actually loads (03:26)

`11_test_backbone_load.py` confirmed Prithvi-EO-2.0-300M loads with **zero missing keys**
and produces the expected output shape. Worth its own script: a silently mismatched
checkpoint produces a model that trains and is meaningless.

### 12 — the measurement that reshaped the project (03:40)

`12_image_proxy_control.py` built an image proxy — six band means plus seven spectral
indices from the chips themselves — and used it as a control block.

```
image proxy alone                          AUC = 0.7105
+ full 10-dimensional conditioning vector  AUC = 0.7080   -0.0025   p = 0.34   FAIL
+ 4 geographic/seasonal dimensions only    AUC = 0.7074   +0.0287   p = 0.010  PASS
image proxy + geography, then weather      AUC = 0.7080   +0.0008   p = 0.31   FAIL
```

Three consequences:

1. `AUC = 0.695` was an illusion. A crude image proxy scores **higher** than the conditioning
   vector. The imagery already had what we thought we were adding.
2. All of the signal is in four dimensions — `lat`, `lon`, `sin(doy)`, `cos(doy)`. The six
   weather dimensions contributed nothing and **diluted** the four that worked, which is why
   the 10-dimensional vector failed while its 4-dimensional subset passed.
3. The claim had to change from "this data helps the model" to "given data that is not
   redundant, where should it enter?"

> **Decision 3.** The conditioning vector is 4-dimensional. Weather is reported as a measured
> negative result, not quietly dropped.

The parallel session then ran the same 4-dimensional test on its datasets: marginal pass on
GPP, fail on flood. The rule is real but not universal — flood has 11 events, so `lat/lon` is
close to event identity there.

### 13 — opening the backbone (04:17)

`13_inspect_backbone.py` was written before any adaLN code, to find out whether the risky
part was even possible: 24 blocks, `embed_dim = 1024`, `norm1` and `norm2` both affine, and
block replacement verified to work end-to-end.

It also found `coords_encoding` in the backbone constructor — Prithvi's own metadata path,
reachable through the TerraTorch factory with a single argument.

> **Decision 4.** Test the riskiest component first. Three days had been budgeted for
> attaching adaLN to a pretrained ViT; the feasibility question was answered in twenty
> minutes.

### The equivalence test, and the bug it did not catch at first (04:38, 11:14)

`test_equivalence.py` checks that the adaLN arm collapses exactly to the baseline when its
modulation is zero. The first run reported success on both equality checks — and also
reported that randomising the modulation weights changed nothing.

The module was inert. `copy.deepcopy` on a TerraTorch model breaks block replacement. Full
account in [`FAILURES.md`](FAILURES.md#f1).

> **Decision 5.** Every equivalence test carries a must-differ branch.

### 14 — the smoke test that caught a silent freeze (11:14)

`14_smoke_cpu.py` runs the whole chain on CPU: conditioning vector present, correct value for
the correct file, shuffle with zero fixed points, validation split not shuffled, two real
training steps, gradients flowing.

It found that the conditioning encoder was not in `task.parameters()` and would have trained
for zero steps in every run, silently. See [`FAILURES.md`](FAILURES.md#f2).

### 15 — the official arm, and the number that sharpened the project (11:32)

`15_arm0_official.py` enabled `coords_encoding` and measured what it adds:

```
encoder.temporal_embed_enc.scale    1 parameter
encoder.location_embed_enc.scale    1 parameter
```

**Two parameters.** Against 25,331,200 for adaLN. Both scalars were confirmed to receive
gradient, so the arm is real rather than an inert flag.

This changed the framing from "we have a better method" to something stronger and more
honest: *a 324 M parameter model gives location and time a two-scalar channel of influence,
and nobody has ablated it.*

It also created the fairness problem that the `shuffle` arm exists to answer.

### 16 — one script, four arms (12:04)

`16_run_arm.py` runs `baseline`, `official`, `adaln`, and `shuffle` from a single code path,
so the only difference between arms is `--arm`. All four were verified on CPU.

---

### 17 — the matrix runs (31 July – 1 August)

Ten runs, one GTX 1070, roughly 30 hours of wall clock. Order chosen so the instrument was
calibrated before anything was measured: `baseline` x3 first, then `official` x3, then
`adaln` x3, then `shuffle` x1.

Two protocol events happened mid-matrix and are recorded in FAILURES rather than smoothed
over:

- **Early stopping was removed after `baseline_s0`.** Best epoch was 36 for seed 0 and 7 for
  seed 1; with epoch-to-epoch noise around 0.05 mIoU, early stopping fires at an arbitrary
  point and partly measures *when training stopped* rather than how good the arm is. The
  completed `baseline_s0` run was discarded (moved to `_superseded_earlystop/`, not deleted)
  and redone under a fixed 50-epoch budget. Seed-to-seed gap fell from 0.0176 to 0.0042.
- **`official_s0` hit a real blocker in Prithvi's own metadata path.** Enabling
  `coords_encoding` makes `strict=True` checkpoint loading fail on two missing keys. Fixed
  with a scoped relaxed-loading context manager plus an assertion that *exactly* those two
  keys are missing and zero are unexpected — a relaxed load that silently drops real weights
  would have produced a plausible, wrong number.

### 18 — the result (1 August)

| Arm | mIoU | seeds | injection params | Δ vs baseline |
|---|---|---|---|---|
| `baseline` | 0.8701 ± 0.0036 | 3 | 0 | — |
| `official` | 0.8697 ± 0.0032 | 3 | 2 | −0.0004 |
| `adaln` | 0.8654 ± 0.0015 | 3 | 25,331,200 | −0.0047 |
| `shuffle` | 0.8653 | 1 | 25,331,200 | −0.0048 |

Threshold locked before any injection arm ran: 0.006. Nothing crosses it.

The `shuffle` arm is what turns this from "we found nothing" into "there is nothing to
find". It has the identical 25 M-parameter architecture but each sample receives *another
sample's* location and date. It landed 0.0001 from the `adaln` mean — fifteen times smaller
than the seed noise within the `adaln` arm itself. Scrambling the conditioning costs
nothing, so the correct conditioning was buying nothing, so the −0.0047 is the price of
capacity and not a statement about the information.

The pre-registered prediction was 25 % adaLN wins / 25 % worse / 50 % lost in noise,
revised to 15/20/65 after seeing `adaln_s0`. The majority branch is what happened.

---

## Where this stands

Complete for HLS Burn Scars. All ten runs finished; full numbers in `MEASUREMENTS.md`.

The finding is a null result with a control that certifies it: on this task, both Prithvi's
own two-scalar metadata path and a 25 M-parameter per-block modulation move mIoU less than
the pre-registered detection threshold, and the shuffled control shows the information is
not contributing at all. The question "where should auxiliary data enter?" collapses here
before it can be answered — there is nothing to route.

What this does *not* establish: anything about flood or GPP (those directories exist and
were never run), anything about adaLN at larger data scale (563 training samples here), and
anything about location and date for EO tasks in general — only for separating burned from
unburned, where the spectral bands nearly settle it alone.

---

> **Gap notice.** This journal jumps from 1 August (burn scars) to 6 August. The whole
> EuroSAT-S1 arc in between — the geo split, the leakage discovery, the six-arm
> pre-registration, Measurements 12–21 — is recorded in `MEASUREMENTS.md` and has not been
> transcribed here yet. Flagged rather than quietly skipped.

### 19 — the night of 5→6 August: from "why it failed" to "here is the fix"

Ali handed over the GPU overnight and went to sleep. Two stages ran unattended.

**What went in.** Six new arms in `22_model.py`, added strictly additively — the six
pre-registered arms and their code paths were not touched, and the new names sit after a
comment marking them as post-registration. Three of them (`gate_std`, `gate_max`,
`gate_coord`) change what the gate *reads* at the original position; one (`gate_late`)
changes *where* it reads; one (`add_mid`) moves Prithvi's scalar to the same later position
without a gate; one (`film`) is multiply-and-shift at the front. Two new tests were added
to `23_test_arms.py` first — T6 (a coordinate-fed gate must depend on coordinates) and T7
(image-fed gates must depend on the image) — and `run_stage5.ps1` was written to abort
before spending a single GPU-minute if any test failed.

**What came out at n = 3.** 27 runs, 6h 34m, zero failures. Three different gate inputs at
the old position moved nothing (+0.22, +0.02, −0.33). The same gate three blocks later
appeared to gain +1.03. Every shuffled-coordinate control landed on baseline.

**What came out at n = 5, four hours later.** `run_stage6.ps1` added seeds 3 and 4.
`gate_late` seed 4 came in at 76.89, the lowest of its five. The arm fell to 77.94 ± 0.64
and **every contrast now fails the rule**: −add +0.77 (p = 0.063), −gate +0.74
(p = 0.067), −add_mid +0.69 (p = 0.092). The apparent result did not survive.

Two mechanisms inflated it, both nameable: `gate_late` was **the best of six new arms**, so
its n = 3 value was a selected maximum; and its first three seeds happened to cluster,
giving s = 0.36 and a small-looking p. This is the second time in this project a 3-seed
result shrank at 5 — the first was the six-arm table (max spread 0.83 → 0.17). At s ≈ 0.56,
three seeds cannot resolve a sub-1-point effect. That is now a rule here, not an anecdote.

**The upside, and it is the larger one.** Measurement 20 predicted — before these six arms
existed — that if location is essentially a class prior then nothing should beat `add`.
Six new mechanisms were built to break that prediction, including two structurally unlike
anything tried before. None did. The null across entry points now rests on an explanation
that made a risky prediction and survived 42 runs aimed at it. The tension flagged at 02:41
between Measurement 20 and `gate_late` simply dissolved: there was no advantage to reconcile.

**Two errors on the record.**
1. `33_analyse_geo.py` crashed at the end of stage 5 with `KeyError: 'add_mid'`. No
   training result was affected. Fixed by making the pre-registered table *skip* unknown
   arms rather than absorb them, with the post-hoc arms moved to a new
   `37_analyse_new_arms.py`. The tempting fix — extending the ARMS tuple — would have
   silently mixed post-hoc arms into a pre-registered table.
2. The gate's quoted dynamic range (2.42) came from 142 degenerate all-black chips. On
   clean chips it is 1.30 (Measurement 21). The −0.90 correlation itself survives at −0.87.
   Every document quoting 2.42 or 2.52 is wrong and is being corrected.

**Stage 7, and an admission.** At 06:30 a third stage was launched: seeds 5–9 for
`gate_late`, `add_mid` and `add`, taking the key contrast to 10-vs-10. The decision to
extend was made *after* seeing a near-miss, which is optional stopping and inflates the
false-positive rate. It is named rather than buried, and three constraints were written
and committed to git before the runs started (`docs/PREREG-stage7-power.md`, commit
`6855e35`): final n is 10 and will not move whatever the outcome; n = 10 comes from a
power calculation for the observed s (n = 5 had ≈ 45 % power for a 0.7-point effect, so
the n = 5 null is an underpowered test, not evidence of absence); and the result stays
labelled exploratory. `33_analyse_geo.py` was hard-limited to seeds 0–4 in the same commit
so stage 7's `add` seeds cannot walk into the frozen pre-registered table.

Prediction recorded before the runs: 60 % fails both halves of the rule, 25 % clears 2SE
but not p, 15 % clears both.

**What the morning summary got wrong.** `docs/MORNING-2026-08-06.html` was written at
06:05 on the 3-seed numbers and led with a result that no longer exists. It was rewritten
at 06:45; the wrong version is in git history and was not deleted. `MEASUREMENTS.md`
Measurement 22 carries a correction banner rather than an edit, for the same reason — how
far the number moved is the useful part.

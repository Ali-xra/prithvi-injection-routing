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

### Incident — queue stalled, fixed (2026-08-07)

The self-driving pipeline stalled a few hours in. Cause: long jobs were launched
detached with PowerShell `-NoNewWindow`, which keeps the child tied to the
launching console; when that transient shell exited, the job died. Run 2 of the
logit-prior died after seed 2, and the runner's q0 (which *waited* for an
external result file) then waited forever. GPU sat idle.

Three fixes, all committed:
1. `39_logit_prior.py` is now **resumable** — it reuses any saved
   `runs/none_geo_logits_s*.npy` instead of retraining (seeds 0-2 were already
   saved, so only 3-4 retrain).
2. `queue.json` q0 is now a **script item** the runner runs itself, instead of
   waiting on an external process. Nothing external to die.
3. The runner is relaunched **independent of the console** (no `-NoNewWindow`),
   so it survives.

Note: the venv `Scripts\python.exe` is a launcher shim, so each logical process
shows as two OS processes (shim + real). Four python.exe with venv paths = one
runner + one child script, not two runners.

### Run 2 — logit prior, per-seed clean result (2026-08-07)

`39_logit_prior.py`, per-seed single-model baseline (matched to how none/add were
measured), alpha=1 pre-registered, geo split. This replaces the confounded Run 1.

```
image-only ViT (mean of 5 single models)   73.21   (= none)
prior recovered at alpha=1, per seed        +2.46 +1.18 +1.24 +1.13 +2.70
recovered mean                              +1.74 +/- 0.77
deranged-coord control                      -10.02   (collapses -> prior is real geography)
trained gain (add - none)                   +3.96
fraction of the gain a prior explains       43.99%   -> verdict PARTLY
alpha sweep (exploratory): peaks +2.46 at alpha 0.5, +1.74 at the fixed alpha 1
[for the record] Run 1 ensemble baseline (76.07) gave -0.02 = the confound
```

**Honest reading.** On the real ViT, a regional class prior explains **~44%** of
the location gain -- not the 86% the weak 12-feature probe (35_prior_test.py)
suggested, and not the ~0 of the confounded ensemble. So location is **partly a
prior**; roughly **half** the gain is something a regional prior cannot express.

This does NOT reject the gate -- it cuts the other way. The earlier "the null is
explained because location is mostly a prior" pillar is now weaker: ~56% of the
gain is non-prior residual, which is exactly what a content-reading mechanism
(`gate_late`) could capture and a global additive vector cannot. It raises the
stakes of the depth12 test rather than lowering them.

Recorded as DIVERGED from the 86% expectation but non-halting (halt_on_divergence
false): the queue correctly noted it and moved on to depth12.

### depth12 — in progress (as of this check-in)

Running. Only the `none` baseline is in so far: 73.38 (seeds 73.52/73.31/73.30),
so a 12-block from-scratch ViT does train. The 7 remaining configs (add, and
add_mid/gate_late at mid 3/6/9) are still to run (~4h). The window verdict will
be collected at the next check-in.


### depth12 — FINAL (2026-08-07, T1 depth resolution)

`40_depth12.py`, 12-block from-scratch ViT, geo split, 3 seeds, same recipe as
the frozen 6-block. Validity gate passed: add − none = **+3.96** (the payload
does reach a 12-block model). Same-depth comparison gate_late vs add_mid at each
mid:

```
                add_mid   gate_late   gate_late − add_mid
mid = 3          76.80      77.94          +1.14
mid = 6          77.57      78.27          +0.70
mid = 9          77.08      78.52          +1.45   <- peak
window = max                               +1.45   at mid 9
```

**Verdict: WINDOW-PERSISTS.** The gate advantage does not vanish when the model
is deepened — it is positive at all three depths and, if anything, larger than
the 6-block reference (+0.82 at mid 3). The queue flagged this as DIVERGED-high
(window 1.45 > expected 0.82+0.6) and halted; that is a *confirming* divergence,
so the decision (logged in DECISION-NEEDED.md, now resolved) was to proceed to
Wave 2, no code rewrite. This upgrades the "window, not a direction" claim from a
5-position 6-block model to a model with real depth resolution.

Caveat kept honest: 3 seeds, exploratory. A 5-seed pre-registered confirm at the
peak depth is still the thing that removes the "exploratory" label.

### Wave 2 — readout zoo (2026-08-07, T2: does HOW the gate reads matter?)

`41_readout_zoo.py`, 6-block mid=3 (the window depth), geo split, 3 seeds. Four
new readouts vs the CLS-token reference `gate_late`; each with its `--shuffle-
coords` twin + must-differ test (brief 7.4).

```
reference gate_late (real)     78.22
                 real    shuf    vs gate_late   must-differ
gate_gem        78.05   72.71      -0.17         PASS
gate_bilinear   77.99   73.07      -0.23         PASS
gate_attn       78.35   73.54      +0.13         PASS
gate_min        78.23   72.91      +0.01         PASS
best new arm: gate_attn  +0.13 vs gate_late  (within 2*SE)
```

**Verdict: READOUT-NULL** (as pre-expected in CAMPAIGN.md). Every readout —
generalised-mean, 2nd-order/texture (the "right" quantity for SAR), attention
pooling, min — lands within noise of the plain CLS read. None beats gate_late by
more than +0.13 (< 2·SE). So on EuroSAT the *readout* is not what buys the window;
the entry point / depth is. And the must-differ twins all PASS: real coords beat
deranged coords by ~5 points in every arm, so these are live location-using arms,
not dead code that a "must-be-same" test would have waved through.

Paper value: this is the axis-A (readout) completeness leg. Combined with T1
(depth), the story tightens to "it's a **window** (where you read/inject), not a
**recipe** (how you pool) and not a **direction** (inject-later alone is nothing)."
Note kept: N=5 arms compared — apply Bonferroni before any single-arm claim
(trap 2); here it doesn't matter because the result is a null.

Queue status after Wave 2: **COMPLETE**, GPU idle. Next per CAMPAIGN.md is either
Wave 1's cheap finish (5-seed pre-reg confirm at the peak depth + the 6-block
add_mid curve at mid 2/4/5) or Wave 3 (T3: a second dataset / region-selective
task — the high-upside one). Wave 3 needs a fresh dataset build and a decision, so
it is not auto-started.

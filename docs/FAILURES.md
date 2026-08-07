# Failures

Every entry here cost real time. Each is kept because the *detection method* is the
transferable part, not the bug.

The common thread: **none of these produced an error message.** All of them would have
produced a complete, plausible, wrong result.

---

## F1 — An equivalence test that certified an inert module

**Symptom.** The adaLN arm was built with zero-initialised modulation, so at step 0 its
output must equal the baseline bit-for-bit. The test confirmed that. It also confirmed the
output was bit-for-bit identical **after randomising the modulation weights**, which is
impossible if the module is connected.

**Why it was silent.** The test only asserted equality. A module that does nothing at all
satisfies every equality assertion.

**Root cause.** `copy.deepcopy` on a TerraTorch model. With a deep-copied model, block
replacement no longer affects the forward pass, even though `blocks[0] is wrappers[0]`
evaluates `True` and the modulation tensors are produced (`|shift/scale| = 562.6`).

**How it was caught.** Instead of guessing, three points were instrumented and measured
directly on a non-copied model: block output changed by `6.8e-01`, backbone output by
`5.2e+00`, model output by `5.5e-01`. The module was fine; the test harness was not.

**Fix.** One model, baseline captured from the same object before injection. Never
`deepcopy` a TerraTorch model.

**Rule extracted.**
> Any equivalence test must contain a **must-differ** branch. A test that only checks
> "these are the same" will pass a no-op.

---

## F2 — A conditioning encoder that trained for zero steps

**Symptom.** None. Training ran to completion.

**Root cause.** `AdaLNInjector.encoder` was a child of the injector, not of the model. The
per-block `to_mod` heads live inside the blocks, so they appeared in `model.parameters()`.
The encoder did not. `configure_optimizers` only sees model parameters, so the encoder
stayed frozen at random initialisation — a fixed random projection for the entire run.

**Consequence if undetected.** The arm would have been reported as "adaLN did not help"
with half of it never trained.

**How it was caught.** An explicit assertion in the CPU smoke test:

```python
task_ids = {id(p) for p in task.parameters()}
assert all(id(p) in task_ids for p in inj.encoder.parameters())
```

**Fix.** `model.add_module("cond_encoder", self.encoder)`.

**Rule extracted.**
> Injected parameters that are not submodules of the model freeze silently. Assert
> membership in the optimiser's parameter set, not just that the module exists.

---

## F3 — Not a bug: zero gradient on the encoder at step 1

Worth recording because it looks exactly like F2.

With adaLN-Zero, `to_mod` starts at zero, so

```
d(loss)/d(encoder) = W^T . g = 0    at step 1
```

The encoder receives no gradient until `to_mod` leaves zero. Measured: `0.000e+00` at step 1,
`1.38e-01` at step 2. adaLN-Zero is self-starting.

Had the assertion been "gradient is non-zero at every step", correct behaviour would have
been chased as a bug.

**Rule extracted.**
> Before treating a zero as a fault, derive what the zero should be.

---

## F4 — Five reversals where an indirect number did not survive direct measurement

| # | Indirect claim | Direct measurement | Outcome |
|---|---|---|---|
| 1 | Weather correlates with burn extent | passed season/geography controls | survived |
| 2 | Therefore weather helps the model | image proxy control | **failed** — `+0.0008`, p=0.31 |
| 3 | `AUC = 0.695` shows the conditioning vector is informative | image proxy alone scores `0.7105` | **failed** — the proxy beats the vector |
| 4 | GPP is a viable second task | image proxy control | **failed** |
| 5 | Flood is a viable second task | within-event permutation; 11 events | **failed** — no generalisation test constructible |

The corrected claim is narrower and measured: weather is redundant with imagery; absolute
position and date are not (burn scars, `+0.0287`, p=0.010).

**Rule extracted.**
> A number from a proxy is a hypothesis about the real measurement, never a substitute
> for it.

---

## F5 — Leakage makes auxiliary data look *less* useful, not more

Counter-intuitive and measured on GPP. On a leaky split the auxiliary block added
`+0.0052`; on a clean split, `+0.0288`.

Site identity substitutes for weather when the same site appears in train and test, so the
auxiliary variable has nothing left to explain. Any ablation published on a leaky split has
therefore **underestimated** the contribution of auxiliary data.

---

## F6 — Environment and tooling failures

These are mundane and are recorded because they consumed hours.

| Failure | Cause | Fix |
|---|---|---|
| `PowerShell` parse error `string is missing the terminator` | non-ASCII text in a `.ps1` written without a BOM; PowerShell 5.1 reads it as ANSI and the parser breaks | keep `.ps1` files ASCII-only; put logic in `.py` |
| `No matching distribution found for torch` from `cu124` | the CUDA 12.4 index has no wheel for Python 3.14 | use `cu126`, which has an exact version match |
| `bf16: True` on a GTX 1070 | `is_bf16_supported()` returns True on Pascal although there are no bf16 tensor cores; mixed bf16 is emulated and slow | verify with a real kernel launch, then select `16-mixed` or `32` |
| `cuda available: True` is not proof of usability | recent PyTorch builds drop old architectures; an unsupported arch fails only at kernel launch | check `get_arch_list()` **and** run a real matmul and backward pass |
| ERA5 via CDS: two separate download paths abandoned | API queueing and request-shape limits | switched to Open-Meteo; 804 requests in 29 s, zero errors |

The ERA5 dead end is visible in the repository as `04_download_era5.py`,
`04b_download_era5_monthly.py`, and the working `04c_fetch_era5_openmeteo.py`. All three are
kept.

---

## F7 — We ran the ablation on the wrong checkpoint, and the error message told us so

**The worst failure in this project.** Found on 1 August, after all ten runs had finished,
because the user asked a question I could not answer from what we had measured:

> "We didn't pretrain, we fine-tuned. Isn't pretraining different?"

IBM ships **two** separate 300M checkpoints:

| checkpoint | `coords_encoding` | pretrained with metadata? |
|---|---|---|
| `Prithvi-EO-2.0-300M` | `[]` | **no** |
| `Prithvi-EO-2.0-300M-TL` | `["time", "location"]` | **yes** |

We used the first one. Read directly from both `.pt` files:

```
300M    : 398 keys, no coords keys at all
300M-TL : 402 keys, including
            encoder.location_embed_enc.scale = 0.05815186
            encoder.temporal_embed_enc.scale = 0.00000128
```

### The part that matters

On 31 July our `official` arm failed with:

```
RuntimeError: Missing key(s) in state_dict for PrithviViT:
              "temporal_embed_enc.scale", "location_embed_enc.scale"
```

We treated that as an obstacle and engineered around it with a relaxed-loading context
manager, and we documented the workaround carefully — including a verification that
*exactly* those two keys were missing. All of that was correct engineering applied to the
wrong problem.

**The message was not an obstacle. It was the system correctly reporting that we had loaded
a checkpoint which does not contain the mechanism we were trying to study.** The
verification we were so pleased with confirmed the symptom and never asked why the symptom
existed.

So the `official` arm did not measure "how much does Prithvi's metadata path help". It
measured "what happens if you bolt that mechanism onto a backbone that never learned to use
it and give it 563 samples to figure it out". Those are different questions and only the
second one was answered.

**Rule extracted.**
> When a load fails on exactly the component you came to study, the first hypothesis is that
> you have the wrong artefact — not that you need a way around the error. Engineering
> around an error suppresses the evidence that produced it.

### What did not become invalid

`baseline`, `adaln` and `shuffle` are unaffected: all three used the same checkpoint as each
other and the shuffled control still shows that 25 M parameters extract nothing from
location and date on this task. What became invalid was the *claim* that we had ablated
Prithvi's official metadata path.

---

## F8 — `unexpected_keys` cannot tell you which checkpoint you loaded

The fix for F7 introduced its own error within the hour.

Two new arms were added — `tl_on` (TL checkpoint, coords on) and `tl_off` (TL checkpoint,
coords off) — and to stop F7 from recurring silently, `tl_off` asserted that the two coords
scalars would appear as `unexpected_keys`: the model does not build them, the checkpoint has
them, so they should be reported as extra.

They were not. Measured: **zero missing, zero unexpected.** TerraTorch filters
non-matching keys before `load_state_dict` ever sees them, so `unexpected_keys` is
structurally incapable of identifying the checkpoint.

The assertion firing was the good outcome — it refused to run rather than proceeding on an
assumption that happened to be false. Had it been written as a warning, the whole TL matrix
would have run on an unverified checkpoint.

**The fix: fingerprint a tensor both checkpoints have but with different values.**

```
blocks.0.attn.qkv.weight   300M    sum = -50.225471
                           300M-TL sum = +118.693604
```

168 units apart; no ambiguity. `_verify_backbone_identity()` now runs on **every** arm, not
just the TL ones — had it existed on day one, F7 could not have happened.

**Rule extracted.**
> Verify identity from the artefact's own content, not from the loader's report about it.
> A loader's diagnostics describe what the loader chose to tell you.

---

## F9 — We validated the premise on a clean split and ran the experiment on a leaky one

Found on 1 August while building the stratified analysis. The per-chip evaluation reported
264 validation chips; the tile-disjoint split has 121. That number should not have existed.

`conditioning_v1.csv` carries **two** split columns:

```
orig_split : training 540 · validation 264      published HLS Burn Scars split
split      : train 563 · val 121 · test 120     our tile-disjoint split
```

`12_image_proxy_control.py` — the script whose `+0.0287` justified the entire project — uses
the `split` column. Verified by grep:

```
12_image_proxy_control.py:153  tr = [r for r in rows if r["split"] == "train"]
12_image_proxy_control.py:154  va = [r for r in rows if r["split"] == "val"]
16_run_arm.py                  (no match for "split" anywhere)
```

`16_run_arm.py` pointed the datamodule straight at the physical `training/` and
`validation/` folders — the published split. Every one of the 11 GPU runs used it.

How leaky:

```
MGRS tiles shared between published train and val : 124
val chips sitting on a shared tile                : 194 of 264   (73 %)
```

### Why this is not a cosmetic problem

Our own **F5** measured the direction of the effect:

> "Leakage makes auxiliary data look *less* useful, not more. On a leaky split the auxiliary
> block added `+0.0052`; on a clean split, `+0.0288`."

The mechanism is direct: when the model has already trained on the same MGRS tile, it has
memorised what that location looks like and has no use for being told the coordinates. That
is precisely the mechanism that manufactures a null result for a location-conditioning
experiment.

So the headline finding — "location and date contribute nothing" — was measured under the
one condition most likely to suppress exactly the effect being measured. We had written
F5 ourselves, eleven runs earlier, and did not connect it.

### What is and is not invalidated

- The `shuffle` control still stands **within** the leaky split: destroying the conditioning
  changed nothing there.
- The claim that this generalises to a clean split does **not** stand, and that was the claim
  the README made.
- All arms shared the same split, so arm-vs-arm comparisons remain internally consistent —
  as statements about the published split.

### Fix

`20_build_split_dirs.py` materialises the tile-disjoint split as real directories via
hardlinks and refuses to write anything until it has proved zero shared tiles across all
three partitions:

```
shared tiles train/val   0
shared tiles train/test  0
shared tiles val/test    0
training 563 · validation 121 · test 120
```

`16_run_arm.py` now takes its roots from `_roots()`, defaults to the tile-disjoint
directories, prints which split it is using on every run, and **puts the split in the run
tag** (`--orig-split` appends `_origsplit`). Round-1 artefacts were relabelled `*_origsplit`
rather than deleted, so the two rounds can be compared and can never overwrite each other.

**Rule extracted.**
> A split defined in a metadata column is not a split until something loads data through it.
> Assert the property you claim — count the shared tiles — inside the training script, not in
> the script that designed the split.

### The accidental upside

Round 1 is now a controlled comparison we did not plan: the identical experiment on a leaky
and a clean split. If the metadata arms behave differently across the two, that is a
pixel-level confirmation of F5 and a concrete warning about every ablation published on the
standard split.

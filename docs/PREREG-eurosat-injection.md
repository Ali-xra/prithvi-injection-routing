# Pre-registration — does the injection point matter when the payload is real?

**Written 2026-08-01, 3 of 30 runs complete.**

## Disclosure first

This document is **not** fully blind, and pretending otherwise would be worse than the
impurity itself. At the time of writing, three seed-0 runs had already been observed:

```
none_s0    81.15
add_s0     88.96      learned scale 0.15437
token_s0   89.50
```

Nothing else exists yet: no second seed of any arm, no `adaln`, no `gate`, no `shuffle`.

What that means for what can be locked here:

- The **noise floor cannot have been gerrymandered**, because it is computed from
  seed-to-seed variance and only one seed per arm exists. The rule below fixes how the
  threshold is derived; the number it produces comes entirely from data not yet seen.
- The **hypotheses were fixed before the experiment was built** — they are visible in the
  arm design of `22_model.py`, committed before any run.
- The `token` vs `add` gap of `+0.54` **has** been seen. Any conclusion about that specific
  pair is therefore weaker than the rest and is flagged as such in the results.

## Why this task, and not the previous one

The burn-scars phase measured a payload of `+0.0287` AUC beyond an image proxy, and it did
not survive into IoU. Copernicus-FM (ICCV 2025, Table 12) predicts exactly that: location
metadata is worth `+0.4` on optical EuroSAT-S2 and `+21.3` on radar EuroSAT-S1. The routing
question — *where* should auxiliary data enter — is meaningless when there is nothing to
route.

Our own tabular probe on EuroSAT-S1, run before any GPU time:

```
image only              70.50
location only           69.94
image + location        87.02      +16.52 points
image + shuffled loc    69.98      -0.52  (control)
```

The payload is real, large, and the shuffle control confirms it comes from the
correspondence rather than from added capacity. Image alone and location alone are nearly
equal, yet together they gain 16 points — they are complementary, not redundant.

## Design

One small ViT (2.7 M parameters, dim 192, depth 6, patch 8). **The payload is byte-identical
across arms**: the same `(lon, lat)` for the same chip, encoded by the same
parameter-free sinusoidal function. Only the entry point changes.

| arm | where location enters | injection params |
|---|---|---|
| `none` | nowhere | 0 |
| `add` | Prithvi-EO-2.0 style: one global learned scalar × fixed embedding, added to all patch tokens | 1 |
| `token` | TerraMind style: location becomes an extra token in the sequence | 37,056 |
| `adaln` | per-block shift/scale on both norms of all 6 blocks | 889,344 |
| `gate` | **ours**: same additive path as `add`, but the scalar is predicted per sample from the image | 9,313 |
| `shuffle` | control: `adaln` with each chip given another chip's coordinates | 889,344 |

5 seeds per arm, 30 epochs fixed, no early stopping, OneCycle schedule, identical
augmentation. Best validation accuracy per run.

The location encoder is deliberately **parameter-free and shared**. If it were learned, arms
would differ in how well they can read coordinates, not only in where they inject them, and
the comparison would no longer isolate routing.

## Verification already performed (`23_test_arms.py`, 20/20)

Every check has a must-differ twin, because the most expensive bug of the previous phase was
an equivalence test that passed while the injection was silently disconnected.

- each injecting arm **must** change its output when coordinates change — all pass
- `adaln` at zero-init **must** equal `none` — passes at `0.000e+00`
- every injection parameter **must** receive gradient and be in the optimiser — all pass
- the `gate` **must** depend on the image and **not** on the coordinates — passes at `0.000e+00`


## Threshold — the rule is locked here, the number comes from unseen data

Only one seed per arm exists, so seed-to-seed variance is unknown and cannot be tuned.

1. Let `s` be the pooled seed standard deviation across the six arms (5 seeds each,
   `ddof=1`, pooled over arms).
2. Standard error of a difference between two arm means: `SE = s · sqrt(2/5)`.
3. **A difference counts only if both hold:**
   - `|Δ| ≥ 2 · SE`, and
   - Welch's t-test on the two sets of 5 seeds gives `p < 0.05`.

This threshold will be reported before any arm comparison is discussed, and it will not be
moved afterwards — the burn-scars phase established that habit and it is kept even when the
result is favourable.

## Hypotheses

**H1 — the payload reaches the network.** `add > none`.
*Status: effectively already answered at seed 0 (`+7.81`). Reported as confirmation, not
discovery.* Refuted if the 5-seed difference falls below threshold.

**H2 — routing matters.** At least one of `token`, `adaln`, `gate` differs from `add` by
more than threshold.
*This is the primary question.* Refuted if all three land within threshold of `add`: that
would mean a single global scalar extracts as much from location as 889,344 parameters of
per-block modulation, which is itself a clean and publishable statement.

**H3 — a per-sample gate beats a global scalar.** `gate > add`.
Prithvi weights location with one constant shared by every sample on Earth. If the value of
location varies by sample — which Copernicus-FM's 50-fold radar/optical gap implies — then a
scalar conditioned on image content should capture more. 9,313 parameters against 1.
Refuted if `gate ≤ add` within threshold.

**H4 — the control behaves.** `shuffle ≈ none`, and `shuffle < adaln` by more than
threshold. If `shuffle` matches `adaln`, the gain is capacity rather than information and
every other comparison in this table is void.

## Pre-registered predictions

Written now, before seeing anything beyond the three disclosed runs:

| outcome | probability |
|---|---|
| H2 holds — at least one arm beats `add` by more than threshold | 60 % |
| H3 holds — `gate` beats `add` | 40 % |
| all injection arms tie (routing does not matter) | 35 % |
| `shuffle` fails its control (invalidates the table) | 5 % |

## Secondary measurements, no threshold attached

- **The learned scalar of `add`.** Initialised at `0.1`, exactly as in terratorch's Prithvi
  implementation. Seed 0 finished at `0.15437` — it went *up* on a task where location pays.
  IBM's released `Prithvi-EO-2.0-300M-TL` carries `0.05815186` — it went *down* during
  pretraining. Same mechanism, same initialisation, opposite direction. Reported across all
  5 seeds.
- **The distribution of the `gate` scalar.** Per-sample values are saved to
  `runs/gate_s*_gates.npy`. A wide distribution means the model decides chip by chip how much
  to trust location — the mechanism H3 depends on. A narrow one means it learned a constant
  and `gate` is `add` with extra steps.

## What gets reported regardless of outcome

The full 6×5 table with means, standard deviations and every individual seed. The threshold
and how it was derived. Which hypotheses survived and which did not. The disclosure at the
top of this file stays at the top.

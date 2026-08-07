# Pre-declaration — stage 9, the depth sweep

**Written 6 August 2026, 08:55 local, BEFORE any depth-sweep run was launched.**
Committed before launch so the timestamp is checkable in git history.

## Why

`MID_AT = 3` was a module-level constant chosen before any of the mid-stack arms existed.
Every `gate_late` and `add_mid` number ever recorded used depth 3 and nothing else. It was
never swept. Objection 5 in Measurement 22 says exactly this, and it is the only one of the
five that a GPU can close.

The model has **6 blocks**, so depth 3 is the middle.

## The question — and it is deliberately not "which depth is best"

Last night's lesson was that picking the maximum of several arms inflates it. A depth sweep
is the same trap wearing a different hat: five depths, take the best, report a bigger
number. **That is not what this is for.**

The question is about **shape**:

- If the gain appears **gradually** with depth, the story is "the gate needs a
  representation, and representations get better as you go deeper".
- If it is a **step** — nothing at depth 1, everything from depth 2 or 3 on — the story is
  "one or two rounds of attention are enough to make the CLS token worth reading", which is
  a sharper and more interesting claim.
- If the profile is **flat** across all depths including 1, then depth was never the
  mechanism and the depth-3 result needs re-examining. **This outcome would damage the
  claim, and it is a real possibility.**
- If the gain **disappears at depth 6** (injection after the last block, immediately before
  the head), that bounds the story from the other side: the payload needs blocks after it
  to be used, not just blocks before it to be read.

## Fixed in advance

1. **Depths 1, 2, 4, 6 for `gate_late`, 5 seeds each.** Depth 3 already has n = 10 and is
   not re-run. 20 runs.
2. **`add_mid` at depths 1 and 6, 5 seeds each.** The two extremes, chosen now, not after
   seeing the profile. These are the controls that separate "reading later" from "injecting
   later" at the ends of the range. 10 runs.
3. **n = 5 per point and it will not be extended.** This is a shape question, not a
   significance test on a single contrast. If some point lands near the threshold, it is
   reported as near the threshold. No stage 10.
4. **The headline stays at depth 3.** No depth found here will be promoted to a new
   headline number, whatever it shows. If some other depth looks better, that is a
   hypothesis for a fresh pre-registration on a task we have not touched — not a result.
5. **A flat or falling profile gets reported as loudly as a rising one.** Written here so
   that it cannot be quietly dropped later.

Total 30 runs, ~6 min each, ~3 hours.

## Prediction on record

- **Shape: 55 % step, 30 % gradual rise, 10 % flat, 5 % falling.**
- **Depth 1 vs depth 3:** I expect depth 1 to recover *most* but not all of the gain —
  the algebra in Measurement 19 only rules out depth 0, and a single attention block
  already lets the CLS token see the whole image. Point estimate: depth 1 lands around
  77.7, i.e. between `add_mid` (77.23) and depth 3 (78.05).
- **Depth 6:** I expect the gain to shrink, because the payload is injected after the last
  block and only the final LayerNorm and the linear head can use it. Point estimate 77.4.
- **`add_mid` at depths 1 and 6:** both flat, near `add` (77.05), as at depth 3 (+0.18).

Recording these because last night my stage-7 prediction was wrong (I said 60 % it would
fail; it cleared at p = 0.0004) and a prediction log is only worth keeping if the misses
stay in it.

## Code change this required

`MID_AT` moved from a module constant to a per-instance `LocViT(mid_at=...)` argument,
default unchanged at 3, plus a `--mid-at` flag on `24_train.py` that writes `_d<N>` into the
run tag so a depth-5 run can never overwrite the depth-3 file the claim rests on.

**T8 was added to `23_test_arms.py` first**: it asserts the default is still 3, that depth 1
and depth 6 produce different outputs from identical weights, and that a depth beyond the
block count is rejected. Without that test, a silently ignored `--mid-at` would produce five
identical arms and render as a beautifully flat profile. All tests pass (T1–T8).

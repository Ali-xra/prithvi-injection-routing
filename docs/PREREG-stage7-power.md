# Pre-declaration — stage 7, the gate_late contrasts at n = 10

**Written 6 August 2026, 06:38 local, BEFORE any seed 5–9 run was launched.**
Committed before launch so the timestamp is checkable in git history.

## The situation this is responding to

At n = 3, `gate_late` looked like a result: 78.22, +1.05 over `add`, p = 0.0137.
At n = 5 it is 77.94 ± 0.64 and every contrast fails the pre-registered rule:

```
gate_late - add       +0.77   2SE = 0.71   p = 0.0633   FAILS (p)
gate_late - gate      +0.74   2SE = 0.71   p = 0.0667   FAILS (p)
gate_late - add_mid   +0.69   2SE = 0.71   p = 0.0919   FAILS (both)
add_mid   - add       +0.08                p = 0.7978   FAILS
```

The two added seeds were 76.89 (`gate_late`, the lowest of its five) and 76.90
(`add_mid`). The mean fell 0.28 and the standard deviation rose from 0.36 to 0.64.

This is the second time in this project that a 3-seed result shrank at 5 seeds. The
first was the six-arm table, where the maximum spread between injection arms fell from
0.83 to 0.17.

## The honest problem with running more seeds

**This extension was decided after seeing a near-miss.** That is optional stopping, and
optional stopping inflates the false-positive rate. Naming it is not a formality — it is
the single most likely thing a referee will catch, and they would be right to.

Three things are being done about it, all of them fixed here in advance:

1. **Final n is 10. It is fixed now and will not move**, whichever way the result lands.
   If the contrast is at p = 0.06 at n = 10, that is the reported result and no seed 10
   will be run.
2. **n = 10 comes from a power calculation, not from the p-value.** For the observed
   pooled s = 0.56 and the observed effect d = 0.7, a two-sided test at alpha = 0.05 with
   80 % power needs

   ```
   n = 2 (z_0.975 + z_0.80)^2 s^2 / d^2
     = 2 (1.96 + 0.84)^2 (0.56)^2 / (0.7)^2
     = 10.0 per arm
   ```

   n = 5 had roughly 45 % power for this effect. The n = 5 result is therefore not
   evidence of absence; it is an underpowered test, and that is the actual reason to
   extend. If the effect is real at 0.7 points, n = 5 was always going to miss it about
   half the time.
3. **The result is exploratory and will be labelled exploratory forever.** `gate_late`
   and the other five arms were added after the pre-registration. Nothing from stage 7
   enters the pre-registered table, and `33_analyse_geo.py` has been hard-limited to
   seeds 0–4 so that adding seeds to `add` cannot silently change the frozen numbers.

## What will run

Seeds 5, 6, 7, 8, 9 for three arms: `gate_late`, `add_mid`, `add`. Fifteen runs, about
95 minutes. `add` is included so the headline contrast is 10-vs-10 rather than 10-vs-5.

`gate`, `token`, `adaln`, `none`, `shuffle` stay at n = 5. The pre-registered table does
not change.

## What each outcome will be called

| outcome at n = 10 | the sentence that gets written |
|---|---|
| `gate_late − add_mid` clears both halves of the rule | reading later, then gating, beats injecting later — the mechanism, not the position. Exploratory, found after a null, with this document attached. |
| it clears 2SE but not p, or the reverse | the contrast is at the edge of what 10 seeds can resolve. Reported as unresolved, with the point estimate and CI. Not called a finding. |
| it fails both | moving the gate later does not measurably help. The null across entry points stands, and Measurement 20's prior explanation is unchallenged. |

The third row is the outcome that is currently most likely on the point estimates, and it
is a perfectly good result — it is the one that makes Measurement 20 a successful risky
prediction rather than a tension.

## The prediction on record

Written before the runs: **60 % it fails both, 25 % it clears 2SE but not p, 15 % it
clears both.** Reasoning: the effect shrank from 1.05 to 0.77 when seeds were added,
which is the signature of a selected maximum regressing, and `gate_late` was picked as
the best of six new arms — so its n = 3 value was the maximum of six draws and was
inflated by selection before any of this started.

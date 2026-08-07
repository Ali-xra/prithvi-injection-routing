# Gate test campaign — what is being tested, in order

Three themes (the user's framing = the design-space axes):
  T1 MOVE the gate  = depth / where it reads+injects   (axis B)
  T2 gate INPUT     = readout: how it summarises the image (axis A)
  T3 other DATA/TASK= a second dataset, and a region-selective task (axis E)
Two more axes noted, run only when relevant: form (C: FiLM/SPADE), payload (D).

Every new arm gets a `--shuffle-coords` twin + a must-differ test (brief 7.4).
The runner halts + writes DECISION-NEEDED.md if any headline changes direction.
No push. Commits touch only the runner's own files.

Honest overall expectation, stated up front: on EuroSAT (location = mostly a
regional prior, low ceiling ~+0.8 non-prior) T1 and T2 will MOSTLY confirm nulls
or confirm/kill the one live positive (`gate_late`, +0.82 p=0.0004). The place a
genuine gate WIN could actually appear is T3 — a task where "which region"
matters (segmentation), because a content-reading gate can express that and a
global prior cannot. So T1/T2 are decisive-but-modest; T3 is high-upside,
high-cost.

## WAVE 0 — running now
- q0 logit_prior (per-seed on the real ViT). RESULT: prior explains ~40% of the
  gain (not 86% of the hand-feature probe). Informational; it does NOT reject the
  gate — it leaves ~60% non-prior residual that only a content gate can capture.
- q1 depth12: `gate_late` vs `add_mid` at mid {3,6,9} on a 12-block model, 3
  seeds. Tests T1 with real depth resolution. HALT if the window vanishes or the
  12-block fails to learn the payload.

## WAVE 1 — T1 finish (after depth12)
- if window survives: 5-seed pre-registered confirm at the peak depth (removes
  the "exploratory" label from gate_late).
- cheap: complete the 6-block window (add_mid at mid 2,4,5) for a clean curve.

## WAVE 2 — T2 readout zoo (after wave 1), new arms at the window depth
- gate_gem (generalised-mean pooling), gate_bilinear (2nd-order / texture — the
  right quantity for SAR), gate_attn (attention pooling), gate_min.
- each vs gate_late, each with its shuffle twin + must-differ test.
- then PROBE each (what quantity it tracks) so the result is "readout determines
  expressivity", not "we tried poolings".
- expectation: mostly null vs gate_late on EuroSAT (low ceiling); value =
  completeness + the probe map.

## WAVE 3 — T3 other data/task (after wave 2, only if gate_late survived)
- So2Sat LCZ (classification, coords, city-disjoint split): does the null + the
  window replicate on a SECOND task (two observations, not one).
- Burn Scars segmentation, clean rebuild (leakage-safe split): the region-
  selective task where a content gate / SPADE-form could actually win. Highest
  upside, highest cost (512x512, leakage rebuild).

## Extra axes (conditional)
- Form (C): FiLM single-point (5-seed confirm); SPADE — only on segmentation.
- Payload (D): incidence angle = NOT in the data (dropped); date = needs a
  time-bearing dataset; wind = needs the fire pipeline rebuilt.

## Check-in schedule (send_later into this session)
- +2.5h: collect logit_prior final + depth12 progress.
- +6h: depth12 done -> record; write & queue WAVE 2 arms (with twins) or, on a
  halt, rewrite the next code.
- +14h and onward: adaptive — each check-in collects, adds the next validated
  wave, and reschedules the next.

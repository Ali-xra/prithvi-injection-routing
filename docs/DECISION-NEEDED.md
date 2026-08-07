# Decision needed — none open

## RESOLVED 2026-08-07 — `depth12`
headline `window` = **+1.45** at mid=9 (expected 0.82 ± 0.6), verdict = **WINDOW-PERSISTS**.
The queue halted because the window landed ABOVE the tolerance band — i.e. the
gate advantage did not vanish on a 12-block model, it **grew** (positive at all
three depths: mid3 +1.14, mid6 +0.70, mid9 +1.45). This is a *confirming*
divergence, not a rethink.

**Decision:** proceed to Wave 2 (T2 readout zoo). Added `readout_zoo` to
queue.json and relaunched the runner. No code rewrite needed.

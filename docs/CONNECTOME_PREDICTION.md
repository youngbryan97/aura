# A prediction, written before the run

Two results in `docs/CONNECTOME.md` point at the same suspect, and it is not the
wiring. It is the frame rate.

ZAPBench records a zebrafish at one volume every 914 ms, and this package copied
that rate so its numbers would be comparable. At that rate a frame of Aura holds
about seventy thousand events. A call and its callee are microseconds apart, so
propagation begins and finishes inside a single frame, and what the frame-to-
frame relationship can carry is whatever survives that averaging.

Two measurements are consistent with that and with nothing else:

- Lag-one autocorrelation on the recording is 0.123. The recent past says almost
  nothing about the next frame.
- The connectome beats its own degree-preserving rewiring at a context of four
  frames, and the effect is a median per-cell MAE difference of 2.5×10⁻⁴ — real,
  significant, and tiny.
- Predicting which cells run next, the connectome rule beats knowing what is hot
  by five times on F1 and loses to knowing what just ran.

## What is predicted

A recording at 20 ms — 45 times finer — should move all three in the same
direction, because propagation would then cross a frame boundary instead of
completing inside one.

1. **Lag-one autocorrelation rises above 0.123.** If it does not, the signal is
   memoryless at every rate and the frame rate was never the problem.
2. **The connectome arm's advantage over its rewiring grows.** Same sign, larger
   median difference, on the same restricted set of cells.
3. **The connectome prefetch rule closes on persistence.** Its recall should
   rise relative to the persistent rule's, because a cell that ran last frame is
   less likely to still be running when a frame is 20 ms long.

If the first fails, the other two are uninterpretable and the honest reading is
that Aura's cell-level activity carries no temporal structure a forecaster can
use, at any rate this instrument can reach.

If the first holds and the second does not, then the wiring genuinely carries
little information about the activity, and the small effect at 914 ms was the
whole of it.

The run is four compute workloads at 150 seconds each, 20 ms frames, which is
30,000 frames. It is not comparable with ZAPBench's numbers and is not meant to
be; it is the experiment that decides whether the comparison was measuring the
instrument.

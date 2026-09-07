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

---

## The outcome

The run: four compute workloads, 150 seconds each, 20 ms frames. 21,355 frames.
It caught only **74 cells**, because compute workloads exercise a narrow slice of
the tree, so every number below rests on 38 connected pairs and is a small
sample. A broader recording at the same frame rate is the check on it.

All three predictions held.

**1. Autocorrelation rose.** Lag-one went from
0.17409 at 914 ms to
**0.38472** at 20 ms, a factor of
2.2.
The frame rate was averaging the propagation away.

**2. The connectome's advantage grew.** Median per-cell MAE difference against
its own degree-preserving rewiring: **-0.0136**, interval
[-0.0240, -0.0083], better on
**86.4%** of the cells it says anything about, against 57.8%
at 914 ms.

**3. The prefetch rule closed on persistence, and passed it.** This was
predicted to close and it overtook:

| rule | precision | recall | F1 |
| --- | --- | --- | --- |
| connectome | 0.7581 | 0.8322 | **0.7934** |
| connectome, contact-weighted | 0.7789 | 0.7871 | 0.7830 |
| persistent | 0.7779 | 0.7806 | 0.7792 |
| frequent | 0.1990 | 0.2097 | 0.2042 |

At 914 ms persistence won by 0.12 on F1. At 20 ms the connectome rule wins by
0.014 and by 5 points of recall.

Like-to-like also holds at this rate: connected pairs correlate at 0.595 against
0.012 for the rewiring, z = 10.3 over 38 pairs.

## What this changed

The prefetch result is the one with a consequence. Warming by the connectome
rule was documented as wiring the worse rule, on the evidence at 914 ms. That
evidence was about the instrument. The warm-up now takes its rule from whichever
measured better on this system's own recording, rather than from either of these
two numbers being written down as the answer.

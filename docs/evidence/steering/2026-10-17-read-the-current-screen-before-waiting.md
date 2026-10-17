# Read a running world before waiting for its scene to settle

The recorded attempt at `0d71007e1` completed its first Play click at
monotonic time 209244.844767. Its first fast picture arrived at
209247.464595 and first key completed at 209248.122881. The intervals are
2.620 seconds and 3.278 seconds. The log records a 1.3-second scene wait
before fast play. These intervals alone do not establish where every
remaining millisecond went or whether earlier input would have succeeded.

The ordinary page reader's wait gate used the preceding reading to decide
whether the screen was active play. A preceding menu therefore permitted
waiting for an animation to settle before reading what the click opened,
even when the place's retained instructions established a moving clock.
Six regressions reproduced this ordering through the actual `OnAPage.read`
route across steering, catching and survival instructions. Each now reads
first. A current NEXT remains a navigation control; a current counter does
not become one. Unknown and turn-based environments retain their wait.

The new priority changes only when to photograph and read. It neither sends
input nor grants body identity. Existing current-label interpretation,
ownership, input and navigation gates remain. It does not restore the
October 13 calibration-interleaving experiment that regressed shooter play.

The first test attempt had an incomplete watcher fixture and failed before
the intended assertion. The corrected red group failed six ordering cases
and passed the three waiting cases. The repaired focused group passed 41
checks. A further 59 navigation and physical-pointer checks passed. Their
supervisor records remain under
`artifacts/flash-campaign/2026-10-17-running-world-read/`.

The protected native checkout and G09 job remain untouched. This change has
not yet been shown to improve an original game's outcome or qualify a demo.

Smoke passed 165 checks with one skip. Lint, compilation, layering,
dependency consistency and writing passed. The combined navigation and gate
group completed in 98.56 seconds under the CPU supervisor.

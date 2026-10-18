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

## Original-game follow-up at `16371505e`

The bounded CPU attempt completed in 140.16 seconds without a win. Endings
were `lost`, `unread`, `unread`; the independently viewed final image was
the instructions screen. There were 149 control observations, no selected
body, 22 fresh reference geometries, no object-bound or continuing source
geometry and no candidate above zero on the two identity statistics. The
record closed without drops (319 images, 80 input-port events). Software
SwiftShader was measured; the current-interpreter classical boundary loaded
no backend and refused six MLX imports. This is not native-app proof.

The first fast image arrived 4.280 seconds after Play completed and the first
key was requested at 5.712 seconds. Those times worsened despite retiring
the scene wait. The startup log measured a 2,151.2-millisecond observation
operation: interpreter contention selected the separate perception process,
whose cold initialization happened after Play. The earlier run selected
local perception. These unmatched attempts do not isolate a causal timing
benefit from the wait change. Next test bounded preparation on a current
menu before input starts the moving phase; do not remove custody checks or
relax identity thresholds to conceal the delay.

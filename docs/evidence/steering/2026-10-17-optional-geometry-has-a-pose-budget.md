# Optional geometry gets one pose per frame

Ordinary CPU play at `3e452109c` still failed. Two diagnostic runs added
timing wrappers around the existing functions, preserving their return values
and policy. The first recording took 74.76 seconds and reported lost/unread;
the finer recording took 77.67 seconds and reported two losses. Neither
established demo readiness. The first instrumentation setup failed before
launching a browser because it named a locally imported function as a module
attribute; the retry instrumented its defining module.

In the first recording, observation's median was 0.183636 seconds, while
capture's median was 0.015495 seconds and action's maximum was 0.033666
seconds. The finer recording identified optional current-source geometry:
median 0.071685, p95 0.362145 and maximum 0.477594 seconds. Ordinary local
motion perception's median was 0.109923 seconds. Bar reading also spiked:
median 0.003784 and maximum 0.245547 seconds. Reading completed OCR words
was quick (maximum 0.006830 seconds). These are nested timings under load,
not independent costs to add together or native-app measurements.

The optional geometry stage could refit four poses in one frame, with up to
24 fine candidates. Three regressions failed on that four-pose call. It now
refits one pose per reference visit, with a separate rotating cursor for each
reference. The other pose hints remain available for later frames. A successful
search updates its own hint without erasing unsearched alternatives. Every
qualified current match is still checked for precision and can be published;
extra matches beyond the retained search hint remain explicitly unretained.
Unsearched alternatives are explicit and no result grants uniqueness, object
identity, task role or control authority. Pixel and precision gates are unchanged.

The existing fair-attention invariant and registered claim now cover the pose
budget and preservation of unsearched hints. The focused geometry group passed
38 checks. The first nearby-contract invocation had a nonexistent test filename
and ran no tests; the corrected invocations retain separate artifacts.

Timing artifacts: `artifacts/flash-campaign/2026-10-17-ordinary-stage-timing/`
and `artifacts/flash-campaign/2026-10-17-ordinary-substage-timing/`.
Repair artifacts: `artifacts/flash-campaign/2026-10-17-optional-geometry-budget/`.
Bar discovery remains the next measured latency issue. Gameplay carryover of
the pose budget has not been measured.

The nearby pose, input-deadline and recording contracts passed 55 checks.
Smoke (165 passed, one skipped), lint, compilation, layering, dependency
consistency and writing passed in 96.57 seconds.

## Ordinary follow-up at `7f5e22242`

The timed CPU campaign completed in 71.70 seconds. The report remained
unread, with no win; the independently viewed final image was active play.
Its complete record closed without drops: 168 images, 44 input-port events
and 124 control observations. Nineteen observations selected body 1. The first
selection had statistic 27.038360 and effect 26.948735, above the unchanged
20/20 thresholds, with two up and two neutral acceleration measurements.
The current image at that selection independently showed the selected box
on the vehicle at the left. No source name supplied this identity.

Fresh geometry's median/p95/maximum were 0.030637/0.106731/0.142820 seconds.
Total observation's p95/maximum were 0.275808/0.316813 seconds. These are
observed follow-up timings on a different actual game state, not paired
hardware measurements or a universal distribution. Source geometry counts
were 27 fresh, four object-bound and zero continuing.

The controlled track was then lost after a 1.25-second disappearance. After
identification it used measured up control, then a left calibration pulse
carried the body to the left edge. The record does not establish a successful
landing or complete steering model. Bar reading still reached 0.168235
seconds, so its scan remains actionable. Preserve current identity proof and
investigate calibration observability and later reacquisition without relaxing
identity requirements.

Artifacts: `artifacts/flash-campaign/2026-10-17-ordinary-pose-budget-timing/`.

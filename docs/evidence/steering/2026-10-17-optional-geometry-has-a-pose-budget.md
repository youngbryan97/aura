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

# Reserve observation time on the first measured interval

The original attempt at `53fc0d28e` retained one settled sample from its first
0.464213-second control pulse. Two capture intervals were already measured.
`observed_response_seconds` discarded both because it required three intervals,
so the ordinary actor used its 0.45-second default. This could not supply
four settled samples at that observed rate.

Three regressions failed before the repair. Two reconstruct a complete
acceleration measurement from one or two measured intervals. The third runs
the actual trial executor with two intervals and an acknowledged delivery;
the delivered deadline was only 0.45 seconds after that acknowledgement.

The existing reservation now uses the slowest valid recent interval as soon
as one exists. It reserves the same settlement and four-sample requirements,
inside the same finite maximum. This estimates time; it does not establish a
stable rate, measured response or controlled identity. No interval, invalid
intervals and an infeasible reservation still return unknown. A delivered
trial retains its deadline when later cadence changes. The existing registered
invariant now also checks one- and two-interval reservations, and the registered
cadence tests cover both cold start and longer histories.

The focused timing group passed 152 checks. Original-game carryover remains
unmeasured. Artifacts are under
`artifacts/flash-campaign/2026-10-17-first-trial-reservation/`.

Smoke (165 passed, one skipped), lint, compilation, layering, dependency
consistency and writing passed in 92.65 seconds. The protected native
checkout and runtime were not updated or restarted.

## Original follow-up at `a702a104e`

The CPU-only campaign completed in 133.09 seconds. Its recorded endings
were unread, unread and unread, with no win. The independently viewed final
image was the instructions screen. The complete recording closed without
drops: 336 images, 74 input-port events and 160 control observations. No body
was selected. Source geometry counts were 39 fresh, five object-bound and
zero continuing; the largest candidate statistic/effect was 2.449049/4.347407.

The first two thrust intervals now retained five and four settled samples,
and each supplied an acceleration measurement. The neutral intervals did
not. A 0.424726-second capture gap discarded all 28 prior point anchors.
The next picture was contradicted, followed by another 0.375659-second gap.
This prevented the independently repeated neutral comparison. The reservation
repair reached play but did not establish controlled identity or a landing.

The 40-frame pixel-only replay reproduced that anchor loss and contradiction.
It retained 343 measured, 87 unavailable and one contradicted observations.
It does not recreate the asynchronous reference worker or new actor decisions.

An additional serial local/separate/local comparison replayed those same 40
captures through ordinary asynchronous observation and source matching.
Total observation times were 2.914524, 2.849373 and 2.931003 seconds; p95
times were 0.112794, 0.099067 and 0.112558 seconds. Every frame published and
none took 0.35 seconds. The separate process's small advantage here does not
explain the much longer live capture intervals or justify a policy switch.
Its child was closed through the supported helper. No model backend loaded.

Artifacts: `artifacts/flash-campaign/2026-10-17-case18-first-reservation/`,
`artifacts/flash-campaign/2026-10-17-first-reservation-replay/` and
`artifacts/flash-campaign/2026-10-17-observation-process-comparison/`.

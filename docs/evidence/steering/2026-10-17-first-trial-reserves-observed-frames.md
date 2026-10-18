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

# Information in the observed control intervals

Date: 2026-10-18. Continues the failed original CPU run at `db593d175`.

## Falsified repairs

A 44-picture first-stream replay retained the original pixels and their
hash checks. At the first broken vehicle path, zero still-measured vehicle
landmarks were evicted by point capacity. Four owned landmarks fell to two
valid correspondences. The interval was 0.207845 seconds, inside the existing
0.35-second limit. Its two survivors disagreed: one moved approximately
0.286/-0.106 working pixels and the other 0.047/3.232. Capacity retention
would not supply the missing independent material evidence.

The current strict-mask admission rejected two possible new marks in the
preceding frame. Their distances to current support were 1.551856 and
0.782575 working pixels. No mask threshold was changed to admit them.

A dense fit of the reduced patch failed at both inspected breaks. Source
pixels gave a qualified 53-point fit at the later break, agreeing with its
two continuing names within 0.027149/0.033360 working pixels. At the first
break only three of 63 queries passed the existing forward/reverse tests;
no rigid fit qualified. A coarse initializer also failed. This is not a
verified material bridge, and none was promoted.

The existing response recommender found no feasible two-phase or four-phase
repeat in the replay. No new exploratory input or deadline was introduced.

OpenCV's documented sparse pyramidal LK uses a 21-pixel default window and
returns per-point status. [Official tracking API](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html).
A serial original/native/native-with-wider-context/original comparison kept
the existing colour and reverse-motion checks. All variants retained 370
measured and 70 unavailable object observations and selected no body.
Durations were 5.003659/3.831646/4.145347/4.019257 seconds. The repeat shows
that a cold first run affects this comparison. No native kernel was promoted;
the primary process also has an existing OpenCV import restriction described
in `core/media/safe_imports.py`.

The present state observer supplies no velocity-error estimate for the
first two settled thrust readings. Simply using absolute pose error and
capture bounds gives a 14.283695 acceleration-error estimate for that short
interval. It cannot be treated as exact state.

## A different measurement model

The raw velocity is an average over an observed displacement interval.
Dating it at that interval's midpoint, rather than its final picture,
corrects a cadence-dependent acceleration bias. Local rigid registration
also distinguishes displacement precision from absolute reference error.
Common reference uncertainty cancels under translation; under rotation its
contribution is bounded by the norm of the rotation minus identity. The
reconstruction includes that term, current-fit target precision, closure
residual and capture-time uncertainty. It uses a conservative working-pixel
resolution floor and never joins a missing interval.

A post-hoc reconstruction of the recorded independent holds found these
acceleration measurements, in working pixels per second squared:

| Input | Intervals | Vertical acceleration | Estimated error |
| --- | ---: | ---: | ---: |
| First thrust | 2 | -35.013122 | 15.235256 |
| First rest | 7 | 20.871459 | 5.467689 |
| Second thrust | 5 | -38.108912 | 7.017300 |
| Second rest | 6 | 21.768771 | 4.892053 |

Adding estimated measurement variance to the between-press noise gives a
candidate statistic of 38.402551. Subtracting the groups' mean estimated
error from their separation gives an effect of 20.793608 over the existing
0.5-second horizon. Both exceed 20. This is a different estimator, not an
execution of the current identity gate. Its errors are estimates, not
guaranteed physical bounds or calibrated probabilities.

The reconstruction neither changed Aura's identity nor replayed new actions
or gameplay outcomes. It suggests a precision-aware route around a fixed
four-velocity-picture proxy. That route must retain two independent physical
holds per compared input, current material correspondence, source custody,
contradiction retirement and the existing statistic/effect thresholds.
Unknown precision must remain unavailable. Synthetic negative controls,
irregular cadence, mixed origins, time-boundary cases and the actual recorded
pixel path must be tested before integration or further game claims.

Receipts are under `artifacts/flash-campaign/` with prefixes
`2026-10-18-point-capacity-replay`, `point-ownership-replay`,
`dense-material-probe`, `source-material-probe`, `source-material-stages`,
`source-material-initialization`, `response-opportunities`,
`native-point-comparison`, `velocity-precision-replay`,
`propagated-precision-replay` and `interval-information`. Each prefix after
the first also begins `2026-10-18-`.

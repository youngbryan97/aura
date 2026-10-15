# Facing trials keep the requested observation time

The original-case-9 run at 93c06efe1 completed normally in
166.403557084 seconds. It made 534 choices with median picture age
0.2469690903 seconds and did not win. Evidence is retained in
`/private/tmp/codex-steer/retained-heading-original9-1014`.

The pixel cohort completed all 24 scheduled presses and supplied 127
ordinary observations. No sampling region qualified. Later ordinary
observations selected bodies for 153 frames. Retained travel reached
the heading estimator: one historical `down` pace was kept, but the
repeated pace and turning measurements remained incomplete.

The expanded trace shows facing trials still being issued for 0.45
seconds at the observed slower cadence. `try_a_facing_control` asks the
cadence calculator for four settled observations, with a one-second
maximum. At a 0.21-second cadence, the request needs 1.04 seconds.
The calculator correctly returns unknown when this exceeds its cap,
but the caller then substitutes the shorter 0.45-second default.
That press can finish before two settled frames are captured.

The caller now gives facing observation the same finite four-times-base
budget as ordinary body trials: up to 1.8 seconds for the existing
0.45-second base. The separate maximum for inferring a stationary turn
from before/after headings remains one second. A longer observed press
can teach pace or a continuously observed travelling arc; it cannot
silently bypass the turn-ambiguity limit.

Four regressions exercise 0.21, 0.25, 0.30 and 0.35-second cadences
through the real receipt-owning delivery path. Their leased duration
must cover the requested samples and remain bounded. The focused
control, deadline, heading and calibration suite passed 99 tests in
11.30 seconds. A subsequent original-game run is still required.

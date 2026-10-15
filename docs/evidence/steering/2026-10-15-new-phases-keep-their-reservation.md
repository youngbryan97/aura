# A newly delivered phase keeps the current observation reservation

Fresh CPU play at `3b4b3e124` did not confirm a body and still lost. Its
visible terminal picture shows zero landings and one crash. The complete
recording contains 548 frames and 150 input events without drops. Raw
evidence is under `artifacts/flash-campaign/2026-10-15-case18-room/`.
The trial-priority mechanism requires an identified body, so this attempt
does not establish whether it improves original-game flight.

The retained body candidate had one usable thrust acceleration trial, then
two more thrust pulses with only three settled observations apiece. In the
second pulse, delivered at 3.020 seconds, valid observations arrived at
3.455, 3.578 and 3.712 seconds. Its release completed at 3.775 seconds.

The cadence helper was reserving time for slower recent frames. The
pulse/rest executor replaced that reservation with `counter.phase_seconds`,
which was frozen when the experiment first began. The reservation therefore
never reached subsequent physical input leases.

A newly delivered counter phase now retains the larger of its planned
minimum and the current observed reservation. Each already delivered
`ControlTrial` stays immutable. The counter programme's overall end remains
unchanged. Durations are validated before the next physical input.

The executor regression starts with a 0.734-second plan and an observed
0.245-second capture gap. The new phase receives the computed 1.18-second
reservation while the previous deadline and overall programme end stay
fixed. Four distinct late observations then support the acceleration fit;
the old request admitted three of them. This is a bounded constructed
timing test based on captured intervals, not a counterfactual game win.

Validation: 117 focused tests passed. Smoke passed 165 tests with one
skipped. Fresh original-game play after this change remains required.
G09, native Aura and protected holdouts were not changed or launched.

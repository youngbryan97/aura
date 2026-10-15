# Calibration can wait for its actual samples within a hard end

The fresh CPU case-18 run at `d0decf11c` selected its body after 5.3 seconds.
It still ended with zero landings and one crash. Its complete recording
contains 158 frames and 68 input events, without drops. The raw journal and
frames are in `artifacts/flash-campaign/2026-10-15-case18-phase-reservation/`.

The first right pulse produced three settled velocity observations; the
second produced five. Both released at their estimated observation times.
Only one pulse therefore qualified for the acceleration model, which
requires two. Pixel replay on the first 156 frames, before pointer input,
retains the own-body histories in `own-control-samples.json`. It is a
reconstruction on recorded actions, not a changed-policy game result.

An identified inertial body's finite trial now separates an expected
observation time from its hard physical end. Its observed history must
contain four distinct settled finite samples spanning at least 0.15 seconds
before acquisition ends early. The current body owner, physical input time
and experiment epoch must match. It then physically releases the input and
keeps a finite neutral observation phase. That phase can finish when its
own required samples arrive.

The hard end is set before observation to the existing maximum trial
allowance, 1.8 seconds with the actor's normal control-trial configuration.
New pictures cannot renew it. Its watchdog still releases the input when
no more pictures arrive. Attempt allowances and control certification
requirements are unchanged. Unknown-body discovery retains its existing
experiment schedule; the acquisition mode applies to an already identified
body and records feedback-chosen inputs as nonindependent.

The executor test supplies three actual model observations after a late
settlement. It verifies that the expected time passing does not release
the key, then supplies the fourth and observes the physical release through
the real control-delivery wrapper. The neutral phase also completes from
observed samples. A real asynchronous lease test verifies physical release
at the hard deadline without a new picture and preserves the separate
expected duration on the rest.

Validation: 155 focused tests passed. Smoke passed 165 tests with one
skipped. Ruff, layering and dependency checks passed. Original-game play
after this change and ordinary native-app proof remain required.

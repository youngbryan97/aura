# Source points follow current owned material, 16 October 2026

The calibrated reference point can now attach to an existing material track
through `SourceImageCoordinates.on_track`. The matcher must own at least six
reference correspondences and 70 percent of their visible material. Another
track claiming any of the accepted material refuses attachment. A current
named-point basis and an estimated precision within one working pixel are
also required. The estimate includes registration residual and the point's
lever arm; it is not a bound on correlated tracking errors.

The source point is expressed in the track's existing material basis. It
does not replace `PointPose.origin`, change the learned motion coordinates,
select the controlled body or supply a task role. A visual kind change can
retain geometry only while the same observer epoch, instance birth and
measured material basis continue. A replacement basis, stale track or absent
picture cannot inherit it. `inspect` reports the failed connection explicitly.

The ordinary `WhatMoves.see` path now keeps a small immutable receipt for its
consumed RGB array: capture time, source shape and pixel digest. The receipt
survives worker publication and the separate observation process. Initial
reference attachment requires these exact pixels, rather than a timestamp
alone. Later projection still requires a consumed current picture and the
current material basis. Unsupported arrays supply no receipt.

The first turning-coordinate test exposed valid NumPy coordinates being
rejected by the instance-query helper. Finite real scalar values are now
canonicalized to Python floats; booleans, strings and nonfinite values remain
invalid. Tests also check the instance-name consumer with NumPy values.

`tools/measure_source_points_on_recorded_material.py` validates a source
reference bundle and replays recorded RGB frames through the ordinary
tracker. It can then follow a bound point without reattaching it, while
independent reference matches measure its error. It checks declaration and
coordinate namespaces, image inventory and byte identities. A finished
replay is a measurement, never a gameplay pass.

On original case 18's already recorded pixels, a 20-frame prefix ending at
sequence 304 attached source point (0, 0) to track 1. All seven reference
correspondences belonged to that track, whose continuing basis had 13 marks.
The point was (207.381876, 36.249549) in working coordinates, with estimated
error 0.378149 working pixels. The track's ordinary box centre was
(207.5, 33); its motion origin was left unchanged.

The following 20 recorded frames produced 16 current source points. Four
were refused: one replacement point basis, one unavailable basis, one stale
track and one missing track. Nine frames also yielded an independent fresh
reference appearance; all nine agreed within one working pixel. The largest
observed difference was 0.216533 working pixels. Missing independent
measurements were excluded. The replay took 7.996223 seconds; it did not run
the game, send input or load a model. The receipts are under
`artifacts/flash-campaign/2026-10-16-source-point-material/`, especially
`custodied-follow.json`.

Focused checks passed 241 tests in 34.16 seconds. They cover attachment,
precision, stale evidence, rotation, the ordinary point-continuation helper,
consumed pixels, separate-process parity, publication deadlines, source
receipt corruption, and existing naming/target contracts. Smoke passed 165,
one skipped, in 48.31 seconds. Lint, compile, layering, DEPS consistency and
writing passed. The registered claims are
`source_points_need_their_current_material_basis` and
`observers_keep_their_consumed_pixel_receipt`. The previously recorded 129
governance regressions were not changed.

Ordinary runtime asset acquisition, delayed matcher delivery, destination
occurrence binding and source-derived contact requirements remain open.
There is no new original-game win or native-app presentation qualification.
G09's exclusive general-reasoning job remains active and untouched; the
resident checkout/runtime, original game 9 and holdouts were not operated.

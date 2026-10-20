# A capture clock can reach instance naming

## Ordinary follow-up at pushed 3830ca8e4

The original-case-18 browser run used the host renderer after process and
memory checks found no active model job. The browser reports ANGLE Metal
on Apple M5 Pro. Tracking and OCR remained CPU-only; the Python probe denied
model backend imports and model-worker starts. G09 and holdouts were excluded.
This is an ordinary browser control-path diagnostic, not native-app proof.

The run finished in 131.80 seconds. It reported one lost round and one unread
round, with no win. The first selected frame, `frame-0000120.png`, was viewed
independently: its selected box is on the vehicle. There are 171 selected
control observations out of 583 recorded observations. All three movement
keys were known on 188 observations. Observation p95/max were
0.092043/0.123722 seconds. This run changes renderer and host contention from
the earlier CPU-only runs; those timings are not a paired effect estimate.

The viewed later `frame-0001315.png` shows zero landings and one crash. All
six recorded planning choices contain the requested relation, on the landing
platform, but none binds its referent. All six choose `wait`. This locates
an open connection after control identification. It does not establish that
naming is the only remaining problem.

The recording closed with 250 dropped events, so the complete run is not
eligible for contiguous control replay. Its manifest records 781 offered
frames and 358 offered input events; the retained log contains 705 frame
events and 344 input events. It is partial evidence. The campaign report
and private stage timings are under
`artifacts/flash-campaign/2026-10-20-ordinary-interval-host/`.

## Reproduced gate failure

`LookingAtThings.look` used `type(at) in (int, float)`. The ordinary loop
passes a `CanvasFrameTime`, a float subclass with immutable capture bounds.
That check silently rejected it before starting any instance query. The
downstream `query_for` already accepts real numeric capture coordinates and
canonicalizes them while preserving current box, birth and instance identity.

The new regression tests reproduced five failures: measured capture clocks
and NumPy real timestamps were rejected, a negative timestamp aged the
observer, and an oversized integer raised during conversion. The correction
canonicalizes finite nonnegative real capture coordinates before the existing
scheduling gate. Booleans, complex numbers, strings, missing/nonfinite values
and overflowing integers still refuse before changing state.

The naming and destination integration group passed 101 tests in 7.04 seconds
(7.61 supervised). The provider in these tests is stood in for; this establishes
that capture timestamps reach the existing request and custody code, not
that a vision model correctly names the original destination. No provider
prompt, identity gate or task-specific rule was changed.

Next: finish checkpoint gates, then use the existing small vision worker in
one bounded ordinary run. Its admission must share the host model-lane
ledger, and only the worker started by that diagnostic may be shut down.
No game win, campaign pass or native demo readiness is established.

Recording-count clarification: the manifest's 781 frames count all saved
picture payloads, including pointer and screen-option pictures. They are
not offered motor frames. Its 358 input events include 344 input events and
14 navigation-input events. The 705 retained events of kind `frame` are the
motor captures. The incomplete-recording conclusion is unchanged.

Checkpoint gates passed in 76.86 seconds: smoke (164 passed, one skipped),
lint, compile, layering, generated DEPS and writing. No baseline changed.

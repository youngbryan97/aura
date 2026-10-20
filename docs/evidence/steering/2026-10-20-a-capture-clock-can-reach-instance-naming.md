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

## Vision admission exposed two installation boundaries

The first owned-vision attempt ended in 1.20 seconds before a model worker
started or gameplay began. The non-live state guard refused a write to the
resident model ledger. It was left intact; no broad live-state override was
set. The next diagnostic uses the supported private development ledger,
with process discovery and current host resource checks. It cannot claim
shared fencing against an unrelated job that starts later.

The registry also resolved the default vision repository identifier to a
missing installation directory, although its snapshot exists in the local
Hugging Face cache. The serving resolver now checks that cache after missing
its local inventory. It uses the installed library's read-only cache lookup,
returns an existing local snapshot directory, and never downloads a missing
one. Explicit inventory locations keep priority. A cache hit confers neither
serving-role authority nor proof of complete weights; normal artifact and
admission checks still apply. This applies to local repository-named models
in general, with no game or character rule.

The cache and existing registry group passed 54 tests in 4.03 seconds
(4.66 supervised), including missing, dangling and negative cache entries.
The default vision identifier now resolves to snapshot
`2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b`, whose existing safetensors occupy
3,093,767,283 bytes. No weights were copied, changed or fetched.

The cache checkpoint passed smoke (164 passed, one skipped), lint, compile,
layering, generated DEPS and writing in 76.29 seconds. The final registered
cache canary and its tests passed 10 checks in 2.51 seconds.

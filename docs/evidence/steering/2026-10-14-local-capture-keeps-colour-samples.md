# Local capture keeps colour samples

The original-case-9 run at b6992dde2 completed normally in 162.376312
seconds. It made 475 choices, selected no body and did not win. Median
picture age was 0.257326 seconds. Its recorder captured every ordinary
vision-worker input, including pictures not published to candidate callbacks.
The first 250 of those inputs replayed with 250 exact entity snapshots and
no missing RGB digests. This establishes replay fidelity for that prefix.
Evidence: `/private/tmp/codex-steer/complete-worker-original9-1014` and
`/private/tmp/codex-steer/actual-worker-fidelity-1014`.

At source time 535509.6395881176, the retained object's point path ended.
The forward registration found the displacement, and reverse registration
returned close to the old points. Three marks failed the 0.025 colour-error
limit: their forward errors were 0.030835, 0.025857 and 0.030615. Two other
marks passed, which was insufficient for a rigid pose. The earlier saved
support was 0.330863 seconds old, within the existing 0.35-second horizon.
This trace identifies the rejecting gate. It does not isolate compression
from rendering, resampling or other causes of the colour difference.

Both local capture routes encoded JPEG at quality 80 before the point
tracker saw the pixels. Chromium's public protocol supports PNG for
`Page.startScreencast`; its documented formats are `jpeg` and `png`:
[official protocol](https://github.com/ChromeDevTools/devtools-protocol/blob/master/json/browser_protocol.json).
A bounded encoding ablation changed only the stream format and direct
canvas data URL to PNG. Decisions, point limits and attribution gates stayed
on b6992dde2. The native payload's PNG signature was verified.

That run completed normally in 175.353931 seconds without a win. It made
786 choices with median picture age 0.257515 seconds. Body 15 was selected
for 60 decisions over 7.170043 seconds; body 62 for 621 decisions over
97.532227 seconds. The first accepted body's statistic was 24.464177 with
effect 34.323108. Its captured position lies on the visible green vehicle
containing the rider. Five historical forward pace measurements reached
heading calibration, but no reverse pace or turning response qualified.
Evidence: `/private/tmp/codex-steer/lossless-worker-original9-1014`, including
the independently viewed native capture `owned-stream-frames/frame-0447.jpg`
(the retained payload is PNG despite that diagnostic filename).

The two runs have different input schedules and trajectories. Their results
do not prove that compression caused every correspondence failure, or that
PNG alone establishes a general gameplay improvement. They support testing
lossless local capture without weakening correspondence or control gates.
Local disk retention reached its 256-MiB bound in the PNG ablation. A later
diagnostic recorder deduplicates identical encoded payloads while retaining
each capture's separate receipt; the old evidence remains unchanged.

Production stream capture and direct canvas capture now request PNG. The
stream still uses the page's CSS extent, current clock and adaptive stride.
Browser rendering and the tracker's later size reduction remain separate
sources of sampling error. The decode path still accepts existing JPEG
receipts. No new model or game-specific code is involved.

Real Chromium tests draw alternating fine colour marks and compare every
captured RGB sample with the drawn values, through both production capture
routes. These tests and capture-ownership/time regressions passed 41 tests
in 7.91 seconds. A subsequent ordinary actor run must verify the committed
path. Native Aura prompt proof and a game win remain open.

The PNG replay matched its first 244 worker inputs exactly. Input 245 has no
retained payload because the diagnostic disk bound had been reached; the
replay stopped rather than supplying guessed pixels. Ruff, writing,
changed-method size, layering, dependency graph, compile and lint checks
passed. Smoke passed 165 tests with one skip. The known module-size debt and
123 governance regressions remained unchanged.

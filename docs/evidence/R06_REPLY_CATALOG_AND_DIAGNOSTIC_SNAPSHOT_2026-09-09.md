# R06 reply catalog and diagnostic snapshot

## Observed answer-path stall

Live turn `aura-chat-47fb6801-06f7-4e7b-90c0-5167f5a670f3` finished model
generation at 17:50:29. Reply stabilization later measured 51006.04 ms.
The runtime traceback in `data/error_logs/stalls/stall_1789001530.txt`
shows `_claims_a_capability_it_does_not_have` calling `build_skill_catalog`
from synchronous reply validation, walking Python source files on the event
loop. This happened for a database explanation that made no tool-use claim.

The capability check now extracts relevant claims before reading capabilities.
When needed, it asks the existing live CapabilityEngine for a locked snapshot
of published skill names. That API does not initialize, discover, or validate
skills. An unavailable catalog retains the existing unknown/non-contradiction
semantics. Offline checks outside an event loop can still inspect source.
There is no new cached copy with a separate invalidation policy: subsequent
reads see the engine's current registry.

Three regression tests failed before repair: ordinary answers requested the
catalog, live claims rebuilt source instead of reading the registry, and an
on-loop missing registry triggered discovery. Twenty-five compatibility and
lock-order tests passed after repair. A further production-engine snapshot
test verifies cold reads do not load and subsequent publications are visible;
the final snapshot/claim selection passed 16 tests in 10.65 seconds.

## Diagnostic report consistency

The periodic OOM report called each non-immune organ's footprint probe twice:
once for its score and again for its displayed footprint. A changing footprint
could therefore contradict its score, while expensive probes did duplicate
work. The report now samples once and passes that observation to the unchanged
scoring formula. Victim selection semantics and immunity are unchanged.

The changing-footprint regression failed with two observations before repair.
The kernel, orchestration, and MLX memory safety suites passed 162 tests in
68.31 seconds after repair. This removes one measured source of repeated work;
it does not establish the cause of every diagnostic overrun.

Deployment and comparable live latency measurement remain required. R06 and
R11 are not closed by these repairs.

Checkpoint gates: smoke 164 passed, one skipped, 139.47 seconds; lint,
compilation, governance, and layering passed. Layering baseline remains 37.

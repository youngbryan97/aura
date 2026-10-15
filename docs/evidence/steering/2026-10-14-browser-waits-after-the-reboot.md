# Browser waits after the reboot

The first original-case-9 run at c293fa586 stopped producing ordinary
actor returns after 171 decisions. No body had been selected. The pixel
search retained eight accepted presses, then reported a capture or
delivery timeout with release confirmed. Its last vision-worker record
arrived later. The outer supervisor terminated the test at 210.073984
seconds. There was no win or presentation proof.

A second run added coroutine and thread snapshots. It stalled before
gameplay, during navigation. An OS sample of that run's browser GPU
process showed its Metal command-queue thread waiting inside
`IOGPUCommandQueueSubmitCommandBuffers`. The resident G09 process was
still running. This identifies an observed browser wait; it does not
establish that G09 caused the stall or the subsequent machine crash.
The diagnostic was stopped after 63.786640 seconds without changing
the model process.

Bryan reported the crash. The next process check showed a fresh boot
and no surviving Aura or model job. Git retained c293fa586 on main.
The reboot removed `/private/tmp/codex-steer`, including those raw
captures, stack samples and diagnostic scripts. Their measurements
above are the retained session record. The missing artifacts cannot
be replayed or independently rechecked. Future run evidence belongs
under `~/.aura/flash-campaign/evidence/`, outside temporary storage.

Several browser observation operations had no caller deadline:
stream setup, acknowledgements, resampling, stop and detach, canvas
evaluation, and both screenshot routes. Each now uses a one-second
request budget. The existing failure paths return no picture or close
the stream. A timeout cannot renew a capture clock. Caller cancellation
still propagates. Browser clock calibration retains its existing
separate bounded queries.

The new tests block each startup stage, stream resampling, stop and
detach, an acknowledgement, direct canvas evaluation and both screenshot
paths. They require cancellation and an unavailable result while another
loop task still runs. The focused capture set passed 75 tests in 10.48
seconds. Smoke passed 165 tests with one skip in 41.39 seconds. Ruff,
compile, layering, dependency graph, writing and changed-method checks
passed after declaring the new pure validation dependency.

These changes bound browser observation waits. They do not repair a
kernel or GPU fault, prove physical release during a browser failure,
or complete the original game. Claude's resumed G09 tests remain outside
this work's process ownership.

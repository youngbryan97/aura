# Bar discovery continues between pictures

The ordinary timed run at `7f5e22242` found its controlled body, then lost it.
Bar reading still reached 0.168235 seconds in that run. An earlier timed run
had reached 0.245547 seconds. The expensive part periodically searched for
new strips; established-strip sampling was cheap.

Two smaller optimizations were tested and rejected. Batching row-join colour
arithmetic preserved 300 random row comparisons and the five captured scenes,
but saved only about six percent in the paired full scans. Moving existing
edge and duplicate checks earlier also preserved all 17 captured/generated
scene outputs, but gave no overall speed gain (1.320603/1.415715/1.411800
seconds before/prototype/before). Neither prototype remains in production.
The pruning fixture initially exceeded the existing thickness limit; its
corrected red run measured 15 repeated colour scans for one accepted strip.

The retained change separates discovery work into finite steps. One owned
capture supplies at most eight sampled rows per incoming picture. The next
picture continues that same scan, rather than restarting or completing a
whole-frame search inline. Established strips are still sampled every picture.
Shape, source-surface, capture-epoch or clock changes retire pending work.
Only one capture is retained by a reader; there is no worker, model or queue.

A discovered candidate is a search hint. Before admission, its colour, current
run and complete thickness predicates are measured again on the current
picture. The old candidate cannot supply a current level, change or reward.
The original full-scan adapter remains available and visits a finite number
of steps. Reward and resource authority gates remain unchanged.

Nine cold-path regressions failed before implementation, while established
bar sampling already passed. The focused integration group then passed 56
checks, including changing sources, dropped candidates and reward authority.
The registered invariant covers the row allowance and current-pixel admission.

The complete scanner's outputs matched the previous implementation exactly on
17 captured/generated pictures. A serial comparison took 0.719879/0.701163/
0.656579 seconds for previous/refactored/previous complete scans. This change
does not claim less total work. Across those pictures, 180 discovery slices
had a maximum of 0.007823 seconds, and 340 ordinary reads had a maximum of
0.012662 seconds. Original-game carryover is still unmeasured.

The comparison driver's retained mode label `pruned` denotes the refactored
complete adapter in this later run; the rejected predicate-pruning prototype
had already been removed. Two additional current-shape refusal cases passed
with the 12-test focused file. Smoke (165 passed, one skipped), lint and
compilation passed. Layering then refused the new canary's unlisted import;
the manual organism DEPS now names that pure meter module explicitly. No broad
perception exception was added. The remaining gates must be rerun.

After the narrow DEPS entry, layering, dependency consistency and writing
passed in 37.59 seconds. Combined with the preceding smoke, lint and
compilation results, the main code gates pass at this revision.

Governance still reports 130 regressions in 9.64 seconds, the same recorded
count as before these changes. Its baseline was not changed. The full-scan
comparison used the scanner from `2a127c0df` as its previous implementation.

Artifacts: `artifacts/flash-campaign/2026-10-17-bar-join-batch/`,
`artifacts/flash-campaign/2026-10-17-bar-discovery-pruning/` and
`artifacts/flash-campaign/2026-10-17-incremental-bar-discovery/`.

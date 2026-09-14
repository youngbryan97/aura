# G10: public calibration and owned lock recursion

The lower-strength development calibration completed on frozen commit
`d69b8ba7a`. At alpha 0.05, all six baseline and six treatment samples produced
nonempty public answers, closed their private channel and stopped at EOS.
The maximum sample used 679 of 1,024 allocated tokens. Decode time was
561.717 seconds. The result explicitly sets qualification to false.

The receipt and public answers remain in
`~/.aura/live-source/.claude/worktrees/codex-steering-public-low-20260914/artifacts/migration/27b/recovery/public-calibration-low-20260914/`.
The supervisor record is in
`~/.aura/experiments/caa-public-calibration-low-20260914/`.
These are exposed development prompts. Completion is not a causal steering
result, a correctness result, or fresh generalization evidence.

Reading the output also exposed lock-order warnings. The steering owner takes
the layer locks in order, then the generation thread re-enters those RLocks.
Lockdep incorrectly recorded each re-entry as a new wait dependency on an
inner lock. This invented reverse edges although the thread already owned the
lock. The earlier ownership tests checked restoration but not the warning
stream, so they missed this defect.

Lockdep now retains recursive acquisition depth without adding order edges
for a reentrant lock already held by the current context and thread. Real
first acquisitions still receive rank and cycle checks. Non-reentrant
self-deadlock, another thread's acquisition, and a later reversed acquisition
still report. No lock names are exempted and no warning channel is disabled.

The reproducer failed three tests before the repair: ranked recursion,
unranked recursion, and the real multi-hook measurement path. The repaired
combined suite passed 52 tests, including reporting reentrancy, kernel lock
discipline, measurement restoration and the replaceable-clock regression.
G10 remains open pending current-basis causal qualification and serving proof.

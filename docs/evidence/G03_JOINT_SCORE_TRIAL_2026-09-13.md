# Joint operation and argument scoring: development trial

The opt-in joint selector compares complete graphs using operation evidence,
argument evidence and the existing length penalty. The default still selects
the first feasible operation chart. No learned coefficients changed.

The unbounded joint trial completed one exposed weave split at 23/24 answers;
the immediate parent had 24/24. Its full weave run and the full-source run were
stopped without completed receipts after the observed regression and excessive
development runtime. A subsequent bounded full-weave run also did not complete.
These are incomplete trials, not full-cohort scores. No control run or serving
promotion is claimed.

Two mechanical changes remain useful independently of that selection policy:

- An optimistic argument-score bound avoids solving a chart that cannot beat
  the incumbent. It relaxes overlap and consistency constraints and counts all
  potentially positive definition evidence. Tests compare it with the optimum.
- A decode-local cache reuses identical reference/definition scalar scores
  across competing charts. It is discarded before another request or model.

The bounded two-case canary produces the same two correct programs with and
without the cache (10.33 and 6.44 seconds). This is a small development timing,
not a throughput claim. The default policy with the cache retains 47/48 answers
and all 48 per-case outcome rows are identical to the immediate parent's
`semantic_feasible_operation_dev_20260913/weave.json`. Its elapsed time was
97.12 seconds; run conditions do not justify attributing a speedup from that
comparison alone.

Evidence is in `artifacts/rlc/semantic_joint_score_dev_20260913/`. Its receipt
records each incomplete trial explicitly and hashes the completed reports.
The joint candidate's receipt is
`a5f434650e51a3359134fb3162066a5a9fa643b8c9fc6e4d52d16b3bc8f04c00`.
Source development, fresh transfer, public answers and current-model serving
qualification remain separate obligations. G03 remains open.

Focused tests: 50 passed. Lint, compile, governance and layering passed. Smoke
reported 163 passed, one skipped and the existing current-manifest activation
alarm. That alarm is not suppressed by this work.

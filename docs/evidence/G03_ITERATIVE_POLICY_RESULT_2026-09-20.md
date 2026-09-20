# Re-mining the actual decoder's competitor

The source-only scan found one incorrect program in 128 training examples.
The first policy-aligned fit satisfied its stored inequalities but did not
repair the autonomous decode. The error exchanged the arguments of a
subtraction following integer division. The selected operation chart and its
primary score were already correct.

Re-mining after the first update found a new argument binding for the same
incorrect program. Its margin was -0.142942 instead of the initial -3.830401.
The second update corrected the selected program. The third mining round
found all eight selected training programs equivalent and made no coefficient
change. This is why one frozen set of latent binding inequalities did not
establish correctness of the decoder after fitting.

| Cohort | Before | After | Gains | Regressions |
| --- | --- | --- | --- | --- |
| Selected training | 7/8 | 8/8 | 1 | 0 |
| Exposed validation | 99/100 | 99/100 | 0 | 0 |

The candidate retains the incumbent `first_feasible_v1` policy. Fitting used
the eight selected training examples, operation and boundary retention on 128
training examples, and no validation labels. All 5,323 final stored
inequalities passed. That count is not exhaustive search or transfer proof.

Before-validation observations were reused from the preceding immutable
source-error trial. The current core identity includes the closed procedure
type changes; the earlier trial did not. Those changes do not participate in
this decoder, but this reused comparison is not a same-snapshot fresh paired
run. After observations and both training observations were executed here.
Full comparisons must bind one source snapshot and replay both candidates.

- Iterative receipt:
  `fe5719bbcd9487a977e647177fb52c8bbb15364682cdcccdfba422c919f18b99`.
- Artifact: `/Users/bryan/.aura/rlc-evidence/semantic-policy-active-iterative-20260920.json`.
- Candidate: the adjacent `.candidate.json` file, with no serving authority.
- Elapsed: 372.685 seconds including publication.
- Prior single-fit receipt:
  `e78ece359b968b33b0dc3a06eeb6d06c0df8a63aff4ff5356aeca713851c1999`.

G03 remains open. A source-only error scan across the full training population
and broader independent replay follow this small mechanism check. No evidence
here establishes fresh transfer, broad reasoning gain, fusion or frontier
performance.

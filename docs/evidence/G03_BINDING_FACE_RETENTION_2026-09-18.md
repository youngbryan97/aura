# Binding faces and retained margins

The full `39d7cd0a4` fit mined all 764 source-training examples. Its first round
retained 28,143 comparisons, including 1,363 wrong or tied comparisons. The
optimizer refined its projection but accepted zero updates and reported
`no_retention_preserving_step_found`.

The run was stopped through its authenticated supervisor after that result.
With no coefficient change, later rounds would repeat the same competitor
search. Its log and numerical checkpoint remain under
`~/.aura/rlc-evidence/g03-projection-full-run-20260918` and
`~/.aura/rlc-evidence/semantic-projection-full-20260918.fit-checkpoints`.
No final candidate, validation result, or successful campaign is claimed.

## Reproduced mechanism

The initial projection treated the nearest retained margins as binding faces
even when their current values exceeded their floors. It therefore forbade
some harmless decreases, rather than forbidding only floor crossings.

For example, let the margins be `x` and `0.5 - x`, with initial `x = 1.1` and
required margin `0.1`. The first margin has a floor of `0.1`; the second is
wrong. Every `x` between `0.1` and `0.4` satisfies both comparisons. Requiring
the first margin's derivative to remain nonnegative prevents reaching any of
those points. Both batched and scalar optimizer tests reproduced this stall.

The initial projection now uses only faces at their floor. Trial updates must
still satisfy every retained nonlinear margin after float32 storage. Rejected
updates can still contribute additional blocking faces. Neither the acceptance
test nor the retained floors were weakened. The policy identity advances to
`binding_face_deficit_backtracking_v3` (and corresponding fixed/logistic modes).

The two reproductions now pass, as do the existing incompatible-objective and
omitted-face tests. A 109-test combined pass includes these checks, argument
learning, and the independent public-answer measurement fixes. A small
source-bound trial is required before another full fit; no transfer gain is
established by the numerical example.

# G03 source-fold misses: reach, interruption, and selection are separate

`tools/profile_semantic_crossfit_misses.py` reads the three archived
construction-held proposer reports, checks the signed plan, report, row
receipts, disjoint fit/calibration/held identities, and shared source basis.
It does not load a model or change a serving decision. The immutable profile
is `~/.aura/rlc-evidence/semantic-crossfit-miss-profile-v3-20260925/report.json`.
These are 764 source-development examples, not an untouched family test.

| Outcome at bounded 4x2, one-second acquisition | Cases |
| --- | ---: |
| Public selected program independently equivalent | 496 |
| Correct candidate observed, public selection interrupted | 58 |
| Correct candidate observed, wrong public program selected | 75 |
| No correct candidate observed; search incomplete | 135 |

The 75 wrong selections divide into 32 with a different operation multiset
from any observed correct candidate and 43 with the same operations but wrong
structure or arguments. This is a coarse mechanism stratum, not a diagnosis
of the intended natural-language meaning. Correct program signatures vary
across supported arithmetic and collection operations; no single missing
operator or domain explains the failures.

Fold 0 had 51 reach-unobserved and 26 wrong selections, but no interrupted
public selection. Fold 1 had five reach-unobserved, 36 interrupted, and 23
wrong selections. Fold 2 had 79 reach-unobserved, 22 interrupted, and 26
wrong selections. The distribution is heterogeneous, so the next training
run must report paired strata rather than an aggregate alone.

On these same source rows, choosing the highest scored diagnostic candidate
would gain 75 over ordinary selection and lose four ordinary successes;
all 58 interrupted cases have a correct highest-scored candidate. Those
labels were read only after acquisition. The result is diagnostic, not a
target-blind serving policy or a guarantee that the pattern transfers.
Increasing time alone cannot establish the 135 are reachable, nor can
ranker training fix a public optimizer that returns no program. G03 remains
open pending a mechanism that preserves incumbent successes and improves
independent held-out selection/reach under a declared compute budget.

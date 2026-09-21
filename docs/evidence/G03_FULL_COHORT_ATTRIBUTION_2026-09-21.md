# Full development cohort attribution

The frozen conditional-retention candidate was evaluated on all 764 training
and 500 exposed validation observations. No test observations were used.

| Split | Semantically equivalent | Reachable alternative not selected | Search unresolved | Total |
|---|---:|---:|---:|---:|
| Train | 760 | 4 | 0 | 764 |
| Validation | 477 | 14 | 9 | 500 |

These are semantic program measurements, not successful public answer counts.
The candidate remains unqualified. All four training errors share the
`arithmetic:nominal_nested` construction; the validation misses span arithmetic,
cataphoric sequence, reserved-alias sequence and role-binding constructions.
The construction names are diagnostic labels, never runtime routing inputs.

The 1,264-row receipt is
`2b4277f6afedf2c5b9ad743c8a88fd751a313f8f882d07e5a35b181f0dc2aab4`.
Candidate: `9da7500600578d9053bc75681deb611de22bdcbd874720bac6890635538bf6bf`.
Artifact: `~/.aura/rlc-evidence/semantic-retention-cohort-audit-20260920/report.json`.
The hour-bounded process stopped with SIGALRM; the next supervised process
verified and reused its immutable row receipts, completed the remaining rows,
and exited successfully. It ran on frozen `d9a645672`, not the later staged
diagnostic optimization. No historical receipt was relabeled.

## Search and scoring are distinct failures

The two-bank/two-scorer replay completed three selected observations. The two
training errors have identical observed program/span sets under both scorers.
The parent chooses an equivalent program; the fitted scorer chooses a different
one on either frozen set. That is a scoring/selection-policy effect, not a
missing candidate on those observed banks.

For validation source `170dac1d065919458053643ba3cac7a7013cd3e445eb2736a520887987fb9c63`,
the fitted scorer chooses the correct program on the parent's bank. Its score
is -12.0146093344, above its own ordinary wrong answer's -13.2359613345. The
fitted search's top-one operation label filter excluded the needed `count_of`
at the source span. The two small banks alone did not prove the full grammar
unreachable; the larger audit also found a semantically equivalent alternative.

Replay receipt: `e5c8e66a03d36483aa5ddc227df719bf27bd0a530a22b196f040d9d15e8a4dfe`.
Artifact: `~/.aura/rlc-evidence/semantic-fixed-bank-replay-20260921/report.json`.

An additional target-blind intervention used the existing typed-chart and
label-alternative policies without changing any coefficient. Typed charts
alone did not recover the missing operation. Two labels per span and all
learned labels both recovered this validation answer. Neither corrected the
two ranking failures. This is a three-case mechanism test, not a full-cohort
improvement claim or permission to promote either search policy.

Intervention receipt: `1a0f1cc48781847ae5edf14f0a0fbf053d54b7da9e308a933e75866f3c978fbc`.
Artifact: `~/.aura/rlc-evidence/semantic-search-intervention-20260921/report.json`.

## Complete-decision retention

The prior auxiliary source-retention path preserved operation-label and pointer
constraints on all 764 training rows, but mined complete graph constraints only
on its active subset. Joint graph ranking can change while these auxiliary
constraints remain satisfied. Four training regressions survived this boundary.

The existing joint trainer now has opt-in `source_graph_retention` (CLI
`--replay-source-graphs`). It replays ordinary decode across the entire declared
source-training population before fitting and after coefficient updates.
Independently witnessed wrong programs become retained constraints for the next
round, including rows outside active mining. Unknown or timed-out observations
stay unresolved. The last replay reports actual complete-decision status even
when the final allowed round introduces a regression. Validation and sealed
test examples are refused by this learning path.

This is counterexample-guided retention over an observed training population,
not a proof of non-regression on unknown tasks. Round coefficients are saved
before the replay so a long audit cannot erase a completed fit. All public,
fresh-transfer and frontier obligations remain separate. G03 is not closed.

Focused replay, acquisition, joint-learning, policy and normalization suites:
77 passed in 33.86 seconds. Smoke: 164 passed, one skipped in 58.99 seconds.
Lint, compile, governance-lint, layering and writing passed their configured
gates. Tests include a regression outside active mining entering the next fit,
unresolved decode remaining unresolved, held-out exclusion and CLI wiring.

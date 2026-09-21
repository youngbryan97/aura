# Fixed-bank semantic replay

Status: diagnostic implementation; no candidate promotion or G closure.

Review obligations Q03/Q04/Q08/Q11 in
[the six-report disposition](../G_LEDGER_REVIEW_DISPOSITION_2026-09-21.md)
require separating ranking changes from changed search hypotheses.

## Source-attribution repair

`argument_graph_program` topologically normalizes a source-ordered operation
chart into an execution-ordered SSA program. Candidate banks previously paired
that program with operation spans still in chart order. They were adequate for
executing/comparing programs but not for reconstructing their source-grounded
scores when a forward reference changed operation order.

The shared `argument_graph_order` now supplies both permutations. Candidate-bank
schema v2 explicitly stores execution-ordered spans. The new replay refuses v1
instead of guessing how historical spans were ordered. Ordinary model decoding
and coefficients are unchanged.

## Two-by-two diagnostic

`semantic_bank_replay.py` builds parent and candidate banks independently,
without source labels. Each frozen bank is then scored by both models through
`score_annotated_graph`, which reuses the deployed argument chart and latent
mention/definition optimization. Semantic annotation is applied only after both
banks exist. A replay record binds bank, scorer, observation, basis, search
allowance and score decomposition. Missing or unfinished scoring does not
declare a winner.

`tools/replay_semantic_candidate_banks.py` loads the existing feature bundles,
accepts explicit development source identities, rejects sealed test rows,
writes immutable resumable per-row records, and binds the implementation.
This distinguishes fixed-bank rank changes from differing observed candidate
sets. It does not prove either bank exhausts the grammar. The comparison keeps
the incumbent model's actual first-feasible or joint-score policy; it does not
replace it with a convenient offline sum.

## Checks

112 focused tests passed in 36.52 seconds across bank replay, candidate banks,
counterexamples, cohort diagnosis, joint graph learning and policy alignment.
The tests cover forward-reference attribution, real same-model score replay,
unchanged candidate identity across two scorers, no candidate regeneration,
source/basis/observation mismatch, legacy receipt refusal, incomplete scoring,
two-by-two null comparison and sealed-test exclusion.

Repository smoke passed 164 tests with one skip in 59.20 seconds. Lint,
compile, governance-lint, layering and writing passed their configured gates.
Governance and writing retain their reported baseline debt; these gates do
not claim that inherited debt is repaired.

The staged diagnostic replay from the preceding checkpoint completed all six
selected observations: three recorded errors and three correct controls kept
their previous failure-stage classifications. Each error needed one completed
small bank rather than the larger alternatives inventory. Loading took
24.7826756 seconds and the audit took 15.7030553 seconds on this run. This is a
selected six-case replay, not a cohort throughput or accuracy claim.

Receipt: `a8b4c476274944f1d17dd27f38447b761703c9bc4a911879aeb1cd60d4af90ba`.
Local artifact: `~/.aura/rlc-evidence/semantic-cohort-prefix-replay-20260921/report.json`.

The original full-cohort audit continues on its frozen source identity;
changing diagnostic code here does not relabel or invalidate its historical
program-equivalence results. Full-cohort and fixed-bank results will be recorded
separately when complete.

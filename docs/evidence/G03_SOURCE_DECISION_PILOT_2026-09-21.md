# Source decision retention pilot

The source-only fit repaired all four training failures identified by the full
development audit. Its declared retention population contained those failures,
16 source-selected controls and the earlier five-example mining population,
with duplicates removed: 23 observations. Validation did not supply gradients
or retained constraints; no test observations were used.

The final ordinary-decision replay was 23/23 semantically equivalent. A separate
paired replay used identical 20-second search allowances and alternating arm
order: starting model 19/23, fitted model 23/23. Four gains, zero regressions,
zero unresolved comparisons. This is training-population repair, not transfer
or successful public-answer evidence.

Candidate: `a365976b653d002de7a256e6506de316a9fb4006970bfa1df42da8e4a6c20448`.
Fit receipt: `1e5a99c28fa2fbbccec6e9be2731cf28f338f69f75a7a9ac8cde9b434daa0f24`.
Paired receipt: `0453d268b5045e7a66ce6ed3801936047cd3dff30f6ce3a0aa52e8f974d61730`.
Artifacts are under `~/.aura/rlc-evidence/semantic-source-decision-pilot-20260921`
and `semantic-source-decision-eval-20260921`.

The fit used typed charts and all learned operation labels, held the operation
classifier fixed, and updated argument/relation/pointer parameters. Its three
numerical steps did not certify global optimality. Several earlier replays
exhausted the three-second search allowance; those records remain unresolved.
The final replay and independent paired measurement completed, without changing
the earlier records. G03 remains open pending the complete development cohort.

The existing cohort auditor now supports `--observe-only`. It records ordinary
decisions for the entire population before costly alternative-bank diagnosis.
A wrong answer is explicitly unattributed in this mode; an unfinished semantic
comparison stays unresolved. Mode and search allowance bind the cached receipts,
so a diagnostic run cannot silently reuse observation-only attribution.

Focused cohort and retention tests: 34 passed. Smoke: 164 passed, one skipped.
Lint, compile, governance-lint and layering passed. No serving promotion occurred.

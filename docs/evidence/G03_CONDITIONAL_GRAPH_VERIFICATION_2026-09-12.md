# G03 conditional graph verification

This is the completion record for the conditional-graph development checkpoint.
It does not close G03 or authorize serving.

The final focused run passed 57 tests covering the conditional scores, pairwise
fitter, structural equivalence, source-order proposals, forward references and
definition boundaries. A second run passed 66 optimizer, prefix-search,
projection-reuse, selector and definition-boundary checks. These suites overlap;
their counts are not added as a unique-test total. The second run includes the
new check that a structurally equivalent answer cannot relax strict selection.
Smoke passed 164 tests with one skip in 297.90 seconds. Touched-file Ruff,
full-tree compile and writing passed. Git diff whitespace checks passed.

The source-neighbor definition candidate completed its 84-example pilot. It
regressed one fork/join program, including under structural equivalence, while
improving one strict reserved-alias result. The existing multiplication-swap
regression remains strict-only. The candidate is rejected for promotion.
Its rows, decision, and implementation hashes are retained as
`source-ordered-definition-pilot-v2.json` and
`definition-boundary-rejection.json` in the conditional-graph artifact directory.
The option remains a separately identified development experiment; neither the
frozen default nor a qualified serving package uses it.

Aggregate governance lint remains red at two unrelated ownership buckets in
`core/subject/snapshot.py::_restore_stores`: two `ensure_directory` calls and
one `write_bytes` call. The preceding full lint run also retained 69 findings
outside these changed files. Neither aggregate gate is reported green and no
baseline was widened.

The next mechanism investigation is the relation scorer. Its low-rank tissue
is trained with multiclass cross-entropy over raw logits, while the runtime
register scorer applies positive-label sigmoid transforms before taking score
differences. No improvement from changing that rule is claimed in this record.

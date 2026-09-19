# Paired public-answer measurement

## Reproduced defects

The CP236 retrieval/depth diagnostic used case-insensitive answer containment.
It credited negated answers, longer identifiers, lists of alternatives, and
private reasoning containing the expected value. Its aggregate verdict could
claim both factors were required when retrieval helped one task and depth
helped a different task. Retrieval was compared at the best observed depth.

The CLI also removed baseline-correct tasks before measuring the factorial,
which concealed their possible regressions. It printed a nonexistent verdict
key after writing its output and could overwrite an earlier receipt.

The first counterexample run produced 13 failures and six passes. These are
measurement defects, not evidence that the model's answers improved.

## Repair

`core/learning/integrated_reasoning_eval.py` now defaults to
`public_terminal_paired_v2`. It extracts the public channel with the shared
channel parser and compares the selected terminal scalar using the existing
answer normalizer. Formatting tolerance does not admit other values, negation,
or alternatives. The explicit `substring_aggregate_v1` policy retains legacy
grading for historical replay; it is not the default for new measurements.

Retrieval and depth effects use the same declared deepest treatment cell and
retain per-task wins, losses, ties, and exact one-sided paired tails. Joint
necessity requires the same task to fail both ablations. Duplicate tasks and
unordered or duplicate depths are rejected. Each solver call receives its own
context list. Retrievers receive the question without a fixture identifier.

New receipts retain the complete task manifest and selected public answers.
`tools/eval_integrated_reasoning.py` keeps every declared task regardless of
its preliminary baseline result, records the selection policy, and publishes
once through the existing atomic writer. A second invocation cannot replace
the receipt or load the model before reporting the collision.

## Verification

The final focused run passed 68 tests across:

- `tests/test_integrated_reasoning_measurement.py`
- `tests/test_integrated_reasoning_eval.py`
- `tests/test_exact_paired_statistics.py`

An independent review then reproduced two additional false positives:
punctuation-only gold normalized to an empty string, and a template-opened
private channel was absent from the generated continuation's text. The default
design now rejects empty normalized gold and requires a nonempty public answer.
The CLI carries the rendered template's channel state into generation, decodes
the accumulated token sequence, and applies stopping rules only after the
public boundary. A private newline cannot end an answer. The combined follow-up
passed 109 tests, including the graph constraint and argument-learning suites.

The CLI test uses a declared fixture solver and the real publication path.
It proves baseline successes remain counted and that the receipt is immutable;
it is not a model-active test. Smoke passed 164 tests with one skip. Lint,
compile, governance and layering passed.

## Scope

This remains a diagnostic using a caller-provided solver. Planted fixture
retrieval is identified as such. The CLI's direct intrinsic-recurrence protocol
does not establish ordinary desktop execution, current-model qualification,
or live retrieval quality. No-retrieval failure is not proof of absent knowledge.
Reports explicitly carry no confirmatory evidence or serving authority.

G08 still requires independent campaign verification, contamination controls,
and cross-domain outcomes. G09 still requires measured broad reasoning gain.
Neither item is closed by these tests.

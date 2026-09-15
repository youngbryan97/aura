# Counterfactual source training

The program-first generator now renders source-training programs with fresh
register names and independently permuted clause order. Operation and reference
spans remain attached to their computational roles, including forward references.
It also rotates single-use addition/multiplication subtrees when the existing
integer-polynomial checker proves equality. Subtraction is not treated as
associative, and shared intermediate values are not rewritten.

Minimal meaning-changing examples change a typed operation or swap its operands.
They enter the corpus only when the universal floor produces a distinguishing
witness. Every emitted program must also execute on its printed inputs and agree
with the independent primitive interpreter. This caught mutations that were
type-correct but undefined on their printed values.

The existing feature materializer accepts `counterfactual_natural_source_v1`.
Its ordinary config, tokenizer projection, checkpoint binding, record checks,
manifest, and reload path apply unchanged. No runtime parser or prompt changed.

## Evidence

[Generation audit](../../artifacts/rlc/semantic_counterfactual_corpus_20260915/audit.json):
seed 43 yields 36 examples from 12 natural-source training rows: 24 equivalent
renamings/reorderings and 12 witnessed changes. All 36 execute with matching
floor/primitive results. This source cohort has no associative rotation site;
addition and multiplication rotations have separate exhaustive small-value tests.
Validation/test examples are excluded, and reserved construction IDs are rejected.

Focused suite: 56 passed, including complete acquisition/reload through a fake
hidden-feature client, independent execution at three seeds, aligned reference
spans, and invalid-input controls. Lint, compile, governance and layering pass.

No 27B features have been acquired for this corpus yet. It has not trained a
candidate or improved an evaluation score. S05, G03 and fresh-transfer gates
remain open.

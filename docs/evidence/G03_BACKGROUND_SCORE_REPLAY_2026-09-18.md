# Background score replay

The optional joint-background runtime policy scores an operation as
`log(mean_view_probability(operation)) - log(mean_view_probability(background))`.
It deliberately excludes the separate operation-pointer score.

Graph learning replayed a different quantity: operation log probability plus
the pointer score. Its variable terms also omitted the background denominator.
Consequently, combining this policy with graph refitting optimized a different
score from the one used for selection.

The existing `OperationEvidenceBank` now carries an optional normalizer label.
Scalar evaluation differentiates both log probabilities, including their
clamps. The shared batched evaluator expands the denominator into a negative
term over the same bank; values, weighted gradients and directional derivatives
remain identical across chunk sizes. Annotated graph replay excludes the
pointer term when the runtime excludes it. Explicit source-boundary supervision
remains available for proposal learning.

Evidence: 89 focused tests pass across background scoring, graph batching,
joint learning, constrained updates and checkpoint replay. Tests compare the
real runtime node score and annotated graph score and check derivatives against
central differences. Smoke passes 164 with one skip; lint, compile, governance
and layering pass.

The full working-face run does not use background log odds and remains frozen
at its launch revision. No candidate scores, held-out accuracy, serving state
or G-ledger completion are inferred from this repair. The earlier background
trial remains a negative result.

# Opt-in relation capacity expansion

The existing directed relation head can now expand its bilinear rank without
replacing the learned component. Its score contains `(reference @ Q) @
(definition @ D)`. Expansion retains the old columns, adds seeded query
directions orthogonal to their actual column space, and initializes the
corresponding definition columns to zero. Thus the new component starts at
zero while its definition gradient can be nonzero. Initializing both factors
to zero would prevent first-order learning in the new component.

The mathematical score is unchanged initially. Floating-point reduction order
can still differ, so numerical and autonomous replay remain required. The
same head, decoder, graph optimizer and serialization are used; there is no
second relation model. The refit CLI exposes this as `--relation-rank` and
`--relation-rank-seed`, only for the joint-graph path. It is not enabled by
default or applied to the active frozen run.

Original training records retain their original rank and selection evidence.
An ordered expansion lineage records each parent, old/new rank and seed.
Invalid geometry or lineage fails deserialization. No expansion grants
serving authority or inherits a measurement on its new dimensions.

Tests cover unchanged old columns and scores, live new-column gradients,
deterministic initialization, three-way definition learning through the
existing constrained optimizer, serialization, malformed lineage and CLI
configuration. The combined run passes 94 checks and fails the historical
frozen-path activation check with `source_contract_drift`. Comparison against
HEAD confirms this patch changes none of its declared bound symbols; four
bound symbols already differ from the historical activation. This remains a
G10 requalification requirement, not permission to reseal old evidence.
Smoke passes 164 with one skip; lint, compile, governance and layering pass.

No source-cohort accuracy benefit or need for rank expansion is established.
The current rank-16 full run has independently satisfied all 28,143 first-round
constraints in 16 steps (minimum margin 0.11748874757671501), so it continues
unchanged into re-mining before any capacity decision.

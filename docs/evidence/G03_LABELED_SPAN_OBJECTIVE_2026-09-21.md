# Joint labeled span-set experiment

This experiment addresses Q07 and Q14 from the September 21 review. The
incumbent's operation classifier learns labels on supplied spans; its pointer
learns boundaries separately. The new source-only objective compares each
annotated set with every bounded, non-overlapping labeled span set, including
the empty set once. It does not add an answer lookup or inspect test labels.

For interval-label scores `s(i,j,c)`, the loss is
`log sum_S exp(sum_(i,j,c in S) s(i,j,c)) - sum_(i,j,c in target) s(i,j,c)`.
Label log-sum-exp followed by the existing interval dynamic program computes
the partition and marginals exactly for this declared finite grammar. Masked
intervals contribute zero probability. The gradient passes through the same
probability clipping as the existing background-odds runtime scorer.

Prefix-summed token projections and their scatter adjoint avoid materializing
every interval's 5,120-dimensional feature vector. This is the same normalized
mean representation, not a changed representation or an approximate top-k
partition. The candidate exports through the existing classifier/background
contract. It changes neither argument nor relation heads. Fits remain opt-in
through `tools/refit_semantic_argument_proposals.py --objective labeled_spans`.

The mathematical statement is limited: the partition includes all permitted
operation sets. It does not include typed argument bindings, definition
ownership, or execution semantics. Lower loss therefore does not entail a
better whole-program decision or generalization. Runtime proposal pruning also
remains a separate process, not a completeness claim made by this objective.

## Small paired replay

Evidence directory:
`~/.aura/rlc-evidence/semantic-labeled-span-pilot-20260921/`.

The immutable plan selects 64 source-training rows and counterbalances the two
arms on 16 exposed controls plus two previously diagnosed challenges. Neither
validation nor the excluded test split enters fitting. This is development
evidence, not fresh transfer or a powered comparison.

- Fit: 54 iterations, 9.508 seconds; loss 10.309603 to 3.214702.
- Candidate: `0b58c515ef8498af9437005a10aa95f0103bd47a8a29a123c4afa0cbbd48a330`.
- Incumbent: 16/18 semantically equivalent; candidate: 14/18.
- Paired changes: zero gains, two losses. Both challenges remain incorrect.
- Report: `4e46a6f78a2e2a743cdbe294b29ad1683424a33891fd2b39a59d18a945add5c9`.

The candidate is not promoted. A full-source fit using the same exposed replay
is recorded separately; it cannot turn this result into a success.

## Full-source paired replay

Evidence directory:
`~/.aura/rlc-evidence/semantic-labeled-span-full-fit-20260921/`.

All 764 source-training examples enter fitting; the same 18 exposed examples
are replayed with counterbalanced arms. The full fit converges in 108
iterations and 265.875 seconds, reducing loss from 10.330637 to 1.428912.
Its candidate identity is
`1e8e3ef7d026a3533b0ba2174dd4ef1154f9453163f1a07edea363c04bbf030e`.
Both arms score 16/18, with zero gains or losses and both challenges still
incorrect. Report identity:
`566d375de05f3fadbda3a78268d6cef1b1bcbaf31bf01689f5260bc56a92c122`.

There is no measured benefit supporting promotion or a larger qualification
run for this candidate. These results do not prove the model class incapable;
they reject the claim that this objective change has repaired the observed
selection errors. The untouched 500-row test split remains excluded.

## Verification

Forty-four focused tests pass across labeled-set learning, the existing span-set
learner and background scoring. Checks include exhaustive small-grammar
enumeration, masked intervals, finite-difference gradients through clipping,
prefix projection/adjoint agreement, current-format serialization and split
exclusion. Duplicate/split overlap, changed model basis, missing operation
classes and an unconverged optimizer are rejected. The small partition identity
is also registered with the invariant verifier. Smoke passes 164 with one skip;
lint, compile, governance, layering and writing pass. These establish numerical
contracts, not G03 closure.

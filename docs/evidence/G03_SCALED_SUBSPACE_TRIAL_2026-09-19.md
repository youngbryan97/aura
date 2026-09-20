# G03 scaled subspace trial

The larger source-only trial tested the parameter-subspace repair after the
eight-training/60-validation diagnostic. No validation labels entered fitting.
The operation classifier remained exactly fixed; the existing relation,
argument and operation-boundary parameters could change.

## Frozen setup and result

The source-hash/geometry selector scanned 128 training examples, then selected
three witnessed errors and thirteen retention examples. Operation and boundary
retention used all 128 training examples. The separate validation cohort had
100 examples. Eight operation charts per training example were searched.

| Cohort | Before | After | Gains | Regressions |
| --- | --- | --- | --- | --- |
| Selected training | 13/16 | 16/16 | 3 | 0 |
| Development validation | 91/100 | 91/100 | 4 | 4 |

All 5,063 retained inequalities pass after one accepted update and five
constraint-discovery rounds. The smallest stored margin is
0.10000001312681839. The operation classifier's 30,726 coordinates are unchanged.
The Euclidean parameter displacement is 4.887710777501376.

Both before and after used the same symbolic verifier. No decodes were refused
and no semantic observations were unmeasured. Operation-retention search was
incomplete; fitting the retained inequalities does not prove that every runtime
competitor was retained.

Artifact:
`/Users/bryan/.aura/rlc-evidence/semantic-subspace-scaled-20260919.json`.
Receipt:
`3ef8afbdcc291705b736a83fa70ef1645d4c55b44da51c8f752b98b94b263227`.

## Decision

The candidate is not promoted. This trial falsifies the stronger expectation
that freezing the operation classifier alone would prevent regressions at
larger source coverage. It establishes source repair, not a net validation
gain. G03 remains open.

## Component interventions

The fit replayed bit-identically before interventions. All eleven examples whose
semantic status changed were decoded under seven coefficient interventions.
Restoring the parent's relation head alone reverses all four validation losses,
but also removes all four validation gains. Restoring the operation pointer,
argument roles, proposals, or both argument components reverses none of the
validation losses. A relation-only update reproduces three of the four gains
and all four losses. Relation learning therefore causes the observed tradeoff;
these exposed examples are diagnostic, not fresh transfer evidence.

Artifact:
`/Users/bryan/.aura/rlc-evidence/semantic-subspace-attribution-20260919.json`.
Receipt:
`2481dd99148e88bb57e69f34f788e9e5df0bd2317e6128ba2d0c68b4b12ac552`.
Neither intervention selects a serving model or fits validation labels.

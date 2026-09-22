# G03: request context, constrained decoding, and retained alternatives

## Change

The source-fold experiment now fits a full-request contextual span recognizer
over the frozen resident features. The exact labeled span-set partition supplies
the loss and gradients. All in-bound spans and operation labels remain available
to decoding; target spans and operation counts are never inference inputs.
The existing typed graph search, argument model, and universal-floor executor
remain the owners of graph feasibility and execution.

One-operation-set commitment and constrained search are measured separately.
The latter orders complete feasible operation sets by learned operation score,
then selects arguments with the existing argument model. This is hierarchical
selection, not a jointly normalized graph model.

The new portfolio wrapper reuses the common pairwise evidence selector. It
retains every complete proposal, executes each on the floor, preserves an
executable incumbent, and repairs missing/failed executions with alternatives.
It compares candidate programs by proved symmetries or counterexample execution.
Conflicting executable meanings remain distinct. Neither comparison nor
selection can access the target program or expected answer.

## Completed measurements

The frozen source-construction fold plan has receipt
`9c6413bfdf573d03b4239ac82577001ee6f7499561ede194b24cc07022801b94`.
Fold zero fits 508 examples and holds out 256 for operation recognition.
The model uses width 32, four attention heads, one layer, eight fixed-order
AdamW epochs, learning rate 0.001, and seed 73. Capacity and settings were
not chosen using the 500-row development-validation set.

| Method | Training operation sets | Held-out operation sets | MAP complete programs | Constrained complete programs |
| --- | --- | --- | --- | --- |
| Full context, absolute position | 508/508 | 144/256 | 176/256 | 240/256 |
| Full context, relative position | 508/508 | 158/256 | 161/256 | 235/256 |
| Token-local control | 508/508 | 180/256 | 180/256 | 200/256 |
| Frozen random context | 5/508 | 3/256 | Unrun | Unrun |

Operation spans are not the semantic grading unit: different predicted spans
can produce the same correct program. Complete-program equivalence uses the
existing structural, polynomial, and typed partial-ring proofs; a finite probe
that finds no counterexample remains unknown rather than passing.

The three MAP methods contain a correct program on 203/256 tasks. The
answer-blind portfolio selects all 203, compared with 180 for its declared
local incumbent: 23 gains, zero regressions. Adding the three constrained
methods yields 239/256 selected, 59 gains, zero regressions against that same
incumbent. Their candidate union contains a correct program on 248/256:
eight tasks lack a correct proposal and nine have one that selection misses.
The best single constrained method scores 240/256, so preserving the local
incumbent is not universally the best selection policy.

All six methods consumed 990.35 seconds of measured graph resolution across
the cohort. Portfolio execution and comparison added 1.44 seconds. These
numbers exclude feature acquisition, training, checkpoint load, and the
earlier MAP operation-score evaluation. They are not an end-to-end latency
or equal-compute gain claim.

The unchanged absolute-context configuration on source fold one fits 510/510
operation sets and predicts 236/254 held-out sets; removing cross-token context
reduces this to 222/254. Its completed constrained full-program replay reaches
246/254, with eight witnessed differences and no refusals.

All eight fold-zero candidate-coverage failures begin with an imperative
subtraction. The training partition has no operation starting at token zero.
The models instead propose division or the wrong nested ordering. This is a
construction-transfer weakness, not a reason to hardcode an imperative phrase.

## Mechanism defects found and repaired

The installed fused encoder evaluation path changed the treatment of learned
real-valued attention masks. Relative-position inference produced NaNs despite
finite training predictions. Two of six train/eval parity cases reproduced
the failure. An explicit block of public attention, normalization, and linear
modules now preserves mask semantics, parameter names, outputs, and gradients.
All six parity cases pass; legacy checkpoint loading and gradient parity pass.
No global framework switch or monkeypatch is used.

The interrupted graph run retained 351 completed rows. Identity-checked recovery
preserved them, including failures, and completed the remaining 417. Recovery
binds the old plan and each retained row; changed populations, checkpoints,
inference sources, or search settings are refused. The initial plan omitted
three context/objective source hashes. The recovery receipt explicitly names
the newly bound paths, rather than claiming they were bound retroactively.

Fold-one training resumed from epoch four with its AdamW state. Earlier epoch
timings/losses absent from the checkpoint remain null. New checkpoints retain
history and RNG state; an interrupted/resumed optimizer test matches an
uninterrupted trajectory exactly.

## Evidence and limits

Artifacts are under `/Users/bryan/.aura/rlc-evidence/`:

| Report directory | SHA-256 of report.json |
| --- | --- |
| semantic-context-graphs-fold0-20260921 | d1ef7f71dc3ec45f24850f3b53222da5eb5263625591454059dce19e32b72e80 |
| semantic-context-portfolio-v2-fold0-20260921 | 94c7c670a6cee0357efc6965d507e4dd16a50647f5206313ea839e38028a5238 |
| semantic-context-typed-recovered-20260921 | e649edccc567ce8440ac526174459ee0dcf351dde04d04e3212dfe64762cd95f |
| semantic-context-absolute-fold1-recovered-20260921 | f4c845406ca5a078ab4730382d0dded7847d9be657e9baf08991d0df3d95bf02 |
| semantic-context-typed-fold1-20260921 | d4fea9304374eb0bbc2f8a86c0414589ccca7faae5f1c10489fbb63c97ad780a |
| semantic-context-combined-fold0-20260921 | e2ceda26bf79fd37775e1c8b6391656d680d64beab8325e3c739eeb7d77e1bb7 |

The combined report is `semantic-context-combined-fold0-20260921/report.json`.
It retains each proposal's correctness, actual execution, selection receipts,
candidate relations, graph-report identities, and measured resolution costs.

The frozen argument model was trained on all 764 source examples, including
the operation-recognition holdouts. Thus complete-program results here are
architectural development diagnostics, not clean end-to-end held-out transfer.
Validation/test examples do not enter training or scoring. The existing bundle
loader checks archive metadata before selecting training examples.

The connected focused suite passes 96 tests. Compile, lint, layering, and
governance checks pass. Smoke passes 164 tests and fails the existing installed
activation alarm for `qualified_recurrent_ingress.py` source drift. The alarm
was not suppressed or relabeled as passing. No new candidate has serving authority. G03, broad
reasoning gain, fusion, and frontier comparison are not closed by these results.

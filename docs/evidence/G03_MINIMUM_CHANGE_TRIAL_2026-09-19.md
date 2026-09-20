# Minimum-change semantic repair: numerical fix, transfer still negative

The source-only trial selects eight graph-training cases from a hash-ordered,
geometry-balanced pool of 64. The pool contains one witnessed decoding error.
Operation-label and boundary retention use all 64 source cases. Sixty
validation cases are separate; none supplies fitting constraints.

The starting model uses joint operation/argument scoring. It gets 7/8 training
cases and 52/60 validation cases semantically correct. This is not the
first-feasible incumbent's accuracy. Pointer capacity remains additive.

| Update | Training after | Validation after | Validation gains/losses | Parameter displacement |
| --- | ---: | ---: | ---: | ---: |
| Minimum change, supervised boundaries | 8/8 | 51/60 | 0/1 | 10.681236071671913 |
| Minimum change, retained boundaries | 7/8 | 52/60 | 0/0 | 0 |
| Stored-precision repair, retained boundaries | 8/8 | 51/60 | 0/1 | 2.2690860943430873 |

The supervised-boundary trial begins with 148 wrong or tied fitting
comparisons despite having only one selected program error. Retaining the
existing boundary ordering reduces that count to one. Whole-program
counterexamples still supply corrective supervision.

## Numerical cause and repair

The first retained-boundary run rejects every proposed update. Of its 2,504
constraints, 153 begin near the required margin. Continuous optimization
nearly resolves the fit, but float32 coefficient storage leaves several
constraints below their exact retained floors. The last full-sized proposal
has three violations, with maximum deficit 5.842544226197788e-8. Eight serial
restoration steps do not resolve the simultaneous faces.

A separate 128-dimensional linear example with 48 binding faces reproduces
the failure. The shared solver now verifies and polishes proposed active
equalities, then adds numerical reserves to comparisons crossed by coefficient
rounding. It rechecks stored coefficients against the original inequalities;
it does not lower the floors or declare an unresolved problem infeasible.

On the real source constraints the repaired optimizer accepts one update,
after two working-set refinements. All 2,504 stored margins pass; the minimum
is 0.10000001426403848. The continuous affine subproblem has numerical
primal/dual gap 1.0658141036401503e-14. This is not a global nonlinear
optimality proof.

## Independent replay

The after-fit replay invokes the autonomous decoder and compares programs
through structural equivalence or a distinguishing universal-floor execution.
The training error is repaired. One validation answer changes from correct
to incorrect; no validation answer improves. There are no decode refusals in
the stored-precision replay. This refutes the claim that small parameter
change plus retained training constraints alone guarantees transfer.

The candidate is not promoted. No fresh transfer, broad reasoning gain,
runtime serving claim or G03 closure follows. The next diagnostic restores
coefficient groups on the exposed changed cases; those cases cannot count as
fresh replication afterward.

## Artifacts

All files are under `~/.aura/rlc-evidence/`:

- `semantic-minimum-change-trial-20260919.json`, receipt
  `0eb9f5e23ba0c002bd07c9e4fbbd8147179aa9fddd291efb96b23a2f098dcc06`.
- `semantic-minimum-change-retained-boundaries-20260919.json`, receipt
  `0577f5627ec6561f63a8bef6604e76d5ba719ad9f4919854960d7a47fad70504`.
- `semantic-minimum-change-stored-replay-20260919.json`, receipt
  `3d4022badce167daeb66cd1de7004887cbf02e73e2480f23c20f26006d4dfed4`.

The last replay reuses the preceding receipt's unchanged before observations,
checks that receipt and the parent identity, fits only source-training rows,
and records a new after observation for every case. It binds the relevant
implementation file hashes. Later metadata/style repairs change the
step-policy label, not the fitted coefficients. These are local development
artifacts, not a frozen prospective campaign.

Verification: 118 focused tests passed before the final metadata check; the
final solver subset passed 32 tests. Smoke passed 164 tests with one skip.
Lint, compile, governance, layering and writing passed. The new regression
tests cover simultaneous boundaries, exact representable equalities,
nonrepresentable intervals and rejected or forged certificates.

## Component intervention completed

Restoring the original operation classifier while keeping the candidate's
other learned coefficients repairs the exposed validation regression and
preserves the repaired training answer. Restoring the pointer, relation head
or argument heads alone does not repair that validation regression.

The training computation changes from multiplication to the requested integer
division. The regressed validation computation changes both the operation
and the count argument. These are actual program changes, not a grading
disagreement. This two-case selected intervention identifies a cause for this
regression; it is not an accuracy estimate or permission to choose a serving
candidate using validation answers.

Artifact: `semantic-minimum-change-stored-attribution-20260919.json`, receipt
`fafcb91add8376768d39d2eb49f1c7262d5b21c24ca40a475b96c95c0857350b`.
The next source-only experiment restricts the optimization subspace rather
than changing task-specific outputs.

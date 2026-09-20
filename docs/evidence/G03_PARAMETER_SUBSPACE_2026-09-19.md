# A Source Repair Without the Measured Classifier Regression

The preceding component intervention implicated operation-classifier changes
in one exposed validation regression. A new source-only fit holds that block
fixed, while relation, argument and operation-boundary parameters remain
trainable. This is a declared optimization subspace, not a post-fit restore or
a rule selecting outputs by task identity.

The same eight graph-training examples, 64 source-retention examples and 60
validation examples are used. Validation supplies no constraints. These are
exposed development cases, not fresh transfer evidence.

| Run | Training equivalent | Validation equivalent | Gains/losses on validation |
| --- | ---: | ---: | ---: |
| Starting joint-score model | 7/8 | 52/60 | -- |
| Frozen classifier, first solver | 7/8 | 52/60 | 0/0 |
| Frozen classifier, active-set repair | 8/8 | 52/60 | 0/0 |

The first solver rejected its proposal before accepting any update. Solving
the optimizer's positive dual support as one equality system produced a
negative multiplier. The old polish discarded that solution but could not
remove the spurious active face. The retained numerical solution missed a
margin by 1.763754155937152e-6.

The shared solver now pivots negative multipliers out and violated inactive
constraints in. Independent primal feasibility and dual-gap verification
remain required. New tests exercise both directions of active-set change.

The repaired run satisfies all 2,504 constraints after one accepted update
and two working-set refinements. The smallest stored margin is
0.10000000360326133. The local continuous projection gap is
8.79296635503124e-14; this is not exact arithmetic or global nonlinear
optimality. The operation classifier's 30,726 coefficients remain unchanged.
No validation answer regresses, unlike the previous unrestricted trial.

The subspace restriction is applied to gradients, constraint normals,
restoration, storage and checkpoint verification. Checkpoint identity binds
the mask. Receipts record per-block displacements and exact frozen-coordinate
retention. Existing callers continue to train all their previously trainable
blocks unless they explicitly request a restriction.

## Artifacts

Under `~/.aura/rlc-evidence/`:

- `semantic-minimum-change-frozen-operation-20260919.json`, receipt
  `21e25d1e76862a3be4f26df84821e3c050dbab1cefb7461aec8706795233864c`.
- `semantic-minimum-change-frozen-operation-pivot-20260919.json`, receipt
  `97f8671f825e8f0836f6ffbf54cabd8033c83f400588dfd5be28cd9cdcce8dcb`.
- Parent: `b8d9e6d785976a923c071ac969b58063bd76ed512eb6f2e472ad123dd99cf12d`.
- Repaired candidate: `cdc7123576bc4bba78601da993b24851cf9012648c9d73a34cf6ceea273628c9`.

The replay reuses the hash-checked prior before observations and runs all
after observations anew. It retains the original negative run. The symbolic
comparison extension was loaded for the final replay; the before observations
predate that extension. No validation outcomes changed, but this comparison
is therefore a development diagnostic, not a frozen before/after experiment.
The next larger trial measures both arms under the same current verifier.

Verification: 42 subspace/constraint tests, 66 solver/trial/joint/checkpoint
tests, 34 final solver/subspace tests, and 101 graph-regression tests passed.
Smoke passed 164 tests with one skip. Lint, compile, governance, layering and
writing passed.

This removes one measured repair regression. It does not establish validation
gain, source-wide no-regression, or G03 closure. No model was promoted.

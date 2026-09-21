# Operation retention policy

The restored joint candidate correctly decodes all 764 source-training
examples. It nevertheless loses five of the original parent's correct answers
on the separate 100-example development-validation cohort.

## Component intervention

On those five observed regressions, coefficient replacement gives:

| Coefficients changed from parent | Correct |
| --- | ---: |
| Relation head only | 5/5 |
| Operation classifier only | 0/5 |
| Operation pointer only | 5/5 |
| Argument role and proposal heads only | 5/5 |

Restoring the parent's operation classifier in the complete candidate also
recovers 5/5. Restoring any of the other three groups recovers 0/5. This
attributes these five failures to the classifier update; it does not establish
the cause of every possible failure or qualify any mixed candidate.

Artifact: `~/.aura/rlc-evidence/semantic-accumulated-second-decode-20260920/component-attribution.json`.
Receipt: `739e315910f38c16cb03e54465e6ff4f6cb5da67df28bde3821c9ad0f43c061e`.
No learning used those validation examples.

## Training distinction

Source-label supervision addresses annotated spans. Runtime decoding may
choose another span for the same operation. Requiring every auxiliary
annotation to acquire a new positive margin can therefore change the
classifier even when its runtime operation choice is already correct.

The source operation-constraint builder now has two explicit policies.
`supervised` retains the existing objective. `retain_existing` preserves
the original auxiliary comparison without demanding its label change.
Complete-program counterexamples remain unchanged and can still train the
classifier. The option is available in the joint trainer, development trial,
and refit CLI; it is recorded in the training receipt.

For initial comparison margin m0 and requested margin t > 0, the retained
comparison is m(theta) + max(t - m0, 0). Requiring that value to reach t is
equivalent to m(theta) >= min(m0, t). This holds for initially positive,
negative, and tied auxiliary comparisons. The initial auxiliary deficit is
zero; lowering the original retained floor still violates the constraint.

This is an opt-in training policy, not a change to program scoring,
validation admission, serving behavior, or evidence requirements. A new
paired fit must establish whether it prevents the observed regression.

Ten new tests cover the margin identity, detection of worsened comparisons,
unchanged supervised behavior, invalid settings, and joint-trainer receipt
integration. The focused regression run passes 68 tests. G03 remains open.

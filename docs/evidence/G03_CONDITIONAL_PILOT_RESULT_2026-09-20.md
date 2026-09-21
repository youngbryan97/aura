# Conditional graph learning pilot

The pilot learns from three historical source-training errors and two source
controls. Auxiliary operation and boundary constraints retain all 764 source
examples. It uses three single-update rounds, complete conditional argument
denominators, and the existing 16-chart operation allowance. No validation
example supplies an update, and no test example is used.

The five diagnostic source examples improve from 2/5 to 5/5. The third round's
final numerical state reduces wrong-or-tied retained comparisons from 1,341
initially to 12, with no reversal of retained positive comparisons. Neither
fact establishes generalization.

The completed, separate development-validation comparison is:

| Arm | Equivalent programs |
| --- | ---: |
| Original parent | 99/100 |
| Conditional scoring, unchanged coefficients | 98/100 |
| Conditional scoring, learned coefficients | 96/100 |

Relative to the original parent, the candidate has three regressions and no
gains. It is not promoted. G03 remains open. This cohort is previously exposed
development evidence, not fresh transfer evidence.

Training finished before the pilot's 1,800-second bound. Evaluation reached
63 complete rows before that bound terminated the process. A separate
resume-only evaluator checked the frozen implementation and observation order,
loaded the already-saved candidate, and completed the remaining 37 rows.
It did not repeat training or alter coefficients. Its elapsed time was 262.20
seconds through the final row; this is not total campaign wall time.

Artifacts: `~/.aura/rlc-evidence/semantic-conditional-shared-pilot-20260920/`.
The immutable completed validation receipt is `validation.result.json`:
`74b2210777014fa706fc5e080e80364a5d261e602ee7eccc75ff1fd8faa961a6`.
Candidate: `b9c1ff0014003a8732afa0349f63d918e35752a127b45c802439c5c5cf13163e`.
Implementation: `b09623c70599d6d7093ce446cc6255ebc924655e8b37f3f717f121caa014b1a2`.

## Component interventions

Each intervention decodes the same five source examples and the three
observed validation regressions. Single-component arms start from the
unchanged conditional model; restored-component arms start from the learned
candidate. No learning occurs in these interventions.

| Intervention | Source correct | Regression cases correct |
| --- | ---: | ---: |
| Relation update only | 2/5 | 3/3 |
| Restore parent relation | 4/5 | 0/3 |
| Operation classifier update only | 2/5 | 2/3 |
| Restore parent classifier | 5/5 | 0/3 |
| Operation pointer update only | 3/5 | 0/3 |
| Restore parent operation pointer | 4/5 | 3/3 |
| Argument update only | 3/5 | 3/3 |
| Restore parent argument heads | 3/5 | 0/3 |

The operation-boundary pointer update reproduces all three regressions;
restoring that pointer recovers them but loses one source repair. This rules
out treating a frozen pointer as the demonstrated complete solution. The
existing source-retention policy is the next controlled training comparison:
retain auxiliary evidence without demanding new annotated-span margins,
while permitting witnessed complete-program errors to train the pointer.

Artifact: `component-attribution.json` in the same directory.
Receipt: `da501a166623319a5cd70ddf3bdb1732301e691823bea30d0864506e49ad7ae2`.
These are diagnostic interventions, not validation-tuned serving candidates.

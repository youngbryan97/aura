# Recognition evidence overruled by binding scores

The joint transition pilot's 29/40 is a selected development diagnostic, not
the complete validation result. Its population contains 24 known incumbent
failures and 16 selected controls. The full 1,264-observation audit is retained
separately; this record does not infer its result from the pilot.

An intervention on a frozen snapshot examined three training regressions and
the eleven validation failures remaining in that pilot. All fourteen archived
programs reproduced. Eleven differ in operation boundaries, two in operation
labels, and one only in argument binding. Supplying source operation boundaries
and labels repairs thirteen; the remaining division-order error stays wrong.
These supplied annotations are diagnostic interventions, not autonomous answers.

The three training regressions expose an objective mismatch. Recognition alone
prefers the source interpretation, but the complete graph score prefers the
wrong interpretation:

| Source prefix | Correct-minus-selected recognition | Binding | Combined |
|---|---:|---:|---:|
| `fb1ff82bbdb2` | 2.167403 | -5.175480 | -3.008077 |
| `8a877e193d4c` | 2.167403 | -4.643663 | -2.476260 |
| `0fafc07219cd` | 2.287116 | -4.347147 | -2.060030 |

The replayed argument scores equal the ordinary decoder's recorded scores.
Thus these are not unexplained solver failures or evidence that the correct
graph is outside the search space. Training an operation-only span likelihood
does not constrain the combined score that chooses the actual answer.

## Coupled objective

The existing labeled-span trainer now accepts explicitly named source-training
identities for runtime-error acquisition. It calls the existing answer-blind
decoder and independent program comparison. A witnessed incorrect graph enters
the same operation-head fit as a complete-selection margin. Binding coefficients
remain frozen; their contributions and choice normalizers remain in the margin.
The original span likelihood still covers every supplied training observation.

The added loss is the existing squared margin deficit, averaged over witnessed
runtime contrasts, with required margin 0.1. Its gradient reuses the shared
complete-graph differentiation implementation. It does not train on validation
failures, replace the search policy, or insert a prompt or case-specific rule.

The receipt names the parent, source identities, mined programs, constraint
identity, initial and stored margins, and whether all witnessed margins meet
the target. Missing witnesses remain unmeasured. Optimizer convergence is not
source-decision retention: changed coefficients can expose new competitors or
change latent bindings. The complete cohort must be decoded again.

Thirty-eight focused tests pass, covering both span features, finite-difference
gradients, probability clipping, source isolation, unresolved mining, storage
round trips, and the case where binding alone is wrong and recognition has no
gradient capable of fixing it. Smoke passes 164 tests with one skip.

## Bound artifacts

- Attribution: `~/.aura/rlc-evidence/semantic-transition-attribution-20260921/`;
  receipt `1029a005eda4953bf268763e55cb7ff2baaf4cc83d6fab16631e4b23ca0ed2ee`.
- Score replay: `~/.aura/rlc-evidence/semantic-transition-score-audit-20260921/`;
  receipt `0918d1cd90a8163b2780137532bbfd104b9556303531b10392dbd12478ecab40`.
- Coupled fit and complete cohort:
  `~/.aura/rlc-evidence/semantic-transition-runtime-fit-20260921/`.
  Tool-session interruptions stopped the optimizer before export. The fit and
  complete-cohort result remain pending; no accuracy result is claimed. The
  continuation uses a detached supervisor and the existing durable row receipts.

No serving authority changes and no G item closes.

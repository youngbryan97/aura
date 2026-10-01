# Joint Development Screen, 2026-10-01

The joint artifact now has a paired development CLI:
`tools/evaluate_semantic_grounded_native.py`. It requires an independently
verified positive-step selected checkpoint before loading the backbone.
The exact source parent, report, fold plan and bank must match the native
source basis. Held requests cannot overlap native fitting or calibration.

The three arms are the unchanged source parent, the existing joint-score
global chart without the new pointer, and the complete native-adapter/pointer
chart. Each arm receives the same public source tokens, parent features,
public inputs, source identity, model identity and search-time bound.
No expected answer, target instruction graph, construction label or teacher
argument register enters the decoder. The existing source-anchor scorer
reads target programs only after decoding. Exact and permitted structural
equivalence scores remain separate.

The evaluator freezes its plan before acquiring an exclusive supported model
lease. Unrelated archived feature matrices are dropped before model loading.
It loads one backbone for the joint arm, preserves per-request program and
refusal outcomes, and links joint results to the decoder's selected-graph
receipts. Backbone references and MLX cache are released before the lease
ends, including the failure path. A changed implementation cannot publish a
completed report.

The development screen advances only when every candidate interpretation is
correct under the declared structural-equivalence rule, at least one gains
over the unchanged parent, and neither comparison arm loses a correct
interpretation. An output report does not authorize serving. The bank is
previously exposed source-fold development data, not a fresh sealed transfer
set. A pass would permit broader development and causal controls; it would
not close G03 or later gates. The historical native v7 16-node/256-node
protocol uses a different artifact and decoder contract and is not silently
reclassified as this screen.

Fixture checks passed 19 tests in 40.48 seconds, including the native fit and
replay path. They check the public-only call boundary, control regressions,
empty/repeated request rejection and actual suffix-parameter movement during
small fitting. These are wiring checks. No 27B development screen has run.

## Corrected Native Run

The corrected preparation ran from frozen commit `86154d49d` and completed
in 107.673372 seconds. Its contained terminal receipt is
`f276bf3923a40868c66d515c6637c87e2a774a05735c587b0bbbb52d8b8aaac4`.
The native plan is
`e1e555f20046d2e3eb53f12f65b30eea68265d2b087320dde3d2b221b99d64f3`.

The exact prepared fit launched with supervisor PID 45972, child PID 45977
and sleep inhibitor PID 45980. The supervisor is parented by launchd; its
target owns a separate process group with fork/escape containment. The
supervisor plan is
`bb1ba56bee04c832556a77516f39034c0339422ab3d97f223fdadea99ae78998`;
its command is
`eea0c335c4ab17f2fa1fc30025335024c091f25e4d8d7e9e2749c9912c85a0b3`.
Source-only prefix logs were moving with about 16.6 GB active MLX residency.
Checkpoint zero had not yet been published at this observation. Detached
custody and a declared resume verifier do not establish actual resumability
until the first complete optimizer generation passes that verifier.

The fit subsequently published durable restart generations through step 480.
An independent check of the step-160 generation passed with the exact native
arithmetic environment (`MLX_ENABLE_TF32=0`). Its generation is
`f71b7ae9c577cc400d2c14e7b2182c073f8345d0aeb6dac63b3e975bca22cdaf`
and verification receipt is
`12eb8a97c99a4811bcfcc70354edccfa77aee15c1e19350eefd73053a982f06b`.
The full native geometry remains checked by the actual resume loader.
This is restart integrity evidence, not semantic success.

The fit uses a 15,000-second outer supervisor bound and the unchanged
14,400-second acquisition/fit allowance. No other historical completed source
fit was repeated, and no held request was scored during preparation or fit.

## Bounded Handoff

`tools/run_semantic_grounded_handoff.py` binds one existing trainer's exact
supervisor plan and command. It waits for a contained successful terminal
receipt and dead training child, verifies the selected positive-step artifact,
then enters the development evaluator in the same supervised process. It
cannot launch another training command. Failure, indeterminate completion,
an initial-only checkpoint or implementation drift stops before model loading.

Per-request start and completion records contain measured elapsed time,
program identity, refusal and the native chart selection. An independent
backbone-free report verifier recomputes nested digests, population counts,
paired gains, both regression comparisons and the advancement verdict. It
checks native selections against the exact fit and selected weights. A
negative scientific screen still produces a verified negative report and
exit two; it is not retried or converted into infrastructure success.

Report integrity is not independent semantic replay. Neither verifier nor
handoff closes G03, proves fresh transfer, or authorizes serving. The fixtures
exercise these boundaries with deliberate faults rather than claiming that
fixture-selected answers demonstrate learned interpretation.

## Completed Joint Fit

The corrected joint fit completed all 512 updates and selected step 288 by
source calibration. It used 1,335 fitting requests and 11 calibration
requests, with no held requests scored. The fit receipt is
`13d01ddb2064137a751e186680a51ae42b8e55491e1b26e9c63acd3a944c136c`;
selected weights are
`6445c617c9a4fe41052f6b5b37dc06b7ef07f0c669e22c9334885c447c759a4f`.
Independent fit verification passed with receipt
`6b434a8457d9a3d0d1488a50ea7984a8f8f073eab5afec1f3e70df08c5693695`.
The contained supervisor passed in 1,641.250472 seconds with receipt
`a13f2016f603596f8ebd7500e66cf93eb598734dc72e4fa39990ca487ee8a273`.
The target, supervisor, process group and descendant lineage were all gone
when inspected. The model lane is no longer owned by this fit.

The expanded focused fixture suite passed 58 tests in 48.06 seconds, followed
by 24 handoff/evaluation checks in 9.01 seconds after the preparation-mode
rejection was aligned with the actual CLI. The final focused run used a
state root outside the live `.aura` root: the earlier fixture run's exit-time
ownership warning came from placing test state underneath that protected
root, not from the trainer. The corrected run had no such warning.

At this checkpoint the real development screen remains unrun. Completed
joint fitting is a prerequisite, not evidence of semantic improvement.

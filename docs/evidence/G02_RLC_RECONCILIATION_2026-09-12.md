# G02: RLC evidence and activation reconciliation

Source inspected: `97aa16aba`. This record reconciles retained experiments;
it does not report a new model experiment or confer serving authority.

## Evidence by mechanism

| Mechanism and vehicle | Retained result | What it supports |
| --- | --- | --- |
| Frozen backbone recurrence, Qwen2.5 1.5B, July 17 | Vanilla 21/72; seven latent arms 7-13/72 | Negative capability result on this checkpoint. The report names its model snapshot and checkpoint fingerprint. |
| CP498, 1.5B learned transitions | T1 0.5417, T16 0.1536; oracle-state training did not match autonomous deployment | Negative acquisition result, not answer-level transfer. |
| CP547 semantic neural machine | 288/288 tasks, 6,624/6,624 transitions | Bounded typed execution with learned arithmetic tissue and structural routing; no LLM answer or frontier claim. |
| CP566 resident 32B | 60/60 treatment; ordinary 16, wire 7, coefficient lesion 5, wrong-state 0 | Bounded state-conditioned decoded gain on four executable families. |
| CP1003/CP1011 resident 27B recovery | 60/60 treatment; ordinary 0, wire 6, coefficient lesion 4, wrong-state 0 | Separate bounded replication on the new model. Different cohorts do not rank the two base models. |
| v14 withheld family | 79/96 exact answers and programs; lesion 0 | Transfer within a shared primitive vocabulary. |
| v16 natural-language replication | 26/96 answers, 22 programs; minimum 48 | Failed absolute capability floor; ordinary arm aborted under its preregistered rule. |
| v19 development repair | 93/96 answers; lesion 0 | Exposed development evidence, not fresh replication. |
| Frozen natural weave replication | 21/48 answers, 19 programs; two lesions 0; ordinary budgets 1/48 and 2/48 | Fresh bounded transfer to six inputs and five steps; exact execution assists the learned language-to-program mapping. Shadow only. |
| September 12 backbone-depth arms | Depth one 21/40, depth two 2/40 | Negative depth comparison. The arm reports omit model identity; do not attribute this observation to the resident 27B. |

The [G01 baseline](G01_RLC_BASELINE_2026-09-08.md) binds and recounts the
32B/27B raw decodes and v14/v16/v19 evidence. This reconciliation adds the
small-model campaign and later frozen-path package using existing adjudicators.
The two answer-correct but program-incorrect weave cases remain visible.

Sources beyond G01:

- `artifacts/current/latent_campaign_1p5b_run2.json`, SHA-256
  `065b2546817bdb054afe2dcad39405b536438f13b757b67f5e745b78192d12a6`.
- `artifacts/closeout/latent_cortex/cp498_factorized_transition_acquisition_verdict.json`.
- `artifacts/closeout/latent_cortex/cp547_semantic_neural_machine/verification.json`.
- `artifacts/rlc/semantic_program_27b_frozen_path_v1/activation.json` and its
  hash-bound mechanism, ordinary-control and preregistration records.
- `artifacts/recurrent_depth/arm_loops{1,2}.json` and the digest-bound response journals.

## Activation observation

At 2026-09-12 23:36:52 UTC, port 8000 refused connection and no `aura_main`
or model worker process matched the process inspection. Aura was stopped.
An earlier ready process is not evidence of current serving.

The configured 27B path was
`/Users/bryan/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15`.
Calling the production activation validators from the canonical launch checkout,
with isolated test logs/state and an explicit authority-key path, returned:

- Exact semantic package: `active=true`, `semantic_neural_serving_active`.
  Package `rlc-27b-recovery-05346acd618d1c925f16`; activation
  `2c6d577c91d28af9ef7992c8a1290bf0c1fee6f7388745e3e0e3f162bdd93c00`.
- Natural composition package: `available=true`, `mode=shadow`,
  `serving_authority=false`. Activation
  `6012af4d2eb529efb7d46bbb3f33e2e2aa1fec01c5dcf36179961a87dda00a76`.
  The default live-shadow switch was false in the audit process.

The first flag authorizes eligible exact-contract execution; it does not prove
that a running process consumed the package. The second exposes an observation
package and cannot replace an ordinary answer. G11 remains open.
An isolated-state import also logged `migration_authority_key_unavailable`;
the explicit-key validator passed. That test-environment warning is not a live
runtime failure.

## Next causal repair

The frozen weave rows contain 23 `typed_argument_chart_empty` failures and four
incorrect emitted answers. The existing compositional transducer first grounds
inputs, proposes operation charts, and assigns typed dependencies. G03 targets
that binding/composition stage. These 48 examples are now exposed diagnostic
material; improvements on them cannot be reported as fresh replication.

Broad reasoning gain, frontier performance, static RLC weight fusion, and
general live serving remain unestablished. Repeating the backbone more times
does not inherit the typed executor's positive evidence.

Acceptance checks: `tests/test_rlc_release_baseline.py`,
`tests/test_rlc_evidence_reconciliation.py`,
`tests/test_semantic_program_frozen_path_replication.py`,
`tests/test_compositional_semantic_qualification.py`, and
`tests/test_compositional_semantic_shadow.py`.

Validation: 40 focused checks passed; smoke passed 164 with one skip.

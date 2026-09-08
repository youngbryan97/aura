# G01: Frozen RLC baseline and claim boundary

Recorded 2026-09-08 by Codex. Source snapshot:
`e081980444c656fd84e1243f530602cf5295d353`.

The [machine-readable baseline](G01_RLC_BASELINE_2026-09-08.json) pins nine
retained evidence files and seven implementation files. It is a baseline of
the evidence available at this source snapshot. No new model benchmark ran to
produce this record. Freezing it closes G01; it does not close G02-G12.

## Model identity

The configured checkpoint is
`Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15`, descriptor
`52d313c2c435d343cf6acfa2b2ca61bc70334c01b8b17877df79f32ec3c5283c`.
The active manifest, checkpoint config, weight index, tokenizer and tokenizer
config were read locally and SHA-256 hashed. Config and index match the
retained 27B adjudication. This pass did not rehash model weight shards or
prove which model an active process had loaded. Current serving activation
requires separate live evidence under G02/G11.

## Retained measurements

| Evidence | Treatment | Controls and boundary |
| --- | --- | --- |
| CP566, 32B, four executable families | 60/60 exact decoded answers | Ordinary 16, wire 7, coefficient lesion 5, wrong-state 0. Historical fixed-contract result. |
| CP1003/CP1011 recovery, 27B | 60/60 exact decoded answers | Ordinary 0, wire 6, coefficient lesion 4, wrong-state 0. A separate cohort; no direct model-ranking inference. |
| v14 endogenous withheld-family replay | 79/96 programs and answers | Coefficient lesion 0/96. The complete target program family was withheld from fitting; its primitive vocabulary was shared. Shadow-only evidence. |
| v16 fresh natural-language replication | 26/96 answers; 22/96 programs | Failed the preregistered 48/96 answer floor. Ordinary decode was not run after the futility stop. |
| v19 clause-local identity repair | 93/96 programs and answers | Coefficient lesion 0/96. Development evaluation of an already exposed cohort; fresh replication remains required. |

Both decoded-answer studies retain all 300 task-arm outputs each. The G01
acceptance check regenerates their cohorts through the existing independent
verifier and regrades all 600 outputs. It also recounts the retained v14 and
v19 boolean outcome rows. This checks the archived counts, not new neural
inference, source-current replay, or independent external validation.

The ordinary 27B arm had no parseable final answers under the recorded decode
contract. That is a result of that evaluation contract, not a statement that
the checkpoint lacks the underlying knowledge. Token limits, formatting,
state conditioning and retry policy must be reported when comparing arms.
The v16 failure is retained even though its treatment beats a coefficient
lesion. Statistical separation from a damaged mechanism does not satisfy an
absolute capability floor or replace the missing ordinary comparison.

## Mechanisms

The historical frozen-backbone latent loop is distinct from both mechanisms
below. Its existence or runtime activity alone is not evidence of a gain.

CP566 and the 27B recovery execute a semantic neural machine over typed
problems, then expose the resulting state to the model's answer decode. The
canary includes state-conditioned inputs and bounded serialization handling.
The learned coefficient lesion targets that machine. This evidence must not
be described as a proof that arbitrary additional middle-layer passes through
the resident language model improve its general reasoning, or as proof that
the machine was fused into the language model's weights.

The semantic-program track learns a transducer from resident hidden features
to typed operations and argument bindings. Accepted programs are executed by
the existing exact executor or universal metered floor. The neural component
supplies the learned language-to-program mapping; the execution substrate
supplies the program's exact computation. Their combined answer score does
not isolate neural execution of every primitive.

## Claim boundary

Supported by the retained studies: bounded state-to-decoded-answer transfer
on two separate cortex generations, and learned composition across a withheld
synthetic program family whose reusable primitives were already available.

Not established here: broad natural-language reasoning gain, unseen-primitive
induction, frontier-model parity, static RLC weight fusion, universal
no-regression behavior, or present live serving. The persona fusion in the
configured model is a separate mechanism from RLC fusion.

The next scientific comparison must freeze the selected candidate and use
fresh tasks after development ends. It needs ordinary and equal-compute
controls, causal lesions, complete outcomes and preregistered stopping rules.
The v19 development score cannot become that comparison by relabeling it.

## Executable acceptance

```sh
AURA_LOG_DIR=/tmp/aura-g01-20260908/logs \
AURA_STATE_ROOT=/tmp/aura-g01-20260908/state \
/Users/bryan/.aura/live-source/.venv/bin/python -m pytest -q \
  tests/test_rlc_release_baseline.py
```

The check binds artifact bytes and schemas, verifies the recorded source
snapshot against git objects, regrades historical decoded answers, preserves
the failed replication and refuses stronger claims in this baseline.

Validation: the baseline suite plus `test_evidence_integrity.py` and
`test_live_claim_language_boundaries.py` passed 44 tests in 5.43 seconds.
`make smoke` passed 164 tests with one skip in 50.84 seconds. Focused Ruff
passed. The skip is retained as unrun coverage.

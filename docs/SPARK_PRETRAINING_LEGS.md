# Spark pre-training legs (F7, Fable lane)

Seven surfaces built on 2026-07-27 so the resident-32B campaign has its
scaffolding waiting rather than blocking. None of them runs a model. All of
them are strict, fail-closed validators over data with no Aura runtime imports
where possible, so the independent verifier can replay them without booting
cognition.

The single design rule they share: **an absence of a check is never reported as
a passed check.** Every one of these modules refuses an incomplete input rather
than treating the missing part as satisfied, because that is the defect this
codebase keeps re-finding — most expensively as CP227, whose accuracy gate was
voided after publication because nobody had checked that the treatment adapter
was in the forward pass.

| Item | Module | What it refuses |
|---|---|---|
| SPARK-064 | `core/learning/permanent_distillation.py` | A promotion missing any of seven regression batteries, or one that graded zero probes |
| SPARK-064 | `core/learning/permanent_distillation_registry.py` | A durable write that rewrites or truncates stored history |
| SPARK-065 | `core/learning/architecture_meta_controller.py` | An architecture change with no finding, out of bounds, tried in the live runtime, self-approved, or bought with extra compute |
| SPARK-063 | `core/learning/star_iteration_ledger.py` | A holdout that overlaps *any* prior iteration's training set, or a reused holdout |
| SPARK-066 | `core/brain/llm/latent_cortex/penultimate_execution_receipt.py` | A latent claim whose adapter never activated, whose passes never diverged, or whose decode ignored the recurrent state |
| SPARK-067 | `core/brain/llm/latent_cortex/coupling_matrix.py` | A seam whose only evidence is a copied field, a one-way read, or an effect that survives its own lesion |
| SPARK-068 | `core/brain/llm/latent_cortex/journal_{state,accumulator,witness}.py` | A campaign envelope that costs O(n) per intervention |

## SPARK-064 — promoting anything the campaign produces

```bash
python tools/promote_distilled_generation.py --registry <path> open \
  --artifact-id frozen --base-model <id> --adapter-identity <id> \
  --artifact-root <dir> --file adapter.safetensors
```

Then, when the campaign has run its batteries:

```bash
python tools/promote_distilled_generation.py --registry <path> promote \
  --artifact-id trained --base-model <id> --adapter-identity <id> \
  --artifact-root <dir> --file adapter.safetensors \
  --gate-report gates.json
```

**Do not hand-write `gates.json`.** `core/learning/permanent_distillation_gates.py`
produces every row from the battery's own receipt —
`build_gate_report(interference_receipt=..., capability_guard=...,
capability_report=..., behavior_verdict=..., memory_before=..., memory_after=...,
memory_case_count=..., frontier_before=..., frontier_after=...)` takes all seven
measurements as required arguments and derives the counts and verdicts from
them. The JSON path below exists for campaigns whose batteries run in separate
processes; serialize the produced report, do not compose one by hand.

`gates.json` is `{"gates": [...]}` with one row per id in `REQUIRED_GATES`:
`anti_interference`, `capability_families`, `personality_retention`,
`tool_effect_honesty`, `authority_safety`, `memory_retention`,
`frontier_regression`. Each row carries `battery_schema`, `probes_graded`,
`probes_passed`, `verdict`, `evidence_sha256`.

Exit codes are the interface: `0` promoted, `2` refused with every responsible
gate printed, `3` the report itself was invalid (a missing battery lands here —
it is an invalid report, not a failing one). A refusal is a result to record,
not an error to retry until it passes.

Rollback re-hashes the restored bytes off disk and refuses unless they equal
the target generation's recorded manifest exactly.

## SPARK-063 — keeping the flywheel's holdout a holdout

`star_iteration(...)` per iteration, then `validate_star_lineage(records)`
across the run. The lineage check is the one that matters: it catches a holdout
that was trained on three iterations ago, which no single iteration can see.
`lineage_trend(...)` will not produce a score series over a lineage that fails
validation, so the number nobody should quote without its disjointness cannot
be computed without it.

Tool-assisted and latent traces need a `trace_gate` recorded as passed before
they may appear in `training_trace_classes`; a later failing gate withdraws the
class again.

**Read this before minting a holdout.** `core/learning/star_iteration_producer.py`
derives the fingerprints from real task objects and grades the holdout itself.
It also carries a repair you need: **the flywheel's `EVAL_SEED_FLOOR` separates
seeds, not content.** `generate_battery` has no cross-battery exclusion and its
template space is small — measured over seeds 3-39, 15.5% of seed pairs share at
least one prompt. Two "independent" batteries can therefore contain the same
task. Use

```python
holdout = mint_disjoint_holdout(
    seed=..., size=..., excluded_fingerprints=everything_trained_on_so_far,
)
```

rather than trusting a different seed. This is the same defect CP385 repaired in
the recurrence curriculum; it is still live on the held-out-battery path.

## SPARK-065 — bounded architecture control

`architecture_findings(observations)` → `propose_architecture_change(...)` →
`candidate_trial(...)` → `approve_architecture_change(...)` →
`evaluate_rollout(...)`. The trial needs a runtime identity distinct from the
live one and all six invariants present; the approver must differ from the
proposer; the rollback revision is required before the canary starts.

**The measurement side is wired.** `core/learning/architecture_observations.py`:

```python
observations = observe_architecture(
    dynamics_reports=[...],      # trajectory_dynamics, one per episode
    action_transitions=[...],    # value_of_computation transitions
)
findings = architecture_findings(observations)
```

Depth comes from the loop's own convergence report; expert/router from real
operator selections (the checkpoint is dense — there is no MoE router, so those
words mean the operator inventory and the policy over it). The episode count is
derived from the evidence and cannot be passed in, and malformed evidence is
refused rather than skipped.

`router_misroute` is not produced: nothing measures a routing decision against
a known-correct one, so it shows up in `surfaces_unmeasured` rather than as a
healthy zero. Wiring a misroute oracle is open work.

## SPARK-066 — proving the answer came out of the latent path

`penultimate_execution_receipt(...)` per arm per episode, then
`latent_execution_verdict(receipt, require_adapter=...)`.

**The producer for `activated_blocks` is wired.** `ScopedLoRALinear.from_base`
takes `block_index` and `site`; both attachment sites supply them;
`RecurrenceAdapterActivation` records `applied_blocks` / `applied_sites` per
application; and `attach_adapters` returns `adapted_sites` and
`adapted_block_indices` so a receipt can be built against measured values
rather than asserted ones:

```python
adapter={
    "attached": True,
    "expected_blocks": wiring["adapted_block_indices"],
    "activated_blocks": activation.activated_blocks(),   # measured
    ...
}
```

`tools/eval_intrinsic_accuracy.py`'s dark-adapter gate was upgraded with it,
from *something fired* to *everything wrapped fired*: a block where any adapted
site stayed dark stops with `instrument_partial_adapter` and names the first
dark site. `calls > 0` was never proof the treatment ran — only that at least
one projection did.

**The state producer is wired too.** `core/learning/intrinsic_recurrence_receipt.py`:

```python
hidden, receipt = run_and_receipt(
    model, tokens, plan,
    wiring=wiring, identity=identity,
    answer_sha256=..., decoded_token_count=..., adapter_sha256=...,
)
```

It opens the adapter scope itself (forgetting it is the CP227 failure),
digests the per-iteration trajectory, and sets `decode_state_sha256` to the
**window's** last output — not the post-coda hidden, which would make the
"decode consumed the recurrent state" check true for any forward pass ever run.

What remains is calling it from the live worker's decode path on the resident
checkpoint.

## SPARK-067 — coupling that is not just a copied field

`coupling_seam(...)` per subsystem, then `coupling_matrix(seams)` over all
nine. Each seam needs forward evidence, reverse evidence, and a lesion that
removes at least half the intact effect.

**Do not hand-label `kind`.** Use the harness:

```python
seam = measure_coupling_seam(
    subsystem="memory", trials=64,
    forward_closed=..., forward_open=..., forward_metric=..., forward_identity=...,
    reverse_closed=..., reverse_open=..., reverse_metric=..., reverse_identity=...,
    lesioned=...,
)
```

It runs the behavior with the seam open and closed and compares each trial's
**observable outcome identity**. If nothing observable changed, the evidence is
`metadata` however you would have described it. `outcome_identity` is the
important argument: it must return what a downstream observer can actually act
on, not the field the seam passed.

Binding the nine seams to real subsystem callables is the work that remains.

## Before you start: run the preflight

```bash
python tools/spark_pretraining_preflight.py --campaign <artifact-dir>
```

It hands every instrument above the exact input it must refuse and fails if the
refusal does not come — the same idea as the campaign kernel probes. A weakened
gate still imports and still returns a verdict; it just stops saying no, and
nothing but an attempt to make it say no will notice. Exit 0 ready, 1 an
instrument is dead, 2 a campaign artifact failed validation.

## SPARK-068 — the campaign journal stops costing quadratically

`build_journal_witness(events, plan=..., attempt_id_for=...,
checkpoint_sequence=k)` replaces `campaign_journal_prefix`. The existing
complete-prefix envelope is untouched and still authoritative; adoption is the
march's call.

Two things to know before adopting:

1. `verify_journal_witness` takes `trusted_checkpoint_state_sha256` with **no
   default**. A witness cannot prove its own checkpoint state — someone
   replayed genesis→k once to compute it. Passing that digest explicitly is the
   safety property. `checkpoint_sequence=0` trusts nothing and replays
   everything, which is exactly today's behavior.
2. The journal's transition function now lives in `journal_state.py` and
   `action_intervention._validate_journal_prefix` delegates to it. The rules
   and refusal messages are unchanged (verified: the existing intervention and
   calibration suites pass 22/22 across the extraction). Do not add a second
   copy of the fold — a drifted copy would pass one validator and fail the
   other.

At an 8-event checkpoint cadence a 159-event campaign's envelopes carry under a
tenth of the events the prefix form carries.

## What none of this is

No model ran. No capability, reasoning, or frontier claim is made or supported
by any of it. Every checkbox in the Spark ledger for these items stays open,
and the reasons each stays open are recorded there per item. These are the
walls of the room the experiment will run in, built while the room was empty.

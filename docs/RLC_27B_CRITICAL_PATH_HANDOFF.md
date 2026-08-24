# 27B critical path: what is done, what is left, what runs it

Status: Handoff · 2026-08-24 · CPU-only, no model launched or loaded

`make rlc-27b-critical-path` is the one command. It reads every CPU gate and
prints either the blockers or the two commands, never both.

## Answered by measurement this round

**The persona is already in the 27B.** The signed migration contract carries
`persona_crsm` as `fused_persona_crsm` with a fusion plan and a fusion receipt:
496 fused modules over 700 cumulative iterations, onto
`Qwen3.8-27B-4bit-3e6447f082e8`. It does not need a recovery run. It needs
verification and behavioural continuity testing.

**Logical parameter counting is exact for the hybrid.** Counted by layer kind
from the real projection shapes, it lands within 0.00002% of the checkpoint's
own index metadata, and the 32B still lands at 32.8B. A hybrid whose recurrent
geometry is not declared returns unsupported rather than a plausible default.

**The campaign's cost is counted, not guessed.** CP566 made 364 decode calls
for 300 arm rows in 4,814.53 s. The ordinary control is 2,457 of those seconds
— half the campaign — because it generates 279 tokens against the typed arms'
39 to 62. It is also the floor the claim is measured against, so its length is
not available as a saving.

## Capability dispositions

| Capability | Disposition | Basis |
|---|---|---|
| `persona_crsm` | **portable** | fused into the checkpoint, signed |
| `steering` | **retrain required** | `model_basis_quarantine` |
| `recurrence_native` | **retrain required** | `model_basis_quarantine` |
| `expert_adapters` | **retired** | `retirement_inventory` |
| `qualified_rlc_serving` | gated outside the contract | descriptor pinned in the activation record |
| `grounding_contracts` | gated outside the contract | re-derived from the live tokenizer |
| `fast_weight_surfaces` | gated outside the contract | identity at attach, proven erase |
| `episodic_plasticity` | gated outside the contract | built and destroyed per episode |

Nothing is uncovered. A capability with its own gate is reported as gated, not
as a gap — reporting a false gap buries the real one.

## Optimizations: adopted, already there, rejected

**Adopted** — five stage commands become one resident worker, 364 task
tokenizations become 60 bound to a digest, and verification, hashing and
adjudication move outside the model-owning process. Decode calls before and
after are identical, on purpose.

**Already there** — prompt-prefix caching. The ordinary arm prefilled zero
tokens on all 60 tasks. Reporting it as a saving would invent a speedup out of
existing behaviour.

**Rejected** — cross-arm batching, because `_arm_order` counterbalances arm
order per task and batching destroys the interleaving counterbalancing exists to
create. Dropping serialization retries, because the 64 extra decodes
concentrate in the two arms designed to fail. Sharing post-treatment state,
because an arm reading another arm's state is not that arm.

The rule is written down and tested: an optimization that changes any arm's
measured compute is rejected, **including one that makes a control cheaper**.

## No wall-clock claim

Every adopted optimization removes process and CPU overhead that no retained
receipt times separately. The saving is counted in loads and tokenizations, and
reported unmeasured in seconds. Training wall time appears in no receipt at all
and is carried as `null`.

## Remaining model-active work

Five contiguous stages inside one residency: calibration → training → canary →
lesion arms → export. Verification and adjudication run after the unload, from
files, in a separate process. No model-active stage may overlap another; a test
fails if one ever claims it can.

## Current blocker

Memory. The receipt reports `insufficient_ram` while the resident lane is held —
which is what an environmental blocker is for, and it clears when the lane frees.
Read the receipt at launch time rather than this paragraph.

## Commands

```bash
make rlc-27b-critical-path
```

Launch and promotion are separate. Training completing is not authorization to
serve: promotion requires independent verification and adjudication to have run
on the exported artifacts, after unload.

## For Codex

`core/brain/llm/decoder_topology.py` is now yours; CP988 routed four training
cache sites through it rather than growing a second abstraction.
`_cache_from_state` in `recurrence_native_objective_v5.py` now takes the model,
because rebuilding a boundary as KVCache-per-entry restores a gated-delta
layer's recurrent state into an attention cache.

One non-reproducing failure to be aware of:
`test_recurrence_native_objective_v5.py::test_legacy_v1_receipt_replays_detached_softmin_without_changing_history`
failed once inside a batch under memory pressure and passed on two subsequent
runs, isolated and batched, here and on the primary checkout. Not reproduced,
not diagnosed.

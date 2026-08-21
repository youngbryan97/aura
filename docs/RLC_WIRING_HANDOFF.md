# RLC wiring handoff (CP234) — seams closed CP236–238

State as of Jul 20 2026 (evening). All seven Anima Rationis components
exist with tests, and the seams this document mapped are now closed:

* **CP236**: learned halting is ATTACHABLE — `CortexConfig.halting`
  loads a trained `HaltingHead` from disk (save/load added), attaches it
  to every branch, and the episode receipt carries `head_was_causal` with
  a `learned_halting_not_causal` flag when the head never fired. A
  requested head that cannot load refuses the episode. Plus the two
  dynamics gaps the evidence demanded: `rotation_pressure` (direct
  training pressure on cos(pass1,pass2)=0.9994; cos² over consecutive
  window-pass increments punishes idempotence AND the period-2 cycle)
  and `trajectory_shaped_rewards` + `latent_step_answer_ce`
  (RLTT-style latent-trajectory credit for GRPO; verifier stays the
  last word, reordering is confessed).
* **CP237**: `record_paired_outcome` has a real caller —
  `tools/schedule_search_campaign.py` runs search → paired holdout
  trials → per-task-committed outcomes → the LIVE library, persisted.
* **CP238**: the integrated evaluation exists —
  `core/learning/integrated_eval_tasks.py` (retrieval-DEPENDENT tasks
  whose answers exist only in organ context) +
  `tools/integrated_rlc_eval.py` (paired context-on/context-off arms,
  distractors in both, ingress-fired receipts, leakage self-check).

Remaining architecture decision (unchanged): whether the live engine
adopts intrinsic recurrence, slot recurrence, or both — CP227's training
result decides it. Original map below for the record.

## What is done

| component | module | live seam |
|---|---|---|
| 1 recurrent depth | `core/learning/intrinsic_recurrence.py` | **none** |
| 2 writable slots | `core/brain/llm/latent_cortex/engine.py` | live |
| 3 schedule search | `core/brain/llm/latent_cortex/schedules.py` (`ScheduleSearch`) | **ALREADY LIVE** via `_resolve_schedule` -> `ScheduleLibrary`. CP235 added held-out separation, generalization gap and a compute cap to it. |
| 4 virtual width | `core/consciousness/parallel_branches.py` | live |
| 5 latent optimization | `core/brain/llm/latent_cortex/latent_opt.py` | **ALREADY LIVE** — engine.py:1372, with matched-random control AND manifold drift. `core/learning/latent_optimization.py` (CP231) duplicated this and has since been removed; the path above is the live one. |
| 6 fast weights | `core/brain/llm/latent_cortex/fast_weights.py` | live |
| 7 adaptive halting | `core/learning/adaptive_halting.py` | **LIVE** — `HaltingController.halting_head` (recurrence.py); None = old policy |
| RLVR | `core/learning/grpo.py` + `tools/train_grpo.py` | n/a (training) |
| verifiable tasks | `core/learning/verifiable_tasks.py` | n/a (data) |

## Next: call the halting bridge from the engine

The engine halts on residual convergence. `steps_taken` and `residual_trail`
are fields on `EpisodeReceipt` in
`core/brain/llm/latent_cortex/types.py:1447`. The ensemble check is
`core/brain/llm/latent_cortex/engine.py:4348` (`ensemble.all_halted()`).

1. Add `halting_mode: str = "residual"` to `RLCExecutionSpec`
   (`execution_spec.py`). Note `adaptive_halting: bool = False` already
   exists and is a **v2-training constraint validator**, not a dead flag --
   do not repurpose it.
2. In the per-branch step loop, replace the residual comparison with
   `should_halt(step=..., residual_trail=branch.halting.residual_trail,
   config=..., head=..., state=branch.z)`.
3. Collect verdicts per branch; attach `bridge_receipt(...)` to the episode
   receipt. **Check `head_was_causal`** -- a learned run whose every stop
   came from the residual floor is the old policy under a new name.
4. Default MUST stay `residual` until a trained head beats it offline.

## Component 5 needs NO wiring — it was already live

`core/brain/llm/latent_cortex/latent_opt.py` is wired at `engine.py:1372` and already has
both honesty controls: `control_mode` applies matched-magnitude random
perturbations sized from the true gradient step, and the objective carries
a manifold term (RMS + cosine drift from the post-prelude seed).

`core/learning/latent_optimization.py` (CP231) was written without finding
this and duplicated it — an audit that searched `latent_optim` missed a
module named `latent_opt`. It has since been **deleted**, with its unique
properties ported into the live module (see "the audit searched guessed
filenames" below). **Search by capability, not by guessed filename.**

`spec.latent_opt_mode` is validated to `"disabled"` for v2 TRAINING only;
that is a training constraint, not the live default.

## Component 3 needed no new module either

`ScheduleSearch` already existed in `core/brain/llm/latent_cortex/schedules.py`, and
`engine._resolve_schedule(domain)` already consults `ScheduleLibrary`.
CP235 ported the missing safety properties INTO it rather than keeping a
parallel implementation:

* `holdout_evaluator` scored once after search, on the winner only
* passing the same callable for both is REFUSED
* `generalization_gap()` / `overfit_warning()` on the result
* `max_layer_apps`, which refuses a budget below the seed schedule (the
  seed enters the pool unconditionally, so such a cap bounds nothing)

Remaining work: feed real verified task scores into `library.record_paired_outcome`.

## AUDIT LESSON (cost real hours on Jul 20)

Three components were reported "NOT BUILT" and rebuilt as duplicates
because the audit searched GUESSED FILENAMES:

    "schedule_search"  -> missed schedules.py::ScheduleSearch
    "latent_optim"     -> missed latent_opt.py
    (verifiable tasks) -> missed heldout_battery.py

`core/learning/schedule_search.py` and `core/learning/latent_optimization.py`
have been DELETED; their unique properties were ported into the live
modules. **Search by capability (grep for the behaviour), never by guessed
module name.**

## Then: intrinsic recurrence (component 1)

`intrinsic_recurrence` currently only runs in `tools/`. Deciding whether the
live engine uses slot-recurrence, intrinsic recurrence, or both is an
architecture decision, not a wiring task. CP227's result should inform it.

## Standing hazards in this codebase

Six bugs on Jul 20, all one species: **a mechanism that appears present but
does not fire.**

- checkpointing delegated to a scope bound to a different module (no-op)
- params snapshotted outside the trace -> all-zero gradients (looks converged)
- `adaptive_depth_loss` returns a DEPTH; treated as an index (silent mislabel)
- metadata `{"ordered": True}` beside a set-equality grader
- held-out eval after a `continue` -> a run never measured itself
- `sqrt(0)` NaN gradient at the optimizer's own starting point

**Test that the mechanism FIRES, not that it exists.** Every bridge here
carries a `*_was_causal` field for that reason.

## Bar for success

Anima Rationis line 658, and do not soften it: +5 broad unseen reasoning,
+15 hard verifiable, 2x at the failure frontier, <=2 point decline
elsewhere, positive d(accuracy)/d(steps), transfer to unseen families,
causal evidence.

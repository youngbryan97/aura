# Intrinsic recurrence

Status: Guide · Reviewed against the tree 2026-08-24 at CP1012

The live front of the [Recursive Latent Cortex](RECURSIVE_LATENT_CORTEX.md)
programme. This page covers the pivot from *talking to* a recurrent workspace
to *being* recurrent, and the training work that follows from it.

Read [RECURSIVE_LATENT_CORTEX.md](RECURSIVE_LATENT_CORTEX.md) first if you
haven't — this page assumes the frozen-loop result.

---

## The finding that caused the pivot

The RLC as originally built was a model that **talked to** a recurrent
workspace. Reading its own `_persist_and_score`: the answer tokens traverse
`layers[prelude:coda]` exactly once, at every depth setting. Only the four
slot positions were ever recurred.

So the computation producing the answer always received the base checkpoint's
64 layers — identical to vanilla. "Depth" changed nothing except what got
written into a scratchpad the answer attends to.

The measurement follows from the architecture, and should have been predicted
from it:

```
RLC depth 1 / 2 / 4 / 8 : 25 / 29 / 25 / 25 %   (8× compute, flat)
vanilla greedy          : 21 %
```

Slots are causal and worth a few points as a prior. **Depth was worth nothing
because no depth was ever applied to the answer's own computation.** The
frozen-loop negative result follows from the architecture, not from tuning.

`core/learning/intrinsic_recurrence.py` (CP226) applies depth where it
actually matters. The real token stream re-enters the middle block `T` times:

```python
h = layers[:prelude](h)                     # prelude, once
for t in range(T):
    h = layers[prelude:coda](h)             # the loop
h = layers[coda:](h)                        # coda, once
```

Effective depth becomes `prelude + T*(coda-prelude) + (L-coda)`. A 64-layer
checkpoint runs **160 layers deep at T=4, with the same weights**. This is the
Ouro / LoopLM architecture, retrofitted onto a checkpoint that was not
pretrained for it — which is the only reason the stabilizers exist at all.

### The load-bearing safety property

> At `T=1` this is bit-identical to the base forward pass.

Recurrence is added by increasing `T` from a known-good starting point, never
by a cutover. Every campaign carries a `T=1` no-recurrence anchor arm for
exactly this reason, and a preflight that finds the anchor missing blocks the
launch before any model loads.

A checkpoint pretrained without recurrence has no reason for its middle block
to be a stable map — iterating it can drift in norm until the coda receives
activations outside anything it was trained on. `anchor_injection` and
`renormalize` exist for that failure mode, and **both default to OFF** so the
plain loop is what gets measured first.

---

## Teaching process instead of output

The first training attempt was answer-only SFT. It taught the model to *stop
reasoning* — recurrence itself became the damage. Output-only transfer is now
rejected quickly and by contract (CP393).

What replaced it is a typed program the recurrence executes and is supervised
on. The controller emits, per recurrent step, a structured action against a
canonical schema (`core/learning/recurrent_action_schema.py`,
`aura.recurrent_action_target.v2`):

| Slot | Meaning |
|---|---|
| `opcode` | which operation this step performs |
| `arg0`–`arg5` | typed operands |
| `terminal` | whether the program halts here |

The narrow opcode vocabulary that was proven first covers exact machine
semantics — `copy value`, `add/multiply/subtract modulo`, `boolean
not/and/or/xor`, `register affine`. CP394 extended it with seven broader
process meanings (`frontier traverse / enumerate / simulate / infer /
schedule / calibrate / audit`), values 9–15.

Because the targets are typed and the operations are exactly checkable, the
supervision signal is a verifier rather than a preference model. There is
nothing to game: a step either computed the right typed state or it didn't.

Alongside the action schema sit a state schema
(`recurrent_state_schema.py`), literal grounding bound to real tokenizer digit
token ids (`recurrent_literal_grounding.py`), an opcode grounding contract
(`recurrent_opcode_grounding.py`), and an answer-emission contract
(`recurrent_answer_emission.py`). `unified_intrinsic_recurrence.py` runs depth,
memory, correction, and halting on **one** resident-transformer trajectory —
additive and identity-initialized, so one iteration remains the base forward
until learned controller parameters are admitted.

---

## What has been established

Working on a 1.5B vehicle so the resident Cortex stays live. Every item below is
bounded to what it measured; none of them authorize a broad claim.

**Trained recurrence is not inert.** Package-depth trained parameters beat
their exact initialization control 3/7 to 2/7 (CP368) — a small margin against
the right control: same architecture, same decode budget, differing only in
whether the parameters were trained.

**The serving policy was the bug.** CP368's campaign failed because fixed
depth four discarded a correct depth-one answer and produced no new success
over ordinary decode.
Decode now produces two separately attested candidates from the same trained
controller — depth one and the package-qualified depth — and public exact
verification considers both. A deeper decode can *add* a success that depth
one missed; it can never *erase* a shallow answer already proven correct
(CP376). Source labels for the shallow arms are rejected unless the resource
receipt records recurrence depth one.

**Typed recurrent serving authority is durable, and narrow.** The resident Cortex
holds `qualified_typed_only` authority for the `khop`, `modular`, and
`register_trace` families at task depths 1, 2, 4 with recurrence depth four
(CP357). Two independent cold loads each decoded 9/9 typed cases exactly; the
durable pointer reopened, rollback completed, and both canary authorities
expired after their requests without exposing token outputs.
`ordinary_chat_authorized=false` and `arbitrary_reasoning_authorized=false`
are the standing boundary.

**Proven tissue can be extended without ambiguous inheritance.** CP396
recovered the exact CP232 parent immutably. CP399 built a certified
append-only semantic migration for the seven new opcode meanings. CP400 then
corrected CP399's own rule: the state, action, and literal codebooks are
*learned tissue*, and requiring equality against a fresh initialization would
have discarded the exact parent CP396 had just recovered. The migration now
starts from all 51 exact parent tensors and overwrites only
`controller.action_value_embeddings[opcode, 9:16]`. Everything else is
verified byte-exact, and independent evaluation recomputes the merge and
checks the reconstructed step-zero controller hash.

**A bounded resident-32B gain replicated, lesion-dependent (CP566).** On a
frozen four-domain semantic cohort — coding, calibration, misleading premise,
scientific inference, 15 tasks each — the trained controller answered 60/60
exactly against 16/60 for ordinary decode, over 300 decodes. The two controls
that matter both fell further: a matched wire base reached 7 and a coefficient
lesion 5, and a matched wrong-state arm reached 0. Forty-four gains, no
regressions in any family, paired one-sided exact *p* = 5.7 × 10⁻¹⁴. The
adjudicator's verdict is `BOUNDED_WOW_SIGNAL`, and its own limitations line is
part of the record: bounded executable families, not open-domain general
reasoning, not static fusion, not frontier performance, not consciousness
evidence.

The lesion result is what makes it a mechanism claim rather than a score.
Every family separated under lesion, so the gain tracks the trained
coefficients rather than the extra decode budget. One family — misleading
premise — gained nothing at all, and the record keeps the reason rather than
averaging it away: ordinary decode was already at ceiling there (15/15), and
treatment preserved all fifteen instead of manufacturing a gain by regressing
its own baseline. The 44 came from the other three families.

**The bounded mechanism survived migration to a different cortex generation
(CP1011).** The fused Qwen3.8-27B uses a different architecture, vocabulary,
and model descriptor; representation-bound 32B adapters and steering vectors
were quarantined rather than relabeled. On a separately seeded cohort with the
same four executable families and five-arm design, the descriptor-bound 27B
treatment answered 60/60, ordinary decode 0/60, matched wire 6/60,
coefficient lesion 4/60, and wrong-state 0/60. Sixty gains, no regressions,
exact one-sided *p* = 8.67 × 10⁻¹⁹. An independent verifier replayed all 300
journal rows before frozen adjudication returned `BOUNDED_WOW_SIGNAL`.

That is evidence for **bounded architectural portability**: the result is not
confined to one checkpoint's representation geometry. It is not evidence that
27B is generally better than 32B, because these were separate cohorts rather
than a paired head-to-head benchmark, and it does not expand the admitted task
grammar.

**It runs in the live serving path (CP568, CP824, CP1011).**
`core/brain/llm/semantic_neural_serving.py` refuses to serve unless the
activation record says `active_by_default` and matches the active model
descriptor. The current package is
`rlc-27b-recovery-05346acd618d1c925f16`. Runtime verification over 120
tasks — four domains, three difficulties, three scientific surface profiles —
was 120/120 exact, 120/120 lesion-disrupted, 120/120 through foreground and
service integrations, with unsupported language refused. Latency is per
package, and each one carries its own measurement:
the CP567 qualification ran a median 46.160 ms and a maximum 83.188 ms, the
CP568 shadow 34.686 / 63.737 ms, and the prior active `r1` package measured
6.501 / 17.102 ms on the 32B. The current 27B package measured
9.229 / 38.696 ms. `make rlc-figures` recomputes retained historical figures;
the 27B evidence remains under `artifacts/migration/27b/recovery/`.

CP824 removed the last thing keeping it out of reach: admission had been
coupled to `desktop_required`, and ordinary sovereign chat does not set that,
so an active certified package was unreachable from the surface it had been
promoted to serve. Qualified recurrence is an answer contract rather than a
desktop-control one, and the lane now admits any foreground turn.

What that does *not* mean is that the recurrence answers ordinary chat.
`ordinary_chat_authorized` and `arbitrary_reasoning_authorized` are both still
pinned `False` in `core/brain/llm/unified_recurrent_qualified_activation.py`,
and an activation record that says otherwise is rejected as
`qualified_activation_authority_invalid`. Admission runs an answer-blind
parser over the turn and lets it through only when the complete public task
grammar is recognised. Unsupported language never acquires the model lane.

**Small-checkpoint falsification ran before resident expense.** The retained
five-arm run passed heldout likelihood transfer against base and every
equal-work control, and passed all teacher-forced regression families. The
generated behavior gate **failed** at 1/12 trained canaries, including three
zero-token answers. That completes the experiment requirement; it does not
authorize resident training.

## What is not established

The ledger restates this at every checkpoint:

- No broad behavioral gain, and nothing open-domain. The measured gain is on
  four named executable families and does not transfer beyond them.
- No static fusion.
- No frontier performance.
- No consciousness evidence.
- No unrestricted promotion: the recurrence is reachable from ordinary chat
  turns and still cannot answer ordinary chat, because admission is decided by
  a parser over the task grammar rather than by the router.

Two entries left this list on 2026-08-20, and it is worth naming which and
why. "No resident-32B reasoning improvement" and "no `WOW Signal`" were both
retired by CP566, whose adjudicated verdict is literally `BOUNDED_WOW_SIGNAL`.
The word *bounded* is doing the work: a replicated, lesion-dependent gain on a
frozen four-domain cohort is a much smaller thing than the phrase suggests to
anyone reading it without the limitations line. Everything above still holds.

The next bounded milestone is the 1.5B adaptation over train depths
`1,3,4,5,6,8,10` with held-out `12,16`, followed immediately by the
already-frozen four-arm behavioral canary.

**This page is a snapshot; the ledger is the record.** It was reviewed at
CP832, and checkpoints land faster than a narrative page can track. Between
CP409 and CP832 this page stood still while the programme moved past two of
its own "not established" bullets — the exact drift the paragraph below
promises would not happen quietly, happening quietly. Everything
above is *mechanism and status*, which changes slowly. For what happened most
recently, read the tail of
[RLC_SPARK_EXECUTION_LEDGER.md](RLC_SPARK_EXECUTION_LEDGER.md) and
[AURA_EXECUTION_TRACKER.md](AURA_EXECUTION_TRACKER.md) — both append-only.
What will *not* have changed silently is the "not established" list: every
checkpoint in this programme restates it explicitly, so if a broad gain is ever
claimed it will be claimed in a named checkpoint rather than drifting into
being true.

---

## The code

| Path | What it is |
|---|---|
| `core/learning/intrinsic_recurrence.py` | The loop itself, `RecurrentDepthPlan`, the `T=1` identity property |
| `core/learning/unified_intrinsic_recurrence.py` | Depth + memory + correction + halting on one trajectory |
| `core/learning/unified_intrinsic_objective.py` | Structured state/action/initial-state losses and accuracy breakdowns |
| `core/learning/recurrent_action_schema.py` | Typed action targets, the opcode vocabulary |
| `core/learning/recurrent_state_schema.py` | Typed state targets from an execution trace |
| `core/learning/recurrent_opcode_grounding.py` · `recurrent_literal_grounding.py` | Tokenizer-bound grounding contracts |
| `core/learning/recurrence_checkpoint_migration.py` | The certified append-only codebook migration |
| `core/learning/recurrent_grpo.py` | The GRPO training path |
| `core/learning/recurrent_sft_falsification.py` | The small-checkpoint falsification battery |
| `tools/train_unified_intrinsic_recurrence.py` | The trainer entry point |
| `tools/run_unified_recurrent_broad_canary.py` | The frozen behavioral canary |
| `tools/run_unified_recurrent_shadow_lifecycle.py` | Cold-load shadow lifecycle and rollback |

Evidence is frozen under `artifacts/closeout/latent_cortex/`. The
checkpoint-by-checkpoint narrative is in
[AURA_EXECUTION_TRACKER.md](AURA_EXECUTION_TRACKER.md) — append-only, read the
tail as current.

**One training protocol at a time.** It is memory-hungry, and launching a
second beside the resident 32B will take the host down.

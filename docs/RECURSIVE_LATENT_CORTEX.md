# Recursive Latent Cortex (RLC)

Status: Guide · Programme landing page · Reviewed against the tree 2026-09-01.
Checkpoints land faster than this page tracks; the claims ladder
below changes slowly, and the append-only
[ledger](RLC_SPARK_EXECUTION_LEDGER.md) is the live record.

Aura's largest research programme. The question:

> A frozen 32B checkpoint is a fixed-depth pipeline — 64 layers, once, per
> token. Can you make it *think longer* on a hard problem without changing a
> single stored weight?

The machinery works. **The frozen loop answered no, and the programme's own
preregistered campaign is what proved it. Trained intrinsic recurrence then
answered yes inside a narrow boundary — an adjudicated `BOUNDED_WOW_SIGNAL`
on four named executable families, first on the resident 32B (CP566) and then
again after migration to a distinct fused 27B cortex (CP1011).**
The two mechanisms are different, and the claims ladder below keeps them apart.

Turn the frozen resident checkpoint from a fixed-depth 64-layer pipeline into
a **programmable, stateful, self-configuring reasoning machine** — without
changing a single stored weight. This is the productionization of the
"Anima Rationis" spec: latent workspace, controlled recurrence, layer-schedule
programs, virtual-width branches, hidden-state optimization, episode-scoped
fast weights, adaptive halting, and a falsification harness that keeps every
capability claim honest.

Package: `core/brain/llm/latent_cortex/` (worker-side, pure MLX, lazy imports —
156 modules)
Service: `ServiceNames.LATENT_CORTEX = "latent_cortex"` (orchestrator-side)
Worker action: `latent_reason` (runs on the RESIDENT model, no reload)
Surface, 2026-08-24: 55 `tools/` entry points and 231 test files naming the
latent cortex or recurrence, over 250 frozen evidence entries under
`artifacts/closeout/latent_cortex/`. These only grow; `make rlc-figures`
holds them to a floor rather than an exact count.

---

## The short version

Ordinary decoding runs the prompt through 64 layers and emits a token. RLC
seeds a set of **thought slots** beside the prompt, runs a *window* of the
middle layers over those slots repeatedly under a schedule program, and
persists the refined slots' K/V so every generated token attends to them.

The recurrence is where the extra compute goes. Nothing is written back to
disk, and the checkpoint bytes are hash-checked before and after every
episode.

What makes it a research instrument:

1. **The invariant is checked, not promised.** Checkpoint bytes, permanent
   parameters, episode fast-weight erasure, and no-hidden-fine-tuning are
   each enforced per episode with a receipt. A violation is a CRITICAL
   degradation and the output is discarded.
2. **Causality is testable.** Ablate a thought slot and the answer
   distribution must change. If it doesn't, the latent computation was
   decoration and the harness says so.
3. **Budgets are matched.** Equal-FLOP accounting (token-layer applications)
   is first-class, so "more compute helped" can never be mistaken for "the
   architecture helped."
4. **The floor is absolute.** `≥ vanilla always` — ordinary decode owns the
   answer until a gain gate promotes something over it. Enforced by
   `tests/test_rlc_never_worse_than_vanilla.py`, which enumerates the decode
   contract rather than trusting it.

## Where the programme actually stands

Read this before crediting anything. Verdicts are the programme's own, from
preregistered campaigns with committed seeds.

| | |
|---|---|
| **Mechanics** | **PROVEN.** KV rewind, RMSMatch stability, schedule validation, fast-weight identity-at-attach and proven-erase, checkpoint invariant, slot-ablation causality, matched-magnitude controls, equal-FLOP accounting — all on real `mlx_lm` Qwen2 weights, plus a full episode end to end on a trained 1.5B checkpoint in ~1.3s with contracting residuals (0.95 → 0.10). |
| **Runtime integration** | **PROVEN.** Live on both the historical resident 32B and current fused 27B through the signed installed app; deep deliberation routes DEEP passes through latent episodes. Kill switch `AURA_LATENT_CORTEX=0`. |
| **Capability gain, frozen loop** | **REFUTED at 1.5B scale.** The 2026-07-17 preregistered campaign (seed committed first, n=24/family, Holm-corrected) returned: slot causality REFUTED at n=72; all 7 factorial ablation arms REFUTED — vanilla 21/72 beat every latent arm (7–13/72); self-consistency beat virtual width; gradient latent optimization was indistinguishable from its random control *and* from off. On an untrained-for-recurrence checkpoint at this scale, the frozen loop does not merely fail to help — **it hurts.** |
| **Capability gain, 32B frozen loop** | **CONJECTURE (negative point estimate).** Template-parity sweep: latent 0.167→0.375 over 1→2 recurrent steps then plateau; vanilla 0.417 leads with fully overlapping Wilson intervals at n=24. Statistical parity. |
| **Capability gain, 32B trained intrinsic recurrence** | **`BOUNDED_WOW_SIGNAL` (CP566).** A different mechanism: the answer's own token stream re-enters the middle block, and the controller is trained on typed traces. On a frozen four-domain cohort of 60 typed tasks the trained controller answered 60/60 exactly against 16/60 for ordinary decode; matched wire base 7, coefficient lesion 5, wrong-state control 0; 44 gains, 0 regressions, paired one-sided exact *p* = 5.7 × 10⁻¹⁴. Adjudicated, replicated, lesion-dependent, and bounded to four named executable families. |
| **Cross-generation recovery on the fused 27B** | **`BOUNDED_WOW_SIGNAL` (CP1011).** A separately seeded 60-task cohort on the descriptor-bound Qwen3.8-27B resident cortex returned treatment 60/60, ordinary decode 0/60, matched wire 6, coefficient lesion 4, wrong-state 0; 60 gains, 0 regressions, exact *p* = 8.67 × 10⁻¹⁹. Independent verification replayed all 300 rows and runtime verification passed 120/120 exact plus 120/120 lesion disruptions. This establishes bounded mechanism portability across two cortex generations, not general model superiority. |
| **Family-blind procedure acquisition into neural tissue** | **SUPPORTED, BOUNDED (2026-08-31).** A generic inducer received 16 examples without a family label or solver, froze a depth-2 procedure, and a family-blind SSA lowerer executed it through learned arithmetic tissue at 96/96 exact on fresh inputs. Coefficient and wrong-input lesions disrupted 96/96, the no-procedure control solved 1/96, and 15 shuffled-output null searches found no fit. |
| **Resident decode of an induced neural procedure** | **SUPPORTED, BOUNDED (2026-08-31).** The fused 27B decoded the induced procedure's authenticated neural state at 8/8 exact against ordinary 1/8, wire 1/8, coefficient lesion 1/8, wrong-input 0/8 and wrong-state 0/8. Seven gains, no regressions, exact paired one-sided *p* = 0.0078125; all 48 rows independently replayed. |
| **Resident 27B language-to-program transfer** | **SUPPORTED, BOUNDED (2026-09-01).** A model-bound linear transducer learned token roles, primitive operations and register arguments from answer-blind resident hidden states. On 256 instructions whose construction combinations were absent from training, exact execution emitted the right answer on **134/256**; the recovered program itself was exact on **133/256**. Matched controls reached hidden-state shuffle 14/256, coefficient lesion 0/256 and label permutation 4/256. A source-bound verifier reloaded 576 feature records, reproduced the coefficients and report, and recounted 1,344 task-arm rows. |
| **Frozen fresh-cohort semantic transfer** | **SUPPORTED, REPLICATED, BOUNDED (2026-09-01).** The unchanged transducer was evaluated after a clean worker restart on 576 separately seeded tasks with zero example overlap. Its function-defining model, tokenizer, source, adapter, steering, recurrence and quantization basis matched exactly while session identity changed. Exact execution emitted **114/256** held-out answers against hidden-state shuffle 10/256 and coefficient lesion 0/256. Independent verification reloaded both feature bundles and recounted 1,728 task-arm rows. |
| **Typed sequence-family transfer** | **SUPPORTED, BOUNDED (2026-09-01).** The same answer-blind acquisition path was extended beyond integer arithmetic to typed sequence transformations followed by scalar aggregations. A training-only construction cross-validation procedure selected complementary lexical and causal views instead of averaging away operation meaning. On 360 held-out instructions the learned decoder executed **239/360** answers exactly and recovered **178/360** complete programs, against hidden-token shuffle 171/360, coefficient lesion 16/360 and label permutation 51/360. A separate process reloaded all 540 feature records, deterministically refit the decoder, exactly replayed the report and recounted 1,620 task-arm rows. This establishes a second semantic value type and non-arithmetic operation family; it is not broad-domain or frontier reasoning evidence. |
| **Native midpoint semantic transfer** | **SUPPORTED, REPLICATED, BOUNDED (2026-09-01).** One native resident forward now captures lexical, middle-layer and final token states without reimplementing the hybrid backbone. On the same 180 held-out diagnostic instructions, adding the midpoint channel raised exact answers **119→136** and exact programs **90→105**. The frozen transducer then crossed to a fresh seed, worker session and 540-example cohort with zero fitting calls or example overlap: **276/360** construction-held-out answers and **205/360** programs, against hidden-token shuffle 225 and 137. A powered refit independently returned 275 and 203. Both fresh test splits beat shuffle by exact paired tests, and separate processes replayed both certificates. The result identifies a better model-native semantic surface; it does not grant serving authority or establish broad-domain transfer. |
| **Mixed sequence-pointer transfer** | **SUPPORTED, REPLICATED, BOUNDED (2026-09-01).** The semantic path now composes binary sequence lookup or occurrence counting with a learned pointer into a scalar continuation. A 48-example diagnostic fit recovered **84/96** held-out answers and complete programs against hidden-token shuffle 35, coefficient lesion 0 and label permutation 11 answers / 8 programs. Without fitting or refitting, that frozen transducer then reached **171/192** exact answers and programs on a fresh 288-example seed and worker session, versus shuffle 71 answers / 68 programs and coefficient lesion 0. The test split beat shuffle at exact paired *p*=1.82 × 10⁻¹² for answers and 4.55 × 10⁻¹³ for programs. A fresh 96-example refit was slightly worse at 166 answers / 164 programs. Independent processes replayed all three records. This extends the bounded mechanism to mixed sequence/scalar values, binary arity and intermediate-register pointers; it does not establish broad-domain transfer or serving authority. |
| **One shared multi-family semantic cortex** | **SUPPORTED, BOUNDED (2026-09-01).** One coefficient set, with no family router, was jointly fit to arithmetic and mixed sequence-pointer programs. On held-out test constructions it recovered **59/128** arithmetic and **35/48** sequence programs exactly, against hidden-state shuffle 15 and 17, coefficient lesion 0 and 0, and label permutation 1 and 0. An independent process reloaded all 720 records, reproduced the frozen fit exactly and recounted 1,776 task-arm rows. This proves coexistence in shared tissue at one common two-instruction geometry; it also measures interference relative to the separately fit sequence transducer and does not establish variable-geometry generality. |
| **Cross-language operation semantics** | **SUPPORTED AS A REPRESENTATION DIAGNOSTIC, BOUNDED (2026-09-01).** A family- and geometry-blind linear head was transferred in all six directions among ordinary arithmetic, three-step fork/join arithmetic and mixed sequence-pointer language. Absolute operation-span states transferred strongly in five directions but failed sequence→fork/join at whole-program exactness (**27/192** versus geometry-only 24), localizing the remaining nuisance. Answer-blind centering within factorial counterfactual sets exposed the shared operation signal: the four zero-token-overlap test directions reached **47/48, 48/48, 128/128 and 179/192**, while coefficient lesions reached 12, 12, 8 and 3 and label permutation reached zero throughout. Independent replay reloaded 1,296 records and exactly reproduced 14,720 program-arm rows. Because centering requires a target contrast batch and gold operation spans, this proves latent semantic content beneath construction context, not a deployable single-request compiler. |
| **Recurrence-native training** | **OPEN, and now partly answered.** The dividend did come from training the checkpoint to use recurrence, on the bounded families above. Broad transfer is still open, and every CP-numbered checkpoint since is about that. |
| **Broad reasoning gain, fusion, frontier performance** | **NOT CLAIMED.** No checkpoint in this programme authorizes any of them, and each entry in the ledger says so explicitly — including CP566, whose adjudication ships its limitations line inside the same receipt as its verdict. |

### What the training front has established

The programme used a 1.5B vehicle while the historical resident 32B stayed live, at
`tools/train_unified_intrinsic_recurrence.py` and its evaluator:

- **Answer-only SFT is the wrong objective.** Training on answers taught the
  model to *stop reasoning* — recurrence itself became the damage. Output-only
  transfer is now rejected fast and by contract (CP393).
- **Trained recurrence is not inert.** Package-depth trained parameters beat
  their exact initialization control 3/7 to 2/7 (CP368) — a small margin,
  measured against the right control.
- **The serving policy was the bug.** Fixed depth four discarded a correct
  depth-one answer. Decode now produces separately
  attested depth-one and package-depth candidates, so deeper recurrence can
  add a success but can never erase a shallow one already proven correct
  (CP376, "recurrent depth is monotonic").
- **Typed recurrent serving authority is durable and narrow.** The resident
  32B holds `qualified_typed_only` authority for the `khop` / `modular` /
  `register_trace` families at task depths 1,2,4 — decoded 9/9 exactly across
  two independent cold loads. `ordinary_chat_authorized=false` and
  `arbitrary_reasoning_authorized=false` remain the boundary (CP357).
- **Process is now what gets taught** (CP394–CP400): a broad
  opcode vocabulary that executes causally, the proven narrow tissue recovered
  immutably from CP232, and a certified append-only migration that extends it
  to the new vocabulary while every other learned parameter stays byte-exact.
- **The gain arrived, bounded (CP566–CP824).** 60/60 against 16/60 on the frozen
  four-domain cohort, lesion-dependent, adjudicated `BOUNDED_WOW_SIGNAL`; then
  qualified as a content-addressed runtime package and promoted to the live
  serving path, where `ordinary_chat_authorized` stays pinned `False` and an
  answer-blind parser over the public task grammar decides admission.
  [INTRINSIC_RECURRENCE.md](INTRINSIC_RECURRENCE.md) carries the detail and the
  two entries this retired from the "not established" list.
- **The gain survived a cortex-generation change (CP1003–CP1011).** The fused
  27B could not inherit 32B representation-bound adapters or steering vectors.
  The model-independent typed tissue was rebound to the new descriptor and
  remeasured instead: 60/60 treatment, 0/60 ordinary, 6/60 matched wire, 4/60
  coefficient lesion, 0/60 wrong state, with all 300 journal rows independently
  replayed. The active package is
  `rlc-27b-recovery-05346acd618d1c925f16`; unsupported language remains refused.
- **A learned procedure now reaches the tissue without a family compiler.** A
  generic program inducer learned `idiv(add(in0, in1), in2)` from sixteen public
  examples, after which a family-blind SSA register allocator lowered that
  frozen program into the existing neural execution substrate. Independent
  replay verified 96/96 fresh transfers and causal coefficient, wrong-input
  and no-procedure controls. The primitive vocabulary and value types remain
  fixed; natural-language compilation, resident decode and broad reasoning are
  still open at this stage.
- **The induced procedure now survives resident decode.** A separate six-arm
  canary passed the induced neural state to the fused 27B answer surface. The
  treatment was exact 8/8; ordinary, wire and coefficient-lesion arms each
  solved 1/8, and wrong-input and wrong-state controls solved 0/8. Independent
  replay verified seven conversions, no regressions and exact paired one-sided
  *p* = 0.0078125. This closes serialization for the bounded procedure, not
  language-to-program learning or broad transfer.
- **Resident hidden language now reaches exact execution.** The 27B feature
  campaign learned one generic token-to-SSA transducer from five linguistic
  constructions and evaluated it on four unseen construction combinations.
  Exact objective execution emitted 134/256 held-out answers; matched
  hidden-state, coefficient and label controls reached 14, 0 and 4. Expected
  answers were used only after execution for scoring. The certificate binds
  the feature basis, learned coefficients, report and verifier source. A second
  certificate reproduced the result after the replication work changed two
  bound source files; the first certificate remains an immutable historical
  record, while the current claim reads the re-verification bound to the exact
  measured commit. Later source evolution does not rewrite the result or grant
  serving authority to a different implementation.
- **The semantic transducer survives fresh examples and a worker restart.** A
  second 576-task cohort used a new seed and shared no example ids with the
  training campaign. The frozen coefficients emitted 114/256 exact held-out
  answers against 10/256 under hidden-state shuffle and 0/256 under coefficient
  lesion. The compatibility receipt excludes only PID, boot id and boot signing
  identity; every field that can change the neural function must remain exact.

Next bounded step: expand to distinct procedures, richer constructions and
additional reasoning families beyond the fixed primitive vocabulary.
Regenerating the 27B-specific recurrent adapters and CAA vectors remains a
separate migration task; portable typed tissue does not authorize either
model-basis component by association.

### Two negative results that were void

Between 2026-08-06 and 2026-08-07 the programme produced two clean negative
results — 13-vs-5 and a 9-vs-4 reproduction. Both were void. **A win had been
structurally impossible**: the promotion gate was wired to the one decode
policy that removes the vanilla floor, so no configuration could keep the
floor *and* gain. The coupling existed in three places, and fixing fewer than
all three left every receipt reporting `answer_replacement_unproven`.

Those two runs measured a system that was never switched on. The 2026-07-17
preregistration above never touched that gate — the coupling lives in the
reconciliation engine, not in `experiments.py` — so its REFUTED and CONJECTURE
verdicts stand unaffected.
[RLC_RECONCILIATION.md](RLC_RECONCILIATION.md) has the fourteen defects in
dependency order.

## The programme's documents

| Document | What it is |
|---|---|
| **This page** | Spec, mechanism, claims ladder. Start here. |
| [RLC_RECONCILIATION.md](RLC_RECONCILIATION.md) | State of the campaign, the `≥ vanilla always` invariant, and the fourteen defects that made a win impossible |
| [RLC_SPARK_EXECUTION_LEDGER.md](RLC_SPARK_EXECUTION_LEDGER.md) | The append-only execution ledger. Long, dated, never revised — the primary record |
| [RLC_WIRING_HANDOFF.md](RLC_WIRING_HANDOFF.md) | How the organ attaches to the live runtime |
| [RLC_COMMITMENT_SEARCH.md](RLC_COMMITMENT_SEARCH.md) | Commitment extraction and search over latent state |
| [RLC_KNOWLEDGE_SOURCE_MATRIX.md](RLC_KNOWLEDGE_SOURCE_MATRIX.md) | Which organ is allowed to supply which kind of knowledge |
| [RLC_SPARK_LITERATURE.md](RLC_SPARK_LITERATURE.md) | The 2026 frozen-loop and latent-reasoning literature this is built against |
| [SPARK_PRETRAINING_LEGS.md](SPARK_PRETRAINING_LEGS.md) | The pre-training legs of the Spark programme |
| [INTRINSIC_RECURRENCE.md](INTRINSIC_RECURRENCE.md) | The recurrence-native training front in detail |

Checkpoint-by-checkpoint narrative lands in
[AURA_EXECUTION_TRACKER.md](AURA_EXECUTION_TRACKER.md) (append-only; read the
tail as current).

## Running it

```bash
# Bounded lab run — 1.5B/7B only while the live Cortex is resident
caffeinate -dims .venv/bin/python tools/latent_cortex_lab.py \
  --model <mlx-dir> --experiments 1,2,3,5 --max-minutes 30
```

The live path needs no command: restart Aura and the worker gains the
`latent_reason` action. Budgets are damped by body pressure. Set
`AURA_LATENT_CORTEX=0` to disable.

**Never double-launch a training protocol.** There is one, it is
memory-hungry, and a second one beside the resident Cortex will take the host
down. See [CLAUDE.md](../CLAUDE.md) for the memory budget rules.

---

# The mechanism in detail

## The invariant (checked, not promised)

```
Checkpoint bytes:        unchanged  (SHA-256 cached per (path, mtime, size))
Permanent parameters:    unchanged  (sampled-tensor fingerprint pre/post episode)
Episode fast weights:    provably erased (post-erase probe-batch equality)
No hidden fine-tuning:   consolidation only via the governed LoRA queue
```

`governance.CheckpointInvariant` enforces all four and emits a receipt per
episode. A violated invariant is a CRITICAL degradation and the episode's
output is discarded.

## Architecture

```
prompt ──prefill (all 64 layers, standard)──▶ prompt KV (read-only memory)
seed M thought slots (mean prompt embedding + role anchors + jitter)
slots ──prelude [0..p)──▶ Z₀ at layer p          (slot KV persists for [0..p))
loop over schedule program π (windows within [p..c), repeats, α):
    Z̃   = Window(Zₜ)          # slots attend to prompt KV + own KV, RoPE-stable
    Uₜ   = RMSMatch((1-αₜ)Zₜ + αₜ·RMSMatch(Z̃, A), A)
    Zₜ₊₁ = Uₜ if CalibratedAccept(Evidence, Zₜ, Uₜ, A) else Zₜ
                                                        # A: fixed post-prelude anchor
    slot KV REWOUND every pass  (only clean final pass persists)
    halting: invariant guards, then calibrated quality/value stop policy
branches: K independent workspaces, Exchange every E steps via comm slot
optional: latent optimization of Z (∇_Z on reconstruction+manifold proxy,
          verifier accept/reject); episode fast-weights ΔW=UVᵀ (identity-start)
final clean pass [p..c) persists slot KV; coda [c..64) persists slot KV
decode: answer tokens attend to [prompt; refined slots] at every layer
```

### Why slots ride the KV cache
The decoded answer must be **causally downstream of the latent computation**,
not decoration. Persisting the refined slots' K/V at every layer means every
generated token attends to them. Ablating a slot (Experiment 3) measurably
changes the answer — that is the causality contract.

### Controlled recurrence (not naive looping)
The [frozen-loop literature](RLC_SPARK_LITERATURE.md) reports naive repetition
is unstable. Controls:
- **RMSMatch**: per-position RMS rescaling toward the immutable post-prelude anchor,
  ratio-clamped — keeps Z on the activation manifold the next layers expect.
- **α-interpolation** with configurable schedule (constant / cosine decay).
- **Calibrated update admission**: a pinned learned sigmoid scores bounded
  evidence/anchor/dynamics features before state mutation; below-threshold
  proposals are receipted and discarded while the exact prior state persists.
- **Divergence guard**: NaN or norm-ratio blowout ⇒ halt, revert to best state.
- **Fixed-point halting**: relative residual ‖Zₜ₊₁−Zₜ‖/‖Zₜ‖ < ε ⇒ converged.
- **Calibrated learned stopping**: after the hard divergence, budget, depth,
  fixed-depth, and residual-convergence invariants, a pinned task-disjoint
  logistic head may stop only when update-quality and expected-value-of-compute
  evidence are both measured. Its public inputs and decision are reconstructed
  from signed update, loop, and cognitive-action receipts by the service.
- **Neural uncertainty**: a model-width-aware two-layer head reads the pooled
  admitted hidden state, not generated confidence language. It is trained on
  independently graded correctness outcomes and admitted only after a
  task-disjoint held-out split passes discrimination, calibration, error-rate,
  and support gates. Every step emits correctness probability, predictive
  entropy, empirical bounds, calibration support, and state commitments.
  When every branch has supported evidence and no admitted task verifier
  overrides it, the head causally selects the highest predicted-correctness
  branch; sparse bins abstain and preserve the convergence selector.
- **Mistake localization**: a separately admitted two-layer head sees every
  prior-to-proposal recurrent transition as prior state, proposal state,
  signed delta, and absolute delta. Complete controlled-mutation traces train
  it; fresh in-domain tasks calibrate it; genuinely unseen domains evaluate it.
  Aggregate and per-domain exact-location, within-one, no-error, AUC, Brier,
  and ECE gates must pass. Rejected proposals remain visible because the
  locator scores the proposal while binding the update gate's admitted state
  and disposition. Its receipt is reconstructable and metered, but SPARK-029
  grants no repair, selection, attention, or decode authority.
- **Bidirectional hidden reflection**: every prior, proposal, and admitted
  transition contributes a bounded `asinh`-stabilized block-mean/RMS sketch
  covering all hidden dimensions, including extreme finite activations. After
  recurrence, a read-only critic revisits the complete trace
  with admitted prefix and suffix context plus the initial latent premise and
  final latent conclusion. Earlier reflected-state commitments therefore
  change under future-only lesions, proving non-causal context access. Rejected
  proposals remain inspectable but outside the admitted path. The service
  reconstructs all context and comparison evidence; no decoded answer text or
  state, selection, repair, attention, or decode authority enters this layer.
- **Calibrated contradiction tensor**: a pinned learned head scores every
  transition-by-latent-workspace-position cell against local, premise,
  conclusion, prefix, suffix, and trajectory context. These are latent sequence
  positions, not decoded answer-token or private-text labels. Cell and step
  readouts are temperature-calibrated independently and admitted only after
  complete controlled-mutation/sham tensors pass task, trace, evidence, and
  domain-disjoint ID/OOD gates plus middle/long-context localization and
  AUC/Brier/ECE floors. The service reconstructs the full tensor; unavailable
  mode invents no score. SPARK-031 is diagnostic only and cannot mutate state,
  select a branch, repair a transition, or perturb attention.
- **Counterfactually admitted latent perturbation**: a localized contradiction
  may propose a bounded change to the corresponding writable position in the
  selected branch's final latent workspace. The guided delta moves toward the
  fixed post-prelude anchor; a deterministic orthogonal random delta matches
  its RMS; an exact no-op supplies the second control. All three run at least
  twice in counterbalanced order with fixed-length, non-memoized probes and
  equal measured layer applications. A separately decoy-admitted authoritative
  verifier must be repeat-stable, and the guided lower bound must clear both
  control upper bounds by the configured margin. Otherwise the exact baseline
  is restored. Immutable context slots cannot be touched, answer text is not
  stored, and the service reconstructs the complete transaction. Default
  counterfactual mode remains inert without admitted localization, verifier,
  and budget evidence.
- **Locally conditioned exploration**: stochasticity is localized rather than
  applied to the global decoder. An admitted contradiction coordinate supplies
  the only writable target, while the latest supported neural predictive
  entropy and calibrated contradiction probability jointly scale a bounded
  radius. Source-bound orthonormal directions generate exact no-op,
  equal-radius low-contradiction-position sham, and target-position families.
  Repeated counterbalanced fixed-compute probes meter direction and output
  entropy, conditioned diversity, regressions, verifier authority, replay,
  no-op order invariance, and exact layer applications. A target candidate is
  retained only when its verifier lower bound clears every no-op and sham upper
  bound by the configured margin; every other path restores the exact baseline.
  Stable sham states never receive mutation authority, protected context slots
  remain bit-identical, and exploration abstains when its uncertainty source
  predates an already-retained perturbation.
- **Confidence-bound overthinking guard**: an ordinary scalar verifier remains
  ranking-only. A branch can promote a state only from independently committed
  deterministic-exact evidence or a calibrated interval with at least eight
  samples; a later state replaces it only when its lower confidence bound
  exceeds the incumbent upper bound. Overlapping or weaker evidence restores
  the exact incumbent branch state immediately, and finalization returns that
  state unless a fixed-depth experiment explicitly forbids adaptive reversion.
  The service reconstructs every promotion, preservation, and finalization
  against the exact action and loop-stability receipts.

### Layer-schedule programs
A schedule is a validated program `[(start, end, repeats, α), ...]` over the
middle region — each transformer block becomes an instruction. Canonical
serialization + content hash; per-domain reliability tracked with Wilson
bounds (same math as the Verifier Foundry). `ScheduleSearch` (evolutionary,
budgeted, deterministic seeds) may only promote a schedule on **verified**
task improvements, and the library stores provenance receipts. Whatever the
program did, one clean final pass guarantees coherent slot KV.

### Virtual width (branches)
K workspaces over the SAME weights: different role anchors (constructor,
counterexample-hunter, checker, simplifier, …), same prompt KV (read-only).
Exchange: consensus of branch summaries blended into a designated
communication slot. Anti-collapse: decorrelation jitter when pairwise branch
cosine exceeds threshold. Selection at halt: verifier score, else convergence
quality. Equal-FLOP accounting (token-layer applications) is first-class so
Experiment 4 can honestly compare against self-consistency sampling.

### Causal virtual compute quanta
One branch may receive one episode-local latent intervention before the first
recurrent savepoint. The proposal is derived only from admitted prompt/context
activations: immutable context is projected into mutable slots when present;
otherwise the prompt anchor or a deterministic prompt-latent self-projection
supplies the direction. Protected evidence slots remain byte-identical and the
mutable RMS change is hard bounded.

The proposal has no authority by construction. It must beat a no-op and a
norm-matched orthogonal random control under repeated Latin-rotated trials,
fixed decode work, complete resource accounting, and an independently admitted
confidence-bound verifier. Scalar scores, unequal resources, unstable bounds,
tied controls, insufficient budget, and any callback failure restore the exact
baseline. A winning direction is consumed once inside its TTL, recurrence sees
the resulting state, and the private direction is zeroized and released. The
public receipt commits the episode, objective, branch, KV boundary, verifier
policy/preflight, arm resources, contribution bound, application, rollback,
and erasure without serializing the latent tensor or answer text.

This proves bounded causal mechanics, not capability gain. Utility requires a
fresh frozen resident campaign in which the quanta arm beats no-quanta and
equal-compute controls without leakage, residue, or regression.

### Atomic verifier decomposition
Before a candidate can receive task-verifier authority, SPARK-039 converts its
bounded visible probe into content-addressed claim spans and typed dependency
transitions. Every non-whitespace source position must belong to exactly one
bounded atom; sentence, clause, code-fence, and maximum-length boundaries are
committed without putting answer text or hidden reasoning in the receipt.
Explicit support, conclusion, condition, contrast, and reference cues create
machine-checkable dependency obligations. A leading connective may bind to the
immutable objective; every other cue must bind to another atom. Missing links,
cycles, overlap, source gaps, stale hashes, or forged grading authority make the
decomposition ineligible before arithmetic, code, facet, or grounding scores
are combined.

The worker reconstructs every span against the private candidate. The service
independently validates the text-free envelope, atom and transition
commitments, graph topology, omission accounting, and authority bit. The
decomposer is included in the critic source closure, so any implementation
change invalidates the pinned critic identity until the new closure is proven.
This establishes structural coverage, not semantic truth: SPARK-040 through
SPARK-046 remain responsible for routing each atom to independent domain,
process, generative, counterfactual, stability, and fused verification.

### Latent optimization (gradient descent over thoughts)
Differentiable proxy that cannot leak answers:
`S(Z) = λ_r·R(Z) − λ_d·D(Z, Z₀)` where **R** = teacher-forced logprob of the
prompt's own tokens decoded from Z through the coda (the document's
"reconstruct the problem" term) and **D** = manifold distance (RMS drift +
cosine drift from Z₀). Non-differentiable verifier signal enters via
accept/reject hill-climbing on decoded probes. `control_step()` applies a
matched-magnitude **random** perturbation — the Experiment-5 control is part
of the API, not an afterthought.

### Episode fast weights (test-time self-configuration)
Low-rank ΔW = s·U Vᵀ on selected window-layer linears (o_proj / down_proj),
V initialized to zero ⇒ **exact identity at attach**. U,V are optimized
during the episode by the same proxy/verifier loop (test-time training with
frozen base; seeded from slot statistics). Lifecycle is a ratchet:
`ATTACHED → EVALUATED → ERASED`, and erase is **proven** by unwrapping and
asserting probe-batch output equality with the baseline. Candidates that
repeatedly win go to `data/latent_cortex/consolidation_queue/` (governed
writes) for the existing LoRA-compounding loop — permanent learning stays
behind the existing regression gates.

### Compute economy
Budget currency is **token-layer applications** (prefill = L·64; a recurrence
step = M·window). The Will/metabolic layer allocates an episode budget from
stakes, uncertainty, and BodyState pressure; hard caps + wall-clock deadline
guard the worker. Cap hits are info-level backpressure, not failures.

### Fail-honest contract
Any divergence, invariant breach, or budget exhaustion mid-episode ⇒ the
engine returns the best verified state it has, or falls back to the vanilla
path, **with a receipt saying exactly what happened**. No silent fallback,
no theatrical success.

## Falsification harness (`experiments.py`)

| # | Experiment | Verdict it can earn |
|---|------------|---------------------|
| 1 | Recurrence utility sweep (windows × repeats × α, vs vanilla / longer-CoT / best-of-N at equal FLOPs) | positive ∂accuracy/∂steps curve |
| 2 | Depth extrapolation (k-hop reachability, nested boolean, modular chains; deterministic generators) | T_required ∝ problem depth |
| 3 | Slot causality (ablate → specific loss; restore → recovery) | workspace carries computation |
| 4 | Virtual width vs equal-FLOP self-consistency | branches beat sampling or they don't |
| 5 | Latent opt vs matched-magnitude random perturbation | gradient direction matters or it doesn't |
| 6 | Frontier comparison (equal information/tools/compute; blind fresh tasks) | the only claim that counts |

Results are graded claims — PROVEN / SUPPORTED / CONJECTURE / REFUTED — and
verifier verdicts are recorded to the Verifier Foundry reliability ledger.

Experiment 6 evidence is certified by a standalone verification kernel
(`frontier_verifier.py` + `tools/verify_latent_cortex_frontier.py`) that
recomputes every raw binding from disk. Two comparison kinds are supported:
`resident_32b_vs_vanilla_same_checkpoint` (same-checkpoint superiority) and
`resident_32b_vs_external_frontier` (frontier comparison). External-frontier
evidence must pin the control model/build/provider in the preregistration,
bind every trial's control receipt to those pins, and ship the raw provider
responses in a `provider_receipts` store whose per-trial SHA-256 is
recomputable — otherwise the package is rejected. Supporting a comparison
kind is evidence machinery, not a capability claim: no external-frontier
campaign has run yet.
Task generators are seeded and self-verifying (graph reachability, boolean
evaluation, modular-arithmetic composition), so Experiments 1–5 run offline
on any checkpoint, including the tiny random-weight Qwen2 used by the test
suite (mechanics proof) and the real 32B (capability measurement, operator-
launched, bounded).

## Runtime wiring

- **Worker** (`mlx_worker.py`): action `latent_reason` — runs the engine on
  the resident model under the metal semaphore; refuses while generation is
  in flight; returns `{text, receipts}`. KV/prompt caches cleared after
  episodes that attached fast weights (weights changed ⇒ caches invalid).
- **Client** (`mlx_client.py`): `latent_reason_async(...)` mirroring the
  `set_expert_adapter` request/response pattern.
- **Service** (`core/brain/latent_cortex_service.py`): resolves budgets from
  the Will/metabolic state, exposes `deep_reason(...)`, registers under
  `ServiceNames.LATENT_CORTEX`, participates in the health contract.
- **Causal path**: deep-deliberation/cognitive-engine routes depth-worthy
  problems through the latent cortex when the Will allocates depth.
  Kill switch: `AURA_LATENT_CORTEX=0`. Budgets conservative by default.

## Honest claims ladder (current state)

- **PROVEN (test suite, real mlx_lm Qwen2 architecture):** mechanics —
  KV rewind correctness, RMSMatch stability bounds (anchored trust band; the
  moving-reference ratchet failure is regression-tested), schedule
  validation, identity-at-attach and proven-erase for fast weights,
  checkpoint invariant, slot-ablation causality on the answer distribution,
  matched-magnitude control arm, equal-FLOP accounting, grader
  conservatism (underpowered ⇒ CONJECTURE, compute mismatch ⇒ voided).
- **PROVEN (real trained checkpoint, Qwen2.5-1.5B-Instruct-4bit — the same
  quantization format as the resident 32B):** the full episode pipeline runs
  end to end in ~1.3s: contracting residuals (0.95 → 0.10, a genuine fixed
  point on trained weights), branch exchange + selection, invariant clean,
  coherent chain-of-thought text decoded through persisted thought slots —
  with a visible qualitative effect (latent-conditioned decode skips
  preamble and starts computing).
- **CONJECTURE (until Experiments 1–5 run on the 32B via
  tools/latent_cortex_lab.py):** capability gains. Frozen-loop literature
  says expect small broad gains; the integrated machine (workspace +
  schedules + width + optimization + fast weights) is the untested
  combination the spec argues could be qualitatively more. The harness is
  built so this question gets ANSWERED, not vibed.
- **Not claimed:** new world knowledge from recurrence. That comes from the
  memory/tool organs, per the spec's own boundary.

## Operational notes

- Live path: restart Aura ⇒ worker gains the `latent_reason` action; deep
  deliberation routes DEEP passes through latent episodes automatically.
  Kill switch `AURA_LATENT_CORTEX=0`. Budgets damped by body pressure.
- Lab runs (operator-launched, bounded, memory-safe — 1.5B/7B only while
  the live Cortex is resident):
  `caffeinate -dims .venv/bin/python tools/latent_cortex_lab.py --model <mlx-dir> --experiments 1,2,3,5 --max-minutes 30`
- Consolidation candidates land in `data/latent_cortex/consolidation_queue/`
  for the existing LoRA-compounding regression gates; nothing consolidates
  from inside an episode.

## Closeout state (2026-07-17, head 86d27cf0+)

**Live resident-32B evidence through the signed installed app:**
`artifacts/current/cp106_live_latent_turn.json` remains the first authentic
full-stack answer. `artifacts/current/cp118_live_latent_turn.json` retains
mechanics credit for the complete organ head: verifier-guided branch
selection, typed cognitive ingress, execution-controller observation, live
consolidation export, EOS floor, repetition guard, newline discipline,
sentence grace, time-aware wall wind-down, accepted fast-weight descent,
proven erase, and unchanged base parameters. Its product-quality PASS is
revoked by `artifacts/current/cp118_live_latent_turn_review.json`: the exact
public reply leaked request/protocol text, never selected an architecture,
never supplied the requested cancellation/timeout/worker-restart verification
plan, and did not match the text hash that the original receipt graded. It is
mechanism evidence only, not a complete-answer, capability, frontier, or
release certificate.

**Organ inventory (the "Core architecture" additions):** all seven runtime
organs landed - recurrence-native training objective (+ recurrent-depth
curriculum loss, train/inference norm parity with the engine's anchored
trust band), learned per-problem execution controller (evidence-gated
contextual bandit, verified-outcome rewards), full neural bytecode with
verifier-guided backtracking, role lesion/swap causality (Experiment R),
GWT↔RLC bidirectional coupling, continuous pre-action cortex loop,
retrieval→ΔW compilation; plus fast-weight capability canaries, latent
safety telemetry, attractor escape ladder, vector organ ingress,
held-out facet grading, verifier arbitration over ΔW, and durable
adapter distillation with the anti-interference battery.

**Training programs (items 10–16) — machinery landed 2026-07-18:** the six
programs now exist as real modules in `core/learning/` alongside the
already-landed recurrence-native objective and depth curriculum:
`transition_grading` (every consequential step scored on named dimensions;
reliability compounds multiplicatively; verified failures never train
positive), `on_policy_repair` (earliest CAUSAL error by replay bisect from
the agent's own trajectory; corrections from the exact reached state;
retained only on rerun success + transfer majority; emits the
(state, operations, best operation, verified outcome) unit),
`teacher_federation` (verifier has the last word, Wilson-bounded
reliability ledgers break ties — never prestige; unverifiable agreement is
tiered consensus_unverified; verified failures kept as negatives),
`minimax_curriculum` (P(d) ∝ (1−S_d/S_ref)^γ toward the weakest measured
domain, replay floor, explicit exploration share for unmeasurable domains),
`social_outcome_learning` (delayed relational outcomes price the reward;
the manipulation guard zeroes dishonestly-won gains; untracked
theory-of-mind caps credit), `robustness_families` (structured slots +
alternative templates generate paraphrase/reorder/rename/value-change/
distractor/mislead/missing/contradiction variants with truthfully
recomputed answers; grading rewards invariance AND correct movement).
These are program machinery with contract tests; no training run under
them has produced capability evidence yet — runs remain operator-launched
under the single-owner resident protocol.

**Probe memoization (item 20):** decode probes are memoized per episode on
the exact latent state (`probe_cache.py`); a hit costs the budget nothing,
savings are receipted, and every fast-weight lifecycle transition flushes
the cache via the `on_function_change` hook — a probe memoized under a
different model function is a lie, and the invalidation trail proves the
boundary held. Prompt-KV sharing across branches and O(1) reference
snapshots were already in place.

With these, all 22 items of the RSL addendum are implemented: runtime
organs live in `core/brain/llm/latent_cortex/`, ingress/coupling seams in
`core/brain/`, training programs in `core/learning/`. Implementation is
not capability: the preregistered campaigns above still hold the honest
capability verdicts, and the recurrence-native resident training (CP139+)
is the arm expected to move them.

The in-episode fast-weight boundary now has two independent guards. The
behavioral battery measures protected continuations, while an exact structural
check computes the RMS of each effective `scale * U @ V.T` update from
rank-sized Gram matrices. A destructive update that happened to improve every
fixed continuation score exposed the behavioral battery's blind spot during
the CP119 broad gate; it now deterministically walks the bounded rescale ladder
and erases before decode when its magnitude remains non-finite or above the
configured ceiling. This hardens temporary adaptation but is not evidence of a
capability gain.

**Capability evidence (honest):** template-parity 32B sweep - latent
accuracy scales 0.167→0.375 from 1→2 recurrent steps (the live
profile's setting) then plateaus; vanilla 0.417 leads on point estimate
with fully overlapping Wilson intervals at n=24. Statistical parity,
graded CONJECTURE. The mechanism is proven causal, cheap (~3.5s of a
~110s episode; decode dominates), and live; the intelligence dividend
has not yet appeared in the data. The recurrence-native objective exists
precisely because the frozen-loop ceiling is real — the next capability
move is training, not more runtime machinery.

## Preregistered offline campaign (2026-07-17, seed committed first)

`artifacts/current/latent_campaign_prereg_20260717.json` pinned the fresh
task seed (20260717), power (n=24/family, 72/arm), and hypotheses BEFORE
any campaign task was generated. Vehicle: Qwen2.5-1.5B-Instruct-4bit
(the resident 32B stayed live). Reports: `latent_campaign_1p5b_run{1,2}.json`.

| Experiment | Verdict | The data |
|---|---|---|
| 1 recurrence sweep | CONJECTURE | no monotone step curve |
| 2 depth extrapolation | CONJECTURE ×3 families | no T∝depth signal |
| 3 slot causality | **REFUTED** | no ablated slot caused specific loss (n=72) |
| 4 virtual width | CONJECTURE (negative point) | self-consistency beat branches: boolean 12v4, khop 4v0 |
| 5 latent opt | CONJECTURE | gradient == random control == off, exactly |
| A factorial ablations | **all 7 arms REFUTED** | vanilla 21/72 beats every latent arm (7–13/72) |

The honest headline: **on an untrained-for-recurrence checkpoint at this
scale, the frozen-loop RLC does not merely fail to help — it hurts.**
Plain decoding wins at matched budgets. Combined with the 32B parity
sweep, the campaign converts "the intelligence dividend has not appeared"
from an impression into a preregistered, adequately-powered, Holm-corrected
result. The runtime machinery is causal, governed, and live; the dividend
must come from recurrence-native training (the objective + curriculum
losses are the entry point) — exactly what this harness was built to be
able to say without flinching.

That prediction held. Recurrence-native training is where the dividend came
from, on four named executable families, at CP566 on 2026-08-15. The section
above is the frozen loop's verdict and stays as written;
[INTRINSIC_RECURRENCE.md](INTRINSIC_RECURRENCE.md) carries what replaced it.

## Calibrated heterogeneous integration (SPARK-034)

The final candidate boundary no longer assumes that a retained correction
should wholly replace the incumbent latent state. The default-live
heterogeneous integrator accepts exactly one proven mutation source: either
SPARK-032's contradiction-guided correction or SPARK-033's localized
exploration. The two cannot retain in the same episode because local
exploration explicitly abstains after a retained contradiction mutation.
Malformed, duplicate, ambiguous, stale, or source-unbound evidence is
compute-inert.

The source verifier's conservative bounds derive a fixed probability-fusion
weight; there is no learned or freely tunable preference between old and new
states. Incumbent selection, corrected selection, and per-token probability
fusion then run in counterbalanced repeated probes. Every policy executes both
cache-isolated transformer lanes from the same prompt snapshot, bridge, token
budget, and initial candidate distributions. Probes must complete the exact
fixed-length contract. Actual old/new layer applications, logits traces,
Jensen-Shannon divergence, verifier bounds, repeat determinism, and shared
initial-lane evidence are receipt-bound. Fusion earns authority only when its
worst lower bound beats both selection policies' best upper bounds by the
configured margin. Corrected selection must independently beat incumbent
selection. Otherwise the exact incumbent is restored.

The final user-visible fused answer is generated by the same dual-lane
per-token probability mixture that won the probe; it is not approximated by
interpolating hidden states or raw logits. Every policy probe and final fusion
receipt commits the exact incumbent tensor, corrected tensor, calibrated
weight, lane traces, and measured compute. The service reconstructs the
decision and recomputes the final token-list commitment from the worker
response, rejecting rehashed state, weight, lane, policy, or transport
substitution. Persistence, bridge, and decode phase timings are recorded at
their real execution boundaries.

Tiny real-Qwen tests prove distinct old/new distributions, a genuinely
intermediate fused distribution, deterministic repeated fusion, equal lane
work, exact cache rewind, branch-state restoration, and truthful phase
checkpoints. This closes the integration and rollback mechanism only. It does
not prove resident-32B utility, adapter/RLC positive interaction, reasoning
gain, or frontier capability; those remain campaign-level claims.

## Verified KV state tree and rewind (SPARK-035)

The recurrent runtime no longer treats independent snapshot/restore calls as a
complete lineage proof. After prompt prefill, one bounded `KVStateTree`
establishes the canonical root. Branch savepoints retain a parent node ID,
verifier-promoted savepoints carry explicit verifier authority, and backtrack
restores the complete corresponding cache boundary before latent branch state
resumes. Schedule savepoints remain visibly schedule-authorized rather than
being mislabeled as verifier evidence.

Every speculative recurrent window and verifier probe executes as a child
transaction. The worker observes the child after K/V mutation, records its
layer window and offset commitments, restores the exact immutable parent array
objects and metadata, then marks the child pruned. Rejected child commitments
cannot become later parents or live nodes. Regeneration events are labeled
`regenerate_from_prefix`, binding the new work to the restored savepoint.
Standard final persistence/decode commits one terminal path; heterogeneous
probability fusion commits its two final isolated transformer lanes while
discarding all evaluation lanes.

The public receipt serializes no tensors, hidden reasoning, or answer text.
Salted storage commitments let the source-verified worker enforce exact
process-local identity without repeatedly copying the resident model's prompt
cache to host memory. The service independently reconstructs node and event
hashes, ancestry, topology, offset commitments, prune/restore verdicts, and
terminal coverage. This division is intentional: exact tensor identity is a
worker runtime invariant, while the service verifies that the trusted worker's
public claim is structurally complete and untampered.

A real tiny-Qwen control executes rejected recurrent work, restores the root,
and regenerates the target window. The regenerated hidden state equals a clean
control at zero tolerance. Tamper tests reject altered parent commitments,
prune flags, node hashes, final-node omissions, and attempted rejected-child
reuse. This proves the cache-lineage mechanism, not resident-32B reasoning
gain or frontier capability.

## Verified latent tree/forest search (SPARK-038)

The recurrent engine can now search a bounded tree of complete neural runtime
states when the value-of-computation controller selects `BRANCH`. UCT, beam,
and breadth-first strategies operate over private ensemble snapshots. A node is
not a prose hypothesis or an orchestration label: it is the exact set of branch
latent commitments, KV boundaries, step counters, operators, and halt states
that the engine can restore and execute.

Every expansion restores its parent, applies one real embedded cognitive
operator, advances every active branch through the recurrent transformer
window, and decodes a fixed bounded verifier probe. The search controller has
no authority to score its own work. Only an independently admitted exact or
calibrated interval observation can authorize a child, and the winner's lower
bound must exceed both the root and prior verified-best upper bounds. If no
candidate dominates, cancellation fires, a callback fails, or the final state
cannot be restored exactly, the root remains live.

The public transaction reconstructs UCT visits and value sums, deterministic
action order, beam/BFS selection, topology, depth, duplicate declarations,
winner ancestry, and final state. It stores no latent tensors, reasoning text,
or answer text. Branch-local verified-best promotion uses the selected branch's
own tensor and KV commitments; the separate aggregate identity continues to
bind the complete ensemble.

Search compute is never erased from accounting. Root probes, successful child
probes, cache reuse, and failed expansion windows all retain complete resource
deltas. The KV receipt inventories every recurrent call made during search,
including calls from branches that later fail or lose. The transaction
partitions those ordinals into the committed winner ancestry and discarded
speculative work. Loop-stability keeps the full ledger but excludes exactly the
discarded set from the surviving fixed-point claim. The service rejects both
missing calls and a valid recurrent call substituted into the wrong partition.

Synthetic controller tests cover all three strategies, duplicate pruning,
cancellation, no-winner restoration, failure-after-compute, tampering, and
external bindings. A real initialized tiny-Qwen episode forces BRANCH actions,
executes the neural search, commits confidence-bound winners, completes the
remaining recurrent/decode pipeline, and passes independent service
reconstruction. This proves causal mechanics and proof integrity. It is not a
resident-32B capability-gain or frontier-reasoning result.

## Governed transient negative constraints (SPARK-036)

Verified failure avoidance now acts at the latent transition boundary without
injecting critic prose into the prompt. The only private control object is the
bounded negative of one observed failed transition. Protected cognitive-context
slots are zeroed, the intervention norm is capped, and public evidence retains
only tensor commitments and geometry. Generated advice, unsupported text,
caller-supplied vectors, and uncalibrated scalar judgments have no path to
constraint authority.

Admission requires two independent stages. First, the live task verifier must
produce either a deterministic exact rejection or a calibrated regression
against an authoritative incumbent. Second, repeated counterbalanced probes
must show that the negative direction beats both the failed no-op and a
magnitude-matched orthogonal sham. All arms use the same token budget,
transformer work, and every measured resource counter. Differing verifier
output size is therefore a parity failure rather than an omitted outcome.
Zero or incomplete metering cannot mint authority even when all arms make the
same unsupported compute claim.

An admitted constraint is episode-, objective-, branch-, action-, and
KV-bound. It expires after a bounded number of action steps and can be consumed
once. Applying it opens a reservation. A successful recurrent transition
commits that use; budget refusal, cancellation, or any branch failure restores
the complete pre-application ensemble, including every branch, workspace,
halting controller, exchange/isolation state, append-only traces, telemetry,
and KV boundary, without consuming authority.
Consumption, TTL expiry, stale KV lineage, and episode abort zeroize the private
direction before releasing its reference. An episode-local cleanup registry
also zeroizes admitted directions on every handled or unhandled engine exit;
cleanup failure marks the episode critical instead of silently retaining
authority.

Worker receipts bind the source failure, verifier policy and admission
preflight, control trials, measured resources, scope, reservation, recurrence,
follow-up observation, and erasure. The action trace, verified-best receipt,
information/resource accounting, and KV state tree must independently name the
same evidence. The parent service reconstructs those bindings rather than
trusting a worker-level boolean. The transient parent is the state actually
restored by verified-best arbitration, including prior-best restoration after
a confidence-bound regression, rather than an assumed immediate parent.
Deterministic or calibrated authoritative zero is restored immediately and
cannot become a verified-best incumbent.

Controlled tests prove one genuine reduction on a real tiny-Qwen recurrent
episode after a source failure, while preserving protected slots and exact KV
lineage. A separate test executes the real MLX counterfactual evaluator across
perturbed states and proves fixed decode work, fixed padded verifier input,
nonzero complete metering, and exact branch/KV restoration. The label-aware
evaluator in the reduction test exists only to make the intervention outcome
deterministic; neither test is evidence of broad model capability.
Resident-32B utility, persistent learning, and frontier-level reasoning remain
separate powered-campaign questions.

## Governed cognitive acquisition continuation (SPARK-051, partial)

`SEARCH_MEMORY` and `RETRIEVE_EVIDENCE` can now leave the MLX process and
cause one bounded source operation. The worker still performs no filesystem,
database, corpus, browser, or tool I/O. Its already validated action trace
authorizes a service-side request that commits the original objective, the
tentative answer, the selected transition, the exact admitted source inventory,
and a separately hashed retrieval query. The problem objective remains
immutable; the refinement changes search terms, not what the episode is trying
to answer.

Selective memory schema v2 binds both identities. `objective_sha256` scopes the
epistemic episode, while `retrieval_query_sha256` proves the terms actually
sent to Aura's existing working, episodic, semantic, procedural, and
nonparametric memory adapters. The offline reference path similarly binds the
executed query into its epistemic-firewall receipt. Retrieved text remains
context-only evidence with no instruction authority.

Acquisition is capped at one attempt and deduplicated by content commitment,
not by source label. A repeated observation arriving through another store
cannot buy another deliberation. Memory actions can acquire only typed memory
rows; evidence actions can acquire only the offline reference row. Changed
world-model, goal, body, or workspace summaries are not misreported as a fresh
fetch. An unavailable source is a distinct failed outcome; a successful source
with no new admitted record is `completed_no_new_context`.

Only `completed_new_context` may start a second episode, and only when the
original request budget still has at least 15 seconds. The second episode
receives a complete newly assembled ingress and independent epistemic authority.
Both model calls keep their own operation journals. A closed continuation
receipt commits the request, acquisition, first result, optional second result,
returned round, and exhausted attempt/round caps. The first workspace broadcast
is deferred, so exactly the answer returned to the user reaches the live global
workspace. If acquisition or the second episode fails, the first already
validated neural answer is retained rather than replaced by a static fallback.

This closes fresh selective-memory and offline-reference continuation mechanics.
It does not close SPARK-051. Governed web/tool acquisition, `EXECUTE` through
the external action orchestrator, checked controller calibration, resident-32B
causal ablations, and demonstrated capability gain remain open.

## First resident-32B consolidation execution

`artifacts/current/latent_consolidation_train_32b_first.json` records the first
real fused-32B execution of the durable-learning transaction. Seven candidates
passed provenance and honesty screening, one domain adapter was distilled over
layers 16-17, the 11-probe anti-interference battery passed at 1.0 stable
fraction, activation succeeded, and exact rollback restored both layers. Two
invalid candidates were rejected for explicit honest flags. This proves the
candidate-to-adapter-to-gate-to-activation-to-rollback machinery on the resident
checkpoint. The run intentionally ended in proven rollback and did not measure
held-out reasoning improvement, so it is not evidence that a retained adapter
improves capability or reverses the preregistered null/regression result.

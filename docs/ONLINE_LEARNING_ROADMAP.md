# Online Continuous Learning — Honest Roadmap

**Status:** engineering roadmap + claim boundaries. This document exists so we
never *casually* claim "Aura learns continuously / online" in the strong sense.
It states precisely what learning Aura does today, what is a bounded next step,
and what is genuine research frontier.

Scope owner task: **#49**. Related: **#37** (LoRA sleep-consolidation, the real
bounded step), **#33** (RSI single-patch proof), **#46** (the honest ablation
that any "learning helps" claim must pass).

---

## 1. The honest baseline: what changes, and when

Aura runs a **frozen** local model. The ~32B MLX core weights do **not** change
during a conversation, a day, or a week of use. Between explicit, scheduled
training jobs the pretrained/fine-tuned weights are immutable. Anything that
*looks* like "learning within a session" is one of the mechanisms below — none
of which is online gradient update of the core weights.

| Mechanism | Code | What it changes | When | Core weights touched? |
| :-- | :-- | :-- | :-- | :-- |
| Retrieval (RAG / vault) | `core/memory/*`, recall telemetry | the **prompt context** for the next turn | every turn | No |
| Conversation/state carry | chat preflight, world state | injected context | every turn | No |
| Auxiliary online plasticity | `core/consciousness/synaptic_plasticity.py` | a small **auxiliary** Hebbian weight matrix (modulation signal), **not** the LLM | per inference | No (its own docstring states the LLM is frozen) |
| LoRA adapter training | `core/learning/lora_trainer.py`, `continual_lora_merge.py`, `tree_lora_manager.py`, `adapter_registry.py` | **adapter** weights merged onto the backbone | batch / scheduled | Adapters only |
| Full-weight training | `core/learning/full_weight_training.py` | every parameter (real backprop), CPU-bounded, eval-gated, hot-swap promoted | scheduled job | Yes, but **offline/batch**, behind promotion gates |
| RL / self-update tasks | `core/learning/rl_train.py`, `scripts/self_update.py` (dispatched via `core/tasks`) | policy / fine-tune artifacts | scheduled background | Indirect, batch |

**Claim boundary (do not cross):** the honest phrasing is *"Aura retrieves and
re-contextualizes continuously, and consolidates into adapter/weight artifacts in
scheduled batch jobs behind evaluation gates."* It is **not** *"Aura's neural
weights learn online from each interaction."* The second is false today.

---

## 2. The bounded, real next step: LoRA sleep-consolidation (#37)

The achievable, provable increment is **batch** LoRA consolidation, not online
learning:

1. **Collect** verified experience shards (`experience_collector.py`,
   `trace_labeler.py`) — only governed, receipted, hidden-eval-safe traces.
2. **Plan** the merge (`continual_lora_merge.py` already builds the mlx-lm
   command + provenance and *refuses promotion without validation evidence*).
3. **Train** a LoRA adapter in a scheduled "sleep" window (`lora_trainer.py`).
4. **Evaluate** on a *sealed held-out pack* before promotion
   (`eval_before_promotion.py`, `hidden_eval_repro.py`) and record the result in
   the tamper-evident behavioral ledger (`core/evaluation/behavioral_ledger.py`).
5. **Promote or reject** via the promotion gate; **TreeLoRA**
   (`tree_lora_manager.py`) branches a new adapter node when a merge would cause
   catastrophic interference, rather than overwriting.

**Definition of done for #37:** one consolidation cycle, end to end, where a
held-out score measurably improves *and the ablation (#46) shows the gain is from
the adapter, not the base model* — recorded in the behavioral ledger, with a
rollback path. Until that runs on real hardware, #37 stays **pending**, not
claimed.

**✅ DONE (2026-06-23).** `tools/generate_lora_consolidation_proof.py` ran one
real cycle on a local model (`Qwen2.5-1.5B-Instruct-4bit`, QLoRA): the frozen
base scored **0.000** on a sealed held-out pack of a synthetic deterministic
convention it cannot know; after training a LoRA on experience shards (the
"sleep" step via `lora_trainer.py`), the **same frozen base + the adapter**
scored **0.950** on the held-out pack — over a **disjoint** vocabulary, so the
gain is generalization of the learned format, not memorization. Attribution is
the inline ablation: identical base, the adapter is the only delta, and base=0
proves the base alone cannot do it. Recorded in the tamper-evident behavioral
ledger (chain verified, held-out-integrity ok), cleared the eval-gate +
promotion gate, and the rollback path (register/activate → baseline) is proven.
Bundle: `artifacts/proof_bundle/latest/LORA_CONSOLIDATION.json`; clears
`tools/proof_fabrication_guard.py` (real measured scores, no hardcoded
baselines). HONEST SCOPE: this proves the consolidation **machinery** on a small
local model; the same pipeline runs on the 32B cortex in production sleep
windows (longer train). It is **batch** consolidation with a sleep metaphor —
**not** online core-weight learning (§3 remains the unclaimed frontier).

**Why this is honest, not online:** the model is unavailable for inference during
the heavy train step (or runs a copy), the update is a discrete artifact, and it
is eval-gated. That is *batch consolidation with a sleep metaphor*, which is a
real and defensible capability — distinct from continuous online plasticity.

---

## 3. The genuine frontier (open research — do not claim)

True **online continuous learning of the core weights without catastrophic
forgetting** is an unsolved research problem, not an engineering task. Honest
status of the candidate directions:

- **Online LoRA / streaming adapters.** Update an adapter from a rolling buffer
  more frequently. Tractable-ish, but frequent updates amplify forgetting and
  reward-hacking; needs strong eval gating and is still batch-ish, not per-token.
- **TreeLoRA / parameter isolation** (`tree_lora_manager.py`). Mitigates
  forgetting by *isolating* new knowledge in branched adapters. Promising and
  partially implemented; it sidesteps online core-weight learning rather than
  solving it.
- **Reward-modulated Hebbian / STDP** (`synaptic_plasticity.py`,
  `meta_plasticity.py`). Real and running, but on a **small auxiliary** matrix —
  it is a modulation/affect signal, **not** learning in the LLM's billions of
  parameters. Claiming otherwise would be the exact overclaim this doc prevents.
- **Full online backprop of the core weights.** Not viable on a 64GB unified-
  memory desktop, and unsolved for catastrophic forgetting / stability-plasticity
  even with unlimited compute. **Out of scope. Not on the roadmap as a claim.**

Hard constraints that bound all of the above on this hardware:
- 64GB unified memory: a full-weight online optimizer state for 32B does not fit
  alongside live inference (see `existential_stakes` memory ceiling work).
- Stability/plasticity dilemma: every weight that adapts to *new* data risks
  degrading *held-out* behavior. This is why every path here is eval-gated.

---

## 4. What "learning works" must mean (no fabrication)

Any claim that learning *helps* must clear the honest bar set this cycle:

- a **sealed held-out** improvement (not the training set),
- recorded in the **tamper-evident** behavioral ledger
  (`core/evaluation/behavioral_ledger.py`),
- with an **ablation** showing the gain comes from the learned component, not the
  base model (`core/evaluation/ablation_harness.py`,
  `tools/agi/run_prompt_baseline_ablation.py`),
- and it must survive `tools/proof_fabrication_guard.py` (no hardcoded scores,
  no assert-victory over invented baselines).

Until a result clears that bar, the honest classification is **"capability
present, lift unproven."**

---

## 5. Roadmap summary

| Step | Status | Honest claim allowed |
| :-- | :-- | :-- |
| RAG/context re-use per turn | shipped | "continuous retrieval & re-contextualization" |
| Auxiliary online plasticity (aux matrix) | shipped | "auxiliary modulation; not core-weight learning" |
| LoRA batch consolidation pipeline | **proven end-to-end (2026-06-23)** | "one real consolidation cycle proven (held-out 0.00→0.95, adapter-attributed)" |
| LoRA sleep-consolidation proof (#37) | **✅ done (2026-06-23)** | "proven on a local model; batch consolidation, not online core-weight learning" |
| TreeLoRA forgetting isolation | partial | "interference-aware branching" |
| Full-weight batch self-training | built, gated | "offline, eval-gated, hot-swap" |
| Online core-weight learning | **research frontier** | **none** |

---

## 6. The learned language substrate (2026-08-20) — what is real, what is not

`core/language/learned_matcher.py` decides what a sentence is from declared
examples rather than from a word list, and `core/language/model_features.py`
reads those sentences off the resident model's hidden states instead of a
topical embedder. First consumer: whether a reply claims a completed action
(`core/conversation/response_reliability.py`).

**What is measured.** A topical sentence embedder was put behind all
twenty-five declared matchers in the runtime. Eight separated their own
examples and none by more than the spread inside its classes — zero usable
boundaries. That is a real negative result and it is why the feature source
is a parameter.

**What the frozen measurement says (2026-08-20).** Fitted on the twelve
declared examples alone, scored on twenty-four held-out wordings that are
never examples, against the live resident model:

| Feature space | AUROC | Boundary gap | Spread | Trustworthy | Abstain |
| :-- | --: | --: | --: | :-- | --: |
| topical embedding | 0.693 | −0.693 | 0.193 | no | 1.00 |
| model hidden state | **0.771** | −0.034 | 0.018 | no | 1.00 |

Two things follow, and both matter.

The resident model's own representation **does** separate this decision better
than a topical embedder — 0.771 against 0.693 on wordings neither was fitted
to. The hypothesis was right about the feature space.

And it is **still not enough to act on**. Neither boundary is trustworthy at
twelve examples, so both abstain on everything and neither decides anything in
production. The hidden-state gap is −0.034 against a spread of 0.018: an order
of magnitude closer to separating than the embedder's −0.693, and on the wrong
side of zero.

So the honest classification is **"better representation confirmed, usable
boundary not yet reached."** What closes it is labels, not cleverness — the
receipt-teaching path adds them from live traffic, and durable storage now
lets them accumulate across restarts. Re-run: the measurement task writes
`artifacts/language_substrate/measurement.json` once per boot, and the numbers
move when the declaration grows.

F1 and false-positive rate stay undefined until a boundary is trustworthy
enough to decide anything; reporting them from an all-abstain run would
describe a system nobody runs.

### Known limitations, in the order they matter

| # | Limitation | Where | Consequence |
| :-- | :-- | :-- | :-- |
| 1 | ~~Learned state is process-local~~ **fixed 2026-08-20** | writes through the governed gateway into `paths.data_dir/language`; verdicts deliberately not restored | phrasings now accumulate across restarts |
| 2 | The fast path caches exact strings | `_decided` is keyed on the stripped sentence | a paraphrase is a first sighting again, even though the *decision* generalizes over vectors |
| 3 | ~~`observe()` keeps stale verdicts~~ **fixed 2026-08-20** | `observe` now clears `_decided` with `_ready` | a new example retires the decisions it predates |
| 4 | ~~The warmer drops what it could not decide~~ **fixed 2026-08-20** | only a settled phrase leaves `_pending` | a phrase deferred while the model was busy is retried |
| 5 | ~~Representational claim unproven~~ **measured 2026-08-20** | `artifacts/language_substrate/measurement.json` | hidden states beat embeddings on held-out paraphrases (AUROC 0.771 vs 0.693) and neither boundary is usable yet at twelve examples |

### Where this could go: the model as a representation organ

The model's roles have been to understand, reason and generate. `encode_hidden`
adds a fourth that does not require it to speak: a reusable representation,
`f(x) → h(x)`, that other systems consume directly.

If the measurement above succeeds, the same surface — independently
calibrated, abstaining, trained on receipts rather than on a list — could
decide request against statement, promise against completed action,
observation against inference, hypothetical against factual, correction
against new instruction, uncertainty, intent, whether a tool is required,
relevance, contradiction, task state, and appraisal categories.

Each of those is a lexical debt in the runtime today. Capability routing still
carries explicit verb classes (run/execute/compute, search/find/browse,
write/create/build) and object classes (code/script/python,
file/document/path) with handcrafted mood logic; the positional solver still
converts English into constraints with regular expressions. The accurate
description of the system as it stands is **deterministic language rules plus
one early learned semantic substrate** — not learned language interpretation.

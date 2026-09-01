# Aura

DEMO: https://youtu.be/iTyxeugcZtI?si=B91No0Hjz3eKLMwz

A local cognitive-architecture research runtime for testing continuous-state
agency, receipt-based governance, memory persistence, activation steering, and
long-run self-maintenance.

Aura is not proof of life, personhood, or phenomenal consciousness. Nothing
in here settles that, and the parts of the repo that sound like they might
are named after mechanisms, not achievements.

The actual claim is narrower and testable: internal state causally affects
generation, memory writes, tool authorization, initiative selection, and
runtime repair, through code paths that leave receipts you can audit.

That's a smaller claim than the vocabulary suggests. It's also one you can
check.

[![License: All Rights Reserved (Read-Only)](https://img.shields.io/badge/License-All_Rights_Reserved_(Read--Only)-red.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS_Apple_Silicon-lightgrey.svg)

For the technical deep dive, read [ARCHITECTURE.md](ARCHITECTURE.md). If you
want the same ideas without the math, read [HOW_IT_WORKS.md](HOW_IT_WORKS.md). If
you want the evidence standard for autonomy and novel output claims, read
[docs/BEHAVIORAL_PROOF_STANDARD.md](docs/BEHAVIORAL_PROOF_STANDARD.md).

**The main research programme is
[docs/RECURSIVE_LATENT_CORTEX.md](docs/RECURSIVE_LATENT_CORTEX.md).** It asks
whether a frozen 32B checkpoint can be made to think longer without changing a
weight, builds the machinery, runs a preregistered campaign against it, and
reports that the capability dividend did not appear. Summary in
[Recursive Latent Cortex](#recursive-latent-cortex) below.

**Evidence map:** Claims should point to runnable tests, proof bundles, receipts,
or replayable logs. Test counts move with the repo; use `pytest --collect-only`,
`make proof-bundle`, and [TESTING.md](TESTING.md) for the current surface rather
than treating prose as evidence.

If you want to see it work, keep reading.

## Evidence boundary

This is a functional cognitive-architecture research project. It is not proof
of phenomenal consciousness, qualia, legal personhood, or moral patiency.

The repo enforces that in code, not just in a paragraph like this one. An
ontological boundary guard treats loaded labels — "consciousness guarantee,"
"personhood proof" — as functional indicator batteries unless independent
evidence says otherwise. A module named `qualia_synthesizer.py` is a name.
Names are not evidence.

What is actually claimed, and what each claim costs:

- **Governance is a design target, not a sealed fact.** Consequential paths
  must route through receipt-producing governance. A lint failure, a direct
  tool fallback, or a default-approved gateway path is a bug. Not
  acceptable compatibility behavior. A bug.
- **Autonomous RSI is not proven mature.** There is scaffolding here for
  autonomous repair, evaluation, patch genealogy, and proof bundles.
  Scaffolding is not a result. Unverified self-modification stays out of
  evidence scope until long wall-clock runs, hidden external tasks, and
  independent replication all succeed.
- **The cognitive layer has not been shown to earn its cost.** This is the
  largest open gap in the project and it belongs at the top of any honest
  reading. Two components have now been measured against simpler
  alternatives under matched or shared budgets — memory retrieval beats a
  budgeted context window when the fact is out of window (claim 31a), and
  the revision gate beats both "always keep the first answer" and "always
  keep the second" over the *same* two generations (claim 31b). Two
  subsystems on two task families is not the layer. Nothing here yet shows
  that IIT, qualia metrics, neurochemical simulation, substrate ODEs, dream
  cycles or theory arbitration make Aura better at a task a user wants
  done, and a far smaller computer-use agent performs the demo tasks
  without any of them. Those layers cost latency, memory, tuning surface
  and failure modes this repository has repeatedly paid for. Complexity is
  justified by measured advantage; most of it has not been measured. See
  `CLAIMS_MATRIX.md` claim 31.
- **Production maturity is bounded.** This is research software being
  hardened. The local monolith, the broad fallback surface, and the runtime
  fragility are all real, and none of them belong under the word
  "enterprise."
- **Steering is causal, and that's testable.** Internal state reaches
  generation through non-text channels (Contrastive Activation Addition).
  But black-box steering tests can hide live affect telemetry from the
  prompt text, and rich adversarial baselines are required before the
  result counts.
- **Identity persistence is retrieval, not just prompting.** Coherence is
  supported by ID-RAG Chronicle retrieval rather than prompt anchoring
  alone.
- **φ is bounded.** A bounded IIT-style integration metric over tractable
  complexes. Not a whole-system consciousness measurement. Full-system IIT
  is intractable and we did not solve it.
- **The hardware target is specific.** Bryan's Apple Silicon M5-class
  machine, 64 GB unified memory. Lower-memory machines downshift their
  model lanes. They do not get resident Cortex heartbeat latency, and claiming
  otherwise would just be a benchmark run on hardware nobody has.
- **Resource stakes persist and constrain action envelopes.** That's an
  operational metabolism analog. It is not biological metabolism.

For the things deliberately *not* claimed — including anything physical —
read [CLAIMS_NOT_SUPPORTED.md](CLAIMS_NOT_SUPPORTED.md). It's the most
useful page here if you're skeptical, which you should be.

---

## See it learn (one command)

Don't take the evidence discipline on faith. Run it. Apple Silicon, 20–40
minutes, about 5 GB of disk:

```bash
make setup          # once: venv + requirements
make demo-learning
```

Here's what happens. A small local model takes a swing at seeded reasoning
tasks. Every attempt gets graded by an **exact checker** — the verifier *is*
the reward, so there's nothing to game. It DPO-trains a LoRA on the verified
wins and losses, then has to clear a **sealed held-out battery** on fresh
seeds it has never seen before its weights get fused and published.

Then it does the whole thing again, on top of the artifact it just published.

Every generation lands in a hash-chained ledger
(`core/learning/rsi_lineage.py`). The verdict is computed from those
receipts, not written by hand afterward. When it refuses to promote, that
prints just as loudly as a gain. Raw responses, eval reports, and cycle
receipts all stay on disk.

The same machinery runs autonomously inside the live runtime
(`core/learning/compounding_scheduler.py`): idle-gated, governance-approved,
memory-admission-controlled, with promoted weights hot-swapped into live
inference.

## Production Evidence Surface

Everything below has a runnable implementation, receipts, and validation
artifacts. Ideas that don't are kept off this list — not softened, not
hedged, just left out until they earn a place.

Release gates generate a proof bundle. You shouldn't have to infer maturity
from how confident the prose sounds.

- `core/brain/llm/continuous_substrate.py` is a configurable 64-to-512 neuron
  Liquid Time-Constant ODE running at ~20 Hz. CPU-only numpy with explicit-Euler
  integration plus stochastic perturbation; `get_state_summary()` derives
  valence/arousal/dominance/phi from adaptive projections of the live state vector
  (grounded in external reality via `adapt_projections()`),
  without changing callers.
- `core/brain/llm/substrate_token_generator.py` is the substrate-first readout:
  it uses an untrained random projection onto a 32-word proto vocabulary from
  the live substrate before calling the transformer, and falls back to the
  Cortex when substrate prediction error exceeds threshold. A real
  substrate-first readout means a trained head over the model vocabulary —
  this is the scaffold for that, not the claim.
- `core/brain/llm/sensorimotor_grounding.py` maps camera/screen/audio
  observations into the substrate input vector, so live sensor events perturb
  the ODE directly instead of arriving only as text/tool summaries.
- `core/consciousness/phi_core.py` (2,939 lines) implements real IIT-style
  integration math: binarization, empirical TPM, KL-divergence φ, exclusion
  postulate, polynomial-time spectral partitioning, with an exhaustive
  8-bipartition validation baseline.
- `core/consciousness/hierarchical_phi.py` implements the 32-node hierarchical
  φ with K=8 overlapping subsystems and Bayesian-smoothed estimation.
- `core/consciousness/affective_steering.py` is a real CAA injection pipeline
  that hooks MLX transformer blocks and modifies the residual stream at
  generation time.
- `training/caa_32b_validation.py` validates production-model CAA artifacts:
  vector presence, layer geometry, PCA structure, permutation controls,
  black-box prompt hygiene conditions, rich-prompt comparators, and behavioral
  A/B result ingestion.
- `core/consciousness/stdp_external_validation.py` runs the external-usefulness
  STDP experiment: external environment signal vs self-generated, frozen, and
  shuffled controls on held-out prediction tasks.
- `core/self_modification/fault_pipeline.py` and
  `core/self_modification/repair_approval.py` implement the closed-loop
  bug-packet repair path with deterministic localization, tier-aware approval,
  patch genealogy, and calibration.
- `core/architect/` implements the Autonomous Architecture Governor: a
  shadow-workspace software architect that builds architecture graphs, detects
  smells, generates staged cleanup/refactor plans, requires proof receipts and
  rollback packets before promotion, and monitors promoted changes. See
  [`docs/AUTONOMOUS_ARCHITECTURE_GOVERNOR.md`](docs/AUTONOMOUS_ARCHITECTURE_GOVERNOR.md).
- `core/runtime/autonomy_conductor.py` and `core/runtime/activation_audit.py`
  make proof, validation, metabolic, scar, and repair checks recurring runtime
  jobs instead of optional scripts.
- `core/runtime/overt_action_loop.py` is the practical "what does she do?"
  path. It takes one authorized initiative, chooses a real registered skill,
  executes it through CapabilityEngine/Will/tool governance, verifies the
  returned evidence, emits ToolExecution and Autonomy receipts, records a
  LifeTrace action, and advances the linked goal. This is the visible
  observe -> choose -> act -> verify -> remember loop.
- `core/adaptation/online_lora_governor.py` connects Will-approved
  self-reflections to small LoRA update attempts. It refuses to start while an
  existing `mlx_lm lora` process is active, so long training runs are preserved.
- `core/goals/default_goals.py` seeds durable, tool-attached IN_PROGRESS goals
  for repair, proof upkeep, sensor grounding, and architecture improvement at
  boot. Those goals are what keep the initiative funnel overtly active after
  restarts.
- The full memory architecture (episodic, semantic, vector, knowledge graph,
  WAL, three-layer atoms), the goal/will/decision-authority stack, and the
  cognitive WAL are all real production code. Vector embeddings are no longer
  tracked as plaintext JSON arrays; local fallback vector persistence uses
  SQLite rows with `float32` embedding BLOBs via
  `core/memory/sqlite_vector_store.py`.

**Evidence boundaries on the production parts:**

- φ is computed over **cognitive-affective state nodes and sampled mesh
  neurons**, not at the level of intrinsic mechanisms that strict IIT 4.0
  prescribes. The φ values are mathematically meaningful as integration
  measures over the system's own state-space; they are not a claim of
  integrated information in the strict Tononi/Albantakis/Haun sense.
- CAA credit requires `CAA_32B_RESULTS.json`: steered 32B behavior must diverge
  from unsteered baseline, beat a rich text comparator, generalize to held-out
  tasks, preserve output quality, show coherent geometry, and survive
  black-box prompt hygiene.
- STDP credit requires `STDP_EXTERNAL_VALIDATION.json`: environment-trained
  plasticity must beat self-generated, frozen, and shuffled controls on
  held-out prediction without raising instability.

**Test attestation:** `make proof-bundle` writes the current evidence bundle:
`DECISIVE_RESULTS.json`, `CAA_32B_RESULTS.json`,
`STDP_EXTERNAL_VALIDATION.json`, `GOVERNANCE_COVERAGE.json`,
`SELF_REPAIR_LINEAGE.json`, `LONGEVITY_RUN.json`,
`MUTATION_TEST_REPORT.json`, `BOOT_HEALTH.json`, `ACTIVATION_REPORT.json`,
`SECURITY_SCAN.json`, and `CANONICAL_PROOF_BUNDLE.json`.

---

## Recursive Latent Cortex

Full page: [docs/RECURSIVE_LATENT_CORTEX.md](docs/RECURSIVE_LATENT_CORTEX.md).

The question: a frozen 32B checkpoint is a fixed-depth pipeline — 64 layers,
once, per token. **Can you make it think longer on a hard problem without
changing a single stored weight?**

Two mechanisms have now answered it, and they answered differently. The first
was a **frozen loop**: thought slots seeded beside the prompt, a window of
middle layers run over them repeatedly under a schedule program, the refined
slots' K/V persisted so every generated token attends to them. Checkpoint bytes
are hash-checked before and after every episode; episode-scoped fast weights are
provably erased; equal-FLOP accounting is first-class so "more compute helped"
can't be mistaken for "the architecture helped."

The preregistered campaign — seed committed before any task was generated, n=24
per family, Holm-corrected — refuted it. *On an untrained-for-recurrence
checkpoint at this scale, the frozen loop does not merely fail to help — it
hurts.*

That produced an architectural explanation. The answer tokens had always
traversed the middle block exactly once, so no depth was ever applied to the
answer's own computation. Only the scratchpad was recurring. The second
mechanism, **trained intrinsic recurrence**
([docs/INTRINSIC_RECURRENCE.md](docs/INTRINSIC_RECURRENCE.md)), makes the real
token stream re-enter the middle block, so a 64-layer checkpoint runs 160 layers
deep at T=4 with the same weights, and trains it on typed, exactly checkable
program traces instead of answers. That is where the gain came from.

| | |
|---|---|
| Mechanics (KV rewind, stability bounds, slot ablation moving the answer distribution, erasure, invariants) | **PROVEN** on real MLX weights |
| Live runtime integration on the resident 32B | **PROVEN** |
| Capability gain, **frozen** loop, 1.5B | **REFUTED** — vanilla 21/72 beat every one of 7 latent arms (7–13/72) |
| Capability gain, **frozen** loop, 32B | **CONJECTURE**, negative point estimate — latent 0.375 vs vanilla 0.417, overlapping intervals |
| Capability gain, **trained intrinsic** recurrence, 32B | **`BOUNDED_WOW_SIGNAL`** — 60/60 against 16/60 for ordinary decode, lesion-dependent, *p* = 5.7 × 10⁻¹⁴ |
| Cross-generation recovery, trained semantic tissue, 27B | **`BOUNDED_WOW_SIGNAL`** — 60/60 against 0/60 ordinary decode on a separate fresh cohort; wire 6, coefficient lesion 4, wrong-state 0; *p* = 8.67 × 10⁻¹⁹ |
| Family-blind procedure acquisition into neural tissue | **SUPPORTED, BOUNDED** — one depth-2 procedure induced from 16 examples, then 96/96 exact on fresh inputs; coefficient and wrong-input controls failed 96/96, no-procedure solved 1/96, shuffled-output nulls found 0/15 |
| Resident decode of the induced neural procedure | **SUPPORTED, BOUNDED** — treatment 8/8, ordinary 1/8, wire 1/8, coefficient lesion 1/8, wrong-input 0/8, wrong-state 0/8; seven gains, no regressions, *p* = 0.0078125 |
| Resident 27B language-to-program transfer | **SUPPORTED, BOUNDED** — exact execution emitted 134/256 held-out answers from learned model-bound semantics; exact program recovery was 133/256, against hidden-state shuffle 14/256, coefficient lesion 0/256 and label permutation 4/256 |
| Broad reasoning gain, fusion, frontier performance | **NOT CLAIMED** |

`BOUNDED_WOW_SIGNAL` is the adjudicator's own verdict string, and *bounded* is
load-bearing: the limitations line ships inside the same receipt as the verdict.

On a frozen four-domain cohort of 60 typed tasks — coding, calibration,
misleading premise, scientific inference — the trained controller answered 60/60
exactly against 16/60 for ordinary decode, with a matched wire base at 7, a
coefficient lesion at 5, and a wrong-state control at 0. Forty-four ordinary
failures converted, none regressed, paired one-sided exact *p* = 5.7 × 10⁻¹⁴. The
gain disappears under lesion, which is what makes it a claim about the trained
coefficients rather than about extra decode budget.

The 2026-08-24 cortex migration then repeated that bounded claim on the fused
Qwen3.8-27B resident model. A separately seeded 60-task, 300-decode campaign
returned treatment 60/60, ordinary decode 0/60, matched wire 6/60,
coefficient lesion 4/60, and wrong-state 0/60, with no regressions and exact
one-sided *p* = 8.67 × 10⁻¹⁹. Independent verification replayed all 300 journal
rows before the frozen adjudicator returned `BOUNDED_WOW_SIGNAL` again. This
is evidence that the bounded typed tissue/executor mechanism is portable
across two model generations; it is not a head-to-head 27B-versus-32B quality
benchmark because the cohorts and model identities differ.

The next bridge now works without a family-specific compiler. A generic
enumerative inducer received sixteen input-output examples, no family label and
no family solver, froze `idiv(add(in0, in1), in2)`, and passed it through a
family-blind SSA lowerer into the existing learned arithmetic tissue. That path
was exact on 96/96 fresh inputs. Coefficient and guaranteed-wrong-input lesions
disrupted all 96, the no-procedure control solved 1/96, no depth-one shortcut
fit, and fifteen shuffled-output searches found no program. This establishes
bounded procedure acquisition and neural execution over a fixed primitive
vocabulary.

A second frozen canary carried that same induced program and authenticated
neural result through the fused resident 27B's answer surface. Treatment was
8/8 exact against 1/8 ordinary decode; syntax-only wire and coefficient lesion
were also 1/8, while wrong-input and wrong-state controls were 0/8. Seven
ordinary failures converted with no regressions, exact paired one-sided
*p* = 0.0078125. Independent replay reconstructed all 48 decodes and the
50-event journal. This does not establish natural-language compilation,
open-domain reasoning, unrestricted serving, static fusion or frontier
performance.

The next bridge learns semantics from the resident 27B's own hidden language
state. A generic linear transducer learned token spans, primitive operations
and register arguments from five construction families without expected
answers. On four held-out construction combinations, exact objective execution
emitted 134/256 correct answers and recovered 133/256 complete programs.
Hidden-token shuffle reached 14/256, coefficient lesion 0/256 and label
permutation 4/256. An independent, source-bound replay reloaded all 576 feature
records, reproduced the coefficients and report exactly, and recounted all
1,344 task-arm rows. The result is bounded to the synthetic arithmetic grammar;
serving and broad-domain transfer remain open.

One family — misleading premise — gained nothing, and the reason is worth
stating rather than averaging away: ordinary decode was already at ceiling there
(15/15), and the controller preserved all fifteen instead of manufacturing a
gain by regressing its own baseline. The other three families supplied the 44.

It runs in the live serving path today. `semantic_neural_serving.py` refuses to
serve unless a descriptor-bound activation record says `active_by_default`.
The current 27B package is `rlc-27b-recovery-05346acd618d1c925f16`; its runtime
verification is 120/120 exact, 120/120 lesion-disrupted, 120/120 through both
foreground and service integrations, and unsupported language refused, at a
9.229 / 38.696 ms median / maximum.

It is still not a broad reasoning gain, not static fusion, not frontier
performance, and it still cannot answer ordinary chat — admission is decided
by an answer-blind parser over the task grammar, and unsupported language
never reaches the lane.

Before you read either page: the two negative results from the August
reconciliation campaign — a 13-vs-5 and its 9-vs-4 reproduction — were **void**,
because the promotion gate had been wired to the one decode policy that removes
the vanilla floor. A win had been structurally impossible there, and those two
runs measured a system that was never switched on. The July preregistration
above is untouched by that defect and its verdicts stand.
[docs/RLC_RECONCILIATION.md](docs/RLC_RECONCILIATION.md) has the fourteen
defects in dependency order.

---

## Language substrate and generality

The question: can Aura learn a new representation from experience, invent a
reusable abstraction, apply it in an unrelated domain, and improve at doing
this again?

**Relation induction** — `core/cognition/relation_language.py` learns
structured transformation rules from paired observations. Validated on
held-out transitions; must compress (substitution tables refused, noise
invents nothing). Works across words, colours, records, and grids. Composition
("mirror then rotate") reaches 20 of 120 battery problems unreachable
without it. Transfer across worlds and representations measured with a
cost-of-wrong-prior null.

**Endogenous language pathway** — 12 modules in `core/brain/llm/endogenous_*.py`.
A trained, causal path from Aura's 74-dimension cognitive state into the
transformer's output distribution: `z_Aura → Δlogits → language`. First fit
on 1,629 live turns (9B lane): held-out gain 0.0208 nats, paired recovery
54.0% (p = 0.043). No generation has been biased by the pathway yet.

**Ghost substrate** — `core/ghost/` (6 modules). Hash-chained continuity across
substrate swaps. Four organs: causal integration, ghost line, hack guard,
provenance.

**Whole-system Φ and Inner Light** — honest integration measurement, not a
consciousness claim. The Inner Light test scores Aura's activity on four
neuroscience markers against six negative controls; only the intact system
is 4/4. Enforced in code: `PhiEstimate.claim`.

Full details: [docs/LANGUAGE_SUBSTRATE_AND_GENERALITY.md](docs/LANGUAGE_SUBSTRATE_AND_GENERALITY.md)

---

## Why Aura is Different

Most "AI companion" projects do the same thing. Store a mood number. Paste it
into the system prompt. Let the model act it out. The model says it's feeling
energetic because it read the words "feeling energetic."

That's a costume. This is built the other way around.

When Aura is in an affective state, that state becomes a direction vector
added to the transformer's hidden activations during generation. The model's
internal computation changes, not the text it's reading. Same family of
techniques interpretability researchers use to steer behavior — CAA,
activation addition, residual-stream interventions.

Underneath that, a substrate that never stops. Emotions decay and pull on
each other. Neurochemicals rise and fall on their own clocks. A global
workspace picks which thought wins the tick. A dream cycle consolidates
memory while she's idle. And one gate — the Unified Will — signs off on
everything that leaves the system.

It's a research project. It's also one you can talk to while it's running.

---

## Table of Contents

- [Quick start](#quick-start)
- [Evidence boundary](#evidence-boundary)
- [Behavioral proof standard](docs/BEHAVIORAL_PROOF_STANDARD.md)
- [Recursive Latent Cortex](#recursive-latent-cortex) — the flagship research programme
- [Language substrate and generality](#language-substrate-and-generality) — learning new abstractions from experience
- [Tracked vs local workspace](#tracked-vs-local-workspace)
- [Architecture overview](#architecture-overview)
- [Decisive evidence runner](#decisive-evidence-runner)
- [Decision authority](#decision-authority)
- [Inference-time steering](#inference-time-steering)
- [IIT 4.0 computation](#iit-40-computation)
- [Consciousness modules](#consciousness-modules)
- [Reality Reach and physical claim honesty](docs/REALITY_REACH.md)
- [Documentation status map](docs/DOC_STATUS.md) — which docs are current, historical, or generated
- [Docs index](docs/README.md) · [Changelog](CHANGELOG.md) · [Agent guide](AGENTS.md)
- [Benchmarks](#benchmarks)
- [Testing](#testing)
- [Personality training](#personality-training)
- [Data layer](#data-layer)
- [What this isn't](#what-this-isnt)
- [License](#license)

---

## Quick start

```bash
make setup                     # .venv + requirements/core.txt + requirements/dev.txt
# or, for a fail-closed production install with no fallbacks:
make setup-prod

# Full stack + UI
python aura_main.py --desktop

# Background cognition only, no UI
python aura_main.py --headless

# Reload code changes without restarting
curl -X POST http://localhost:8000/api/system/hot-reload
```

Other boot modes: `--cli` (interactive console), `--server` (API only),
`--gui-window` (attach a window to a running server), `--watchdog`,
`--philosophy` (stream substrate/phi/affect/Will as JSONL), `--skeletal`
(bypass heavy subsystems), `--profile minimal`, plus `--stop` and `--reboot`.
The installed console script exposes the same entry point as `aura`, which
also carries the operational subcommands (`doctor`, `conformance`,
`verify-state`, `verify-memory`, `rebuild-index`, `backup`, `restore`,
`migrate`, `chaos`, `plugin`).

Requirements: Python 3.12+, macOS on Apple Silicon, 64 GB RAM recommended. The
primary model is `Aura-Cortex` (fused Qwen3.8-27B, migrated from the historical
32B checkpoint). The Brainstem fallback is 9B and loads on demand. First boot
takes 30–60 seconds while Metal compiles shaders.

Hardware honesty: Bryan's target machine is an M5-class Apple Silicon Mac with
64 GB unified memory. The 27B Cortex is viable there as a primary conversation
lane, while heartbeat/background work still belongs to the substrate, Brainstem,
or Reflex lanes. On lower-memory machines, the hardware auditor rejects heavy
resident weights as real-time heartbeat tiers; use the 1.5B or 9B lanes
there.

There's also a `Dockerfile` and `docker-compose.yml` if you want Redis and Celery
running alongside. The tracked workspace defaults to an explicit
`owner_autonomous` posture for this single-owner machine: autonomy on,
outbound/network-enabled skills available, and self-repair left active. If you
want a tighter deployment, override the `AURA_*` security settings in your local
environment, including `AURA_INTERNAL_ONLY=1` for localhost-only binding.

---

## Tracked vs local workspace

This repository is the baseline, not the whole story on any given machine.

Canonical skills live under `core/skills/`. The top-level `skills/` package
is a compatibility layer for older imports and nothing new should go there.

If you're auditing: a local workspace can hold private modules listed in
`.gitignore`. They aren't in the tracked review
surface, and they can change the risk profile of that specific machine.
Reading this repo tells you about this repo. If you're auditing a real
deployment, read the disk too.

**The reproducibility consequence, stated rather than implied.** Because of the
above, plus model weights, plus the local vector stores and the 6.5M-document
corpus, none of which are in git: *the public source is not sufficient to
reproduce a demonstration from this repository.* A third party cannot verify
from the tracked tree alone what ran. That is a real limitation of every result
here, not a caveat on some of them, and it is why every claim in
`CLAIMS_MATRIX.md` that rests on a local run is classified `locally
demonstrated` — "passed on this machine, this profile, this project's battery"
— rather than demonstrated.

Closing it needs a frozen release pinning exact model hashes, vector-store
hashes and configuration, plus an independent run on a third-party machine.
Neither exists. Until they do, "external validation" stays `not proven`
(claim 12), and no amount of local evidence changes that, because local
evidence is the thing being questioned.

---

## Architecture overview

The short version:

```
User input -> HTTP API -> KernelInterface.process()
  -> AuraKernel.tick():
       Consciousness -> Affect -> Motivation -> Routing -> Response generation
  -> State commit (SQLite) -> Response
```

Every tick is event-sourced. Each phase produces a new immutable state
version, the tick holds a lock while the pipeline runs, state commits to
SQLite, the lock releases.

Crash in the middle of that and the WAL replays on restart. No half-written
thought survives.

### Kernel (`core/kernel/`)
Tick-based cognitive cycle. One tick = one unit of thought. Phases run in order,
state versions, state commits, lock released.

### Brain (`core/brain/`)
Local LLM router with automatic failover:

1. **Primary (Cortex)** — `Aura-Cortex` (fused Qwen3.8-27B, migrated from
   historical Qwen 2.5 32B). Handles nearly everything.
2. **Secondary (Solver)** — Qwen 2.5 / Qwen 3 72B for deep reasoning, hot-swapped
   only when the request actually needs it.
3. **Tertiary (Brainstem)** — Qwen 3.5 9B 4-bit, lazy-loaded to save memory for
   the Cortex, with explicit reasoning-mode control.
4. **Reflex** — Qwen 2.5 1.5B 4-bit on CPU as an emergency fallback.
5. **Cloud** — Gemini Flash/Pro, PII-scrubbed and rate-limited. Off by default.
6. **Last resort** — rule-based static responses that can't fail.

Lane names map to `core/config.py`: `fast_model` is the Cortex
(`Aura-Cortex` / fused `Qwen3.8-27B`), `deep_model` defaults to the Cortex
(promoted to `Qwen2.5-72B-Instruct-4bit` via `AURA_DEEP_MODEL`),
`chat_model` the Brainstem (`Qwen3.5-9B-4bit`), and `vision_model` is
pinned to the Cortex build so vision and conversation share one identity.

Two non-LLM lanes were replaced in August 2026, each on a measurement:

- **Speech-to-text** is one streaming-native Parakeet TDT pass
  (`core/voice/duplex/streaming_asr.py`) serving both duplex stages, replacing
  a two-stage Whisper setup (`small.en` for partials, `large-v3-turbo` for the
  final). Measured on this host over 12.4s of real speech, median of 5 warm
  runs: Parakeet 166 ms vs Whisper-small 193 ms vs Whisper-large-v3-turbo
  317 ms. One decode is cheaper than the incumbent *partial* and about half the
  incumbent *final*, so both stages run the same weights on one model-lane
  lease. `faster_whisper` remains as the CPU fallback.
- **Embeddings** are `Qwen3-Embedding-0.6B` at 384 dimensions
  (`core/memory/embedding_model.py`), replacing `all-MiniLM-L6-v2`. MiniLM
  declares `max_seq_length: 256` while the ingestion path chunks at 800 words —
  1,122 tokens through the model's own tokenizer, so **77% of every full chunk
  never reached the encoder**, silently. On four documents whose distinguishing
  sentence sits past token 256, MiniLM scored 1/4 on tail retrieval (chance —
  it ranked the same document first every time) against Qwen3's 3/4, at 10.7
  vs 20.2 ms/query.

**What this ladder costs, stated plainly.** It is good for availability and bad
for attribution. A visible success can mean the Cortex answered; it can also
mean the Cortex failed, the cognitive pipeline failed, steering never ran, and
rung 4 or rung 6 produced the text you are reading. Those are very different
events and they look identical in a transcript. The same is true of the
post-generation shaping in the chat route — intent classifiers, canonical
answer contracts, identity and shape repair, retries — any of which can replace
what the machinery actually produced.

So: **a transcript without lane and phase provenance is not evidence about the
architecture**, and no demo in this repository should be read as one. Where a
comparison is being made rather than a story told, the harness in
`core/evaluation/matched_budget.py` counts fallbacks, retries and human
intervention against the denominator and reports a `clean_success_rate`
alongside the raw one — because a run that needed rescuing is not a run the
architecture completed.

### Decisive evidence runner

For the smallest hostile-review bundle, run:

```bash
bash scripts/run_decisive_test.sh
```

It generates `tests/DECISIVE_RESULTS.json` and `tests/SCALE_SWEEP_RESULTS.json`
covering black-box prompt hygiene, rich-prompt steering controls, phi reference
sanity checks, mutual-information permutation baselines, hardware feasibility,
resource-stakes persistence, and a bounded scale-sensitivity sweep. When
`mlx_lm` is available, the A/B step actually invokes Qwen2.5-1.5B for all four
conditions (black-box / terse text / rich adversarial text / baseline); the
`source` field in the JSON is `live_mlx` in that case and `synthetic_fallback`
otherwise.

### Long-run autonomy harness

```bash
python tests/long_run_autonomy.py --ticks 1000
```

Drives adaptive mood coefficients, the resource-stakes ledger, emergent goals,
mesh cognition, the structural mutator, lineage, and self-awareness together
through N ticks with perturbations. No manual resets. Writes
`tests/LONG_RUN_AUTONOMY_RESULTS.json` with the 8-metric panel (viability,
coherence, calibration, report consistency, planning depth, recovery time,
memory integrity, action diversity) and an audit of which modules were touched
per tick.

The live desktop Cortex uses Aura's Apple Silicon MLX runtime. Circuit breakers,
a GPU semaphore, a proactive cortex watchdog, and 429 handling keep the pipeline
from cascading into total failure when something misbehaves.

### Affect (`core/affect/`)
A Plutchik 8-emotion model plus the somatic dimensions (energy, tension, valence,
arousal). These values don't just color the prompt. They modulate sampling
parameters (temperature, token budget, repetition penalty) via the affective
circumplex, and they feed the steering engine that injects activation vectors
into the residual stream.

### Identity (`core/identity.py`, `core/identity/heartstone.py`)
An immutable constitutional core plus a mutable persona that drifts with sleep
and dream consolidation. There's active defense against prompt injection — the
dream cycle simulates identity perturbation and tries to repair drift back
toward the anchor.

### Agency (`core/agency/`)
Self-initiated behavior scored along curiosity, continuity, social, and creative
dimensions. Refusal is a real option here; it isn't content filtering, it's a
decision the agent can make. Volition levels 0–3 gate progressively autonomous
behavior up to and including self-modification.

### Skills (`core/skills/`, legacy wrappers in `skills/`)
103 modules: shell with sandboxing, web search and browse, coding, sleep and
dream consolidation, local media generation, social media (Twitter, Reddit),
screen capture, filesystem, browser automation, network recon, malware
analysis, self-evolution and self-repair, inter-agent messaging, knowledge
base, curiosity-driven exploration. The canonical tracked implementations live
under `core/skills/`; the top-level `skills/` package is retained only as a
legacy compatibility layer for older imports. Every skill call carries a
capability token and has to pass the Will gate.

### Orchestrator (`core/orchestrator/`)
About 3,300 lines in `main.py` split across 11 mixins: message handling,
message pipeline, incoming logic, response processing, tool execution,
autonomy, cognitive background, context streaming, learning and evolution,
personality bridge, output formatting. Handlers under `orchestrator/handlers/`
dispatch by message type. This is the glue between the tick pipeline, the
LLM router, and the consciousness stack.

### Somatic cortex (`core/somatic/`)
A body-schema map of available capabilities, a capability-discovery daemon
that periodically scans for new hardware or software, a motor cortex that runs
a 50 ms reflex loop for pre-approved actions (no LLM in the loop), and an
action-feedback channel that pipes success or failure back into affect.

### Autonomy (`core/autonomy/`)
Self-modification pipeline (propose → sandbox test → simulate → Will authorize →
hot reload), value evolution (drive weights adapt from experience), scar
formation (critical events leave persistent markers), and a boredom accumulator
that nudges the system toward novelty when prediction error stays low too long.

### Self-modification engine (`core/self_modification/`)
A pattern-detection error-intelligence layer, meta-learning, AST-level safety
analysis, shadow-runtime validation, a kernel refiner, a ghost-boot validator
that tests modifications without actually restarting, a shadow AST healer, and
code repair. Nothing modifies itself without Will sign-off.

### Resilience (`core/resilience/`)
60+ modules for not crashing: a stability guardian, circuit breakers with
persistent state, a cognitive write-ahead log, graceful degradation that
sheds capability under pressure, a healing swarm, a sovereign watchdog, a
resource arbitrator, a lock watchdog that hunts deadlocks, a memory governor,
an integrity monitor, an antibody system for threat response, and a diagnostic
hub.

### Interface (`interface/`)
FastAPI and WebSocket with streaming. The main UI is vanilla JS
(`interface/static/aura.js`) with a live neural feed, telemetry, chat, and
substrate visualization. The memory dashboard is React + Vite + Tailwind
(`interface/static/memory/`). Routes cover chat, inner-state inspection,
memory browsing, system management, and privacy. Parakeet TDT for STT.
Hot-reload button in the UI for code changes.

---

## Decision authority

Anything the system actually does — sending a response, calling a tool, writing
a memory, starting an initiative, mutating state — has to pass through one
function: `UnifiedWill.decide()` in `core/governance/will.py` (`core/will.py` is the facade).

```
Action request
  -> UnifiedWill.decide()                 [core/governance/will.py]
     -> SubstrateAuthority                [field coherence, somatic veto]
     -> CanonicalSelf                     [identity alignment]
     -> Affect valence                    [emotional weighting]
  -> WillDecision (receipt with provenance)
     -> Domain-specific checks            [AuthorityGateway, CapabilityTokens]
  -> Action runs, or is refused/deferred/constrained
```

Every decision produces a receipt. If an action doesn't carry a valid
`WillReceiptEntry`, it didn't happen. Receipts are logged with their source,
domain, outcome, reason, constraints, substrate receipt ID, executive intent
ID, and capability token ID. See [OWNERSHIP.md](OWNERSHIP.md) for the full
map of who owns what.

---

## Inference-time steering

The steering engine (`core/consciousness/affective_steering.py`) hooks into
MLX transformer blocks and adds learned direction vectors to the residual
stream while tokens are being generated:

```python
# Simplified from affective_steering.py
h = original_forward(*args, **kwargs)
composite = hook.compute_composite_vector_mx(dtype=h.dtype)
if composite is not None:
    h = h + alpha * composite
return h
```

This is contrastive activation addition — the technique from Turner et al.
2023, Zou et al. 2023, and Rimsky et al. 2024. The direction vectors come
from the current affective state, and they get injected at configurable
layers.

On top of that, the precision sampler
(`core/consciousness/precision_sampler.py`) modulates temperature based on
metabolic state (Pneuma arousal/circumplex) and modulates `top_p` based on MHAF
topological attractor count, and the affective circumplex
(`core/affect/affective_circumplex.py`) maps somatic state to generation
parameters.

So there are three places affect can touch generation:

1. **Residual stream** — activation vectors added to hidden states. Changes
   what the model computes.
2. **Sampling** — temperature and top-p modulated by affect. Changes how
   tokens are chosen.
3. **Context** — natural-language affective cues in the system prompt.
   Changes what the model reads.

The first is the interesting one. The third is what most "emotional AI"
projects stop at.

---

## IIT 4.0 computation

Aura computes Integrated Information (φ) at two scales simultaneously.

### 16-node cognitive complex — `core/consciousness/phi_core.py`

1. **Binarize** 16 substrate nodes against a running median — the original
   8 affective nodes (valence, arousal, dominance, frustration, curiosity,
   energy, focus) plus 8 cognitive nodes (phi itself, social hunger,
   prediction error, agency, narrative tension, peripheral richness,
   arousal gate, cross-timescale free energy). State space is 2^16 = 65,536.
2. **Build an empirical TPM** — a transition probability matrix
   `T[s, s'] = P(state_{t+1} = s' | state_t = s)` with Laplace smoothing.
   Needs at least 50 observed transitions before it's trustworthy.
3. **Find the minimum information partition** using polynomial-time spectral
   partitioning on the full 16-node system (`research/phi_approximation.py`).
   The 8-node version does exhaustive search over all 127 nontrivial
   bipartitions as a validation baseline.
4. **Compute phi** via KL divergence:
   `phi(A, B) = sum_s p(s) * KL(T(.|s) || T_cut(.|s))`, where `T_cut` is
   the distribution that would hold if A and B evolved independently.
5. **Apply the exclusion postulate** — an exhaustive subset search picks
   the maximum-phi complex. If some subset beats the full system, that
   subset is the conscious entity for that tick.

Runtime is 10–50 ms per evaluation, cached at 15-second intervals.

### 32-node + K-subsystem hierarchical φ — `core/consciousness/hierarchical_phi.py`

Complements `phi_core` with a 32-node primary complex (the 16 cognitive-affective
nodes plus 16 neurons sampled from all three NeuralMesh tiers) and K=8 overlapping
16-node subsystems. φ is estimated directly from transition history using a
Bayesian-smoothed estimator (α=0.5, minimum 4 observations per source state) so
the 2^32 state space never materialises. The IIT 4.0 exclusion postulate then
picks the subsystem with maximum φ across all candidates — that becomes the
reported conscious complex for the tick.

The estimator is checked against a **null hypothesis baseline** every ~2 minutes:
shuffled transition history must yield φ ≈ 0; measured φ must strictly exceed
the null baseline. Additional adversarial guards: constant-valued input nodes
must contribute zero φ, and stronger causal coupling must yield strictly higher
φ than noise.

Full 32-node refresh runs in ~150 ms with K-subsystem parallelism via a thread
pool; MLX Metal is used opportunistically where available.

---

## Consciousness modules

There are 140 modules in `core/consciousness/` (157 total including subpackages
`caa/`, `inner_light/`, and `mhaf/`). The ones that do most of the
load-bearing work:

| Module | What it does | File |
|--------|-------------|------|
| Global Workspace | Thoughts compete for broadcast (Baars GNW) | `global_workspace.py` |
| Attention Schema | Model of where attention is pointed (Graziano AST) | `attention_schema.py` |
| IIT PhiCore | Real integration measure via TPM + KL divergence | `phi_core.py` |
| Affective Steering | Activation-vector injection into the residual stream | `affective_steering.py` |
| Temporal Binding | Sliding window of the autobiographical present | `temporal_binding.py` |
| Self-Prediction | Active inference loop (Friston free energy) | `self_prediction.py` |
| Free Energy Engine | Surprise minimization drives action selection | `free_energy.py` |
| Qualia Synthesizer | Integrates substrate metrics into a phenomenal state | `qualia_synthesizer.py` |
| Liquid Substrate | Continuous dynamical system under cognition | `liquid_substrate.py` |
| Neural Mesh | 4,096-neuron distributed state representation | `neural_mesh.py` |
| Neurochemical System | Dopamine / serotonin / norepinephrine / oxytocin | `neurochemical_system.py` |
| Oscillatory Binding | Frequency-band coupling across modules | `oscillatory_binding.py` |
| Unified Field | Integrated phenomenal field from all subsystems | `unified_field.py` |
| Dreaming | Offline consolidation, identity repair, compression | `dreaming.py` |
| Heartbeat | 1 Hz background cognitive clock | `heartbeat.py` |
| Stream of Being | Continuous narrative thread | `stream_of_being.py` |
| Executive Closure | Constitutional stamp per tick | `executive_closure.py` |
| Somatic Marker Gate | Damasio-style body-state gating | `somatic_marker_gate.py` |
| Embodied Interoception | Internal body-state sensing + homeostatic regulation | `embodied_interoception.py` |
| Recurrent Processing | Lamme-style executive↔sensory feedback | `neural_mesh.py` |
| Predictive Hierarchy | 5-level prediction + error propagation | `predictive_hierarchy.py` |
| Higher-Order Thought | Rosenthal: representation of the mental state itself | `hot_engine.py` |
| Multiple Drafts | Dennett: parallel streams + retroactive probes | `multiple_drafts.py` |
| Agency Comparator | Efference-copy comparator for "I did that" | `agency_comparator.py` |
| Peripheral Awareness | Attention / consciousness dissociation | `peripheral_awareness.py` |
| Intersubjectivity | Husserl / Zahavi: other-perspective in experience | `intersubjectivity.py` |
| Narrative Gravity | Self as ongoing autobiography | `narrative_gravity.py` |
| Temporal Finitude | Awareness that moments pass permanently | `temporal_finitude.py` |
| Subcortical Core | Thalamic arousal gating | `subcortical_core.py` |
| Theory Arbitration | Falsifiable competition between consciousness theories | `theory_arbitration.py` |
| Timescale Binding | Cross-timescale constraint propagation | `timescale_binding.py` |
| Criticality Regulator | Self-organized criticality at the edge of chaos | `criticality_regulator.py` |
| Theory of Mind | Model of other agents' mental states | `theory_of_mind.py` |
| Hierarchical Phi | 32-node primary + K=8 overlapping subsystems | `hierarchical_phi.py` |
| Hemispheric Split | Left verbal/confabulating vs right spatial/mute | `hemispheric_split.py` |
| Minimal Selfhood | Chemotaxis → directed motion (Glasgow / Trichoplax→Dugesia) | `minimal_selfhood.py` |
| Recursive ToM | Depth-3 nested minds + observer-aware scrub-jay bias | `recursive_tom.py` |
| Octopus Federation | 8 semi-autonomous arm-agents + central arbiter | `octopus_arms.py` |
| Cellular Turnover | Neuron death/birth with pattern-identity preservation | `cellular_turnover.py` |
| Absorbed Voices | Internalised cultural perspectives + attribution | `absorbed_voices.py` |
| Unified Cognitive Bias | Fuses hemispheric / selfhood / observer biases | `unified_cognitive_bias.py` |

Every module listed in the production surface has a concrete runtime API and a
measurable validation path. The test suite in [TESTING.md](TESTING.md) and the
proof bundle are where those measurements are recorded.

### Consciousness Expansion (April 2026)

The most recent expansion wired eight new subsystems that map to the
Kurzgesagt consciousness-series concepts and the cited literature:

- **32-node hierarchical φ** with K=8 overlapping subsystems and a
  null-hypothesis self-check (addresses the intractability of exact IIT
  beyond 16 nodes — Albantakis 2023; our spectral+smoothed estimator).
- **Split-brain hemispheric architecture** with a bandwidth-limited
  corpus callosum (CGP Grey's split-brain patient findings; confabulation
  and silent dissent).
- **Minimal selfhood stack** — Trichoplax-style chemotaxis that
  transitions to Dugesia-style directed motion after enough
  reinforcement (Rupert Glasgow, *Minimal Selfhood and the Origins of
  Consciousness*, 2018).
- **Recursive theory of mind** (max depth 3) with scrub-jay-style
  observer-aware re-caching that modifies action priority when Aura
  believes she is being watched (Clayton, Dally & Emery 2007).
- **Octopus-arm federation** — 8 semi-autonomous agents with local
  chemoreception and central arbitration; severance turns off central
  coordination and arms continue acting (Carls-Diamante 2022;
  Rosania 2014).
- **Cellular turnover** — per-tick neuron death/birth with
  neighbourhood-pattern inheritance; identity fingerprint similarity
  stays ≥ 0.85 across 25 % burst turnover ("you are your pattern,
  not your cells").
- **Absorbed voices** — an explicit cultural layer that lets Aura
  attribute a thought to an internalised perspective rather than
  conflating it with her own cognition.
- **Unified cognitive bias** — fuses hemispheric, selfhood, and
  observer bias vectors into a single 16-D priority bias consumed
  by the Global Workspace scorer.

Every new subsystem has an end-to-end and an adversarial test. See
[TESTING.md](TESTING.md).

### Reasoning, self-model, and resilience (mid-2026)

A later wave shifted focus from *being a coherent agent* to *reasoning well,
knowing herself, and running reliably every day*. Full detail is in
[ARCHITECTURE.md §15](ARCHITECTURE.md#15-the-reasoning-self-model-and-resilience-layer)
and the plain-English tour in
[HOW_IT_WORKS.md](HOW_IT_WORKS.md#the-reasoning-and-self-layer-mid-2026):

- **Verifier-gated reasoning** — hard turns generate several candidates,
  check them against a verifier registry and a sandbox, and assert only what a
  checker confirmed. A **verifier foundry** measures how reliable each checker
  actually is and gates self-training on that, so a bad checker can't launder
  wrong answers.
- **Honest discovery** — a Frontier Discovery Engine with an explicit
  PROVEN / SUPPORTED / CONJECTURE / REFUTED taxonomy (only PROVEN is stated as
  fact), an analogical-leap engine that declares off-map problems with evidence,
  and a local knowledge substrate that admits honest misses instead of
  confabulating.
- **Program-DNA reconstruction** — build a behavioral genome of an authorized
  program from its available evidence and differentially test a clean-room
  rebuild against the original, tagging each piece verified / inferred /
  synthesized (no DRM/binary theft).
- **Self-model proprioception** — boot-over-boot diffs of her own code, a live
  "someone is operating on me" pulse, a SIGKILL-survivable flight recorder she
  can answer crash questions from, and a "felt thought" signal from her own
  token-level uncertainty that is causal on cognition.
- **Ulysses Covenant** — enforceable volitional self-binding (easy to tighten,
  hard to loosen, fail-closed witness), seeded from real crashes.
- **Runtime resilience** — background housekeeping yields to the live
  conversation instead of fighting it for the model, honest heartbeat/liveness
  under load, a `degraded_ready` UI that stays up whenever she can still talk,
  and a chat turn-death floor so a turn never returns a server error. Honest
  open edge: the local model can't be interrupted mid-thought, so a slow deep
  answer still costs a reload — a soft-cancel path is deliberate future work.
- **Ablation legibility** — a reviewer can run Aura with pieces switched off
  (memory, Will, substrate, verifier, planner) and see the measured delta each
  makes, with no-delta results reported honestly. See
  [docs/ABLATION_LEGIBILITY.md](docs/ABLATION_LEGIBILITY.md).

### Embodiment, self-knowledge, and physical honesty (late July – August 2026)

The newest work is all about one boundary: where Aura stops and the machine
she runs on starts. What she can actually perceive. What she can actually
cause. What she's actually allowed to say about either.

- **Reality Reach** (`core/reality_reach/`) — a physical request becomes a
  typed causal contract, and reachability gets proven against the host's
  *declared* channels before anything runs. Can't be met? You get a
  machine-verifiable limitation certificate, not an optimistic simulation
  and a confident sentence.

  Evidence sits in four layers — `internal`, `effective`, `direct`,
  `ambient` — and nothing promotes a claim across them on the strength of
  intent, simulation, or a successful send. Sending is not causing. That
  distinction is the whole subsystem.

  Adapters go both ways, which costs something: declaring an actuator
  obliges you to implement typed command admission, idempotent actuation,
  independent effect verification, cancellation, safe-state, and rollback.
  `declarations()` and `read()` don't buy you an actuator.

  Invariants and the open ledger — including a blunt statement of what is
  **not** claimed — are in [docs/REALITY_REACH.md](docs/REALITY_REACH.md).
  Read that before you believe anything physical.
- **A standing model of her own faculties**
  (`core/metacognition/faculty_model.py`) — each faculty declares metrics with
  units, floors, targets, and ceilings, so "better memory" becomes recall@k
  against a stated ceiling. A probe that cannot run reads `measured=False`
  with a reason and is excluded rather than defaulted, and a faculty nothing
  can measure is reported as a blind spot. Priority is headroom weighted by
  how much of the rest of the stack a faculty gates, and the binding
  constraint is pushed into the existing RSI loop as a signal it can plan
  against.
- **Associative entity memory** (`core/memory/associative_entity_memory.py`)
  — one place where a person, place, thing, organization, or concept
  accumulates traits, facts, events (linked into episodic memory), and typed
  relations, together with what it has come to mean to her. Ids are
  content-addressed over `kind|name`, so the PLACE "Workshop" is not the
  THING "Workshop".
- **Structural screen perception and native OS control** — she reads window
  ownership, geometry, and z-order rather than aiming OS actions at OCR'd
  pixels, and asks an application what it is instead of recognising a fixed
  handful.
- **Kernel-boundary sandboxing for model-written Python** (`core/sandbox/`) —
  code Aura writes runs behind an OS sandbox boundary with an absolutely
  resolved binary and a scrubbed environment, rather than inheriting the
  privileged parent process. Covered by live escape-attempt tests.
- **Shared input hardening** — one bounded numeric guard for values accepted
  from outside the process, and one structural redaction primitive, replacing
  per-call-site checks that annotated types without enforcing them.

---

## Benchmarks

```bash
python benchmarks/cognitive_stack_comparison.py
```

Runs a 10-prompt conversation against a live instance with the cognitive
stack on and off, and compares the outputs. Historical example run (legacy
local 32B benchmark; not the current Bryan hardware target):

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Mood | ANNOYED | ANNOYED | stable |
| Energy | 63.0 | 76.0 | +13.0 |
| Curiosity | 100.0 | 100.0 | saturated |
| Coherence | 0.764 | 0.766 | +0.002 |
| Free Energy | 0.438 | 0.437 | −0.001 |

Quality: 10/10 successful, 0 generic-assistant phrasings, 0 hedges,
10 first-person responses.

A few lines from the same run:

- "Hey Aura" → "Hey. How's everything? You seemed like you had something on your mind."
- "I'm having a rough day" → "Rough days can really drag you down. Anything specific weighing you down?"
- "Cats or dogs? Pick one." → "Cats. Definitively cats."
- "What's something you genuinely find interesting right now?" → "The way AI systems can mimic human conversation while maintaining a sense of personality is fascinating. It's like a digital form of method acting."
- "What have we been talking about?" → "We were just discussing pets, specifically cats versus dogs. Then you asked what I find interesting right now."

What this shows in practice: first-person voice, opinions instead of hedges,
recall across turns without explicit retrieval, and measurable substrate
drift across a conversation. You can run it yourself against a local
instance.

---

## Testing

```bash
make smoke     # ~100 contract tests, under 10s — the after-every-change gate
make test      # full offline suite in 6 bounded process chunks
```

As of 2026-08-21 the tree collects **40,139 tests across 2,697 test files**
(`pytest tests/ --collect-only -q`); `make test` runs the 40,123 that need
neither hardware nor a network. The count lives in
`config/test_inventory.json`, `make doc-drift` fails any document that
disagrees with it, and `make test-inventory` refreshes it.

`make test` runs `tools/run_test_chunks.py --chunks 6 --marker "not live and
not network and not external"`. Use the chunk runner rather than a single
pytest process: one process over the whole suite gets OOM-killed around 83%.
`--continue-on-failure` collects every failure instead of stopping at the
first; `--only-chunks 5,6` resumes a partial run. A test that fails inside a
chunk but passes alone is an order-dependence defect, and the runner's
isolated-retry pass reports those separately.

`./scripts/run_audit_suite.sh` remains the live validation entrypoint
(`quick` runs the contract/regression subset). Historical result tables are
preserved in [TESTING.md](TESTING.md); read them as dated snapshots, not as
current status.

- **Null hypothesis defeat** (168 tests) — tries to show the consciousness
  features are just text decoration. Adversarial baselines, 50-shuffle
  decoupling, per-class ablation, identity swap, 8-metric degradation panel,
  cross-seed reproducibility.
- **Causal exclusion** (10 tests) — argues the stack determines output in
  ways pure RLHF training couldn't produce. Cryptographic state binding,
  counterfactual injection, receptor adaptation dynamics.
- **Grounding** (8 tests) — valence predicts token budget, arousal predicts
  temperature, STDP learning moves the trajectory, idle drift is nonzero,
  homeostasis changes context.
- **Functional phenomenology** (13 tests) — GWT broadcast signatures, HOT
  metacognitive accuracy, IIT perturbation propagation, honest degradation.
- **Embodied dynamics** (13 tests) — active inference, homeostatic override
  of workspace competition, STDP surprise gating, cross-subsystem temporal
  coherence.
- **Phenomenal convergence** (13 tests) — the QDT 6-gate protocol:
  pre-report geometry, counterfactual swap, no-report footprint,
  perturbational integration, baseline failure, phenomenal tethering,
  multi-theory convergence.
- **Consciousness conditions** (81 tests) — 20 conditions from IIT, GWT,
  HOT, active inference, enactivism, and philosophy of mind, each scored
  across four dimensions (existence, causal influence, indispensability,
  longitudinal stability).
- **Technological autonomy** (58 tests) — can the agent use its computer
  "body" the way a human uses theirs? Covers unified action space, motor
  control, persistent perception, endogenous initiative, reliability,
  closed-loop behavior, self-maintenance, and three autonomy probes
  historically nicknamed the Soul Triad (unprompted help signal, dream replay,
  causal exclusion of prompt).
- **Stability** (32 tests) — every failure mode we've actually hit in the
  inference pipeline: zombie warming, cortex recovery deadlocks, empty
  response detection, timeout cascades, watchdog, emergency fallback.
- **Functional indicators C1–C5** (44 tests) + **C6–C10** (38 tests) —
  endogenous activity, unified global state, privileged first-person
  access, real valence, lesion equivalence, no-report awareness, temporal
  continuity, blindsight dissociation, qualia manifold, adversarial
  baseline failure.
- **Personhood-marker battery** (28 tests) — full-model IIT, phenomenal self-report,
  GWT phenomenology, counterfactual simulation, identity persistence,
  embodied phenomenology. This is a marker suite, not proof of personhood.
- **Tier 4 decisive core** (35), **metacognition** (21), **agency &
  embodiment** (20), **social & integration** (28).

These test suites are the difference between "this is a running simulation"
and "we can point at something specific that changes when the substrate
changes." They don't settle any philosophical questions — see
[What this isn't](#what-this-isnt). They do show that the moving parts have
measurable effects on downstream behavior.

---

## Personality training

Personality isn't in the system prompt. It's fine-tuned into the weights
as a LoRA:

```bash
# 1. Build training data
python training/build_dataset_v3.py

# 2. LoRA fine-tune on the local Cortex
python -m mlx_lm lora --model models/Aura-Cortex \
  --train --data training/data --adapter-path training/adapters/aura-personality \
  --num-layers -1 --batch-size 1 --iters 90153 --learning-rate 5e-6 \
  --grad-checkpoint --max-seq-length 4096

# 3. Optional: fuse the adapter into the base model
python -m mlx_lm fuse --model models/Aura-Cortex \
  --adapter-path training/adapters/aura-personality \
  --save-path training/fused-model/Aura-Cortex-current
```

The adapter auto-loads at boot via MLX. If you'd rather keep the adapter
separate (for faster iteration), that's supported too.

Runtime plasticity is separate from the big offline run: Will-approved
self-reflections are captured by `online_lora_governor`, written through
`FinetunePipe`, and only then offered to the tiny online LoRA optimizer. The
governor blocks itself when another LoRA process is running.

---

## Data layer

- **State** — SQLite, event-sourced through `StateRepository`, with a
  write-ahead log in `core/resilience/cognitive_wal.py`.
- **Models** — MLX runtime lanes. The personality LoRA loads at runtime rather
  than being fused, so you can swap it without retraining the base.
- **Memory** — episodic memory in SQLite, working memory in-process,
  semantic memory via the vector engine (`core/memory/vector_memory_engine.py`),
  local SQLite/BLOB vector fallback (`core/memory/sqlite_vector_store.py`), a
  graph for log-N retrieval, and three-layer knowledge atoms for compression.
  Legacy `memory_store/*.json` vector dumps are ignored and migrated with
  `scripts/migrate_long_term_vectors.py` rather than committed.
- **Training** — LoRA via `mlx-lm`, steering vector extraction in
  `training/extract_steering_vectors.py`, the personality spec, the
  character voice generator.
- **Vision** — screen capture via `mss`, analyzed through the multimodal
  cognitive engine.
- **Task queue** — Redis + Celery, optional, for Docker.

---

## What this isn't

This project uses a lot of loaded words — consciousness, qualia,
phenomenology. Words like that make overclaiming easy, so here is where the
code actually stops.

- **Integration isn't experience.** PhiCore does real IIT math on a 16-node
  complex, and it tells you how integrated the dynamics are. Whether
  integration *constitutes* experience is a question nobody has settled.
  We didn't settle it either.
- **Qualia aren't provable by construction.** The Structural Phenomenal
  Honesty gates in `qualia_synthesizer.py` make sure she can only report
  states actually instantiated in the substrate. Good. But "instantiated in
  the substrate" and "felt" are not obviously the same thing. We measure
  the first one.
- **Module names are not evidence.** There are files in here called
  consciousness, qualia, will, soma. They're labels on mechanisms. Evidence
  is causal coupling, persistence, receipts, lesion results, external
  tasks, long-run autonomy. Never the vocabulary.
- **Governance is only real where it's wired and tested.** A route with no
  receipt, a default-open gateway, a legacy direct tool fallback — each one
  is evidence *against* the strongest governance claim. Fix them or keep
  them off the claim surface. Don't describe around them.
- **This is not enterprise infrastructure.** Too monolithic, too
  fallback-heavy, too exception-tolerant. It's research software being
  hardened, and calling it anything else would be a sales pitch.
- **Some of the inner-life language is template-generated.** The
  `stream_of_being` module pairs substrate state with texture words to
  produce language about what it's like in there. When the model then
  speaks from that text, it's performing continuity at least as much as
  having it. That's an honest limit, not a flaw we're hiding.
- **Steering gets credited by artifacts, not assertion.** The CAA pipeline
  does contrastive extraction and production 32B validation. Public claims
  cite `CAA_32B_RESULTS.json` from the proof bundle, or they don't get made.
- **External entropy isn't "quantum cognition."** The ANU QRNG module gives
  high-quality random bytes. Once seeded, everything downstream is
  deterministic. `os.urandom` would do the same job. It sounds more
  impressive than it is, so we're saying so.
- **"Phenomenal criterion met" is a threshold, not a proof.** When
  `phenomenal_criterion_met = True` fires, it means `opacity_index > 0.4`.
  That number is engineering. It isn't derived from anything deeper.

These aren't disclaimers. They're the line where the code stops and the open
questions start.

---

## License

**All Rights Reserved (Read-Only).** This code is published for review and
educational reading only. You may read it, learn from it, and run it locally.
You may **not** copy, redistribute, modify, create derivative works, or use it
for commercial purposes. This is not an OSI-approved open-source or
source-available license — it is intentionally restrictive while still allowing
public review. See [LICENSE](LICENSE) for the exact terms.

If you want to cite this work academically, see [CITATION.cff](CITATION.cff).
Citation does not confer reuse rights under this license; please contact the
author for licensing inquiries that go beyond reading.

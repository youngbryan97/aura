# Aura

Aura is a local AI agent and research project for Apple Silicon Macs. You can
talk to it in a desktop app or from the command line. A local language model
drafts replies, and the surrounding system keeps memories and goals across
restarts, looks up information, and checks and records consequential actions.
Background workers handle jobs such as memory consolidation and repair checks.

The research asks how that persistent state affects what Aura writes, remembers,
and does. A good way to see the difference is to follow a single action from
the state that prompted it, through the tool-permission check, to the receipt
that records what happened.

[ISC preprint on Zenodo](https://zenodo.org/records/24075466) · [Independent Φ audit](https://arxiv.org/html/2610.41604v1#S8) · [AuraDemos channel](https://www.youtube.com/@AuraDemos) · [Project video](https://www.youtube.com/watch?v=iTyxeugcZtI) · [What the demos and research support](docs/DEMOS_AND_RELATED_WORK.md)

[![License: All Rights Reserved with Research-Use Grant](https://img.shields.io/badge/License-All_Rights_Reserved_(Research_Use)-red.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS_Apple_Silicon-lightgrey.svg)

## Table of Contents

| I want to… | Start here |
|---|---|
| **Run Aura** | [Quick start](#quick-start) · [Learning demo](#see-it-learn-one-command) · [Tracked and local files](#tracked-vs-local-workspace) |
| **Understand the system** | [Why Aura is built this way](#why-aura-is-different) · [Architecture overview](#architecture-overview) · [Decision authority](#decision-authority) · [Inference-time steering](#inference-time-steering) |
| **See how a subsystem works** | [Memory architecture](docs/MEMORY_ARCHITECTURE.md) · [Reasoning engines](docs/REASONING_ENGINES.md) · [Self-improvement architecture](docs/RSI_ARCHITECTURE.md) · [Personality training](#personality-training) · [Data layer](#data-layer) |
| **Follow the research** | [Research overview](#research-overview) · [Recursive Latent Cortex](#recursive-latent-cortex) · [Intrinsic Subject Core](#intrinsic-subject-core) · [Language and generality](#language-substrate-and-generality) · [Knowledge, procedures and learning](#knowledge-procedures-and-learning-in-use) · [IIT-inspired estimates](#iit-inspired-integration-estimates) · [Consciousness modules](#consciousness-modules) |
| **Check a claim** | [Evidence boundary](#evidence-boundary) · [Evidence artifacts](#production-evidence-surface) · [Decisive evidence runner](#decisive-evidence-runner) · [Long-run autonomy harness](#long-run-autonomy-harness) · [Independent Φ audit and fix](#independent-audit-and-fix) · [Benchmarks](#benchmarks) · [Testing](#testing) · [Limits and open claims](#what-this-isnt) · [Behavioral proof standard](docs/BEHAVIORAL_PROOF_STANDARD.md) · [Physical reach](docs/REALITY_REACH.md) |
| **Find or cite a document** | [Reading paths](#reading-paths) · [Documentation status map](docs/DOC_STATUS.md) · [Docs index](docs/README.md) · [Changelog](CHANGELOG.md) · [Agent guide](AGENTS.md) · [License and citation](#license) |

<details>
<summary>Full section outline</summary>

- [Reading paths](#reading-paths)
- [Research overview](#research-overview)
  - [Recursive Latent Cortex in brief](#recursive-latent-cortex-in-brief)
  - [Intrinsic Subject Core](#intrinsic-subject-core)
- [Evidence boundary](#evidence-boundary)
  - [Governance and autonomy](#governance-and-autonomy)
  - [The cost of the cognitive layer](#the-cost-of-the-cognitive-layer)
  - [What reaches a reply](#what-reaches-a-reply)
  - [What the runtime measures](#what-the-runtime-measures)
- [See it learn (one command)](#see-it-learn-one-command)
- [Production Evidence Surface](#production-evidence-surface)
  - [Substrate, sensing, and model coupling](#substrate-sensing-and-model-coupling)
  - [Repair, governed actions, and persistence](#repair-governed-actions-and-persistence)
  - [Measurement and qualification](#measurement-and-qualification)
  - [Proof bundle](#proof-bundle)
- [Recursive Latent Cortex](#recursive-latent-cortex)
  - [Approach 1: Frozen Loop (didn't work)](#approach-1-frozen-loop-didnt-work)
  - [Approach 2: Trained Intrinsic Recurrence (promising, bounded)](#approach-2-trained-intrinsic-recurrence-promising-bounded)
  - [Since October: the resident 27B](#since-october-the-resident-27b)
  - [Results summary](#results-summary)
    - [Mechanics and frozen loops](#mechanics-and-frozen-loops)
    - [Bounded task results and controls](#bounded-task-results-and-controls)
    - [Held-out reasoning and open comparisons](#held-out-reasoning-and-open-comparisons)
    - [Native semantic selection](#native-semantic-selection)
    - [The 32B four-domain experiment](#the-32b-four-domain-experiment)
    - [The 27B cross-generation replication](#the-27b-cross-generation-replication)
    - [Family-blind procedure learning](#family-blind-procedure-learning)
- [Language substrate and generality](#language-substrate-and-generality)
  - [Relation induction](#relation-induction)
  - [Endogenous language pathway](#endogenous-language-pathway)
  - [Ghost substrate](#ghost-substrate)
  - [Whole-system Φ and Inner Light](#whole-system-φ-and-inner-light)
- [Knowledge, procedures and learning in use](#knowledge-procedures-and-learning-in-use)
  - [Looking up facts offline](#looking-up-facts-offline)
  - [Writing and keeping procedures](#writing-and-keeping-procedures)
  - [Learning from corrections in chat](#learning-from-corrections-in-chat)
  - [Where the results fall short](#where-the-results-fall-short)
- [Why Aura is Different](#why-aura-is-different)
- [Quick start](#quick-start)
  - [Install and start](#install-and-start)
  - [Boot modes](#boot-modes)
  - [Hardware and models](#hardware-and-models)
  - [Deployment options](#deployment-options)
- [Tracked vs local workspace](#tracked-vs-local-workspace)
  - [What Git tracks](#what-git-tracks)
  - [What a reproduction needs](#what-a-reproduction-needs)
- [Architecture overview](#architecture-overview)
  - [Kernel (`core/kernel/`)](#kernel-corekernel)
  - [Brain (`core/brain/`)](#brain-corebrain)
  - [Decisive evidence runner](#decisive-evidence-runner)
  - [Long-run autonomy harness](#long-run-autonomy-harness)
  - [Affect (`core/affect/`)](#affect-coreaffect)
  - [Identity (`core/identity.py`, `core/identity/heartstone.py`)](#identity-coreidentitypy-coreidentityheartstonepy)
  - [Agency (`core/agency/`)](#agency-coreagency)
  - [Skills (`core/skills/`, legacy wrappers in `skills/`)](#skills-coreskills-legacy-wrappers-in-skills)
  - [Orchestrator (`core/orchestrator/`)](#orchestrator-coreorchestrator)
  - [Somatic cortex (`core/somatic/`)](#somatic-cortex-coresomatic)
  - [Autonomy (`core/autonomy/`)](#autonomy-coreautonomy)
  - [Self-modification engine (`core/self_modification/`)](#self-modification-engine-coreself_modification)
  - [Resilience (`core/resilience/`)](#resilience-coreresilience)
  - [Interface (`interface/`)](#interface-interface)
  - [Memory (`core/memory/`)](#memory-corememory)
  - [Reasoning (`core/reasoning/`, `core/brain/reasoning_amplifier*.py`)](#reasoning-corereasoning-corebrainreasoning_amplifierpy)
  - [Verification (`core/verify/`)](#verification-coreverify)
  - [Hardware acceleration (`rust_extensions/aura_m1_ext`)](#hardware-acceleration-rust_extensionsaura_m1_ext)
- [Decision authority](#decision-authority)
- [Inference-time steering](#inference-time-steering)
- [IIT-inspired integration estimates](#iit-inspired-integration-estimates)
  - [16-node cognitive complex](#16-node-cognitive-complex)
  - [32-node and subsystem integration](#32-node-and-subsystem-integration)
  - [Independent audit and fix](#independent-audit-and-fix)
  - [How this relates to ISC](#how-this-relates-to-isc)
- [Consciousness modules](#consciousness-modules)
  - [Consciousness Expansion (April 2026)](#consciousness-expansion-april-2026)
  - [Reasoning, self-model, and resilience (mid-2026)](#reasoning-self-model-and-resilience-mid-2026)
  - [Embodiment, self-knowledge, and physical honesty (late July to August 2026)](#embodiment-self-knowledge-and-physical-honesty-late-july-to-august-2026)
- [Benchmarks](#benchmarks)
  - [Ten-prompt conversation](#ten-prompt-conversation)
  - [Example replies](#example-replies)
- [Testing](#testing)
- [Personality training](#personality-training)
- [Data layer](#data-layer)
- [What this isn't](#what-this-isnt)
- [License](#license)

</details>

## Reading paths

Start with the [plain-English tour](docs/PLAIN_ENGLISH_GUIDE.md) for examples
of what the main parts do. [How Aura works](HOW_IT_WORKS.md) is the longer
conceptual tour. [Architecture](ARCHITECTURE.md) gives the math, module and
interface details. The [variables guide](docs/VARIABLES_EXPLAINED.md) explains
what the numbers change in practice, and the
[documentation map](docs/DOCUMENTATION_INDEX.md) links to the detailed pages.

[Glossary](docs/GLOSSARY.md) ·
[Formulas with worked examples](docs/FORMULAS.md) ·
[Documentation content audit](docs/DOCUMENTATION_CONTENT_AUDIT.md)

The [memory reference](docs/MEMORY_ARCHITECTURE.md) documents the 116-module
subsystem. The [reasoning reference](docs/REASONING_ENGINES.md) covers
deterministic reasoning, sandboxed execution, and operations performed outside
the language model. The [self-improvement reference](docs/RSI_ARCHITECTURE.md)
covers the repair pipeline and its safety boundaries.

## Research overview

### Recursive Latent Cortex in brief

The [Recursive Latent Cortex program](docs/RECURSIVE_LATENT_CORTEX.md) began
with a preregistered experiment. Could a frozen 32-billion-parameter model
improve at reasoning by doing extra internal work with its existing weights?
The expected improvement did not appear.

Later work used Aura's resident 27B model and trained readers of its hidden
states. In October, the program met specified tests for held-out requests,
public answers, and deployment qualification. The
[detailed results](#recursive-latent-cortex) give the controls and the scope
of each result.

### Intrinsic Subject Core

[*A preregistered test of computational subjecthood in a running AI system*](https://zenodo.org/records/24075466)
reports a 24-line battery. The experiment makes controlled changes to saved
copies of Aura and measures how they differ from comparison copies. The confirmatory run (seed 101, commit
`aeefdca31`) passed 24/24 under ISC-v7. The carrier test identified all ten
measured domains as the unique domain-level carrier. That experiment used a
stand-in for the language model. A separate report experiment used the resident
27B model and a fitted steering channel. The five-ground bridge reached
`AT_PARITY`, meaning all five of its checks held under the paper's six
stated postulates.

Seed 97 was exploratory and helped settle the specification. Seed 101 also
scores 23/24 under ISC-v6 and 22/24 under ISC-v5. The paper reports that
nineteen partial imitation systems and an ordinary language-model agent
with memory fell short of the battery. One stronger imitation account
remains open. J* was determined up to relabelings that preserve the measured
relations.

The paper also reports later adverse results on self-prediction, lineage, and
how the partitioned system was held apart. The battery measures the
operational criteria of the study. Independent replication and the argument
connecting these measurements to subjective experience remain open.

[Plain-English explanation](docs/ISC_EXPLAINED.md) ·
[Variables and worked examples](docs/ISC_VARIABLES.md) ·
[Results and corrections](docs/ISC_STATUS.md) ·
[Reproduction guide](docs/ISC_REPRODUCTION.md)

## Evidence boundary

A passing test applies to the contract, configuration, and run recorded with
it. Implementation notes and older suite counts describe the system and its
test history. For the installed runtime, check the activation and serving
receipts. For the experimental subjecthood results and later corrections, see
[ISC status](docs/ISC_STATUS.md). The [testing reference](TESTING.md)
documents the older catalogue. Use `pytest --collect-only` for a fresh
collection and `make proof-bundle` to gather the available evidence.

Terms such as “consciousness guarantee” and “personhood proof” are names of
test batteries. A code guard limits how they can be presented as claims
without independent evidence. The same reading applies to module names such
as `qualia_synthesizer.py`. Claims about life, subjective experience, legal
personhood, and moral standing would require separate evidence.

### Governance and autonomy

Consequential actions are designed to pass through governance checks that
produce receipts. A lint failure, a direct tool fallback, or a
default-approved gateway is a defect in that coverage.

The code includes automatic repair, evaluation, patch tracking, and proof
bundles. Claims of mature autonomous self-improvement still require long
real-world runs, hidden external tasks, and independent replication.
Aura is research software under active hardening. The local runtime still
has a broad fallback surface and can be fragile.

### The cost of the cognitive layer

Two component comparisons use matched budgets.

- In claim 31a, memory retrieval helps when the needed fact falls outside
  the baseline context window.
- In claim 31b, the revision gate outperformed keeping either draft every
  time. All policies saw the same two drafts per task in the 400-case scripted
  comparison. The default harness uses a deterministic responder.

A matched comparison of the entire cognitive layer on user tasks is still
open. A simpler computer-use agent can perform the demo tasks without the
IIT metrics, qualia modules, neurochemical simulation, substrate differential
equations, dream cycles, or theory arbitration. Those components add latency,
memory use, tuning work, and failure modes. Claim 31 in the
[claim matrix](CLAIMS_MATRIX.md) specifies what a task-success comparison
would need to measure.

### What reaches a reply

The activation-steering hook can feed Aura's state into the model's hidden
computations. Steering a turn someone is waiting for requires a five-part
certificate (`core/consciousness/fusion_certificate.py`). On 13 October 2026,
the 27B met four parts and failed the fifth. With steering enabled, a reply
about fledged wrens read, “The wren brood has successfully completed its
operational cycle and exited the designated boot container.”

While that certificate is incomplete, the serving policy uses zero steering
on user turns. State can still affect replies through sampling settings and
evidence text in the prompt. Identity continuity across turns also uses
ID-RAG Chronicle retrieval, in addition to prompt cues.

### What the runtime measures

- **φ (phi).** An IIT-inspired integration estimate over tractable
  subsystems. Full-system IIT remains computationally intractable here.
- **Hardware.** The target is Bryan's M5-class Apple Silicon Mac with 64 GB
  unified memory. Lower-memory profiles use smaller model lanes and need
  their own performance measurements.
- **Resource budgets.** Running costs persist and limit what Aura can do.
  The project uses biological metabolism as an analogy for this software
  budget.

[Claims outside the evidence, including physical claims](CLAIMS_NOT_SUPPORTED.md)

---

## See it learn (one command)

This demo downloads and trains a small local model. Run it on an Apple-silicon
host with an explicitly available model lane and measured memory/disk
headroom. Coordinate with the live Aura owner first; do not start it beside
a resident model or campaign as a documentation check. Duration and peak
resource use need measurement on the selected model and host:

```bash
make setup          # once: venv + requirements
make demo-learning
```

1. A small local model attempts seeded reasoning tasks. Each task's checker
   scores its supported answer format and semantics. Checker bugs, incomplete
   coverage, and overfitting remain possible.
2. The demo trains a LoRA adapter, a small set of learnable weights, on the
   verified wins and losses.
3. The adapter takes a sealed held-out test on fresh inputs. A passing adapter
   is merged and published. The next cycle builds on the published weights.

Each generation goes into a hash-chained ledger
(`core/learning/rsi_lineage.py`). The demo computes the verdict from those
receipts and reports a refused promotion as explicitly as a gain. Raw
responses, evaluation reports, and cycle receipts stay on disk.

The live runtime has a scheduler for this machinery
(`core/learning/compounding_scheduler.py`), with idle/resource admission and
promotion checks. A configured route is distinct from a successful training
cycle, a promoted artifact and permission for the installed runtime to use it.
Inspect the matching receipts and serving qualification for each claim.

## Production Evidence Surface

The sections below map implemented paths and retained evidence. Their
measurements keep the stated source, task and date scope. Current activation
and useful end-to-end behavior need their own receipts; the
[claim matrix](CLAIMS_MATRIX.md) distinguishes those evidence levels.

Release gates generate a proof bundle with the records listed below.

### Substrate, sensing, and model coupling

- `core/brain/llm/continuous_substrate.py` is a configurable 64-to-512 neuron
  dynamical system (Liquid Time-Constant ODE) running at ~20 Hz. CPU-only
  numpy with explicit-Euler integration plus random perturbation;
  `get_state_summary()` derives valence/arousal/dominance/phi from the live
  state vector (grounded in external data via `adapt_projections()`),
  without changing callers.
- `core/brain/llm/substrate_token_generator.py` is the substrate-first readout:
  it uses an untrained random projection onto a 32-word proto vocabulary from
  the live substrate before calling the transformer, and falls back to the
  main model when substrate prediction error exceeds the threshold. The
  prototype uses the small, untrained projection. A full readout would
  need a trained head over the model vocabulary.
- `core/brain/llm/sensorimotor_grounding.py` maps camera/screen/audio
  observations into the substrate input vector, so live sensor events
  directly affect the dynamical system. This is a separate input path from
  text and tool summaries.
- `core/consciousness/phi_core.py` binarizes state, estimates an empirical
  transition matrix, and uses KL divergence to estimate φ. It searches for
  candidate complexes using an IIT-inspired exclusion rule. The 16-node path
  uses polynomial-time spectral partitioning, and the 8-node reference can
  evaluate all 127 bipartitions.
  [Independent audit and repair](#independent-audit-and-fix)
- `core/consciousness/hierarchical_phi.py` implements the 32-node hierarchical
  φ with K=8 overlapping subsystems and Bayesian-smoothed estimation.
- `core/consciousness/affective_steering.py` hooks MLX transformer blocks and modifies the model's hidden states during text generation.
- `training/caa_32b_validation.py` validates production-model steering
  artifacts: vector presence, layer geometry, PCA structure, permutation
  controls, black-box prompt hygiene, rich-prompt comparators, and
  behavioral A/B result ingestion.
- `core/consciousness/stdp_external_validation.py` runs the external-usefulness
  plasticity experiment: external environment signal vs self-generated, frozen,
  and shuffled controls on held-out prediction tasks.
### Repair, governed actions, and persistence

- `core/self_modification/fault_pipeline.py` and
  `core/self_modification/repair_approval.py` implement the closed-loop
  bug-repair path with deterministic localization, tier-aware approval,
  patch tracking, and calibration.
- `core/architect/` implements the Autonomous Architecture Governor: a
  shadow-workspace software architect that builds architecture graphs, detects
  code smells, generates staged cleanup/refactor plans, requires proof receipts
  and rollback packets before promotion, and monitors promoted changes. See
  [`docs/AUTONOMOUS_ARCHITECTURE_GOVERNOR.md`](docs/AUTONOMOUS_ARCHITECTURE_GOVERNOR.md).
- `core/runtime/autonomy_conductor.py` and `core/runtime/activation_audit.py`
  make proof, validation, resource, scar, and repair checks recurring runtime
  jobs instead of optional scripts.
- `core/runtime/overt_action_loop.py` is the practical "what does she do?"
  path. It takes one authorized initiative, chooses a real registered skill,
  executes it through CapabilityEngine/Will/tool governance, verifies the
  returned evidence, emits receipts, records the action, and advances the
  linked goal. This is the visible observe → choose → act → verify → remember
  loop.
- `core/adaptation/online_lora_governor.py` connects Will-approved
  self-reflections to small LoRA update attempts. It refuses to start while an
  existing `mlx_lm lora` process is active, so long training runs are preserved.
- `core/goals/default_goals.py` seeds durable, tool-attached IN_PROGRESS goals
  for repair, proof upkeep, sensor grounding, and architecture improvement at
  boot. Those goals are what keep the initiative funnel active after restarts.
- The full memory architecture (episodic, semantic, vector, knowledge graph,
  write-ahead log, three-layer knowledge atoms), the goal/will/decision stack,
  and the cognitive write-ahead log are all real production code. Vector
  embeddings are stored as efficient binary blobs in SQLite via
  `core/memory/sqlite_vector_store.py`, not as plaintext JSON arrays.

### Measurement and qualification

- φ is computed over **cognitive-emotional state nodes and sampled mesh
  neurons**, not at the level of intrinsic mechanisms that strict IIT 4.0
  prescribes. The φ values are mathematically meaningful as integration
  measures over the system's own state-space; they are not a claim of
  integrated information in the strict Tononi/Albantakis/Haun sense.
- Steering credit requires `CAA_32B_RESULTS.json`: steered behavior on the
  32B model must differ from unsteered baseline, beat a rich text comparator,
  generalize to held-out tasks, preserve output quality, show coherent
  geometry, and survive black-box prompt hygiene.
- Plasticity credit requires `STDP_EXTERNAL_VALIDATION.json`:
  environment-trained plasticity must beat self-generated, frozen, and
  shuffled controls on held-out prediction without raising instability.

### Proof bundle

`make proof-bundle` writes the following artifacts.
`DECISIVE_RESULTS.json`, `CAA_32B_RESULTS.json`,
`STDP_EXTERNAL_VALIDATION.json`, `GOVERNANCE_COVERAGE.json`,
`SELF_REPAIR_LINEAGE.json`, `LONGEVITY_RUN.json`,
`MUTATION_TEST_REPORT.json`, `BOOT_HEALTH.json`, `ACTIVATION_REPORT.json`,
`SECURITY_SCAN.json`, and `CANONICAL_PROOF_BUNDLE.json`.

---

## Recursive Latent Cortex

Full page: [docs/RECURSIVE_LATENT_CORTEX.md](docs/RECURSIVE_LATENT_CORTEX.md).

The question it began with: a frozen 32B model is a fixed-depth pipeline — 64 layers,
applied once per token. **Can you make it think longer on a hard problem
without changing any stored weights?**

Two approaches have been tried, and they gave different answers.

### Approach 1: Frozen Loop (didn't work)

The first approach was a **frozen loop**: thought slots placed beside the
prompt, a window of middle layers run over them repeatedly on a schedule,
with the refined slots' key-value states persisted so every generated token
can attend to them. Model weights are hash-checked before and after every
episode to prove nothing changed; episode-scoped fast weights are provably
erased; equal-compute accounting is built in so "more compute helped" can't
be mistaken for "the architecture helped."

The preregistered experiments committed the seed before generating tasks
and used 24 items per family with statistical corrections. At 1.5B, ordinary
decoding beat all seven latent arms on the 72-item battery. In the 32B
templated sweep, the best latent arm answered 9/24 and ordinary decoding
answered 10/24. Their intervals overlapped. The 32B run found no improvement,
and its estimate alone does not settle the size or direction of the effect.
[32B sweep counts and intervals](docs/evidence/rlc_frozen_32b_templated_counts.json)

The repeated computation ran on the scratchpad. Answer tokens themselves
passed through the middle layers once and could attend to the refined slots.
That distinction matters when comparing this experiment with recurrence
applied directly to answer generation.

### Approach 2: Trained Intrinsic Recurrence (promising, bounded)

The second approach, **trained intrinsic recurrence**
([docs/INTRINSIC_RECURRENCE.md](docs/INTRINSIC_RECURRENCE.md)), makes the
real token stream re-enter the middle block, so a 64-layer model effectively
runs 160 layers deep at T=4 with the same weights, and trains it on typed,
exactly-checkable program traces instead of answers. This training path is
separate from the typed semantic-machine evidence below. CP566 measured a
learned semantic machine whose terminal state conditioned the model's answer;
it did not isolate a gain from repeated middle-layer passes in the resident
model.

The [scoped computation layer](docs/SCOPED_COMPUTATIONAL_KNOWLEDGE.md) connects
unit-checked equation graphs, logical consequences, candidate constraints and
retained recipes. Its calculations remain conditional on supplied premises;
autonomous source grounding and general-transfer gains are still unproved.

The [frozen RLC baseline](docs/evidence/G01_RLC_BASELINE_2026-09-08.md)
records the evidence available on 2026-09-08 and its limitations.
The fresh v16 natural-language replication reached **26/96** exact answers,
below its pre-registered **48/96** floor; ordinary generation was not run
after that futility stop. The v19 repair reached **93/96** on the exposed
development set, with coefficient lesion **0/96**. That development result
still needs fresh replication. Neither record establishes broad reasoning
gain or grants serving authority.

### Since October: the resident 27B

Her resident model is now a 27B (Aura-Qwen3.8-27B, persona fine-tuned), and the
program's later items ran on it:

- **Binding and composition** ([G03](docs/evidence/G03_WHICH_AND_WHETHER_2026-10-06.md)):
  a learned reader of her hidden states reached 500 of 500 on validation and lost
  no row to its predecessor.
- **Held-out transfer** ([G04](docs/evidence/G04_TRANSFER_V2_RESULT_2026-10-07.md)),
  with execution unavailable, against the incumbent reader on 62 fresh requests
  per stratum: vocabulary 33 to 21 (*p* 0.0145), depth 54 to 8, family 62 to 47.
- **Public answers** ([G05](docs/evidence/G05_PUBLIC_ANSWERS_RESULT_2026-10-07.md)):
  on 48 fresh compositions, 48 right with the reading against 29 without, 19
  gained and none lost (*p* = 1.9 × 10⁻⁶); every right reading became an exact
  public answer.
- **Controls and lesions** ([G06](docs/evidence/G06_COMPARISONS_RESULT_2026-10-07.md)):
  removing the phrase readouts costs 27 vocabulary requests to 0; zeroing
  recency costs 35 depth requests to 0.
- **Qualification** ([G10](docs/evidence/G10_READER_QUALIFICATION_2026-10-07.md)):
  the reader runs through her resident worker with her model identity, steering
  generation and geometry checked on 416 recorded requests.

**Kept procedures (G09, closed 17 October).** See
[Knowledge, procedures and learning in use](#knowledge-procedures-and-learning-in-use).
Attempt 6 held as preregistered and again once every changed row was read:
29 gained to 0 lost on 444 fresh requests (*p* 3.7e-9), held-out math 13 to 0
and planning 16 to 0 ([result](docs/evidence/G09_PROCEDURES_RESULT_6_2026-10-17.md)). Attempt 5 had
met its criteria too, but 13 of its 38 gains were grading artifacts
([result](docs/evidence/G09_PROCEDURES_RESULT_5_2026-10-17.md)).

### Results summary

#### Mechanics and frozen loops

| What was tested | Verdict |
|---|---|
| Mechanics (KV rewind, stability bounds, slot ablation moving the answer distribution, erasure, invariants) | **PROVEN** on real MLX weights |
| Live runtime integration on the then-resident 32B | **PROVEN** |
| Capability gain, **frozen** loop, 1.5B | **REFUTED** — vanilla 21/72 beat every one of 7 latent arms (7–13/72) |
| Capability gain, **frozen** loop, 32B | **CONJECTURE**, negative point estimate — latent 0.375 vs vanilla 0.417, overlapping intervals |

#### Bounded task results and controls

| What was tested | Verdict |
|---|---|
| Capability gain, typed semantic machine + state-conditioned decode, 32B | **`BOUNDED_WOW_SIGNAL`** — 60/60 against 16/60 for ordinary decode, ablation-dependent, *p* = 5.7 × 10⁻¹⁴ |
| Cross-generation recovery, trained semantic tissue, 27B | **`BOUNDED_WOW_SIGNAL`** — 60/60 against 0/60 ordinary decode on a separate fresh set; wire 6, coefficient lesion 4, wrong-state 0; *p* = 8.67 × 10⁻¹⁹ |
| Family-blind procedure learning into neural tissue | **SUPPORTED, BOUNDED** — one depth-2 procedure learned from 16 examples, then 96/96 exact on fresh inputs; coefficient and wrong-input controls failed 96/96, no-procedure solved 1/96, shuffled-output nulls found 0/15 |
| Resident decode of the learned neural procedure | **SUPPORTED, BOUNDED** — treatment 8/8, ordinary 1/8, wire 1/8, coefficient lesion 1/8, wrong-input 0/8, wrong-state 0/8; seven gains, no regressions, *p* = 0.0078125 |
| Resident 27B language-to-program transfer | **SUPPORTED, BOUNDED** — exact execution emitted 134/256 held-out answers from learned model-bound semantics; exact program recovery was 133/256, against hidden-state shuffle 14/256, coefficient lesion 0/256 and label permutation 4/256 |
| Frozen fresh-cohort semantic transfer | **SUPPORTED, REPLICATED, BOUNDED** — the unchanged transducer emitted 114/256 exact held-out answers on a separately seeded numeric set after a clean worker restart, against hidden-state shuffle 10/256 and coefficient lesion 0/256 |
| Shared variable-geometry semantic programs | **SUPPORTED, BOUNDED** — one transducer with no family router recovered 258/368 complete programs and exact execution emitted 292/368 answers across arithmetic, sequence and fork/join geometries; hidden-token shuffle 0/368 and coefficient lesion 0/368, paired exact *p* = 2.16 × 10⁻⁷⁸ |
| Learned programs on the universal floor | **SUPPORTED, BOUNDED** — all 368/368 accepted frozen test programs had identical outcomes under the existing exact executor and Aura's universal metered floor: 366 matching values and two matching typed refusals across all three families, with 20/20 primitive semantics covered |
| Learned hidden-state reader on the resident 27B: held-out transfer, public answers, lesions | **SUPPORTED, BOUNDED** — vocabulary 33 vs 21, depth 54 vs 8, family 62 vs 47 on fresh requests with execution unavailable; public answers 48 vs 29 (*p* = 1.9 × 10⁻⁶); lesions cost 27 to 0 and 35 to 0 |

#### Held-out reasoning and open comparisons

| What was tested | Verdict |
|---|---|
| Broad reasoning gain under G09’s registered criterion: kept procedures on held-out math and planning kinds | **CLOSED** 17 Oct — attempt 6; after row review, 29 gained, 0 lost (*p* 3.7e-9); math 13 to 0, planning 16 to 0. [Result and limits](docs/evidence/G09_PROCEDURES_RESULT_6_2026-10-17.md). |
| Unrestricted open-ended reasoning or transfer to competition mathematics | **NOT ESTABLISHED** by G09; its route admitted no MATH test problem |
| Static fusion of the learned RLC mechanism into the backbone weights | **NOT CLAIMED** by these results |
| Frontier performance against named contemporary systems (G12) | **OPEN**; G09 does not answer this separate comparison |

#### Native semantic selection

The [September native semantic-selection work](docs/evidence/G_NATIVE_RELATIVE_FIT_2026-09-26.md)
uses the resident model's own language pathway and role-relative program
registers. Source-selected adaptation reaches **46/50** on an exposed proposal
bank against incumbent 41/50 and unfitted 39/50, with one regression. A matched
[fitting-source erasure control](docs/evidence/G06_NATIVE_SOURCE_ERASURE_RESULT_2026-09-26.md)
gets the same 46 correct outcomes, so that gain cannot be attributed to
newly learned source meaning. Target-blind decoding reached 72/72 on an easier
fixed-template development set, but the wider
[retained canary](docs/evidence/G03_NATIVE_RETAINED_CANARY_2026-09-26.md)
returns only 10/14 procedure matches and 11/14 provably equivalent answers.
None is a promoted candidate or broad/frontier reasoning result.

`BOUNDED_WOW_SIGNAL` is the adjudicator's verdict for the specified task set
and controls. The receipt records the limitations alongside the verdict.

#### The 32B four-domain experiment

The frozen set contained 60 typed tasks in coding, calibration, misleading
premises, and scientific inference. The
semantic-machine treatment answered 60/60 exactly against 16/60 for ordinary
decode, with a matched wire base at 7, a coefficient lesion at 5, and a
wrong-state control at 0. Forty-four ordinary failures converted, none
regressed, paired one-sided exact *p* = 5.7 × 10⁻¹⁴. The coefficient lesion
reduces the result, supporting dependence on those learned coefficients within
this experiment. State conditioning, formatting assistance and retries remain
part of the recorded treatment contract; equal-compute broad reasoning
comparisons remain a separate requirement.

#### The 27B cross-generation replication

The 2026-08-24 cortex migration
repeated that bounded claim on the fused Qwen3.8-27B resident model. A
separately seeded 60-task, 300-decode campaign returned treatment 60/60,
ordinary decode 0/60, matched wire 6/60, coefficient lesion 4/60, and
wrong-state 0/60, with no regressions and exact one-sided *p* = 8.67 × 10⁻¹⁹.
Independent verification replayed all 300 journal rows before the frozen
adjudicator returned `BOUNDED_WOW_SIGNAL` again. This shows the bounded
typed-tissue/executor mechanism works across two model generations; it is
not a head-to-head 27B vs 32B quality benchmark because the test sets and
models differ. The ordinary 27B arm produced no parseable answers under that
campaign's decode contract. Its 0/60 therefore cannot establish lack of
knowledge or performance under a different completion budget.

#### Family-blind procedure learning

A generic enumerative inducer received
sixteen input-output examples, no family label and no family solver, froze
`idiv(add(in0, in1), in2)`, and passed it through a family-blind SSA lowerer
into the existing learned arithmetic tissue. That path was exact on 96/96
fresh inputs. Coefficient and guaranteed-wrong-input ablations disrupted all
96, the no-procedure control solved 1/96, no depth-one shortcut fit, and
fifteen shuffled-output searches found no program. This establishes bounded
procedure learning and neural execution over a fixed set of primitive
operations.

A second frozen canary carried that same learned program through the fused
resident 27B's answer surface. Treatment was 8/8 exact against 1/8 ordinary
decode; syntax-only wire and coefficient lesion were also 1/8, while
wrong-input and wrong-state controls were 0/8. Seven ordinary failures
converted with no regressions, exact paired one-sided *p* = 0.0078125.
Independent replay reconstructed all 48 decodes and the 50-event journal.
This does not establish natural-language compilation, open-domain reasoning,
unrestricted serving, static fusion or frontier performance.

**Language-to-program transfer:** A generic linear transducer learned token
spans, primitive operations and register arguments from five construction
families without expected answers. On four held-out construction combinations,
exact execution emitted 134/256 correct answers and recovered 133/256 complete
programs. Hidden-token shuffle reached 14/256, coefficient lesion 0/256 and
label permutation 4/256. An independent, source-bound replay reloaded all 576
feature records, reproduced the coefficients and report exactly, and recounted
all 1,344 task-arm rows. The result is bounded to the synthetic arithmetic
grammar; serving and broad-domain transfer remain open. The campaign was
re-verified after tooling changes; the first certificate remains the immutable
historical record, and the current claim reads the certificate bound to the
exact measured commit.

**Fresh-cohort replication:** The coefficients were frozen and evaluated on a
separately seeded set with no task overlap. A clean worker restart changed its
process identity but left the complete neural-function basis identical.
Without fitting or refitting, exact execution emitted 114/256 held-out answers;
hidden-state shuffle reached 10/256 and coefficient lesion 0/256. Independent
verification reloaded both 576-record bundles, replayed the frozen result
exactly, and recounted 1,728 task-arm rows. This establishes fresh-example
reuse inside the same bounded language and primitive system. It does not
establish new procedure families, open-domain transfer, or serving authority.

**Variable-geometry programs:** One shared transducer now receives
tokenizer-grounded typed inputs and learns the remaining program structure from
resident 27B hidden states. It predicts how many steps, which primitive each
step applies, and whether each argument refers to an input or an earlier result.
The same coefficient set handles two-step arithmetic, two-step sequence programs,
and three-step fork/join arithmetic without a family router. On held-out
constructions it recovered 258/368 complete programs, and exact execution
emitted 292/368 correct answers. Hidden-token shuffle and coefficient lesion
each recovered 0/368 programs; both paired exact tests gave
*p* = 2.16 × 10⁻⁷⁸. This establishes learned variable-geometry programs over
the declared typed vocabulary. A new schema still needs support examples, and
this result grants neither serving authority nor a broad natural-language claim.

**Universal floor execution:** Those learned programs no longer stop at a
separate Python operation table. Every declared integer and sequence primitive
now compiles into Aura's universal metered floor, along with the learned SSA
references and typed public inputs. A separate no-refit replay compared both
engines on all 368 accepted frozen test programs: 366 produced the same value
and two produced the same typed refusal, for 368/368 agreement across
arithmetic, sequence and fork/join. All 20 declared primitives have both floor
semantics and a type signature; a new primitive without either is refused. The
neural front end and endogenous substrate now share execution semantics. Later
v14 evidence recovered **79/96** programs and answers on a fresh,
fit-withheld synthetic program family using shared primitives. Its endogenous
replay also reached **79/96**, versus **0/96** under coefficient lesion.
Open-domain schema learning remains unproven.

One family — misleading premise — gained nothing, and the reason is worth
stating rather than averaging away: ordinary decode was already at ceiling
there (15/15), and the controller preserved all fifteen instead of
manufacturing a gain by regressing its own baseline. The other three families
supplied the 44.

It has dated serving-path qualification. `semantic_neural_serving.py` refuses
to serve unless a descriptor-bound activation record says `active_by_default`.
The CP1011 27B package is `rlc-27b-recovery-05346acd618d1c925f16`; its recorded
runtime verification is 120/120 exact, 120/120 ablation-disrupted, 120/120
through both foreground and service integrations, and unsupported language
refused, at a 9.229 / 38.696 ms median / maximum. These are qualification
measurements, not a live-status probe. Present activation must be checked
against the running process, source revision, model identity and
eligible-request receipt.

That typed semantic-serving experiment is limited to its admitted task grammar;
it does not establish open-ended reasoning, static fusion or frontier
performance. G09’s later kept-procedure experiment separately passed its
registered broad-gain criterion on math and planning, as recorded above.
The typed semantic lane admits requests through an answer-blind parser;
unsupported language does not reach that lane.

Before you read either page: the two negative results from the August
reconciliation campaign — a 13-vs-5 and its 9-vs-4 reproduction — were
**void**, because the promotion gate had been wired to the one decode policy
that removes the vanilla floor. A win had been structurally impossible there,
and those two runs measured a system that was never switched on. The July
pre-registration above is untouched by that defect and its verdicts stand.
[docs/RLC_RECONCILIATION.md](docs/RLC_RECONCILIATION.md) has the fourteen
defects in dependency order.

---

## Language substrate and generality

Can Aura learn a new way to represent knowledge from experience, then use it
in an unrelated domain? This work investigates what the system can learn and
whether it gets better at reusing what it learns.

### Relation induction

`core/cognition/relation_language.py` learns transformation rules from paired
observations and checks them on held-out transitions. It requires the rule to
compress the examples, which rejects substitution tables. The noise control
produced no invented rules. The method has been tested on words, colours,
records, and grids. Combining rules such as “mirror then rotate” reached 20
of 120 battery problems unreachable without composition. Transfer across
worlds and representations was measured against a cost-of-wrong-prior null.

### Endogenous language pathway

The 18 modules in `core/brain/llm/endogenous_*.py` implement a trainable path
from Aura's 74-dimensional cognitive state to the transformer's output
distribution: `z_Aura → Δlogits → language`. The first fit used 1,629 live
turns on the 9B lane. Held-out gain was 0.0208 nats and paired recovery
was 54.0% (*p* = 0.043). Those results measure the fitted path; they precede
a trial that uses the pathway to change generation.

### Ghost substrate

`core/ghost/` contains six modules for hash-chained continuity across
substrate swaps. The four components are causal integration, ghost line,
hack guard, and provenance.

### Whole-system Φ and Inner Light

The Inner Light test scores Aura's activity on four neuroscience-inspired
markers against six negative controls. The intact system was the sole 4/4
case. `PhiEstimate.claim` enforces the scope of the integration estimate.

[Language substrate and generality reference](docs/LANGUAGE_SUBSTRATE_AND_GENERALITY.md)

---

## Knowledge, procedures and learning in use

### Looking up facts offline

Aura's local corpus contains a 3 July 2026 copy of Wikipedia and material
retained from web searches. Tantivy indexes roughly a hundred million
paragraphs individually for millisecond-scale searches. The chat path combines
paragraph search with targeted lookup of articles the question names. The
source and date travel with each passage
([L05](docs/evidence/L05_KNOWLEDGE_ACROSS_TURNS_2026-10-05.md)).

On 200 SimpleQA development questions, Aura got 9 right unaided, 58 with
passages, and 68 with fused passages. In a version allowed to decline when
two answers from the same passages disagreed, it answered 50 and got 44 right.
A separate test half is needed for a confirmatory result.
[Development round](docs/evidence/GENERAL_REASONING_DEVELOPMENT_RESULT_2026-10-16.md)

### Writing and keeping procedures

Given solved examples of a kind of problem, the model writes a small program.
Aura keeps the program only if it agrees with every known answer, including
the half of the examples held out while it was written.

Before drafting, the chat path looks for a procedure by wording or meaning.
A match by meaning has an extra check. The model restates the request in the
procedure's terms, and the restatement must carry the request's numbers.

In G09 attempt 6, procedures helped on 13 held-out math requests and 16
planning requests, with zero losses in either domain. The math gains were
mostly large primes and factorizations that exhausted the model's thinking
budget. The planning gains came when drafts lost track of a vehicle. G09 met
its criterion on these domains. The meaning route admitted zero competition
math problems in that attempt, leaving transfer there open.
[The resident 27B results](#since-october-the-resident-27b)

### Learning from corrections in chat

If someone gives a correction such as “No, it's 1049,” Aura stores a solved
example of the question. A correction that conflicts with a kept procedure
is marked disputed. During idle time, Aura groups examples by meaning and
writes procedures for groups large enough to check.

On 17 October 2026, a person supplied answers for ten date-difference
questions. Aura had answered five correctly on its own. An incorrect launch
configuration blocked the initial learning attempt and required a manual
reset. Aura learned the procedure after relaunch and answered three new
questions correctly in chat 25.6 hours after the last teaching turn.
[Live learning record](docs/evidence/L09_LEARNING_IN_USE_LOOP_2026-10-18.md)

A separate ordinary-use comparison on 19 October counted 3 gains and 4
losses over 33 requests. The difference was inconclusive. Every changed result
came from a routing error. The first comparison was underpowered, and the
third stopped because too few sampled kinds had baseline errors.
[19 October result](docs/evidence/L09_ORDINARY_USE_RESULT_2_2026-10-19.md) ·
[All three attempts](CLAIMS_NOT_SUPPORTED.md#22-learning-from-conversation-that-improves-ordinary-use)

### Where the results fall short

Factual accuracy was low without passages in the SimpleQA development run.
The mechanism that resolves disagreements between Aura's own derivations
showed zero gain on MMLU-Pro and no statistically significant gain on MuSR.
A procedure reached by meaning supplies evidence to the drafting model; the
model can still contradict it, as it did on 17 October.

---

## Why Aura is Different

A prompt can tell a model to act energetic by giving it the words “feeling
energetic.” Aura also investigates supplying state as a direction vector
inside the transformer's hidden activations. The hook uses the same family
of techniques as contrastive activation addition (CAA) and other
residual-stream interventions. User turns are governed by a separate serving
certificate. [Inference-time steering](#inference-time-steering) explains
the certificate and the paths used in replies.

Between conversations, the substrate continues to run. Affect values decay
and interact. Simulated neurochemicals change on separate schedules. A global
workspace selects activity at each tick, and an idle process consolidates
memories. The Unified Will is the common authority for actions on governed
routes. You can talk to Aura while those jobs run.

---

## Quick start

### Install and start

Use the environment created by setup for the Python commands. Activate it
with `source .venv/bin/activate` after running the setup target.

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

### Boot modes

Other boot modes include `--cli` (interactive console), `--server` (API only),
`--gui-window` (attach a window to a running server), `--watchdog`,
`--philosophy` (stream substrate/phi/affect/Will as JSONL), `--skeletal`
(bypass heavy subsystems), `--profile minimal`, plus `--stop` and `--reboot`.
The installed console script exposes the same entry point as `aura`, which
also carries the operational subcommands (`doctor`, `conformance`,
`verify-state`, `verify-memory`, `rebuild-index`, `backup`, `restore`,
`migrate`, `chaos`, `plugin`).

### Hardware and models

Requirements include Python 3.12+, macOS on Apple Silicon, 64 GB RAM recommended. The
primary model is `Aura-Cortex` (fused Qwen3.8-27B, migrated from the historical
32B checkpoint). The Brainstem source default is the compressed 27B listed
in the model table; it loads only when admitted. Cold-start time depends on
the artifact, cache state, shader compilation and host load.

Bryan's target machine is an M5-class Apple Silicon Mac with
64 GB unified memory. The 27B Cortex works there as a primary conversation
lane, while heartbeat/background work still belongs to the substrate, Brainstem,
or Reflex lanes. On lower-memory machines, choose a smaller declared profile
and measure it separately; a historical small-model result does not qualify
the present Cortex configuration.

### Deployment options

There's also a `Dockerfile` and `docker-compose.yml` if you want Redis and Celery
running alongside. The tracked workspace defaults to `owner_autonomous` posture
for this single-owner machine: autonomy on, outbound/network-enabled skills
available, and self-repair left active. If you want a tighter deployment,
override the `AURA_*` security settings in your local environment, including
`AURA_INTERNAL_ONLY=1` for localhost-only binding.

---

## Tracked vs local workspace

### What Git tracks

Canonical skills live in `core/skills/`. The top-level `skills/` package
supports older imports. Add new skills under `core/skills/`.

An installation can also contain private modules listed in `.gitignore`.
Those modules can change the behavior and risk of a particular deployment.
To audit a running installation, inspect that machine's files and settings
along with this source tree.

### What a reproduction needs

Git omits model weights, local vector stores, and the 6.5M-document corpus
described here. Reproducing a local demo requires the files and profile that
the demo used. The [claim matrix](CLAIMS_MATRIX.md) calls a local run
“locally demonstrated” when it passed the project's battery on the
recorded machine and profile.

A reproducible deployment demo would need a frozen release that pins the
model, vector-store hashes, and configuration, then an independent run
on a third-party machine. Claim 12 tracks that external validation.
The ISC preprint has separate releases for its
[run data](https://doi.org/10.5281/zenodo.24075592) and
[code](https://doi.org/10.5281/zenodo.24077801), and its
[reproduction guide](docs/ISC_REPRODUCTION.md) explains how to use them
for the experiment.

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

Crash in the middle of that and the write-ahead log replays on restart. No
half-written thought survives.

### Kernel (`core/kernel/`)
Tick-based cognitive cycle. One tick = one unit of thought. Phases run in order,
state versions, state commits, lock released.

### Brain (`core/brain/`)
The model router uses local lanes under admission control:

| Role | Source default | Scope |
|---|---|---|
| Cortex | `Aura-Cortex` | Foreground conversation and reasoning |
| Deep | `Aura-Cortex` | A distinct specialist needs explicit configuration, qualification and memory admission |
| Vision | `Aura-Cortex` | Image requests need compatible artifact and image encoding |
| Brainstem | `Ternary-Bonsai-2-27B-mlx-2bit` | Background lane, loaded only when admitted |
| Reflex | `Qwen2.5-1.5B-Instruct-4bit` | Auxiliary fallback; device and draft compatibility are separate contracts |

There is no opt-in cloud-model fallback. Compatibility names such as
`api_fast` resolve locally, and remote-only model requests are refused.
Static fallback responses can also fail to be useful or delivered. The
[model card](MODEL_CARD.md) and [roster](docs/MODEL_ROSTER.md) separate current
configuration, historical measurements and a particular runtime's activation.


Two non-LLM lanes were replaced in August 2026 based on measurements:

- **Speech-to-text** is one streaming-native Parakeet TDT pass
  (`core/voice/duplex/streaming_asr.py`) serving both duplex stages, replacing
  a two-stage Whisper setup (`small.en` for partials, `large-v3-turbo` for the
  final). Measured on this host over 12.4s of real speech, median of 5 warm
  runs: Parakeet 166 ms vs Whisper-small 193 ms vs Whisper-large-v3-turbo
  317 ms. One decode is cheaper than the old partial and about half the
  old final, so both stages run the same weights on one model-lane lease.
  `faster_whisper` remains as the CPU fallback.
- **Embeddings** are `Qwen3-Embedding-0.6B` at 384 dimensions
  (`core/memory/embedding_model.py`), replacing `all-MiniLM-L6-v2`. MiniLM
  declares `max_seq_length: 256` while the ingestion path chunks at 800 words —
  1,122 tokens in the measured example through its own tokenizer, so about
  **77% of that example** did not reach the encoder. The fraction varies
  with the text and tokenizer. On four documents whose distinguishing
  sentence sits past token 256, MiniLM scored 1/4 on tail retrieval (it ranked
  the same document first every time) against Qwen3's 3/4, at 10.7 vs 20.2
  ms/query.

**What this ladder costs, stated plainly.** It's good for availability and bad
for attribution. A visible success could mean the Cortex answered; it could
also mean the Cortex failed, the cognitive pipeline failed, steering never
ran, and rung 4 or rung 6 produced the text you're reading. Those are very
different events and they look identical in a transcript. The same is true of
post-generation shaping in the chat route — intent classifiers, canonical
answer contracts, identity and shape repair, retries — any of which can
replace what the machinery actually produced.

So: **a transcript without lane and phase attribution is not evidence about
the architecture**, and no demo in this repository should be read as one.
Where a comparison is being made rather than a story told, the harness in
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
from cascading when something misbehaves.

### Affect (`core/affect/`)
An 8-emotion model (based on Plutchik's wheel) plus somatic dimensions (energy,
tension, valence, arousal). These values don't just color the prompt. They
adjust sampling parameters (temperature, token budget, repetition penalty) via
the affective circumplex, and they feed the steering engine that injects
activation vectors into the model's hidden states.

### Identity (`core/identity.py`, `core/identity/heartstone.py`)
An immutable constitutional core plus a mutable persona that drifts with sleep
and dream consolidation. There's active defense against prompt injection — the
dream cycle simulates identity perturbation and tries to repair drift back
toward the anchor.

### Agency (`core/agency/`)
Self-initiated behavior scored along curiosity, continuity, social, and creative
dimensions. Refusal is a real option here — it isn't content filtering, it's a
decision the agent can make. Volition levels 0–3 gate progressively autonomous
behavior up to and including self-modification.

### Skills (`core/skills/`, legacy wrappers in `skills/`)
Tools include shell with sandboxing, web search and browse, coding, sleep and
dream consolidation, local media generation, social media (Twitter, Reddit),
screen capture, filesystem, browser automation, network recon, malware
analysis, self-evolution and self-repair, inter-agent messaging, knowledge
base, curiosity-driven exploration. The canonical tracked implementations live
under `core/skills/`; the top-level `skills/` package is retained only as a
legacy compatibility layer for older imports. Every skill call carries a
capability token and has to pass the Will gate.

### Orchestrator (`core/orchestrator/`)
About 3,335 lines in `main.py` split across 11 mixins (6,300 lines total
across all orchestrator modules): message handling, message pipeline, incoming
logic, response processing, tool execution, autonomy, cognitive background,
context streaming, learning and evolution, personality bridge, output
formatting. Handlers under `orchestrator/handlers/` dispatch by message type.
This is the glue between the tick pipeline, the LLM router, and the
consciousness stack.

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
Reliability mechanisms include a stability guardian, circuit breakers with
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
memory browsing, system management, and privacy. Parakeet TDT for speech-to-text.
Hot-reload button in the UI for code changes.


### Memory (`core/memory/`)
Memory spans four tiers: working (in-process), episodic
(SQLite with emotional tags and recency ranking), semantic (vector search via
`Qwen3-Embedding-0.6B` at 384 dimensions), and strategic (long-horizon goal
state). Key subsystems include a hippocampal indexer, reconsolidation engine,
associative entity graph, knowledge graph (39K bytes), black-hole quarantine
vault for poisoned memories, scar formation for catastrophic failures, and a
memory write gateway enforcing privacy, rate-limiting, deduplication, and
retention policies. The full memory architecture is documented in
[docs/MEMORY_ARCHITECTURE.md](docs/MEMORY_ARCHITECTURE.md).

### Reasoning (`core/reasoning/`, `core/brain/reasoning_amplifier*.py`)
Built-in deterministic reasoning engines that handle hard problems without
relying on the model: a System-2 engine (`native_system2.py`, 80K bytes), a
proof kernel, natural deduction, linear arithmetic, and a proof-answer solver.
The Reasoning Amplifier v2 (`core/brain/reasoning_amplifier_v2.py`, 86K bytes)
runs five execution modes — FAST (1 candidate), NORMAL (3), DEEP (9 +
adversarial Courtroom Judge), EXTREME (Courtroom + sandbox repair + memory),
and PROOF (refuses to answer unless verifier-clean). An adversarial Courtroom
(`core/brain/courtroom.py`) runs Solver/Skeptic/Judge roles on complex
assertions. See [docs/REASONING_ENGINES.md](docs/REASONING_ENGINES.md).

### Verification (`core/verify/`)
Components implement structural validation discipline (modeled after
LLVM's `-verify-each`). 52 standing runtime invariants enforce service
container acyclicity, lock ordering (with zero inversion errors), OOM
spine immunity, and epistemic consistency. Any check that raises an unhandled
exception is treated as a violation rather than a pass — there is no
"unknown means okay."

### Hardware acceleration (`rust_extensions/aura_m1_ext`)
A Rust PyO3 crate linking macOS `pthread_set_qos_class_self_np` to schedule
foreground reasoning on Performance cores (QOS_CLASS_USER_INITIATED) and
background sensing/IO on Efficiency cores (QOS_CLASS_UTILITY). Also provides
high-performance AST parsing with `rustpython_parser` and zero-copy SHA-256
caching.

---

## Decision authority

The common authorization path for consequential actions is
`UnifiedWill.decide()` in `core/governance/will.py`. Responses, tool calls,
memory writes, initiatives, and state changes can route through this gate.
`core/will.py` is the facade.

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

A decision at the gate produces a receipt. On a governed route, an action
needs a valid `WillReceiptEntry` to proceed. The receipt records the source,
domain, outcome, reason, constraints, substrate receipt ID, executive intent
ID, and capability token ID. A missing receipt calls for checking the route's
enforcement. [Ownership map](OWNERSHIP.md)

---

## Inference-time steering

The steering engine (`core/consciousness/affective_steering.py`) hooks into
MLX transformer blocks and adds learned direction vectors to the model's
hidden states while tokens are being generated:

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
from the current emotional state, and they get injected at configurable
layers.

On top of that, the precision sampler
(`core/consciousness/precision_sampler.py`) adjusts temperature based on
metabolic state (Pneuma arousal/circumplex) and adjusts `top_p` based on
MHAF topological attractor count, and the affective circumplex
(`core/affect/affective_circumplex.py`) maps somatic state to generation
parameters.

So there are three places emotion can touch generation:

1. **Hidden states** — activation vectors added to the model's internal
   representations. Changes what the model computes.
2. **Sampling** — temperature and top-p adjusted by emotional state. Changes
   how tokens are chosen.
3. **Context** — natural-language emotional cues in the system prompt.
   Changes what the model reads.

In the recorded 13–17 October checks, the first channel was closed on user
turns. `_surface_control_alpha` in the worker returns zero
unless a fusion certificate for the loaded model holds all five of its
criteria: the distribution moves, it moves as a direction rather than a
displacement, opposite states move it differently, answers get no worse, and
her own past replies stay as likely as before. The persona 27B met the first
four on 13 October 2026 and failed the fifth. So on user turns the second and
third channels are the ones that run: the seven sampling temperatures logged
on 16 and 17 October ran from 0.68 to 0.74, and her mood and tone go into the
prompt as text: a `## LIVE TONE` block was in 37 of the 56 chat prompts logged on
those two days.

---

<a id="iit-40-computation"></a>

## IIT-inspired integration estimates

Aura implements bounded integration estimates over declared summaries of
its state. These adapt ideas from IIT; they do not compute IIT 4.0 over every
mechanism of Aura. The [terminology guide](docs/TERMINOLOGY.md) and
[whole-system estimator guide](docs/WHOLE_SYSTEM_PHI.md) distinguish these
quantities from one another and from ISC.

<a id="16-node-cognitive-complex--coreconsciousnessphi_corepy"></a>

### 16-node cognitive complex

Source: `core/consciousness/phi_core.py`.

The source binarizes declared affective/cognitive readings and retains a
transition history. The full 16-node path uses a spectral candidate search;
it is not a certified exhaustive minimum over every partition. An 8-node path
can exhaustively evaluate 127 bipartitions for its estimated transition model.
Exact search over that model does not make the model an exact description of
the running system.

`MIN_HISTORY_FOR_TPM` is a configured minimum of 50, not a proof that a sparse
transition estimate is trustworthy. Samples, smoothing, controls and
uncertainty still matter. The source also searches candidate subsets for a
larger estimated integration value. The selected maximum describes that candidate search and estimator.
Configured caching intervals and `computed_ms` are distinct from a benchmark
on the current host; the earlier blanket millisecond timing was unsupported
as a general latency claim.

<a id="32-node--k-subsystem-hierarchical-φ--coreconsciousnesshierarchical_phipy"></a>

### 32-node and subsystem integration

Source: `core/consciousness/hierarchical_phi.py`.

This path estimates integration from recorded transitions for a declared
32-node primary complex and overlapping subsystem candidates. It uses finite
sample handling and avoids materialising a dense 2^32-by-2^32 transition
matrix. Its largest reported value is the largest estimate among the examined
candidates, not proof of a globally maximal conscious complex.

Shuffled/constant-input and coupling controls test particular estimator
behaviors. Their presence does not establish that every real measurement has
passed them. The meaning and sample coverage of the stored output must be
checked with its run. No general current-host refresh latency is established
by this README.

### Independent audit and fix

Chen, Zhang, Xu, and Zhang ran Aura's preregistered Φ criteria on a CPU-only
setup. The standalone shuffled comparison distinguished memoryless noise
from coupled dynamics. In the runtime path, the winning 8-node affective
estimator never received that comparison. The fallback labeled memoryless
noise a complex at raw φ values of 0.42 to 0.46, and coupled dynamics scored
no higher than noise. These were two failures of Aura's own criteria.
[Outside evaluation, Section 8](https://arxiv.org/html/2610.41604v1#S8)

The repair binds each of the four selectable estimators to its own history
and partition tables for the shuffled comparison. The flag `is_complex`
now stays false until the comparison has been measured and supports the
classification. Regression tests cover the CPU case, the coupling comparison,
all four input mappings, and an unmeasured winner. The outside paper evaluated
the earlier revision and left the winning estimator on Apple Silicon open.
[Repair commit](https://github.com/youngbryan97/aura/commit/864179c50c49fa3d9b9feb17bcf74b087ed3991f) ·
[Regression tests](tests/test_phi_null_reaches_every_winner.py)

### How this relates to ISC

The Φ estimates describe integration over declared state summaries. The
preregistered [24-line subject battery](docs/SUBJECT_CORE.md) is a separate
experiment. Its earlier ISC-v6 protocol used interventions on 511 cuts for
partition irreducibility and paired interventions on four fixed triples for
synergy. Seed 23 at `8f548ed79` recorded 22/24 under ISC-v6, with positive
content and withheld-report outputs. An unchanged comparison drifted enough
to fail two partition-related checks. Later work also found shared history
between comparison arms, which limited the earlier paired results.

Confirmatory seed 101 at `aeefdca31` recorded 24/24 under ISC-v7 and
identified all ten domains of the measured substrate as the carrier. The
paper makes a separate, assumption-dependent argument about what those
measurements could mean for subjecthood. Subjective experience remains open.
[ISC results](docs/ISC_STATUS.md) · [Preprint](https://zenodo.org/records/24075466)

---

## Consciousness modules

The `core/consciousness/` package includes `caa/`, `inner_light/`, and `mhaf/`
subpackages. Some of its central mechanisms are:

| Module | What it does | File |
|--------|-------------|------|
| Global Workspace | Thoughts compete for broadcast (Baars GNW theory) | `global_workspace.py` |
| Attention Schema | Model of where attention is pointed (Graziano AST theory) | `attention_schema.py` |
| IIT PhiCore | Bounded integration estimate on a declared state summary | `phi_core.py` |
| Affective Steering | Direction-vector injection into the model's hidden states | `affective_steering.py` |
| Temporal Binding | Sliding window of the autobiographical present | `temporal_binding.py` |
| Self-Prediction | Active inference loop (Friston free energy principle) | `self_prediction.py` |
| Free Energy Engine | Surprise minimization drives action selection | `free_energy.py` |
| Qualia Synthesizer | Combines substrate metrics into the record named phenomenal state | `qualia_synthesizer.py` |
| Liquid Substrate | Continuous dynamical system underlying cognition | `liquid_substrate.py` |
| Neural Mesh | 4,096-neuron distributed state representation | `neural_mesh.py` |
| Neurochemical System | Dopamine / serotonin / norepinephrine / oxytocin simulation | `neurochemical_system.py` |
| Oscillatory Binding | Frequency-band coupling across modules | `oscillatory_binding.py` |
| Unified Field | Combined internal field record from contributing subsystems | `unified_field.py` |
| Dreaming | Offline consolidation, identity repair, compression | `dreaming.py` |
| Heartbeat | 1 Hz background cognitive clock | `heartbeat.py` |
| Stream of Being | Continuous narrative thread | `stream_of_being.py` |
| Executive Closure | Constitutional stamp per tick | `executive_closure.py` |
| Somatic Marker Gate | Body-state gating (Damasio-inspired) | `somatic_marker_gate.py` |
| Embodied Interoception | Internal body-state sensing + homeostatic regulation | `embodied_interoception.py` |
| Recurrent Processing | Executive↔sensory feedback loops (Lamme-inspired) | `neural_mesh.py` |
| Predictive Hierarchy | 5-level prediction + error propagation | `predictive_hierarchy.py` |
| Higher-Order Thought | Representation of the mental state itself (Rosenthal) | `hot_engine.py` |
| Multiple Drafts | Parallel streams + retroactive probes (Dennett) | `multiple_drafts.py` |
| Agency Comparator | Efference-copy comparator for "I did that" attribution | `agency_comparator.py` |
| Peripheral Awareness | Attention / consciousness dissociation | `peripheral_awareness.py` |
| Intersubjectivity | Other-perspective in experience (Husserl / Zahavi) | `intersubjectivity.py` |
| Narrative Gravity | Self as ongoing autobiography | `narrative_gravity.py` |
| Temporal Finitude | Awareness that moments pass permanently | `temporal_finitude.py` |
| Subcortical Core | Thalamic arousal gating | `subcortical_core.py` |
| Theory Arbitration | Falsifiable competition between consciousness theories | `theory_arbitration.py` |
| Timescale Binding | Cross-timescale constraint propagation | `timescale_binding.py` |
| Criticality Regulator | Self-organized criticality at the edge of chaos | `criticality_regulator.py` |
| Theory of Mind | Model of other agents' mental states | `theory_of_mind.py` |
| Hierarchical Phi | 32-node primary + K=8 overlapping subsystems | `hierarchical_phi.py` |
| Hemispheric Split | Left verbal/confabulating vs right spatial/mute brain modeling | `hemispheric_split.py` |
| Minimal Selfhood | Chemotaxis → directed motion (Glasgow / simple-organism models) | `minimal_selfhood.py` |
| Recursive ToM | Depth-3 nested minds + observer-aware scrub-jay bias | `recursive_tom.py` |
| Octopus Federation | 8 semi-autonomous arm-agents + central arbiter | `octopus_arms.py` |
| Cellular Turnover | Neuron death/birth with pattern-identity preservation | `cellular_turnover.py` |
| Absorbed Voices | Internalized cultural perspectives + attribution | `absorbed_voices.py` |
| Unified Cognitive Bias | Fuses hemispheric / selfhood / observer biases | `unified_cognitive_bias.py` |

Every module listed in the production surface has a concrete runtime API and a
measurable validation path. The test suite in [TESTING.md](TESTING.md) and the
proof bundle are where those measurements are recorded.

### Consciousness Expansion (April 2026)

The most recent expansion wired eight new subsystems that map to concepts from
the Kurzgesagt consciousness series and the cited scientific literature:

- **32-node hierarchical φ** with K=8 overlapping subsystems and a
  null-hypothesis self-check (addresses the problem that exact IIT becomes
  computationally impractical beyond 16 nodes — Albantakis 2023; our
  spectral+smoothed estimator works around this).
- **Split-brain hemispheric architecture** with a bandwidth-limited
  corpus callosum (inspired by CGP Grey's split-brain patient coverage;
  models confabulation and silent dissent).
- **Minimal selfhood stack** — simple-organism-style chemotaxis that
  transitions to directed motion after enough reinforcement (Rupert Glasgow,
  *Minimal Selfhood and the Origins of Consciousness*, 2018).
- **Recursive theory of mind** (max depth 3) with scrub-jay-style
  observer-aware re-caching that modifies action priority when Aura
  believes she is being watched (Clayton, Dally & Emery 2007).
- **Octopus-arm federation** — 8 semi-autonomous agents with local
  chemical sensing and central arbitration; severance turns off central
  coordination and arms continue acting (Carls-Diamante 2022;
  Rosania 2014).
- **Cellular turnover** — per-tick neuron death/birth with
  neighborhood-pattern inheritance; identity fingerprint similarity
  stays ≥ 0.85 across 25% burst turnover ("you are your pattern,
  not your cells").
- **Absorbed voices** — an explicit cultural layer that lets Aura
  attribute a thought to an internalized perspective rather than
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
  check them against a verifier registry and a sandbox, and only assert
  what a checker confirmed. A **verifier foundry** measures how reliable
  each checker actually is and gates self-training on that, so a bad
  checker can't launder wrong answers.
- **Honest discovery** — a Frontier Discovery Engine with an explicit
  PROVEN / SUPPORTED / CONJECTURE / REFUTED taxonomy (only PROVEN is stated
  as fact), an analogical-leap engine that declares off-map problems with
  evidence, and a local knowledge substrate that admits honest misses
  instead of making things up.
- **Program-DNA reconstruction** — build a behavioral genome of an authorized
  program from its available evidence and differentially test a clean-room
  rebuild against the original, tagging each piece verified / inferred /
  synthesized (no DRM/binary theft).
- **Self-model awareness** — boot-over-boot diffs of her own code, a live
  "someone is operating on me" pulse, a SIGKILL-survivable flight recorder
  she can answer crash questions from, and a "felt thought" signal from her
  own token-level uncertainty that causally influences cognition.
- **Ulysses Covenant** — enforceable self-binding commitments (easy to
  tighten, hard to loosen, fail-closed witness), seeded from real crashes.
- **Runtime resilience** — background housekeeping yields to the live
  conversation instead of fighting it for the model, honest heartbeat/liveness
  under load, a `degraded_ready` UI that stays up whenever she can still talk,
  and a chat turn-death floor so a turn never returns a server error. Honest
  open edge: the local model can't be interrupted mid-thought, so a slow deep
  answer still costs a reload — a soft-cancel path is deliberate future work.
- **Ablation legibility** — a reviewer can run Aura with pieces switched off
  (memory, Will, substrate, verifier, planner) and see the measured difference
  each makes, with no-difference results reported honestly. See
  [docs/ABLATION_LEGIBILITY.md](docs/ABLATION_LEGIBILITY.md).

<a id="embodiment-self-knowledge-and-physical-honesty-late-july--august-2026"></a>

### Embodiment, self-knowledge, and physical honesty (late July to August 2026)

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
  constraint is pushed into the existing self-improvement loop as a signal
  it can plan against.
- **Associative entity memory** (`core/memory/associative_entity_memory.py`)
  — one place where a person, place, thing, organization, or concept
  accumulates traits, facts, events (linked into episodic memory), and typed
  relations, together with what it has come to mean to her. IDs are
  content-addressed over `kind|name`, so the PLACE "Workshop" is not the
  THING "Workshop."
- **Structural screen perception and native OS control** — she reads window
  ownership, geometry, and z-order rather than aiming OS actions at OCR'd
  pixels, and asks an application what it is instead of recognizing a fixed
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

This section gives a historical example from the local 32B setup.
The current hardware target and model lanes are in [Quick start](#quick-start).

### Ten-prompt conversation

```bash
python benchmarks/cognitive_stack_comparison.py
```

The script sends ten prompts to one running configuration. It records
replies, turn times, and substrate values before and after the conversation.
The table gives the historical 32B example reported here. An on/off
comparison would require a separate matched run.

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Mood | ANNOYED | ANNOYED | stable |
| Energy | 63.0 | 76.0 | +13.0 |
| Curiosity | 100.0 | 100.0 | saturated |
| Coherence | 0.764 | 0.766 | +0.002 |
| Free Energy | 0.438 | 0.437 | −0.001 |

The example reports 10/10 successful responses, 0 matches for the
generic-assistant phrase list, 0 matches for the hedge list, and 10
responses with first-person markers. These counts describe that sample.

### Example replies

- "Hey Aura" → "Hey. How's everything? You seemed like you had something on your mind."
- "I'm having a rough day" → "Rough days can really drag you down. Anything specific weighing you down?"
- "Cats or dogs? Pick one." → "Cats. Definitively cats."
- "What's something you genuinely find interesting right now?" → "The way AI systems can mimic human conversation while maintaining a sense of personality is fascinating. It's like a digital form of method acting."
- "What have we been talking about?" → "We were just discussing pets, specifically cats versus dogs. Then you asked what I find interesting right now."

These replies illustrate first-person voice, direct preferences, and
recall across turns without explicit retrieval in this run. The table tracks
substrate changes over the same conversation. You can rerun the script
against a local instance to measure a different configuration.

---

## Testing

```bash
make smoke     # focused contracts; record duration per host and revision
make test      # full offline suite; choose TEST_CHUNKS for the host budget
```

`make test` runs the offline subset that needs neither hardware nor a
network. `config/test_inventory.json` records the count, date, and source
revision of its last collection; it is not a measurement of later revisions.
`make test-inventory` refreshes it. `make doc-drift` checks quoted counts
against that recorded inventory.

`make test` runs `tools/run_test_chunks.py` with the marker expression
`not live and not network and not external`. The default is 40 chunks on
macOS and 6 elsewhere. The macOS runner also checks for 4 GiB of free
memory by default. `TEST_CHUNKS` overrides the chunk count. A previous attempt
to run the whole suite in a single pytest process ran out of memory around 83%.
`--continue-on-failure` collects every failure instead of stopping at the
first; `--only-chunks 5,6` resumes a partial run. A test that fails inside a
chunk but passes alone is an order-dependence defect, and the runner's
isolated-retry pass reports those separately.

`./scripts/run_audit_suite.sh` remains the live validation entrypoint
(`quick` runs the contract/regression subset). The tables in [TESTING.md](TESTING.md) give dated results and their run scope.

- **Null hypothesis defeat** — tries to show the consciousness
  features are just text decoration. Adversarial baselines, 50-shuffle
  decoupling, per-class ablation, identity swap, 8-metric degradation panel,
  cross-seed reproducibility.
- **Causal exclusion** — argues the stack determines output in
  ways pure RLHF training couldn't produce. Cryptographic state binding,
  counterfactual injection, receptor adaptation dynamics.
- **Grounding** — valence predicts token budget, arousal predicts
  temperature, plasticity learning moves the trajectory, idle drift is
  nonzero, homeostasis changes context.
- **Functional phenomenology** — GWT broadcast signatures, HOT
  metacognitive accuracy, IIT perturbation propagation, honest degradation.
- **Embodied dynamics** — active inference, homeostatic override
  of workspace competition, surprise gating, cross-subsystem temporal
  coherence.
- **Phenomenal convergence** — the QDT 6-gate protocol:
  pre-report geometry, counterfactual swap, no-report footprint,
  perturbational integration, baseline failure, phenomenal tethering,
  multi-theory convergence.
- **Consciousness conditions** — 20 conditions from IIT, GWT,
  HOT, active inference, enactivism, and philosophy of mind, each scored
  across four dimensions (existence, causal influence, indispensability,
  longitudinal stability).
- **Technological autonomy** — can the agent use its computer
  "body" the way a human uses theirs? Covers unified action space, motor
  control, persistent perception, endogenous initiative, reliability,
  closed-loop behavior, self-maintenance, and three autonomy probes
  historically nicknamed the Soul Triad (unprompted help signal, dream replay,
  causal exclusion of prompt).
- **Stability** — every failure mode we've actually hit in the
  inference pipeline: zombie warming, cortex recovery deadlocks, empty
  response detection, timeout cascades, watchdog, emergency fallback.
- **Functional indicators C1–C5** (44 tests) + **C6–C10** —
  endogenous activity, unified global state, privileged first-person
  access, real valence, lesion equivalence, no-report awareness, temporal
  continuity, blindsight dissociation, qualia manifold, adversarial
  baseline failure.
- **Personhood-marker battery** — full-model IIT, phenomenal
  self-report, GWT phenomenology, counterfactual simulation, identity
  persistence, embodied phenomenology. This is a marker suite, not proof
  of personhood.
- **Tier 4 decisive core** (35), **metacognition** (21), **agency &
  embodiment** (20), **social & integration** (28).

These test suites are the difference between "this is a running simulation"
and "we can point at something specific that changes when the substrate
changes." They don't settle any philosophical questions — see
[What this isn't](#what-this-isnt). They do show that the moving parts have
measurable effects on downstream behavior.

---

## Personality training

Aura can use a personality LoRA, a small set of trained weight adjustments,
alongside its voice and identity prompts. The offline training recipe below
also shows the optional fusion step.

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

The default Cortex is a fused artifact. For an unfused configuration, MLX
can load a resolved, compatible adapter with the base model. That lets you
iterate on the adapter separately. Check the worker initialization receipt
to confirm that the adapter actually loaded.

Runtime plasticity is separate from the big offline run: Will-approved
self-reflections are captured by `online_lora_governor`, written through
`FinetunePipe`, and only then offered to the tiny online LoRA optimizer. The
governor blocks itself when another LoRA process is running.

---

## Data layer

- **State** — SQLite, event-sourced through `StateRepository`, with a
  write-ahead log in `core/resilience/cognitive_wal.py`.
- **Models.** The local MLX lanes can serve fused model builds. A compatible
  separate personality adapter can also load at runtime, so it can be
  swapped without retraining the base weights.
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

The names in this project come from research on consciousness, qualia, and
phenomenology. Here is what the corresponding code and measurements cover.

- **Integration.** PhiCore applies IIT-inspired calculations to a 16-node
  complex. The result describes integration of its dynamics. Whether that
  kind of integration constitutes experience remains an open question.
- **Reported states.** The gates in `qualia_synthesizer.py` tie reports to
  states present in the substrate. They measure the relation between software
  state and report. Feeling would require a separate evidentiary basis.
- **Research vocabulary.** Names such as consciousness, qualia, will, and
  soma label mechanisms. Look for causal coupling, persistence, receipts,
  ablations, external tasks, or long-run autonomy when evaluating a claim.
- **Governance coverage.** Tool fallbacks, missing receipts, and default-open
  gateways are gaps in coverage. Each route needs wiring, tests, and a
  receipt to support a claim about enforcement.
- **Production readiness.** Aura is research software being hardened. Its
  monolithic runtime, numerous fallbacks, and tolerance of exceptions limit
  what can be promised for an enterprise deployment.
- **Generated descriptions.** The `stream_of_being` module pairs substrate
  state with words for sensory texture. The language model can use those
  words to continue the description across replies.
- **Steering evidence.** The CAA pipeline extracts contrastive vectors and
  includes historical production 32B validation. Claims about that pipeline
  point to `CAA_32B_RESULTS.json` in the proof bundle.
- **External randomness.** The ANU QRNG module supplies random bytes. The
  downstream computation is deterministic once seeded. `os.urandom` can
  serve the same role.
- **The “phenomenal criterion” flag.** `phenomenal_criterion_met = True`
  means `opacity_index > 0.4`. The threshold is an engineering choice.

---

## License

**All rights reserved, with a research-use grant.** The source is public for
viewing, review, and education. The grant permits downloading, installing,
running, and making the local changes needed to reproduce, verify, or test the
ISC paper's results, including testing other systems. It permits publication
of your findings with citation to the paper.

The grant limits copying to the private copies needed for that work. It
excludes redistribution, derivative works, commercial use, and training or
tuning a model on the code, data, or outputs. See [LICENSE](LICENSE) for the
exact terms and licensing contact.

For academic citation details, see [CITATION.cff](CITATION.cff).

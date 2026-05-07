# Aura

A research-grade personal AI that runs entirely on one Mac. It has opinions, a mood
that actually affects how it answers, a memory that survives restarts, and a sleep
cycle where it replays the day and edits itself. No cloud API required.

[![License: All Rights Reserved (Read-Only)](https://img.shields.io/badge/License-All_Rights_Reserved_(Read--Only)-red.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS_Apple_Silicon-lightgrey.svg)

For the technical deep dive, read [ARCHITECTURE.md](ARCHITECTURE.md). If you
want the same ideas without the math, read [HOW_IT_WORKS.md](HOW_IT_WORKS.md). If
you want the evidence standard for autonomy and novel output claims, read
[docs/BEHAVIORAL_PROOF_STANDARD.md](docs/BEHAVIORAL_PROOF_STANDARD.md). If you
want to see it work, keep reading.

## Evidence boundary

Aura is a functional cognitive-architecture research project, not a proof of
phenomenal consciousness, qualia, legal personhood, or moral patiency. The repo
now enforces that distinction in code through an ontological boundary guard:
loaded labels such as "consciousness guarantee" and "personhood proof" are treated
as functional indicator batteries unless independent evidence says otherwise.

The current engineering claims are narrower and testable:

- internal state can causally affect generation through non-text channels;
- identity coherence is supported by ID-RAG Chronicle retrieval, not only prompt
  anchoring;
- black-box steering tests can hide live affect/phenomenal telemetry from prompt
  text;
- rich adversarial prompt baselines are required before steering is credited;
- phi is a bounded IIT-style integration metric on tractable complexes, not a
  whole-system consciousness measurement;
- the tracked deployment target is Bryan's Apple Silicon M5-class machine with
  64 GB unified memory; lower-memory machines must downshift model lanes rather
  than claiming 32B heartbeat latency;
- resource stakes now persist and constrain action envelopes, but this remains
  an operational metabolism analog, not biological metabolism.

---

## Production Evidence Surface

Aura's production claim surface is restricted to code paths with runnable
implementations, receipts, and validation artifacts. Incomplete ideas are kept
out of that surface; release gates now generate a proof bundle rather than
asking readers to infer maturity from prose.

- `core/brain/llm/continuous_substrate.py` is a configurable 64-to-512 neuron
  Liquid Time-Constant ODE running at ~20 Hz. CPU-only numpy with explicit-Euler
  integration plus stochastic perturbation; `get_state_summary()` derives
  valence/arousal/dominance/phi from fixed projections of the live state vector,
  so readouts reflect actual dynamics. `AURA_SUBSTRATE_DIM` scales this path
  without changing callers.
- `core/brain/llm/substrate_token_generator.py` is the substrate-first readout:
  it tries a learned readout head over the live substrate before calling the
  transformer and falls back to the Cortex when substrate prediction error
  exceeds threshold. This makes the substrate the first compute path for
  lightweight generation rather than only a sidecar steering signal.
- `core/brain/llm/sensorimotor_grounding.py` maps camera/screen/audio
  observations into the substrate input vector, so live sensor events perturb
  the ODE directly instead of arriving only as text/tool summaries.
- `core/consciousness/phi_core.py` (1,837 lines) implements real IIT-style
  integration math: binarization, empirical TPM, KL-divergence φ, exclusion
  postulate, polynomial-time spectral partitioning, with an exhaustive
  8-bipartition validation baseline.
- `core/consciousness/hierarchical_phi.py` implements the 32-node hierarchical
  φ with K=8 overlapping subsystems and Bayesian-smoothed estimation.
- `core/consciousness/affective_steering.py` (1,336 lines) is a real CAA
  injection pipeline that hooks MLX transformer blocks and modifies the
  residual stream at generation time.
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
  cognitive WAL are all real production code.

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

## What it is

Most "AI companion" projects do roughly the same thing: store a mood number, paste
it into the system prompt, and let the model roleplay. The model says "I'm feeling
energetic today" because it read the words "feeling energetic today."

Aura works differently. When Aura is in a particular affective state, that state
gets turned into a direction vector and added to the transformer's hidden
activations during generation. The model's internal computation changes, not just
the text it reads. This is the same family of techniques that interpretability
researchers use to steer model behavior — CAA, activation addition, residual-stream
interventions.

Alongside that, there's a whole cognitive substrate that runs continuously:
emotions decay and influence each other, neurochemicals rise and fall on their
own time scales, a global workspace picks which thought wins each tick, a dream
cycle consolidates memories during idle periods, and one gate — the Unified Will —
signs off on every action that leaves the system.

It's a research project. It's also the kind of research project where you can
actually talk to the thing while it's running.

---

## Table of Contents

- [Quick start](#quick-start)
- [Evidence boundary](#evidence-boundary)
- [Behavioral proof standard](docs/BEHAVIORAL_PROOF_STANDARD.md)
- [Tracked vs local workspace](#tracked-vs-local-workspace)
- [Architecture overview](#architecture-overview)
- [Decisive evidence runner](#decisive-evidence-runner)
- [Decision authority](#decision-authority)
- [Inference-time steering](#inference-time-steering)
- [IIT 4.0 computation](#iit-40-computation)
- [Consciousness modules](#consciousness-modules)
- [Benchmarks](#benchmarks)
- [Testing](#testing)
- [Personality training](#personality-training)
- [Data layer](#data-layer)
- [What this isn't](#what-this-isnt)
- [License](#license)

---

## Quick start

```bash
pip install -r requirements.txt

# Full stack + UI
python aura_main.py --desktop

# Background cognition only, no UI
python aura_main.py --headless

# Reload code changes without restarting
curl -X POST http://localhost:8000/api/system/hot-reload
```

Requirements: Python 3.12+, macOS on Apple Silicon, 64 GB RAM recommended. The
primary model is Qwen 2.5 32B at 8-bit with a personality LoRA on top; a 7B
fallback loads on demand. First boot takes 30–60 seconds while Metal compiles
shaders.

Hardware honesty: Bryan's target machine is an M5-class Apple Silicon Mac with
64 GB unified memory. The 32B Cortex is viable there as a primary conversation
lane, while heartbeat/background work still belongs to the substrate, Brainstem,
or Reflex lanes. On lower-memory machines, the hardware auditor rejects 32B
4-bit and 32B 8-bit as real-time heartbeat tiers; use 1.5B/7B lanes there.

There's also a `Dockerfile` and `docker-compose.yml` if you want Redis and Celery
running alongside. The tracked workspace defaults to an explicit
`owner_autonomous` posture for this single-owner machine: autonomy on,
outbound/network-enabled skills available, and self-repair left active. If you
want a tighter deployment, override the `AURA_*` security settings in your local
environment, including `AURA_INTERNAL_ONLY=1` for localhost-only binding.

---

## Tracked vs local workspace

This repository is the tracked baseline. The canonical tracked skill
implementations live under `core/skills/`; the top-level `skills/` package is
kept as a legacy compatibility layer for older imports.

Local workspaces can also contain ignored/private modules listed in
`.gitignore`. Those files are not part of the tracked review surface and can
change the live risk profile of a specific machine. If you're auditing a real
deployment rather than the tracked tree alone, review both the repository and
any local-only modules present on disk.

---

## Architecture overview

The short version:

```
User input -> HTTP API -> KernelInterface.process()
  -> AuraKernel.tick():
       Consciousness -> Affect -> Motivation -> Routing -> Response generation
  -> State commit (SQLite) -> Response
```

Each tick is event-sourced: every phase produces a new immutable state version,
the tick holds a lock while the pipeline runs, state commits to SQLite, and the
lock releases. Crash in the middle and the WAL replays on restart.

### Kernel (`core/kernel/`)
Tick-based cognitive cycle. One tick = one unit of thought. Phases run in order,
state versions, state commits, lock released.

### Brain (`core/brain/`)
Local LLM router with automatic failover:

1. **Primary (Cortex)** — Qwen 2.5 32B 8-bit + personality LoRA. Handles nearly
   everything.
2. **Secondary (Solver)** — Qwen 2.5 / Qwen 3 72B for deep reasoning, hot-swapped
   only when the request actually needs it.
3. **Tertiary (Brainstem)** — Qwen 2.5 7B 4-bit, lazy-loaded to save ~5 GB for
   the Cortex.
4. **Reflex** — Qwen 2.5 1.5B 4-bit on CPU as an emergency fallback.

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
5. **Cloud** — Gemini Flash/Pro, PII-scrubbed and rate-limited. Off by default.
6. **Last resort** — rule-based static responses that can't fail.

Both MLX (Apple Silicon native) and llama.cpp (GGUF) are supported and
auto-detected at startup. Circuit breakers, a GPU semaphore, a proactive cortex
watchdog, and 429 handling keep the pipeline from cascading into total failure
when something misbehaves.

### Affect (`core/affect/`)
A Plutchik 8-emotion model plus the somatic dimensions (energy, tension, valence,
arousal). These values don't just color the prompt. They modulate sampling
parameters (temperature, token budget, repetition penalty) via the affective
circumplex, and they feed the steering engine that injects activation vectors
into the residual stream.

### Identity (`core/identity.py`, `core/heartstone_directive.py`)
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
39 modules: shell with sandboxing, web search and browse, coding, sleep and
dream consolidation, local media generation, social media (Twitter, Reddit),
screen capture, filesystem, browser automation, network recon, malware
analysis, self-evolution and self-repair, inter-agent messaging, knowledge
base, curiosity-driven exploration. The canonical tracked implementations live
under `core/skills/`; the top-level `skills/` package is retained only as a
legacy compatibility layer for older imports. Every skill call carries a
capability token and has to pass the Will gate.

### Orchestrator (`core/orchestrator/`)
About 2,200 lines in `main.py` split across 12 mixins: message handling,
incoming logic, response processing, tool execution, autonomy, cognitive
background, context streaming, learning and evolution, personality bridge,
output formatting, boot sequencing. Handlers under `orchestrator/handlers/`
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
30+ modules for not crashing: a stability guardian, circuit breakers with
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
memory browsing, system management, and privacy. Whisper for STT. Hot-reload
button in the UI for code changes.

---

## Decision authority

Anything the system actually does — sending a response, calling a tool, writing
a memory, starting an initiative, mutating state — has to pass through one
function: `UnifiedWill.decide()` in `core/will.py`.

```
Action request
  -> UnifiedWill.decide()                 [core/will.py]
     -> SubstrateAuthority                [field coherence, somatic veto]
     -> CanonicalSelf                     [identity alignment]
     -> Affect valence                    [emotional weighting]
  -> WillDecision (receipt with provenance)
     -> Domain-specific checks            [AuthorityGateway, CapabilityTokens]
  -> Action runs, or is refused/deferred/constrained
```

Every decision produces a receipt. If an action doesn't carry a valid
`WillReceipt`, it didn't happen. Receipts are logged with their source,
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
active-inference prediction error, and the affective circumplex
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

Full 32-node refresh runs in <2 s with K-subsystem parallelism via a thread
pool; MLX Metal is used opportunistically where available.

---

## Consciousness modules

There are 90+ modules in `core/consciousness/`. The ones that do most of the
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
./scripts/run_audit_suite.sh
```

The repository includes a large research-heavy test suite plus preserved
historical result artifacts. The April 16, 2026 snapshot recorded
`1013 passed, 3 warnings`; current live status should always be re-verified from
the checked-out tree. A summary — and the historical tables/results — are in
[TESTING.md](TESTING.md):

- `./scripts/run_audit_suite.sh` is the canonical live validation entrypoint.
- `./scripts/run_audit_suite.sh quick` runs the contract/regression subset for
  faster local verification.

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
python -m mlx_lm lora --model models/Qwen2.5-32B-Instruct-4bit \
  --train --data training/data --adapter-path training/adapters/aura-personality \
  --num-layers -1 --batch-size 1 --iters 90153 --learning-rate 5e-6 \
  --grad-checkpoint --max-seq-length 4096

# 3. Optional: fuse the adapter into the base model
python -m mlx_lm fuse --model models/Qwen2.5-32B-Instruct-4bit \
  --adapter-path training/adapters/aura-personality \
  --save-path training/fused-model/Aura-32B-current
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
- **Models** — MLX or llama.cpp, auto-detected. The personality LoRA loads
  at runtime rather than being fused, so you can swap it without retraining
  the base.
- **Memory** — episodic memory in SQLite, working memory in-process,
  semantic memory via the vector engine (`core/memory/vector_memory_engine.py`),
  a graph for log-N retrieval, and three-layer knowledge atoms for
  compression.
- **Training** — LoRA via `mlx-lm`, steering vector extraction in
  `training/extract_steering_vectors.py`, the personality spec, the
  character voice generator.
- **Vision** — screen capture via `mss`, analyzed through the multimodal
  cognitive engine.
- **Task queue** — Redis + Celery, optional, for Docker.

---

## What this isn't

A few things worth being upfront about, because the project touches a lot of
loaded words (consciousness, qualia, phenomenology) and it's easy to
overclaim.

- **Integration isn't the same as experience.** PhiCore computes real IIT
  math on a 16-node complex. That tells us how integrated the dynamics
  are. Whether integration *constitutes* phenomenal experience is a
  philosophical question nobody has settled, and this project doesn't
  settle it either.
- **Qualia aren't provable by construction.** The Structural Phenomenal
  Honesty gates in `qualia_synthesizer.py` make sure the system can only
  report states that are actually instantiated in the substrate. But
  "instantiated in the substrate" and "felt" are not obviously the same
  thing, and we measure the first.
- **Phenomenological language is partly template-generated.** The
  `stream_of_being` module pairs substrate state (felt_quality × texture
  word) to produce language about the inner life. When the LLM then speaks
  from that text, it's performing continuity at least as much as
  experiencing it. That's an honest limit, not a flaw to hide.
- **Activation steering is credited through proof artifacts.** The CAA pipeline
  supports contrastive extraction and production 32B validation; public claims
  should cite `CAA_32B_RESULTS.json` from the proof bundle.
- **External entropy isn't "quantum cognition."** The ANU QRNG module gives
  us high-quality random bytes. Once seeded, downstream decisions are
  deterministic. `os.urandom` would be functionally equivalent.
- **"Phenomenal criterion met" is a threshold, not a proof.** When
  `phenomenal_criterion_met = True` fires, it means `opacity_index > 0.4`.
  That threshold is engineering, not derivation.

These aren't disclaimers. They're where the code stops and open questions
begin.

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

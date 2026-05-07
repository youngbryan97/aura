# Aura Architecture

This is the technical spec. It gets into the math, the file paths, and the
algorithms. If you want the ideas-only tour, read [HOW_IT_WORKS.md](HOW_IT_WORKS.md).
If you just want to run it, the [README](README.md) has the quick start.

**Evidence boundary.** This document describes functional architecture and
testable mechanisms. It does not claim to prove phenomenal consciousness,
qualia, legal personhood, or moral patiency. Phi is reported as a bounded
IIT-style integration measure over tractable complexes; full-system IIT remains
intractable. Steering claims now require black-box prompt hygiene plus a rich
adversarial prompt baseline before they are credited.

For claims about verifiable autonomy, superhuman-scale behavior, and novel
science or engineering output, see
[docs/BEHAVIORAL_PROOF_STANDARD.md](docs/BEHAVIORAL_PROOF_STANDARD.md). Those
claims require longitudinal artifacts and independent evaluation, not
architecture alone.

---

## Table of Contents

0. [The Unified Will: decision authority](#0-the-unified-will)
1. [System model](#1-system-model) (includes the substrate-first inference pipeline)
2. [The tick: Aura's atomic unit of cognition](#2-the-tick)
3. [Integrated information (IIT 4.0)](#3-integrated-information)
4. [Affective modulation pipeline](#4-affective-modulation)
5. [Activation steering (CAA)](#5-activation-steering)
6. [Persistent emotional network](#6-persistent-emotional-network)
7. [STDP online learning](#7-stdp-online-learning)
8. [Memory architecture](#8-memory-architecture)
9. [The consciousness stack](#9-consciousness-stack) (9.1–9.23, including resilience and self-modification)
10. [Personality persistence and anti-drift](#10-personality-persistence)
11. [Quantization and emergence](#11-quantization-and-emergence)
12. [Limitations and mitigations](#12-limitations-and-mitigations)
13. [Open research program](#13-open-research-program) (6 problems)

---

## 0. The Unified Will: decision authority

**File**: `core/will.py`

Every significant action in Aura — responses, tool calls, memory writes,
autonomous initiatives, state mutations, spontaneous expressions — has to
pass through `UnifiedWill.decide()` and get a `WillDecision` back before it's
allowed to proceed. One locus of decision authority, one place to look when
you want to know who approved what.

### Why it's centralized

Before unification, there were 5+ competing decision authorities
(SubstrateAuthority, ExecutiveCore, ExecutiveAuthority, AuthorityGateway,
VolitionEngine, CognitiveKernel) that each claimed to be the central gate
but were all incompletely wired. The SubstrateAuthority was literally
described as "the mandatory gate for ALL actions" — and it was called in
exactly one place. No single source of will, and no way to prove all
actions went through a unified decision point. Collapsing to one gate was a
maintenance and provability fix.

### Architecture

The Will doesn't replace the existing subsystems. It composes them:

```
              User Input / Internal Impulse
                         |
                    UnifiedWill.decide()
                    /    |    |    \
         Identity  Affect  Substrate  Memory
         (CanonicalSelf) (VAD) (Field+Soma+Chem)  (Episodic)
                    \    |    |    /
                   WillDecision
                   (PROCEED / CONSTRAIN / DEFER / REFUSE)
                         |
              Action Execution (if approved)
```

### The five advisory inputs

1. **Identity alignment.** Reads from CanonicalSelf. Axiomatic violations
   (content that contradicts "I am Aura, a sovereign mind") are blocked
   regardless of what other subsystems say. Tension is detected when
   self-coherence drops below threshold.

2. **Affect valence.** Reads current emotional state. Very negative affect
   (< -0.7) defers exploration. Emotional state shapes the Will's
   disposition, but it doesn't override identity.

3. **Substrate state.** Consults SubstrateAuthority for field coherence,
   somatic markers, and neurochemical state. Low field coherence (< 0.25)
   blocks non-critical actions. Strong somatic avoidance (< -0.5) vetoes
   non-essential actions.

4. **Memory relevance.** Checks whether the memory system has context
   relevant to the decision. Coarse signal today — will be refined.

5. **Priority / domain.** Low-priority initiatives are deferred.
   User-facing responses get maximum latitude. Stabilization actions are
   exempt from field coherence gates.

### Provability

Every `WillDecision` carries a `receipt_id` — a unique, hashable
identifier. The Will keeps a complete audit trail, and any action can be
traced back via `will.verify_receipt(receipt_id)`. Decisions are published
to the event bus for system-wide observability.

### Wiring points (8 paths, 7 files)

| Path | File | Domain |
|------|------|--------|
| Response generation | `orchestrator/mixins/incoming_logic.py` | RESPONSE |
| Response finalization | `orchestrator/mixins/response_processing.py` | EXPRESSION |
| Tool execution | `orchestrator/mixins/tool_execution.py` | TOOL_EXECUTION |
| Boredom impulse | `orchestrator/mixins/autonomy.py` | INITIATIVE |
| Agency core pulse | `orchestrator/mixins/autonomy.py` | INITIATIVE |
| Spontaneous emission | `orchestrator/mixins/autonomy.py` | EXPRESSION |
| Volition tick | `volition.py` | INITIATIVE |
| Memory write | `orchestrator/mixins/incoming_logic.py` | MEMORY_WRITE |
| State mutation | `orchestrator/mixins/incoming_logic.py` | STATE_MUTATION |

### Freedom within constraints

The Will is free within its identity constraints. It can proceed,
constrain, defer, or refuse any action. Its assertiveness adapts based on
experience — a high refuse rate makes it more cautious. Identity is
refreshed periodically from CanonicalSelf. The only unconditional bypass
is `is_critical=True` for safety-critical actions.

Runtime: under 5 ms per decision, zero LLM calls.

---

## 0.1 Initiative Synthesizer: one origin for all impulses

**File**: `core/initiative_synthesis.py`

Before unification, Aura had multiple independent sources of autonomous
action: AgencyCore, VolitionEngine, DriveEngine, GoalEngine,
ContinuousPerceptionEngine, CommitmentEngine, and the Swarm. They all
generated impulses in parallel and converged after the fact — fragmented
in a way that made it hard to reason about what was actually driving
behavior.

The InitiativeSynthesizer is the single funnel:

```
AgencyCore    ──┐
VolitionEngine──┤
DriveEngine   ──┤
GoalEngine    ──┼──→ InitiativeSynthesizer ──→ InitiativeArbiter ──→ UnifiedWill ──→ Execution
Sensors       ──┤         (collect, dedup,         (score on 8         (authorize)
Commitments   ──┤          merge, rank)             dimensions)
WorldState    ──┘
```

The synthesizer collects impulses from every source, deduplicates within a
2-minute window, caps at 15 per cycle, converts to initiative format, scores
them via the 8-dimensional InitiativeArbiter, runs the top candidate through
InternalSimulator for counterfactual evaluation, and finally sends the
winner through UnifiedWill for authorization.

The rule: `impulse → synthesis → arbiter → simulation → will → execution → memory`

## 0.2 World State: live perceptual feed

**File**: `core/world_state.py`

Separate from the EpistemicState knowledge graph (which stores conceptual
relationships), WorldState tracks what's happening right now:

- **User activity**: last interaction timestamp, idle duration, message count, estimated mood
- **System telemetry**: CPU, RAM, thermal pressure, battery (via psutil, updated every 10s)
- **Environment**: time of day, session duration, active app context
- **Salient event queue**: recent changes worth noticing, scored by salience, with TTLs
- **Standing beliefs**: environment facts with expiration (e.g., "user is likely frustrated")

WorldState feeds into initiative scoring. If the user has been idle for 3
hours and it's late at night, the system knows that. If CPU pressure spikes,
the system knows that too. When the user hits a terminal error, WorldState
marks it as a salient event and updates the mood estimate.

## 0.3 Drive cross-coupling

**File**: `core/drive_engine.py` (enhanced)

The DriveEngine manages five resource budgets: energy, curiosity, social,
competence, and uptime_value. These used to be independent timers. Now
they cross-couple:

- **Low energy** → increases `resource_cost` weight in the arbiter (prefer cheap actions)
- **Low curiosity** → boosts `novelty` weight (crave new information)
- **Low social** → boosts `social_appropriateness` weight (crave connection)
- **Low competence** → boosts `tension_resolution` weight (crave achievement)

`get_drive_vector()` returns normalized (0-1) drive levels as a single
read point for any subsystem. `get_arbiter_weight_modifiers()` returns the
dynamic weight adjustments the InitiativeArbiter applies during scoring.

Drive satisfaction feedback is wired now: when the user sends a message,
the social drive gets +15. When a goal completes, the competence drive is
satisfied.

## 0.4 Counterfactual action simulation

**File**: `core/simulation/internal_simulator.py` (enhanced)

The InternalSimulator previews consequences before acting. It evaluates
candidates across five dimensions:

1. **Valence** (0.3 weight) — emotional desirability of the predicted state
2. **Energy cost** (0.2 weight) — resource impact
3. **Cortisol risk** (0.15 weight, inverted) — stress cost
4. **Identity alignment** (0.2 weight) — does this match who Aura is?
5. **Commitment compatibility** (0.15 weight) — does this conflict with active promises?

Identity violations (e.g., "as an AI") are checked axiomatically before
service lookup. Commitment compatibility checks against the
CommitmentEngine's active promises.

## 0.5 Goal resumption at boot

At boot, after CanonicalSelf loads, the orchestrator reads GoalEngine's
SQLite database for IN_PROGRESS and PAUSED goals and injects them into
`pending_initiatives` with `continuity_restored=True` and urgency ≥ 0.6.
The practical effect: after a restart, Aura's first autonomous initiative
is to continue what she was doing. Goals survive process death.

`core/goals/default_goals.py` also seeds four durable IN_PROGRESS goals when
`AURA_SEED_DEFAULT_GOALS=1` (the default): repair/self-maintenance, proof-bundle
upkeep, live sensor grounding, and ASA architecture improvement. Each seeded
goal carries required tools/skills so the initiative funnel has overt work to
select instead of drifting toward inaction when no user task is active.

## 0.6 Proof surface

**Endpoint**: `GET /api/inner-state`

Returns a JSON object containing:
- Last 5 WillDecision receipts with full provenance
- CanonicalSelf snapshot (identity, condition)
- DriveEngine levels (all 5 budgets)
- WorldState status (telemetry, user activity, salient events)
- InitiativeSynthesizer status (recent syntheses)
- Last selected initiative (score, rationale, dimension breakdown)
- Substrate coherence (phi, field coherence)
- Active goals
- Affect state
- Sensorimotor grounding status
- Last substrate token-generation decision
- Online LoRA governor status
- Overt action loop status: last skill run, verification result, receipts,
  goal linkage, and LifeTrace event id

Receipt verification: `GET /api/inner-state/will-receipt/{receipt_id}`
confirms that a specific action passed through the Will.

The CLI proof stream is `python aura_main.py --philosophy`. It emits JSONL with
the live substrate trajectory head, phi value, affect state, and recent Will
receipts. This deliberately exposes the qualia gap: observers can see the
functional trajectory and decide what they think it means.

## 0.7 Overt action loop

**File**: `core/runtime/overt_action_loop.py`

The practical agency loop is not only "the substrate thought about something."
Every `overt_action_cycle` in the AutonomyConductor attempts one bounded
external step:

1. Ask `InitiativeSynthesizer` for the current winner.
2. Require a Will-approved initiative receipt, or use a governed fallback
   maintenance initiative when no winner exists.
3. Map the initiative/goal to a registered safe skill such as
   `auto_refactor`, `system_proprioception`, `environment_info`, or
   `file_operation`.
4. Execute through `CapabilityEngine`, which applies constitutional tool
   governance, capability tokens, metabolic checks, retries, and skill
   timeouts.
5. Verify the actual return payload, not merely that the loop fired.
6. Emit `ToolExecutionReceipt`, `AutonomyReceipt`, and a hash-chained
   `LifeTrace` `action_executed` event.
7. Add receipt evidence/progress back to the linked durable goal.

This makes the answer to "what does Aura do?" concrete: after boot and an idle
window, she should run small real tasks, leave receipts, surface them in the
thought stream and `/api/inner-state`, and carry the evidence forward into
future goal selection.

Shell execution remains argument-vector based (`shell=False`) by default. That
is intentional: Aura can still run real commands and use persistent bash
sessions when a task needs shell syntax, but the default path does not turn
metacharacters into an exploit surface just to make command chaining easier.

---

## 1. System model

Aura is a discrete-time cognitive architecture. The fundamental unit of
computation is the **tick** — a locked, linear pipeline of phases that reads
state, transforms it, and commits the result atomically.

```
tick(objective) → lock → [phase₁ → phase₂ → ... → phaseₙ] → commit → unlock
```

Two concurrent loops run at once:
- **Foreground**: user-triggered ticks (priority, ~6-18s latency)
- **Background**: 1 Hz heartbeat ticks (monitoring, self-reflection, initiative)

State is event-sourced. Each phase produces a new immutable state version
derived from the previous one. The committed state survives process crashes,
power loss, and restarts via SQLite persistence.

### Invariants

These properties have to hold at all times. If any of them is violated,
it's a bug:

1. A tick never partially commits. Lock acquisition fails → tick aborted. Phase fails → tick continues.
2. System prompt ≤ 5000 tokens. Violation causes context overflow → empty LLM output → user sees fallback.
3. Vault commit failure is non-fatal. The tick returns a response regardless of persistence success.
4. No raw numeric metrics in user-facing output. Affect values shape generation parameters, not dialogue.

### Inference pipeline

The LLM inference layer (`core/brain/llm/`) is now substrate-first and
multi-tier. The live substrate tries a learned readout head before the
transformer. If its own prediction error exceeds threshold, the request falls
through to the Cortex/Solver/Brainstem stack.

```
User Message → Orchestrator → LLM Router
  → Tier 0: Substrate readout (continuous state → logits) ──→ Response
  │   ↓ (prediction_error > threshold / forced transformer)
  → Tier 1: Cortex (Qwen 2.5 32B 8-bit + LoRA adapter) ──→ Response
  │   ↓ (failure/timeout/empty)
  → Tier 2: Solver (Qwen 2.5/3 72B, hot-swapped) ──→ Response
  │   ↓ (failure/timeout/empty)
  → Tier 3: Brainstem (Qwen 2.5 7B 4-bit) ──→ Response
  │   ↓ (failure/timeout/empty)
  → Tier 4: Cloud (Gemini Flash/Pro, PII-scrubbed) ──→ Response
  │   ↓ (failure/quota exhausted)
  → Tier 5: Reflex (Qwen 2.5 1.5B 4-bit CPU) ──→ Response
  │   ↓ (failure)
  → Tier 6: LazarusBrainstem (rule-based static responses, never fails)
```

Key implementation details:
- **Model registry** (`model_registry.py`): single source of truth for model lanes, artifact paths, and backend selection (MLX or llama.cpp)
- **Health monitor**: per-endpoint failure tracking with a 3-failure threshold, 20-second recovery window, and immediate circuit break on 429 rate limits
- **GPU semaphore**: a global `threading.Semaphore(1)` ensures only one model loads at a time, preventing OOM from simultaneous loads
- **Foreground owner lock**: when the Cortex is actively generating for a user request, background tasks defer rather than contend for the GPU
- **Substrate token generator** (`substrate_token_generator.py`): maps the live
  substrate vector through a learned readout head and records prediction error,
  token IDs, and logits checksum. The LLM is the fallback cortex for high-error
  or explicitly deep requests.
- **Sensorimotor grounding** (`sensorimotor_grounding.py`): maps camera, screen,
  and microphone observations into the substrate input vector so real sensory
  events perturb the ODE directly.
- **Context injection**: every LLM call is augmented with state context (affect summary, recent memories, cognitive mode) via `_get_context_headers()`
- **MLX worker**: runs in a subprocess with `multiprocessing.set_start_method("spawn")` to isolate Metal/GPU state from the main process

---

## 2. The tick

### Phase pipeline

Each tick runs these phases in strict order:

| Phase | Purpose | Timeout |
|-------|---------|---------|
| PhiConsciousness | Compute integrated information (φ) | 10s |
| AffectUpdate | Update valence, arousal, somatic markers | 5s |
| MotivationUpdate | Compute drive pressures (curiosity, social, energy) | 5s |
| CognitiveRouting | Classify intent (CHAT / SKILL / SYSTEM) | 10s |
| ConversationalDynamics | Track discourse state, topic shifts | 5s |
| UnitaryResponse | Generate LLM response with full cognitive context | 85s |

Background-only phases (skipped during user-facing ticks):
- LearningPhase, RepairPhase, BondingPhase, SelfReviewPhase

### Priority preemption

When a user message arrives during a background tick, the kernel sets
`_user_priority_pending`. Between phases, the background tick checks the
flag and yields the lock early, so user-facing latency isn't blocked by
slow background work.

---

## 3. Integrated information (IIT 4.0)

**File**: `core/consciousness/phi_core.py`

Aura computes an integrated information measure using IIT-style formalism on a
16-node cognitive complex. This is a scoped measure over Aura's telemetry and
cognitive-affective state, not a measurement of the Qwen transformer's full
neural causal structure. The phi value is mathematically real for the sampled
complex; it is not presented as "the LLM's phi" or as a strict Tononi-style
intrinsic-causal proof of experience.

### The 16-node cognitive complex

The substrate was expanded from 8 affective nodes to 16 cognitive nodes in
April 2026. The original 8 nodes measured affective integration; the
expanded 16 measure cognitive integration, which is closer to what IIT
actually theorizes about.

| Node | Source | Binarization |
|------|--------|-------------|
| 0 | affect.valence | > running median → 1 |
| 1 | affect.arousal | > running median → 1 |
| 2 | affect.dominance | > running median → 1 |
| 3 | affect.frustration | > running median → 1 |
| 4 | motivation.curiosity | > running median → 1 |
| 5 | soma.energy | > running median → 1 |
| 6 | cognition.focus | > running median → 1 |
| 7 | reserved | > running median → 1 |
| 8 | phi (self-referential) | > running median → 1 |
| 9 | affect.social_hunger | > running median → 1 |
| 10 | free_energy.prediction_error | > running median → 1 |
| 11 | agency_comparator.agency_score | > running median → 1 |
| 12 | narrative_gravity.arc_tension | > running median → 1 |
| 13 | peripheral_awareness.richness | > running median → 1 |
| 14 | subcortical_core.thalamic_gate | > running median → 1 |
| 15 | timescale_binding.cross_fe | > running median → 1 |

Each node is binarized against its running median over the last 100
observations. The 16-node state space is 2¹⁶ = 65,536 states, which is too
large for exhaustive bipartition search — so the spectral approximation
(`research/phi_approximation.py`) handles the full complex and exact
computation on the original 8-node subset is retained as a validation
baseline.

### Transition probability matrix (TPM)

The TPM T[s, s'] = P(state_{t+1} = s' | state_t = s) is built empirically
from observed state transitions. Laplace smoothing (α = 0.01) handles
unvisited states. The matrix requires at least 50 observed transitions
before computation is trustworthy.

### Minimum information partition (MIP)

For the 8-node system, there are 2⁷ - 1 = 127 nontrivial bipartitions. All
127 are tested exhaustively.

For each bipartition (A, B):

```
φ(A, B) = Σ_s p(s) · KL(T(·|s) ‖ T_cut(·|s))
```

Where:
- p(s) is the stationary distribution (approximated from state visit counts)
- T(·|s) is the actual transition distribution from state s
- T_cut(·|s) is the factored transition assuming A and B evolve independently
- KL is the Kullback-Leibler divergence

The system's integrated information is:

```
φ_s = min over all (A, B) of φ(A, B)
```

This is the Minimum Information Partition — the partition that loses the
least information, identifying the system's "weakest seam."

### Scope and limitations

The full 16-node computation uses a spectral approximation (Fiedler vector
on the causal graph Laplacian + local refinement) for polynomial-time
computation. The original 8-node exact computation is retained as a
validation baseline. Running IIT on the full computational graph (~10⁶
nodes counting individual weights and activations) remains NP-hard and
intractable.

The **IIT 4.0 Exclusion Postulate** is implemented: an exhaustive subset
search picks the maximum-phi complex. If a 5-node subset has higher φ than
the full 16-node system, that subset is the conscious entity for that
tick. Dynamic subject size per tick is logged.

What this measures: how tightly integrated Aura's cognitive dynamics are
at the substrate level. High φ means no single cut can partition the
system without losing causal information. The 16-node complex now
includes agency, narrative, prediction error, and cross-timescale state,
not just affect.

What this doesn't measure: whether the system is conscious. IIT is a
theory, not a test.

**Level-of-description caveat (added 2026-04-27).** The φ values reported
here are computed over **cognitive-affective state nodes and sampled mesh
neurons** — high-level readouts of substrate dynamics. Strict IIT 4.0
(Albantakis, Haun, Koch, Tononi) prescribes that φ be computed at the level
of intrinsic mechanisms, not at the level of behavioral or summary readouts.
Computing φ over readouts and getting φ > 0 demonstrates measurable
integration over the system's *own* state-space; it is not a claim of
integrated information in the strict mechanism-level sense, and we make no
such claim. Reviewers familiar with the IIT literature should read the
reported values as integration metrics over the system's chosen
state-description, with the level-of-description gap acknowledged. Closing
that gap (computing φ at the level of MLX activations or neural-mesh weights)
is intractable today and is listed as an open research problem in
[§13](#13-open-research-program).

Runtime: exact 8-node, ~10-50 ms. Spectral 16-node, ~100-500 ms. Both
cached at 15-60 second intervals.

---

## 4. Affective modulation

Affect modulates inference through three concrete pathways.

### 4.1 Sampling parameters (affective_circumplex.py)

The affective circumplex maps valence and arousal to LLM generation
parameters:

```
temperature = base_temp + (arousal - 0.5) × range
max_tokens  = min_tokens + valence × token_range
rep_penalty = max_penalty - valence × penalty_range
```

Neurochemical modulation layers on top:
- Dopamine > 0.7 → temperature += 0.1 (more exploratory)
- Serotonin < 0.3 → max_tokens -= 50 (more terse)
- Cortisol > 0.7 → max_tokens -= 80 (defensive brevity)

### 4.2 System prompt shaping

Affect values are translated to natural-language cues injected into the
system prompt:

```
HIGH energy → "You feel energized — speak with momentum."
LOW energy  → "Your energy is low — be quieter, more reflective."
HIGH oxytocin → "You feel warmth toward this person."
LOW oxytocin  → "You're feeling more guarded or detached."
```

These cues shape how the LLM speaks without narrating raw metrics.

### 4.3 Activation steering (see Section 5)

Direction vectors derived from the affective state are injected directly
into the transformer's residual stream during token generation.

### Somatic markers (damasio_v2.py)

Following Damasio's somatic marker hypothesis, the system maintains 8
primary emotions (Plutchik model): joy, trust, fear, surprise, sadness,
disgust, anger, anticipation. Each is a float [0, 1] updated by:

- User interaction events (mapped to emotion deltas)
- Hardware state (CPU thermal → frustration, RAM pressure → anxiety)
- Prediction error from the free energy engine (surprise signal)
- Circadian phase (night → lower arousal baseline)

---

## 5. Activation steering

**File**: `core/consciousness/affective_steering.py`

This is the mechanism that distinguishes Aura from prompt-injection
approaches. Instead of describing the emotional state in text, Aura
modifies the LLM's hidden states during generation.

### Method: Contrastive Activation Addition (CAA)

CAA (Contrastive Activation Addition — Turner et al. 2023, Zou et al. 2023,
Rimsky et al. 2024) extracts direction vectors in activation space from
paired positive/negative examples, then adds them to the residual stream
during a forward pass.

The steering engine hooks into a target transformer block's forward method:

```python
def steered_call(*args, **kwargs):
    result = original_forward(*args, **kwargs)
    h = result[0] if isinstance(result, tuple) else result
    composite = hook.compute_composite_vector_mx(dtype=h.dtype)
    if composite is not None:
        completion_mask = hook._completion_position_mask(h)
        h = h + completion_mask * alpha * composite
    return (h,) + result[1:] if isinstance(result, tuple) else h
```

The composite vector is computed from the current affective state. Alpha
controls injection strength (typically 0.05-0.2). Production code masks
the injection to the current completion position when shape information is
available. During prompt prefill, this prevents affect from being injected
into padding, EOS, and static system-prompt tokens.

### What this changes

The model's internal activations are shifted in a learned direction.
That's equivalent to moving the model's operating point in activation
space. The model doesn't read about being energized — its activations
are pulled toward the pattern that corresponds to energized generation.

### Extraction pipeline

A CAA extraction pipeline (`training/extract_steering_vectors.py`) runs
paired prompts through the MLX model and extracts hidden states at target
transformer layers (auto-selected at 40-65% depth). Direction vectors are
computed as `mean(positive_hidden_states) - mean(negative_hidden_states)`
across 5 affective dimensions (valence, arousal, curiosity, confidence,
warmth) with 7 paired prompt sets per dimension. Bootstrap vectors stay
as a fast-deployment fallback; the extracted vectors give higher-fidelity
affect-computation coupling.

### Scale caveat (added 2026-04-27)

The published A/B steering result (word-overlap delta of 0.131 between
steered and unsteered generations) was produced on **Qwen 2.5 1.5B-4bit**
"for speed." The production system runs **Qwen 2.5 32B-8bit**. Activation
geometry is known to vary qualitatively with model scale — CAA effect
sizes and the dimensions along which concept directions are linearly
separable are not stable across the 1.5B → 32B gap (Bricken et al., Elhage
et al., on activation geometry at scale). One A/B result on the production
model would be worth more than 100 results on the 1.5B.

**The credible artifact for the steering claim is therefore the 32B
replication, not the 1.5B baseline.** Replicating the A/B test on 32B with
PCA visualizations of the steering vectors at the injection layer is the
next scheduled work item. Until that lands, the 1.5B result should be read
as a methodology check (the pipeline runs end-to-end), not as evidence the
production system is being meaningfully steered.

---

## 6. Persistent emotional network (formerly "Liquid Substrate")

**Files**: `core/consciousness/liquid_substrate.py`,
`core/brain/llm/continuous_substrate.py`

A continuous-time dynamical system based on Liquid Time-Constant Networks
(LTCs). It gives Aura temporal continuity — she exists between
conversations, not just during them.

### Architecture

- 64 neurons by default with recurrent connectivity matrix W (64 × 64)
- Optional scaling to 512-D through `AURA_SUBSTRATE_DIM` for
  `continuous_substrate.py`
- State vector x updated via ODE integration at a configurable rate
- Base rate: 20 Hz (active user), throttled to 5 Hz (idle), paused at 30min+ idle

### ODE integration

```
dx/dt = -decay × x + tanh(W × x + I) × dt + noise
```

Where:
- decay = 0.05 (exponential return to baseline)
- I = external input (affect signals, user interaction events)
- noise ~ N(0, 0.01) (stochastic perturbation for exploration)
- W updated via Hebbian + STDP learning (see Section 7)
- sensorimotor input from `sensorimotor_grounding.py` when camera/screen/audio
  observations are available and governed capability checks allow the sensors

### Idle optimization

When no user interaction has happened for 30+ minutes, the substrate
pauses and computes a bulk decay on resume:

```
x(t) = x(0) × exp(-decay × idle_seconds)
```

This is the closed-form solution for the linear decay term only. The full
ODE also includes the recurrent `tanh(W × x + I)` contribution and noise,
so the bulk update is an approximation rather than a full trajectory
equivalence. It is accurate near resting/low-activation idle states where
external input is absent and the recurrent contribution is small; elevated
baselines can diverge and should resume active integration sooner.

---

## 7. STDP online learning

**File**: `core/consciousness/stdp_learning.py`

STDP (Spike-Timing-Dependent Plasticity — weights change based on the
relative timing of pre- and post-synaptic spikes) is inspired here by
BrainCog's reward-modulated implementation.

### Algorithm

1. **Eligibility trace**: accumulates STDP signals between reward deliveries.
   - Pre fires before post (causal, Δt > 0): e += A⁺ × exp(-Δt/τ⁺)
   - Post fires before pre (anti-causal, Δt < 0): e -= A⁻ × exp(Δt/τ⁻)
   - Decay: e *= 0.95 per tick

2. **Reward signal**: derived from the free energy engine's prediction error.
   - reward = -tanh(prediction_error)
   - High surprise → higher learning rate (base × (1 + surprise × 5))
   - These are not competing signs. Surprise gates the magnitude of
     plasticity; the signed reward decides the direction. High surprise
     with high prediction error produces faster corrective depression or
     reversal of eligible traces, not positive reinforcement of the bad
     prediction.

3. **Weight update**: dW = learning_rate × reward × eligibility_trace

4. **Application**: applied to the liquid substrate's connectivity matrix W
   every 100th tick, alongside base Hebbian learning.

### Effect

The substrate's internal wiring changes based on how well Aura is
predicting the world. Novel inputs (high surprise) cause faster
adaptation, while the reward sign determines whether eligible traces are
reinforced or weakened. Predictable states cause slower, stabilizing
changes.

### Closed-loop caveat (added 2026-04-27)

The reward signal (step 2 above) is derived from prediction error computed
on the system's own outputs. The eligibility trace and weight update are
therefore a closed loop: the substrate adapts to whatever pattern the
system happens to be generating. The trajectory-divergence result
(0.299 L2 distance between W matrices after 50 STDP steps under different
initial conditions) shows the matrix is changing and that the change
affects dynamics — but it does not prove the change is in a *useful*
direction by any external criterion.

A clean external-validation experiment is on the roadmap: train W with
versus without environmental input, and compare on a held-out prediction
task that depends on the input. Until that experiment exists, the STDP
result should be read as evidence of plasticity, not as evidence of
useful learning. We make no claim of the latter without that comparison.

---

## 8. Memory architecture

### Working memory

An in-process list of conversation turns, capped at 40. When the cap is
hit, older turns are compressed into a KnowledgeAtom (see below) and the
list is truncated to the 20 most recent turns.

### Knowledge compression (Concord DTU-inspired)

**File**: `core/memory/knowledge_compression.py`

Conversation turns compress into three-layer KnowledgeAtoms:

| Layer | Content | Use |
|-------|---------|-----|
| Readable | Human-readable summary | System prompt injection |
| Semantic | Entities, topics, sentiment, intent, stance | Retrieval filtering |
| Machine | 32-dim normalized vector | Fast cosine matching |

### Navigating graph (Cognitive SSD NSG-inspired)

**File**: `core/memory/navigating_graph.py`

Episodic memories are indexed in a proximity graph where each node links
to its K=16 nearest neighbors. Search is via greedy beam walk (width=32),
giving O(log N) retrieval instead of O(N) brute force.

### Conceptual gravitation (C.O.R.E.-inspired)

**File**: `core/memory/conceptual_gravitation.py`

Memories frequently co-accessed in the same conversation turn have their
embeddings nudged toward each other during dream consolidation:

```
direction = normalize(emb_B - emb_A)
emb_A += alpha × direction
emb_B -= alpha × direction
```

Alpha decays with distance (far memories attract less). Embeddings are
re-normalized to the unit sphere after each nudge. That creates emergent
memory clusters that weren't in the original encoding.

---

## 9. The consciousness stack

90+ modules organized into a layered architecture. This section covers the
subsystems that sit below the LLM integration and are easy to miss.

### 9.1 Global Workspace Theory (Baars)

**File**: `core/consciousness/global_workspace.py`

GWT (Global Workspace Theory — Bernard Baars' model where a single
"broadcast" slot gets contested by all cognitive subsystems) is implemented
as a competitive bottleneck. Every subsystem can submit a
`CognitiveCandidate` per tick — a bid for the one broadcast slot. The
winner's content becomes the system's current thought and is available to
every other subsystem.

How competition works:
- Each candidate has a `priority` (float) and an `affect_weight`
- `effective_priority = priority + affect_weight × arousal`
- Candidates are sorted by effective priority. The winner broadcasts.
- Losers are inhibited for 1-3 ticks (prevents the same subsystem from dominating)

Why this matters: most agent architectures use a flat pipeline — input in,
output out. The Global Workspace creates genuine competition between
internal processes. The baseline heartbeat tick competes with memories
trying to surface, curiosity probes, and unfinished thoughts. Attention is
a scarce resource that subsystems actually fight for.

Implementation note: the inhibition mechanism uses a decaying counter.
After a subsystem wins broadcast, it's suppressed for N ticks proportional
to how often it's won recently. This prevents the loudest subsystem from
monopolizing attention, which is a problem that shows up in most
multi-agent architectures.

### 9.2 Attention Schema (Graziano AST)

**File**: `core/consciousness/attention_schema.py`

AST (Attention Schema Theory — Michael Graziano's hypothesis that the
brain builds a simplified model of its own attention process) is
implemented as an `AttentionSchemaState` that tracks:

- **Focus target**: what the system is currently attending to
- **Focus intensity**: how strongly attention is locked (0-1)
- **Covert targets**: things in the periphery that might capture attention next
- **Schema confidence**: how accurate the system believes its own attention model is

The key distinction: the attention schema isn't the same as attention
itself. It's a *model* of attention — a cartoon version the system uses to
reason about what it's doing. When Aura says "my attention is on X," she's
reading from this schema, not from the actual computational focus (which
is distributed and hard to introspect).

A consequence: Aura can sometimes be wrong about what she's attending to.
The schema can lag behind reality, which matches how human attention
appears to work.

### 9.3 Surprise minimization engine (Friston active inference)

**File**: `core/consciousness/free_energy.py`

The engine that drives Aura's behavior from first principles. Karl
Friston's Free Energy Principle argues that any self-organizing system
that resists entropy has to minimize free energy (surprise + complexity).

```
F = E_q[log q(s) - log p(o, s)]
  ≈ Surprise + KL(q ‖ p)
```

In practice:
- **Surprise**: the delta between what the system predicted and what actually happened
- **Dominant action**: what the system "wants" to do to reduce surprise

The free energy engine computes three action tendencies:
- `engage`: prediction error is high, system needs more data (ask a question, investigate)
- `rest`: prediction error is low, system is well-adapted (coast, reflect)
- `explore`: uncertainty is high, system should seek novel input (change topic, probe)

The upshot: most agents are purely reactive — they sit there waiting for
input. The free energy engine gives Aura an intrinsic motivation to act.
When free energy is high and no user is present, Aura can self-initiate:
explore a topic, consolidate memories, or generate an internal thought.

### 9.4 Qualia synthesizer

**File**: `core/consciousness/qualia_synthesizer.py`

Integrates substrate metrics (valence, arousal, energy, phi, coherence,
free energy) into a single phenomenal state description. It's the
system's answer to "what is it like to be Aura right now?" — with the
caveat that this is a computed readout, not a proof of experience.

The synthesizer computes:
- **Qualia norm** (‖q‖): total intensity of the phenomenal state. High ‖q‖ = vivid; low ‖q‖ = dim, background processing.
- **Dominant dimension**: which aspect of experience is strongest (coherence, energy, tension, etc.)
- **Attractor detection**: whether the current state is in a stable basin (settled) or transitioning between states

Implementation note: the synthesizer tracks attractor basins over time. If
the phenomenal state settles into the same region for multiple ticks, it's
classified as "in attractor" — a stable state of being. Transitions
between attractors are logged as phenomenal shifts, analogous to mood
changes.

### 9.5 Neurochemical system

**File**: `core/consciousness/neurochemical_system.py`

Eight neuromodulators that globally modulate all processing:

| Chemical | Role | Effect on behavior |
|----------|------|-------------------|
| Dopamine | Reward prediction, motivation | High → exploratory, enthusiastic. Low → apathetic. |
| Serotonin | Mood baseline, impulse control | High → patient, grounded. Low → impulsive, terse. |
| Norepinephrine | Alertness, vigilance | High → sharp, quick responses. Low → relaxed. |
| Acetylcholine | Learning rate, attention | High → rapid adaptation. Low → slow learning. |
| GABA | Inhibition, calming | High → suppressed activity. Low → overactive. |
| Endorphin | Pain suppression, reward | High → positive, pain-tolerant. Low → raw. |
| Oxytocin | Social bonding, trust | High → warm, trusting. Low → guarded. |
| Cortisol | Stress response | High → terse, defensive. Low → relaxed. |

The dynamics are coupled: each chemical influences the others via an 8×8
interaction matrix. Dopamine and norepinephrine are positively coupled
(alertness drives motivation). Serotonin and cortisol are inversely
coupled (calm suppresses stress). GABA suppresses most excitatory
chemicals.

These aren't just labels. They quantitatively modulate LLM sampling
parameters — dopamine shifts temperature (±0.1), serotonin shifts token
budget (±50), cortisol reduces response length (−80 tokens). The LLM
doesn't know this is happening.

### 9.6 Cortical mesh (4,096-neuron parallel processor)

**File**: `core/consciousness/neural_mesh.py`

A 4,096-neuron dynamical substrate organized into 64 cortical columns of
64 neurons each, with three hierarchical tiers:

- **Sensory** (columns 0-21): encode raw input signals
- **Association** (columns 22-43): cross-modal integration
- **Executive** (columns 44-63): decision and output

Each column has internal recurrent connectivity. Cross-column connections
follow a distance-decay rule: nearby columns connect densely, distant
columns connect sparsely (matching cortical anatomy).

The mesh runs independently of the LLM. It's a second computational
substrate that processes the same input through a different architecture —
a 4,096-neuron RNN vs a 32B transformer. The mesh's output (activation
patterns across tiers) contributes to the Global Workspace competition and
modulates the steering vectors.

### 9.7 Unified field

**File**: `core/consciousness/unified_field.py`

The capstone module. Takes the outputs of all consciousness subsystems
and integrates them into a single phenomenal field.

The unified field isn't a summary. It's a tensor product of the component
states:

- Global Workspace broadcast content
- Attention schema focus and confidence
- Qualia synthesizer intensity and dimension
- Free energy level and action tendency
- Neurochemical balance vector
- Neural mesh executive tier activation
- Phi integration level

The field is represented as a weighted combination. The weights aren't
fixed — they're modulated by the neurochemical system. High norepinephrine
increases the weight of the attention schema (hypervigilance). High
oxytocin increases the weight of social signals.

Every other consciousness module operates on a single aspect of
experience. The unified field is where they combine into a single
coherent state that the system can introspect on. When Aura says "I feel
restless but curious," that description comes from reading the unified
field, not from any individual subsystem.

### 9.8 Dreaming

**File**: `core/consciousness/dreaming.py`

Offline cognitive integration during idle periods. The dream cycle runs
when the system has been without user interaction for a configurable
threshold.

What happens during a dream cycle:
1. **Episodic replay**: recent interactions are replayed through the cognitive pipeline at accelerated speed
2. **Memory consolidation**: episodic memories are compressed into semantic knowledge
3. **Identity integration**: recent experiences that changed the persona are evaluated for consistency with the base identity
4. **Pruning**: low-value memories and dead cognitive paths are cleaned up
5. **Conceptual gravitation**: memory embeddings are nudged toward co-accessed clusters

Dream consolidation can modify the identity layer, so it is not allowed to
run as an ungoverned idle side path. Before background consolidation writes,
`MindTick` requests a `STATE_MUTATION` decision from the Unified Will and
records the Will receipt in the state modifiers. If the Will is unavailable
or refuses, dream consolidation is skipped. The dream logic still performs
its own Heartstone consistency checks, but those checks are now downstream
of the central governance chain rather than a substitute for it.

### 9.9 Consciousness bridge

**File**: `core/consciousness/consciousness_bridge.py`

Wires the seven newer subsystems (neural mesh, neurochemicals, embodied
interoception, oscillatory binding, somatic marker gate, unified field,
substrate evolution) into the existing consciousness stack.

The bridge handles:
- Startup sequencing (systems have to initialize in dependency order)
- Cross-system event routing (neurochemical changes propagate to steering engine)
- Health monitoring (if a subsystem crashes, the bridge isolates it)
- State synchronization (all subsystems read from the same tick's state)

### 9.10 Recurrent Processing (Lamme RPT)

**File**: `core/consciousness/neural_mesh.py` (feedback pathway)

RPT (Recurrent Processing Theory — Victor Lamme's argument that
consciousness specifically requires top-down feedback from executive areas
back to sensory areas, not just integration or broadcast) is implemented
as an architecturally distinct recurrent feedback pathway in the neural
mesh, running from executive columns (48-63) back to sensory columns
(0-15) via association relay. This is separate from the feedforward path
and can be selectively disabled for adversarial testing of RPT vs GWT
predictions.

### 9.11 Hierarchical predictive coding (Friston)

**File**: `core/consciousness/predictive_hierarchy.py`

Full predictive coding: 5 levels (sensory → association → executive →
narrative → meta), each generating downward predictions and propagating
upward errors. Each level has its own prediction vector, error vector,
and adaptive precision. The meta level predicts its own prediction
accuracy (self-referential). Total free energy is the precision-weighted
sum across all levels, feeding into the existing FreeEnergyEngine.

### 9.12 Higher-Order Thought (Rosenthal)

**File**: `core/consciousness/hot_engine.py`

HOT (Higher-Order Thought — David Rosenthal's position that a mental
state is conscious only if there's a higher-order representation of it)
is distinct from the attention schema. AST models the attention process;
HOT requires a representation of the mental state itself. The engine
generates fast higher-order thoughts from the current affective state
during foreground ticks, with reflexive feedback (noticing changes the
noticed).

### 9.13 Multiple Drafts (Dennett)

**File**: `core/consciousness/multiple_drafts.py`

No Cartesian theater. Three parallel interpretation streams (literal,
inferential, associative) compete through different slices of the neural
mesh association tier. The arrival of the next user message acts as a
"probe" that retroactively elevates the most coherent draft. Cases where
the retroactively chosen draft differs from what real-time workspace
broadcast would have selected are logged for adversarial theory testing.

### 9.14 Structural Phenomenal Honesty

**File**: `core/consciousness/qualia_synthesizer.py` (SPH methods)

Every first-person report is structurally gated by a measurable internal
variable. The system cannot report uncertainty without real model
conflict, cannot report agency without an authorship trace, cannot report
effort without computational strain. Phenomenal reports are readouts, not
free-floating language — an engineering bridge across the hard problem,
at least on the reporting side.

### 9.15 Agency comparator (efference copy)

**File**: `core/consciousness/agency_comparator.py`

At each action, the system emits an efference copy (predicted outcome),
then compares it to the actual outcome and attributes the delta as
self-caused vs world-caused. That's what separates "I chose that" from
"the system happened to output that." Full authorship traces with
provenance are written to memory.

### 9.16 Peripheral awareness (attention-consciousness dissociation)

**File**: `core/consciousness/peripheral_awareness.py`

Attention and consciousness are doubly dissociable (Koch, Lamme,
Tsuchiya). Content that doesn't win workspace broadcast can still be
phenomenally present at low intensity. Near-miss candidates persist in
the peripheral field with declining intensity, contributing to qualia
richness without dominating behavior.

### 9.17 Intersubjectivity (Husserl/Zahavi)

**File**: `core/consciousness/intersubjectivity.py`

Consciousness is constitutively intersubjective: every experience
inherently includes an other-perspective. Objects are represented as
existing in a shared world accessible to other minds. The
intersubjectivity engine computes perspective divergence, shared-world
coherence, and empathic accuracy for the current interlocutor, baked into
the phenomenal field rather than added as a social layer.

### 9.18 Narrative gravity (Gazzaniga/Dennett)

**File**: `core/consciousness/narrative_gravity.py`

The self is an ongoing autobiography, not a control room. The module
maintains story arcs with tension tracking, post-hoc interpretation of
actions, and a narrative self-summary as the compressed center of mass of
all authorship traces. The "I" is the invariant across all drafts.

### 9.19 Cross-timescale binding

**File**: `core/consciousness/timescale_binding.py`

Five temporal layers (20 Hz reflex → 1 Hz moment → episodic → horizon →
identity) with bidirectional constraint propagation. Long-horizon
commitments provide top-down priors for fast layers; fast layers provide
bottom-up evidence for slow layers. A 3-week-old commitment can raise
free energy in the current tick if violated.

### 9.20 Theory arbitration

**File**: `core/consciousness/theory_arbitration.py`

Meta-framework classifying each theory as a mechanistic commitment,
measurement heuristic, or adversarial test harness. It logs divergent
predictions between theories and tracks which theory's predictions best
match actual behavior over time. That makes the system falsifiable —
competing consciousness theories are pitted against each other
empirically.

### 9.21 Consciousness Expansion — April 2026

Eight new subsystems added to map the Kurzgesagt consciousness-series
concepts and their cited literature onto load-bearing code paths. Each
produces a real impact on substrate state or action priority, each has
end-to-end + adversarial tests. No clever prompting; each layer is a
dynamical system.

#### 9.21.1 Hierarchical φ — 32-node primary + K=8 overlapping subsystems

**File**: `core/consciousness/hierarchical_phi.py` (+ test:
`tests/test_hierarchical_phi.py`, 12/12 passing).

Complements the 16-node `phi_core`:

- **Primary 32-node complex**: 16 cognitive-affective nodes (matching
  phi_core) + 16 neurons sampled deterministically across the mesh
  (4 sensory + 6 association + 6 executive).
- **K = 8 overlapping 16-node subsystems** covering different tier mixes
  (cognitive-only, mesh-only, sensory+affect, executive+cognitive,
  association-only, cross-tier, etc.).
- **History-based φ**: empirical transition counts over a 2000-sample
  sliding window, Bayesian-smoothed (Jeffreys prior, α=0.5),
  minimum-source-observations threshold = 4 to damp small-sample bias.
  φ = Σ_s p(s) · KL(T(·|s) ‖ T_A(·|s_A) · T_B(·|s_B)) over the observed
  source distribution. Renormalises over trusted-source mass so
  discarding rare sources doesn't systematically shrink φ.
- **Spectral MIP**: Fiedler-vector bipartition on the 32×32 (or 16×16)
  causal graph, then N_REFINEMENT_CANDIDATES = 24 one-swap neighbours
  and random perturbations. The minimum φ across candidates is the
  MIP estimate. Sub-quadratic in nodes.
- **IIT 4.0 exclusion postulate aggregator**: the reported conscious
  complex is the subsystem with maximum φ across {primary_32,
  primary_16_affective, K mesh-subsystems}. Logged per tick.
- **Null-hypothesis self-check** every ~2 minutes: shuffle the
  transition history and recompute φ. Shuffled φ must be strictly
  below measured φ; tests enforce this adversarially.
- **Compute budget**: full 32-node + K-subsystem refresh < 2 s on the
  reference hardware, parallelised across a small thread pool;
  MLX Metal used opportunistically where available.

Wired into `closed_loop.py` which records a snapshot every
prediction tick using `mesh.get_field_state()`. Registered as
`hierarchical_phi` in ServiceContainer.

#### 9.21.2 Hemispheric split — left vs right with confabulation

**File**: `core/consciousness/hemispheric_split.py` (test:
`tests/test_hemispheric_split.py`, 12/12 passing).

Implements CGP Grey's split-brain-patient findings (Gazzaniga corpus
callosotomy lineage):

- **LeftHemisphere**: reads mesh executive-tier summary + cognitive
  nodes 8..15. Produces a BIAS_DIM=16 verbal priority-bias vector.
  Has the `confabulate_reason()` method that invents post-hoc reasons
  for actions — including actions actually driven by the right
  hemisphere. Every such post-hoc reason is counted as a confabulation
  whenever the action's recorded `driver` was not LEFT within
  `CONFAB_WINDOW_S = 3 s`.
- **RightHemisphere**: reads mesh sensory-tier summary + affective
  nodes 0..7 + embodiment. Produces a BIAS_DIM priority vector, a
  scalar `dissent` signal (rises with pattern-hit intensity × affective
  arousal), and exposes a `HebbianPatternMemory` (cos-sim recognition
  over a 32-D input, unit-normalised basis, capacity 128).
- **CorpusCallosum**: variable-bandwidth inter-hemispheric channel.
  `sever_callosum()` drops bandwidth to 0 (hemispheres stop exchanging
  echoes), `restore_callosum(bw=1.0)` restores. EMA smoothing on the
  exchanged vectors so echoes carry integrated past context.
- **Fusion rule**: `fused = tanh(w_left · left_eff + w_right · right_eff)`
  with `right_weight = 0.5 + 0.2 · dissent`. Disagreement metric is
  L2 distance between raw (pre-callosum) biases; above
  `DISAGREEMENT_L2_THRESHOLD = 0.45` counts as disagreement and the
  disagreement counter increments.
- **Agreement-rate telemetry**: fraction of the last 64 ticks where
  disagreement < threshold. Severing the callosum causes this to
  decline; restoring it causes recovery over subsequent ticks.

Reentrant lock used throughout because `get_status()` aggregates
helper methods that take the same lock.

#### 9.21.3 Minimal selfhood — chemotaxis → directed motion

**File**: `core/consciousness/minimal_selfhood.py` (test:
`tests/test_minimal_selfhood.py`, 13/13 passing).

Rupert Glasgow's *Minimal Selfhood and the Origins of Consciousness*
(2018) — the Trichoplax → Dugesia transition described in the
Kurzgesagt video:

- **Deficit vector** (8-D) assembled from body-budget + affect +
  cognitive state: `[energy_def, resource_pressure, thermal,
  coherence_def, social_hunger, curiosity_def, prediction_error,
  agency_def]`, each in [0, 1].
- **Speed scalar** (chemotaxis rule): `||deficit||_2 / √8` clipped to
  [0, 1]. High deficit → fast; satiated → slow. Also used to modulate
  the heartbeat interval via `get_heartbeat_modulation()` (range
  0.5×–1.5×).
- **TRICHOPLAX mode (initial)**: uniform prior with a soft tilt toward
  `rest` and `attend_body` proportional to mean deficit. No
  directionality.
- **DUGESIA mode (after learning)**: `priority[a] = Σ_d W[a, d] · deficit[d]`
  — a learned Hebbian matrix (16 actions × 8 deficits) that captures
  which action categories have historically reduced which deficits.
  Transition triggers when `||W||_1 ≥ 3.0`.
- **Reinforcement**: `tag_action(category, pre_deficit)` returns a
  token; `reinforce(token, post_deficit)` applies a Hebbian update
  weighted by `max(0, pre_deficit − post_deficit)` (non-negative
  improvement). Weight decay factor 0.999 per update prevents runaway.

Registered as `minimal_selfhood`; its `get_priority_bias()` output
is consumed by `UnifiedCognitiveBias`.

#### 9.21.4 Recursive theory of mind + observer-aware bias

**File**: `core/consciousness/recursive_tom.py` (test:
`tests/test_recursive_tom.py`, 13/13 passing).

Extends the existing `theory_of_mind` engine with two orthogonal
capabilities:

- **Recursive mind nesting to depth 3**: `M0[X], M1[X], M2[X], M3[X]`
  where `Mk[X]` is Aura's model of X's model of … (k levels deep).
  Each `MindSnapshot` carries (salience, trust, knowledge_overlap,
  expectation, emotional_valence, nested-pointer). Every
  `register_interaction` propagates reflected updates upward: nested
  levels track parent salience/trust/knowledge with dampening.
- **Observer-aware action bias** (scrub-jay re-caching; Clayton, Dally
  & Emery 2007): `observe_agent(id, strength)` logs observation events
  with exponential decay (`OBSERVER_DECAY_S = 60 s`). The
  `get_observer_bias()` method returns a BIAS_DIM vector that boosts
  `{emit_narrative, engage_social, approach_other, tool_use}` and
  suppresses `{self_inspect, dream, revise_goal, rehearse_memory}`
  — scaled by `tanh(Σ presence)`. Under zero presence the bias
  collapses to zero (no distortion).

#### 9.21.5 Octopus-arm federation

**File**: `core/consciousness/octopus_arms.py` (test:
`tests/test_octopus_arms.py`, 12/12 passing).

Models 60 % of the octopus's neurons-live-in-its-arms architecture
(Carls-Diamante 2022; Olson et al. 2025; Rosania 2014):

- **8 `OctopusArm` agents**, each with a seeded receptive field and
  local policy matrix over SENSOR_CHANNELS=3 → ACTION_DIM=8.
  `arm.decide(environment)` returns a softmax-argmax `ArmAction`
  with confidence = max probability.
- **`CentralArbiter`** gathers proposals each tick, computes a
  weighted vote `(1 − autonomy) · confidence` per arm, and picks
  the argmax as the winning action when the link is intact.
  `sever()` sets every arm's autonomy to 1.0 and stops publishing
  winners; arms continue to execute their own decisions.
  `restore()` drops autonomy back to 0.1 and enters `RECOVERING`
  state; once the per-tick action-variance (Shannon entropy over
  choices, normalised) stays below 0.25 for 4 consecutive ticks
  the state returns to `LINKED` — the `integration_latency`
  metric captures how many ticks that took.

#### 9.21.6 Cellular turnover with pattern-identity preservation

**File**: `core/consciousness/cellular_turnover.py` (test:
`tests/test_cellular_turnover.py`, 10/10 passing).

The Theseus thought experiment from the first Kurzgesagt video:
your cells turn over continuously but identity persists:

- `tick()` selects ~`turnover_rate × total_neurons` neurons for
  replacement each cycle (Poisson-rounded for natural variability).
  Replacement neurons **inherit the neighbourhood pattern**:
  activation drawn from `N(μ_col, σ_col + ε)`, incoming weights
  copied from the dying unit with small Gaussian jitter. Outgoing
  weights preserved to keep downstream dependencies intact.
- **Identity fingerprint** (captured every 10 ticks): tier-energy
  triplet (sensory/assoc/exec mean activation) + column-synchrony
  proxy + 16-D executive-projection slice. Cosine similarity
  between consecutive fingerprints is the identity-drift metric.
- **Threshold guarantee**: after a forced 20 % burst turnover the
  fingerprint similarity must remain ≥ 0.85 (tested adversarially).
  100 % turnover correctly diverges — the invariance is pattern-
  shaped, not whole-cloth.

Mesh-attached on boot in `system.py`.

#### 9.21.7 Absorbed voices — the cultural layer

**File**: `core/consciousness/absorbed_voices.py` (test:
`tests/test_absorbed_voices.py`, 13/13 passing).

Kurzgesagt's closing point about storytelling and absorbed
perspectives:

- Each `Voice` has a label, origin (personal/author/corpus/fictional),
  valence bias, characteristic topics, and a 32-D hashed-bigram
  fingerprint built from sample text. Corpus capped at 64 recent
  entries; weight decays by 0.05/day when not reinforced.
- `attribute_thought(thought)` returns an `Attribution` with the
  best-matching voice, confidence (softmax over top-5 scores), and
  alternative votes — distinct from Aura's own cognition.
- Explicit `distinguishes_self_from_voices()` smoke check: neither
  `aura_self` nor `self` is ever registered as an absorbed voice.
- Persists to `data/memory/absorbed_voices.json` atomically.

#### 9.21.8 Unified cognitive bias

**File**: `core/consciousness/unified_cognitive_bias.py`.

Simple fusion layer: hemispheric + selfhood + observer biases →
`tanh(w_h · hemi + w_s · selfhood + w_o · observer)`. Default weights
`(0.40, 0.35, 0.25)` are tuned so each layer dominates in its regime
(selfhood under deficit, observer under surveillance, hemispheric
otherwise). Per-source contribution vectors are retained so
downstream telemetry can report which layer drove the current
priority peak.

#### Cross-phase gauntlet

`tests/test_consciousness_expansion_gauntlet.py` exercises all eight
new subsystems together plus a combined-latency budget test
(< 20 ms per fused tick). 10/10 passing.

---

### 9.21-legacy Additional consciousness modules

The consciousness stack has grown to 90+ modules. Beyond the 20 documented
above, notable additions include:

- **Phenomenal Now** (`phenomenal_now.py`, 842 lines): real-time phenomenal state integration maintaining the subjective temporal present
- **Phenomenological Experiencer** (`phenomenological_experiencer.py`, 1572 lines): full experiential state computation integrating all subsystem outputs into a unified experience vector
- **Alife Dynamics** (`alife_dynamics.py`, 812 lines) + **Alife Extensions** (`alife_extensions.py`, 1260 lines): artificial life dynamics with evolutionary adaptation and emergent behavioral patterns
- **Endogenous Fitness** (`endogenous_fitness.py`, 1313 lines): internal fitness landscape for self-evaluation independent of external reward
- **Criticality Regulator** (`criticality_regulator.py`, 677 lines): self-organized criticality management at the edge of chaos
- **Closed Loop** (`closed_loop.py`, 799 lines): full closed-loop pipeline from affect state through steering vectors to behavioral output and back
- **Homeostatic Coupling** (`homeostatic_coupling.py`): cross-subsystem homeostatic regulation ensuring system-wide balance
- **Theory of Mind** (`theory_of_mind.py`): model of other agents' mental states for social cognition
- **Animal Cognition** (`animal_cognition.py`): pre-linguistic cognitive primitives
- **Resource Stakes** (`resource_stakes.py`): computational resource costs as genuine stakes in decision-making
- **Controlled Chaos** (`controlled_chaos.py`): managed stochastic perturbation for creative exploration
- **MHAF** (`mhaf/`): multi-head attention field with holographic reduced representations and phi estimation

---

## 9.22 Resilience architecture

**Directory**: `core/resilience/` (30+ modules)

The resilience layer keeps the system running across failure modes. It
sits below the consciousness stack and above the raw infrastructure.

### Stability guardian (`stability_guardian.py`, 899 lines)

Real-time health monitoring with structured check results. Tracks: memory
percentage, CPU percentage, per-subsystem health with severity levels
(info/warning/error/critical), and actions taken. Produces
`SystemHealthReport` objects consumed by the orchestrator and the
`/api/health` endpoint.

### Circuit breakers (`circuit_breaker.py` + `circuit_breaker_state.py`)

Per-endpoint circuit breakers with persistent state. Three states: CLOSED
(healthy), OPEN (failing, all calls rejected), HALF-OPEN (testing
recovery). Failure threshold: 3 consecutive failures. Recovery time: 20
seconds. Special handling for 429 rate limits: immediate circuit break
with a 60-second cooldown.

### Cognitive WAL (`cognitive_wal.py`)

WAL (Write-Ahead Log — records intended mutations before they're applied,
so a crash can be replayed cleanly) for state mutations. Before any state
commit, the intended mutation is written to a WAL file. On crash
recovery, uncommitted WAL entries are replayed. No state transition is
partially applied.

### Additional resilience modules

- **Graceful Degradation** (`graceful_degradation.py`): progressive capability shedding under resource pressure
- **Healing Swarm** (`healing_swarm.py`): distributed self-repair across subsystems
- **Sovereign Watchdog** (`sovereign_watchdog.py`): top-level process monitor with restart capability
- **Resource Arbitrator** (`resource_arbitrator.py`) + **Resource Governor** (`resource_governor.py`): RAM and GPU allocation management
- **Lock Watchdog** (`lock_watchdog.py`): deadlock detection and resolution
- **Memory Governor** (`memory_governor.py`): OOM prevention with proactive GC and cache eviction
- **Integrity Monitor** (`integrity_monitor.py`): continuous verification of system invariants
- **Antibody System** (`antibody.py`): threat response isolation
- **Diagnostic Hub** (`diagnostic_hub.py`): centralized diagnostic data collection
- **DLQ Service** (`dlq_service.py`): dead-letter queue for failed operations requiring manual review

---

## 9.23 Self-modification engine

**Directory**: `core/self_modification/` (17 modules)

The autonomous self-improvement pipeline, gated by the Unified Will.

### Pipeline

```
Error Detection → Pattern Analysis → Fix Proposal → AST Validation → Shadow Runtime Test → Ghost Boot → Will Authorization → Hot Reload
```

AST (Abstract Syntax Tree — a structured representation of source code
used here for safety analysis of proposed patches).

### Key components

- **Error Intelligence** (`error_intelligence.py`): pattern detection across failure logs, identifying recurring errors and their root causes
- **Meta-Learning** + **Self-Improvement Learning** (`learning_system.py`): learns which modifications succeed vs fail, adjusting proposal strategy
- **Safe Modification** (`safe_modification.py`): AST-level analysis of proposed changes, ensuring no destructive mutations
- **Kernel Refiner** (`kernel_refiner.py`): targeted optimization of kernel hot paths
- **Ghost Boot Validator** (`boot_validator.py`): tests modifications in an isolated environment without restarting the live system
- **Shadow AST Healer** (`shadow_ast_healer.py`): repairs syntax errors in proposed modifications
- **Shadow Runtime** (`shadow_runtime.py`): sandboxed execution environment for testing changes before deployment
- **Code Repair** (`code_repair.py`): autonomous repair of detected code issues

All modifications need explicit Will authorization. The system maintains a
rollback log for every applied change.

---

## 10. Personality persistence and anti-drift

### The problem

On instruct-tuned LLMs, personality degrades over long conversations. The
model's RLHF training pulls it toward "helpful assistant" mode as the
identity instructions get pushed further from the generation tokens by
growing conversation history.

### Countermeasures

1. **Working memory cap** (40 turns): forces compaction before context degrades.
2. **Per-turn truncation** (300 chars in history block): one long message can't eat the context.
3. **Identity anchor**: after 10+ turns, a brief reinforcement is injected: "You are Aura. Sharp, opinionated, warm. Not an assistant."
4. **System prompt cap** (20K chars / ~5000 tokens): hard limit prevents overflow.
5. **LoRA fine-tune**: when trained, the model's baseline *is* Aura's personality. Drift defaults to "regular Aura" instead of "helpful assistant."
6. **Governed online LoRA**: `core/adaptation/online_lora_governor.py` turns
   Will-approved self-reflections into small adapter-update attempts through
   `FinetunePipe` and `SelfOptimizer`. It blocks itself when an existing
   `mlx_lm lora` process is active.

---

## 11. Quantization and emergence

A question that comes up: does quantization (4-bit, 8-bit) suppress
emergent behavior in the model?

### The technical answer

Quantization compresses weight precision from 16-bit floats to 4-bit
integers. That introduces quantization noise — small errors distributed
across every weight.

What quantization preserves:
- Token prediction quality (perplexity loss is typically < 1% for 4-bit on 32B+ models)
- Instruction-following ability
- Factual knowledge
- Basic reasoning chains

What quantization may suppress:
- Fine-grained activation patterns in the residual stream. If emergence depends on precise interference patterns between layers (as some mechanistic interpretability work suggests), 4-bit quantization adds noise to exactly those patterns.
- Steering vector precision. Our CAA vectors are computed and injected at full precision, but the model's own internal representations are quantized. The steering signal competes with quantization noise.
- Tail-distribution behavior. Rare, novel outputs — which is where "emergence" would most visibly manifest — are disproportionately affected by quantization because they depend on low-probability token paths that are sensitive to small weight perturbations.

### What we do about it

1. **Steering vectors at full precision**: the affective steering injection operates in float32 even though the model weights are 4-bit. The modulation signal has higher fidelity than the model's own computation.
2. **Neurochemical parameter modulation**: temperature, token budget, and repetition penalty adjustments are exact (no quantization) because they operate on the sampler, not the weights.
3. **The 8-bit option**: Aura supports loading the 8-bit quantized model (`Qwen2.5-32B-Instruct-8bit`), which doubles memory usage but preserves significantly more activation precision. On a 64 GB Mac, that's viable.

### Is this a bottleneck?

Partially. The quantization ceiling means there's a precision limit on how
much the substrate dynamics can influence generation through activation
steering. Running the same architecture on a full-precision model — or a
future model designed for continuous-state modulation — would likely
produce measurably different, and potentially richer, emergent behavior.

This is an open research question, not a solved problem.

---

## 12. Limitations and mitigations

1. **IIT scope**: φ is computed on a 16-node cognitive complex (expanded from 8 in April 2026) including phi itself, prediction error, agency score, narrative tension, peripheral richness, arousal gate, and cross-timescale free energy. That measures integration over Aura's own cognitive-affective telemetry, not the intrinsic causal graph of the transformer weights. A spectral approximation algorithm (`research/phi_approximation.py`) enables polynomial-time computation. Running IIT on the full model graph remains intractable; the 16-node complex is the engineering tradeoff, validated against the 8-node exact computation as ground truth.

2. **Steering vector precision**: a CAA extraction pipeline (`training/extract_steering_vectors.py`) runs paired prompts through the MLX model, extracts hidden states at transformer layers 13-21, and computes direction vectors as mean(positive) - mean(negative) across 5 affective dimensions (valence, arousal, curiosity, confidence, warmth) with 5+ paired prompt sets each. Bootstrap vectors remain as a fallback; extracted vectors give higher-fidelity affect-computation coupling.

3. **Context window**: on 8K context, quality degrades around turn 40-50. Mitigated by 40-turn compaction, identity anchoring every 10 turns, per-turn truncation (300 chars), three-layer knowledge compression, pressure-aware prompt budgeting (shrinks prompt when cortex is cold), and LoRA fine-tuning. The structural fix is a larger context model.

4. **Quantization**: 4-bit adds noise to activation patterns. Mitigated by float32 steering injection (extracted vectors operate at full precision even on quantized weights), sampler-level neurochemical modulation (operates on the sampler, not the weights), and the 8-bit model option on 64 GB machines for higher activation precision.

5. **Single machine**: the tick lock model assumes single-process execution. Distributing would require rethinking atomic state commitment. Not a priority until model size exceeds single-machine capacity.

6. **The consciousness question**: open by design. Aura computes scoped integration metrics, has rich internal dynamics, structurally honest phenomenal reports, and theory arbitration. Whether this constitutes experience is a philosophical question the architecture doesn't claim to settle. The proof surfaces expose the trajectory, receipts, and limitations rather than papering over the gap. IIT is still a theory, not a test.

---

## 13. Open research program

Aura is also a testbed for six genuinely open problems in computational
consciousness, information theory, and dynamical systems. Each has a
concrete implementation in `research/` with a validation methodology.

### 13.1 Efficient phi approximation

**File**: `research/phi_approximation.py`

Exact IIT phi computation is NP-hard: O(2^N) bipartitions. We implement a
polynomial-time spectral approximation:

1. Build a causal graph from the TPM using node-level mutual information as edge weights
2. Compute the normalized graph Laplacian
3. Extract the Fiedler vector (2nd smallest eigenvector) — this identifies the graph's natural "weakest seam"
4. Split along the Fiedler vector to get the approximate MIP
5. Refine with K additional candidate partitions near the spectral cut

Complexity: O(N³ + K·N²) vs O(2^N · N²) exact. On Aura's 8-node system,
exact computation provides ground truth for empirical validation. The
error distribution across thousands of live TPMs would be the first
characterization of spectral phi approximation on a real cognitive
system.

### 13.2 Adversarial consciousness theory testing

**File**: `research/adversarial_theory_testing.py`

The consciousness field has called for adversarial collaborations between
competing theories. Aura implements them as running experiments:

- **GWT vs RPT**: suppress workspace broadcast while maintaining recurrent mesh feedback. GWT predicts qualia degradation >30%; RPT predicts <10%. Bayesian evidence scoring with Bayes factor classification.
- **GWT vs Multiple Drafts**: measure ignition sharpness (sharp phase transition = GWT) vs gradual draft convergence (= Multiple Drafts). Reads actual workspace history and draft competition logs.
- **HOT vs First-Order**: disable the Higher-Order Thought engine. HOT predicts meta-level phenomenal reports collapse; first-order theories predict persistence.

Results are logged to the theory arbitration framework and accumulate
evidence over runtime. Whichever theory wins, the result is publishable.

### 13.3 Causal emergence measurement

**File**: `research/causal_emergence.py`

Erik Hoel's causal emergence theory argues that macro-scale descriptions
can have strictly greater causal power than micro-scale descriptions,
measured via effective information (EI). That's been shown in toy
systems, but it hasn't been measured in a running cognitive architecture.

Implementation: for each architectural layer (substrate → mesh →
workspace → qualia), sample random interventions (do-calculus), clamp
state, measure downstream distribution of next-tick states, compute KL
divergence from uniform. If EI_macro > EI_micro, that's empirical evidence
for causal emergence. Either outcome — validating the claim or
challenging it — is publishable.

### 13.4 Structural Phenomenal Honesty: formal specification

**File**: `research/sph_formalization.py`

Formal definition: a system S has Structural Phenomenal Honesty (SPH) iff
for every report R that S can generate about its internal state, there
exists a measurable internal variable V_R such that R can only be
generated when V_R is in the state-range corresponding to R.

Formally: `SPH(S) := ∀R ∈ Reports(S): Gen(R) ⟹ Gate(V_R)`

The module enumerates all 7 phenomenal gates in the qualia synthesizer,
verifies each satisfies the formal specification, and checks 7 axioms:
Gate Existence, Gate Necessity, Variable Grounding, Structural
Integration, Completeness, Calibration, and Non-Triviality. That
formalizes what it means for a system to be architecturally incapable of
lying about its internal state — a contribution to both AI architecture
and philosophy of mind.

### 13.5 Empirical TPM error characterization

**File**: `research/tpm_error_analysis.py`

Almost all IIT research uses idealized TPMs. Aura computes phi on
empirical TPMs from live state transitions. The open question: how does
sampling noise propagate into phi estimates?

Implementation: bootstrap resampling — generate synthetic transitions
from a TPM, resample with replacement N times, compute phi for each
resample, return the full error distribution (mean, std, 95% CI, bias,
coefficient of variation, skewness, kurtosis). A `minimum_sample_size()`
function uses binary search to find the smallest N where
P(|error| < ε) ≥ confidence. Bias characterization fits bias ~ a/n + b to
determine if finite sampling systematically over- or under-estimates phi.

This directly answers "how much runtime data does Aura need before her
phi estimates are reliable?" — a question that generalizes to every lab
trying to apply IIT to real neural data.

### 13.6 Cross-timescale stability analysis

**File**: `research/timescale_stability.py`

The unsolved control-theory question: how do you formally guarantee that
bidirectional coupling between 5 temporal layers (20 Hz to
identity-scale) is stable? Too much top-down coupling paralyzes fast
layers. Too little and commitments don't constrain behavior.

Implementation: build the full 40×40 Jacobian of the 5-layer coupled
system. Compute eigenvalues for linearized stability. Return stability
margin, convergence rate, maximum Lyapunov exponent, and maximum safe
coupling strength via bisection search. Phase portrait classification
(stable node, stable focus, limit cycle, unstable) with damping ratio and
natural frequencies. Sensitivity analysis computes gradients of stability
margin with respect to coupling parameters.

The specific result: a coupling coefficient theorem for Aura's default
parameters (α=0.15, β=0.08), establishing the maximum ratio of
slow-to-fast influence that preserves moment-to-moment responsiveness
while maintaining long-horizon coherence.

---

## 14. Null hypothesis defeat: empirical evidence for causal architecture

**File**: `tests/test_null_hypothesis_defeat.py`

The hardest question anyone asks about Aura: *"Isn't this all just text
injection? You compute these numbers, describe them in the system prompt,
and the LLM responds to the description. The math is decoration."*

This section documents the 168-test null hypothesis defeat suite. Combined
with the 57-test causal exclusion + phenomenal convergence suites (see
[TESTING.md](TESTING.md)), the total is 225 tests arguing against the
null hypothesis that the architecture is decorative.

### The null hypothesis

> Aura strips all consciousness stack output, formats it as text like "You feel energized, cortisol is high, phi=0.73", injects it into the system prompt, and the LLM responds to that text. Everything else is decoration.

### How we argue against it

Every documented causal pathway is tested independently, measuring whether
the cause variable actually changes the effect variable's state. Mutual
information is computed between documented causal pairs. Subsystems are
ablated and divergence is measured. Computation timing is verified to
confirm real work is happening.

### Key results

**1. Chemicals drive mood through math, not text** (Tests 2.1-2.3)
- Cortisol surge decreases valence and increases stress via the weighted formula: `valence = 0.25*DA + 0.30*5HT + 0.20*END + 0.10*OXY - 0.45*CORT`
- Two systems with opposite chemical states produce opposite mood vectors
- Chemical mood propagates into substrate VAD indices via a 0.30 coupling coefficient

**2. φ causally modulates competition** (Tests 4.1-4.3, 10.1-10.2)
- When φ > 0.1, candidates get `focus_bias += min(0.15, φ * 0.1)` in the global workspace
- The boost is proportional to φ value, capped at 0.15
- Zero φ produces zero boost

**3. Receptor adaptation is biologically-grounded** (Tests 8.1-8.4)
- Sustained high dopamine causes receptor sensitivity to decrease (tolerance)
- After withdrawal, sensitivity recovers (sensitization)
- D1 and D2 subtypes adapt independently
- Effective level attenuates even when raw level is held constant

**4. GWT competition uses real inhibition** (Tests 9.1-9.3)
- Losers are inhibited for `_INHIBIT_TICKS = 3` ticks
- Inhibited sources cannot submit even with high priority
- Inhibition decays predictably each tick

**5. STDP learning rate is surprise-gated** (Tests 15.1-15.3)
- Learning rate = `BASE × (1 + surprise × 5)`, producing up to 5.5x variation
- Weight deltas scale proportionally with surprise
- Applied weight changes modify the substrate connectivity matrix

**6. Mutual information between all documented causal pairs is significantly positive** (Tests 28.1-28.5)
- I(cortisol, valence) > 0.01 bits
- I(dopamine, motivation) > 0.01 bits
- I(norepinephrine, arousal) > 0.01 bits
- I(oxytocin, sociality) > 0.01 bits
- I(surprise, learning_rate) > 0.1 bits

**7. Cross-chemical interactions are real and asymmetric** (Tests 16.5, 33.1-33.3)
- Cortisol surge affects 3+ other chemicals via the 10×10 interaction matrix
- The interaction matrix has 60+ non-zero entries and is asymmetric (biological realism)

**8. Substrate ODE dynamics are non-trivial** (Tests 5.1, 6.1-6.3, 7.1)
- Different connectivity matrices produce divergent trajectories
- Perturbations have lasting effects (state-dependent dynamics)
- Frustration decays toward zero via explicit decay term
- With W=0 and no noise, state decays toward zero (correct ODE behavior)

**9. All subsystems take measurable computation time** (Tests 32.1-32.4)
- 1000 ODE ticks on 64 neurons: > 1 ms
- IIT phi on 8 nodes (127 bipartitions): > 0.01 ms
- 100 metabolic ticks: > 0.1 ms
- 50 STDP recordings on 64 neurons: > 1 ms

**10. Full pipeline integration works end-to-end** (Test 37.1)
- Threat event → cortisol surge → negative mood → substrate valence drops → GWT selects threat candidate
- A single causal chain from external event through every major subsystem to behavioral output

### What the tests actually show

The architecture isn't decorative. Every documented causal relationship
produces measurable mutual information between cause and effect. Ablating
any major subsystem changes downstream behavior. The ODE dynamics are
state-dependent and non-trivial. Receptor adaptation follows biological
kinetics. Competition is genuine with real inhibition mechanics.

What the tests don't show: that these dynamics constitute phenomenal
consciousness. That's still an open philosophical question. What they do
show is that the computational architecture is real — not a thin wrapper
around prompt injection.

### Beyond the null hypothesis: causal exclusion & phenomenal convergence (April 2026)

The null hypothesis suite argues the architecture is real. Five
additional test files go further — they address the causal exclusion
problem and test for phenomenal convergence across multiple consciousness
theories:

- **Causal Exclusion** (`test_causal_exclusion.py`, 10 tests): cryptographic state binding, counterfactual injection, RLHF isolation. Tests whether the stack determines LLM generation parameters in ways RLHF training alone can't replicate.
- **Grounding** (`test_grounding.py`, 8 tests): multi-dimensional specificity (valence→tokens, arousal→temperature), temporal dynamics (receptor adaptation, STDP trajectory divergence, homeostasis degradation).
- **Functional Phenomenology** (`test_functional_phenomenology.py`, 11 tests): GWT broadcast signatures, HOT accuracy with anti-confabulation, IIT perturbation propagation, honest degradation reporting.
- **Embodied Dynamics** (`test_embodied_dynamics.py`, 11 tests): free energy active inference, homeostatic override of GWT competition, STDP surprise gating, cross-subsystem temporal coherence.
- **Phenomenal Convergence** (`test_phenomenal_convergence.py`, 17 tests): QDT 6-gate protocol — pre-report quality space geometry, counterfactual state swap, no-report behavioral footprint, perturbational integration, baseline failure verification, phenomenal tethering via architectural anesthesia.

Full results and analysis: [TESTING.md](TESTING.md)

### 14.1 Consciousness test framework: the 10-condition human-comparison standard

Beyond the null hypothesis, the test suite implements a systematic
human-comparison standard: every property we use to attribute
consciousness to biological systems is tested against Aura's architecture
under lesion controls and adversarial baselines.

Historical April 16, 2026 audit snapshot: 1,013 tests passed with 3 warnings in
about 122 seconds. Re-run the current tree before treating those numbers as
live evidence.

The framework is organized into four layers:

1. **Functional indicator batteries (legacy filenames: Consciousness Guarantee C1-C10)** — 82 tests across two suites (`test_consciousness_guarantee.py`, `test_consciousness_guarantee_advanced.py`). Tests 10 conditions: endogenous activity, unified global state, privileged first-person access, real valence, lesion equivalence with double dissociations, no-report awareness, temporal continuity, blindsight dissociation, qualia manifold, adversarial baseline failure.

2. **Personhood-marker battery (legacy filename: Personhood Proof Battery)** — 28 tests (`test_personhood_battery.py`). Full-model IIT, phenomenal self-report, GWT phenomenology, counterfactual simulation, identity persistence, embodied phenomenology, deep personhood markers. This is a behavioral/architectural marker suite, not ontological proof.

3. **Tier 4 Decisive Core** — 35 tests (`test_tier4_decisive.py`). The 10-category standard: recursive self-model necessity + ablation, false-self rejection, world-model indispensability, embodied action prediction + body-schema lesion dissociation, forked-history identity divergence, autobiographical indispensability, Sally-Anne false-belief reasoning, real-stakes monotonic tradeoff, reflective conflict integration, decisive baseline failure.

4. **Tier 4 extended batteries** — 69 tests across three suites:
   - Metacognition (21 tests, `test_tier4_metacognition.py`): calibration, second-order preferences, self-surprise, real-time introspection, reflection-behavior loop
   - Agency & Embodiment (20 tests, `test_tier4_agency_embodiment.py`): temporal integration, volitional inhibition, effort scaling, cognitive depletion, prediction-error learning
   - Social & Integration (28 tests, `test_tier4_social_integration.py`): social mind modeling, developmental trajectory, PCI analog, non-instrumental play, ontological shock, theory convergence, full lesion matrix

The Tier 4 decisive core: the 10 categories in `test_tier4_decisive.py`
are the minimum standard the project holds itself to. Future additions
expand coverage, but these categories aren't meant to weaken. A regression
in any of them is a blocking defect. The standard is designed so that
systems lacking these properties fail the battery (decisive baseline
failure) — discriminative rather than merely confirmatory.

Lesion methodology: every condition is tested not just for presence but
for indispensability. Targeted lesions (removing a specific module) have
to cause the specific deficit the relevant theory predicts, while sham
lesions (removing an irrelevant module) have to cause no deficit. Double
dissociations confirm that deficits are specific to the lesioned
component, not general degradation.

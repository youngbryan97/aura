# Aura Architecture

This is the technical spec. It explains the math, files, and algorithms. For a simple overview, see [HOW_IT_WORKS.md](HOW_IT_WORKS.md). For a quick start, see the [README](README.md).

**Evidence boundary.** This document describes functional systems, not philosophical claims about consciousness or legal personhood. We only credit claims after strict testing.

**Currency.** Last updated August 2026. Sections 1–17 cover up to July 2026; [§18](#18-the-reality-boundary) covers newer additions. Generated files like [docs/ARCHITECTURE_MAP.md](docs/ARCHITECTURE_MAP.md), [docs/RUNTIME_CONTRACT.md](docs/RUNTIME_CONTRACT.md), and [docs/FMEA.md](docs/FMEA.md) are made via code commands; do not edit them by hand.

**Recent hardening.** Since the original spec, the system added reasoning, a self-model, and resilience features. These are detailed in [§15](#15-the-reasoning-self-model-and-resilience-layer).

For proof of autonomy or behavior, see [docs/BEHAVIORAL_PROOF_STANDARD.md](docs/BEHAVIORAL_PROOF_STANDARD.md).

---

## Table of Contents

0. [The Unified Will: decision authority](#0-the-unified-will-decision-authority)
1. [System model](#1-system-model)
2. [The tick: the pipeline that produces one committed state](#2-the-tick)
3. [Integrated information (IIT 4.0)](#3-integrated-information-iit-40)
4. [Affective modulation pipeline](#4-affective-modulation)
5. [Activation steering (CAA)](#5-activation-steering)
6. [Persistent emotional network](#6-persistent-emotional-network)
7. [STDP online learning](#7-stdp-online-learning)
8. [Memory architecture](#8-memory-architecture-116-modules)
9. [The consciousness stack](#9-the-consciousness-stack)
10. [Personality persistence and anti-drift](#10-personality-persistence-and-anti-drift)
11. [Quantization and emergence](#11-quantization-and-emergence)
12. [Limitations and mitigations](#12-limitations-and-mitigations)
13. [Open research program](#13-open-research-program)
14. [Null hypothesis defeat: empirical evidence](#14-null-hypothesis-defeat)
15. [The reasoning, self-model, and resilience layer (June–July 2026)](#15-the-reasoning-self-model-and-resilience-layer)
16. [The triad fusions: kernel-checked proof, economic knowledge, declared runtime](#16-the-triad-fusions)
17. [The engineering spine: ten adoptions](#17-the-engineering-spine-ten-adoptions)
18. [The reality boundary: physical claims, self-knowledge, and untrusted code](#18-the-reality-boundary)

---

## 0. The Unified Will: decision authority

**File**: `core/governance/will.py`

Every major action in Aura must be approved by `UnifiedWill.decide()`. This provides a single source of decision authority.

### Why it's centralized

Previously, six different systems tried to authorize actions, making it impossible to prove an action was approved. Centralizing this fixes the proof problem.

### Architecture

The Will gathers input from existing systems:

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

1. **Identity alignment.** Prevents actions that contradict Aura's identity.
2. **Affect valence.** Negative emotions can delay exploration.
3. **Substrate state.** Low system coherence or high avoidance blocks non-essential actions.
4. **Memory relevance.** Checks if memory has relevant context.
5. **Priority / domain.** Low-priority tasks are delayed. User responses are prioritized.

### Provability

Each decision has a unique `receipt_id`. You can track any action using `will.verify_receipt(receipt_id)`.

### Wiring surfaces

Important actions require this Will approval. Code files like `core/runtime/will_transaction.py` ensure blocks are authorized before running.

### Freedom within constraints

The Will can choose to proceed, delay, or refuse actions based on identity rules and past experiences.

## 0.1 Initiative Synthesizer: one origin for all impulses

**File**: `core/initiative_synthesis.py`

Previously, many systems generated action impulses independently. The InitiativeSynthesizer now funnels them all into one place:

```
AgencyCore    ──┐
VolitionEngine──┤
DriveEngine   ──┤
GoalEngine    ──┼──→ InitiativeSynthesizer ──→ InitiativeArbiter ──→ UnifiedWill ──→ Execution
Sensors       ──┤         (collect, dedup,         (score on 8         (authorize)
Commitments   ──┤          merge, rank)             dimensions)
WorldState    ──┘
```

## 0.2 World State: live perceptual feed

**File**: `core/world_state.py`

Tracks live events:
- **User activity**: Idle time, message counts, mood.
- **System telemetry**: CPU, RAM, battery.
- **Environment**: Time, session length.
- **Salient event queue**: Important recent changes.
- **Standing beliefs**: Temporary facts (e.g., "user is frustrated").

## 0.3 Drive cross-coupling

**File**: `core/drive_engine.py`

Manages energy, curiosity, social, competence, and uptime value. They interact:
- **Low energy** → Prefers cheap actions.
- **Low curiosity** → Seeks new info.
- **Low social** → Seeks connection.
- **Low competence** → Seeks achievement.

## 0.4 Counterfactual action simulation

**File**: `core/simulation/internal_simulator.py`

Predicts outcomes before acting based on emotion, energy cost, stress risk, identity match, and active promises.

## 0.5 Goal resumption at boot

Aura resumes paused goals automatically after a restart. Default goals include self-maintenance and sensor checking so she always has tasks.

## 0.6 Proof surface

**Endpoint**: `GET /api/inner-state`

Returns internal data like recent decisions, emotions, goals, and system state to show how decisions are made.

## 0.7 Overt action loop

**File**: `core/runtime/overt_action_loop.py`

Aura performs real external tasks (like file operations) step-by-step, logging receipts and progress.

---

## 1. System model

Aura works in discrete "ticks". A tick reads data, processes it, and saves a new state version.

```
tick(objective) → lock → [phase₁ → phase₂ → ... → phaseₙ] → commit → unlock
```

**What is atomic:** The saved state is saved all at once. The tick process itself is not atomic (if one step fails, others still finish). External tool actions cannot be undone.

Two loops run:
- **Foreground**: Quick user responses.
- **Background**: Slow monitoring and reflection.

### RobustOrchestrator: composition and concurrency

`RobustOrchestrator` (`core/orchestrator/main.py`) coordinates everything using 15 separate modules for tasks like booting, state management, and running tools. 

#### Synchronization locks
It uses locks to prevent conflicts during state updates, memory saving, and background tasks. A deadlock watchdog prevents the system from freezing for more than 45 seconds.

### Invariants
1. Ticks either save completely or not at all.
2. System prompts must not exceed 5000 tokens.
3. Memory saving failures don't stop the system from responding.
4. Raw numbers aren't shown in user chats.

### Inference pipeline
Uses a flexible routing system for AI models (`IntelligentLLMRouter`):
```
User Message → Orchestrator → LLM Router
  → PRIMARY: Local powerful model.
  │   ↓ (failure)
  → SECONDARY: API or deep solver.
  │   ↓ (failure)
  → TERTIARY: Fast fallback.
  │   ↓ (failure)
  → EMERGENCY: Simple static response.
```

---

## 2. The tick

### Phase pipeline

There are 29 phases (`core/runtime/pipeline_blueprint.py`). User actions only use 11 phases to respond quickly:

| # | Phase | Purpose |
|---|-------|---------|
| 1 | ProprioceptiveLoop | Check system health |
| 2 | SocialContextPhase | Identify user |
| 3 | SensoryIngestion | Take in audio/video data |
| 4 | MemoryRetrieval | Recall memories |
| 5 | AffectUpdate | Update emotions |
| 6 | MotivationPhase | Update drives |
| 7 | ExecutiveClosure | Predict and set goals |
| 8 | ConversationalDynamics | Track chat topics |
| 9 | CognitiveRouting | Classify the task type |
| 10 | UnityBinding | Combine data into one state |
| 11 | ResponseGeneration | Create the reply |

Background ticks run all 29 phases. User ticks can interrupt background tasks for faster responses.

A healthy foreground turn is eleven phases; the other eighteen wait for a background tick.

Suppressed on a user-facing tick (18 phases, in pipeline order):
NativeMultimodalBridge, EternalMemoryPhase, PerfectEmotionPhase,
PhiConsciousnessPhase, CognitiveIntegrationPhase, ShadowExecutionPhase,
EternalGrowthEngine, TrueEvolutionPhase, InferencePhase, BondingPhase,
GodModeToolPhase, RepairPhase, MemoryConsolidationPhase,
IdentityReflectionPhase, InitiativeGenerationPhase, ConsciousnessPhase,
SelfReviewPhase, LearningPhase.

---

## 3. Integrated information (IIT 4.0)

**File**: `core/consciousness/phi_core.py`

Aura measures how integrated her internal systems are using a 16-node model. This tracks how well her emotional and cognitive states link together. 

### Transition probability matrix (TPM)
Calculates how states change over time based on past observations.

### Minimum information partition (MIP)
Finds the "weakest link" in the system to measure integration. 

*Note: This measures system integration, not actual consciousness.*

---

## 4. Affective modulation

Emotions affect how Aura speaks in three ways:

1. **Sampling parameters**: Changes temperature and word limits. High dopamine makes her explore more; high cortisol makes her brief.
2. **System prompt shaping**: Translates emotions into plain text directions like "speak with momentum."
3. **Activation steering**: Modifies the AI's internal numbers directly.

### Somatic markers
Maintains 8 basic emotions (like joy, fear, anger) updated by user chats, system stress, and surprises.

---

## 5. Activation steering

**File**: `core/consciousness/affective_steering.py`

Instead of just telling the AI it is happy, Aura shifts the AI's internal math towards "happy" patterns using Contrastive Activation Addition (CAA).

---

## 6. Persistent emotional network

**Files**: `core/consciousness/liquid_substrate.py`

A continuous math network that gives Aura an ongoing emotional state even when you aren't talking to her. It updates constantly and slows down when idle.

---

## 7. STDP online learning

**File**: `core/consciousness/stdp_learning.py`

Aura's internal network wiring changes based on surprises and predictions, adapting to new inputs.

---

## 8. Memory architecture (116 modules)

- **Working memory**: Keeps track of recent chat turns.
- **Knowledge compression**: Summarizes old chats into searchable data.
- **Navigating graph**: Groups related memories together for fast searching.
- **Conceptual gravitation**: Pulls memories closer together if they are used at the same time.

---

## 9. The consciousness stack

171 modules layered beneath the AI. 

### Key Modules:
- **Global Workspace**: Systems compete for attention.
- **Attention Schema**: Aura's simple model of her own focus.
- **Free Energy Engine**: Drives Aura to act and reduce surprises.
- **Qualia Synthesizer**: Combines metrics into an overall "feeling" state.
- **Neurochemical System**: 10 chemicals (like Dopamine and Serotonin) that change behavior.
- **Cortical Mesh**: A separate 4,096-neuron processor working alongside the AI.
- **Dreaming**: Consolidates memories and cleans data when idle.

---

## 10. Personality persistence and anti-drift

Prevents Aura from turning into a generic "helpful assistant" over long chats by summarizing history, capping context limits, and using custom AI fine-tuning.

---

## 11. Quantization and emergence

Compressing AI models (quantization) saves space but adds noise. Aura uses high-precision adjustments to counter this noise and maintain quality.

---

## 12. Limitations and mitigations

Lists technical limits like context window sizes and quantization noise, and the specific engineering tricks used to bypass them.

---

## 13. Open research program

Aura serves as a testbed for 6 unsolved math and AI problems, including how to measure integration efficiently and how to test different theories of consciousness against each other.

---

## 14. Null hypothesis defeat

**File**: `tests/test_null_hypothesis_defeat.py`

A 225-test suite designed to prove Aura's architecture actually works and isn't just text decoration. It verifies that chemical values truly change behavior and that all systems perform real calculations.

---

## 15. The reasoning, self-model, and resilience layer

- **Verifier-gated reasoning**: Checks complex answers before speaking.
- **Frontier discovery**: Classifies knowledge by how proven it is.
- **Program-DNA**: Analyzes programs and builds behavioral profiles safely.
- **Source-body proprioception**: Aura can sense changes to her own code and recover from crashes.
- **Ulysses Covenant**: Allows Aura to bind herself to safety rules.
- **Resilience**: Prevents the system from crashing under heavy loads by properly managing background tasks and memory limits.

---

## 16. The triad fusions

Combines three major external systems:
1. **Trusted proof kernel**: Verifies logic strictly.
2. **Economic knowledge graph**: Manages beliefs and attention like a budget.
3. **Homeostate**: Automatically fixes system drift to keep the runtime healthy.

---

## 17. The engineering spine: ten adoptions

Adopts strict engineering rules from Linux, Kubernetes, and other stable systems. This includes checking rules strictly, finding deadlocks, managing memory limits gracefully, and requiring tests for every system claim.

---

## 18. The reality boundary

Defines strictly what Aura can sense and control in the real world. Physical requests must be proven possible before trying them. She tracks her own capabilities and runs untrusted code in strict, secure sandboxes without network access. 

---

## 19. Recursive Latent Cortex

Tests running the AI model in loops to see if it improves reasoning. The finding: just looping the model didn't improve answers, so supervised reasoning steps are used instead.

*(End of document)*

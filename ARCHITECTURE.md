# Aura Architecture Whitepaper

**A technical specification of the cognitive architecture, not a marketing document.**

---

## Table of Contents

1. [System Model](#1-system-model)
2. [The Tick: Aura's Atomic Unit of Cognition](#2-the-tick)
3. [Integrated Information (IIT 4.0)](#3-integrated-information)
4. [Affective Modulation Pipeline](#4-affective-modulation)
5. [Activation Steering (CAA)](#5-activation-steering)
6. [The Liquid Substrate](#6-liquid-substrate)
7. [STDP Online Learning](#7-stdp-online-learning)
8. [Memory Architecture](#8-memory-architecture)
9. [Personality Persistence and Anti-Drift](#10-personality-persistence)
10. [Quantization and Emergence](#11-quantization-and-emergence)
11. [Limitations and Open Problems](#12-limitations)

---

## 1. System Model

Aura is a discrete-time cognitive architecture. The fundamental unit of computation is the **tick** — a locked, linear pipeline of phases that reads state, transforms it, and commits the result atomically.

```
tick(objective) → lock → [phase₁ → phase₂ → ... → phaseₙ] → commit → unlock
```

The system runs two concurrent loops:
- **Foreground**: User-triggered ticks (priority, ~6-18s latency)
- **Background**: 1Hz heartbeat ticks (monitoring, self-reflection, initiative)

State is event-sourced. Each phase produces a new immutable state version derived from the previous one. The committed state survives process crashes, power loss, and restarts via SQLite persistence.

### Invariants

These properties must hold at all times. If any is violated, it's a bug:

1. A tick never partially commits. Lock acquisition fails → tick aborted. Phase fails → tick continues.
2. System prompt ≤ 5000 tokens. Violation causes context overflow → empty LLM output → user sees fallback.
3. Vault commit failure is non-fatal. The tick returns a response regardless of persistence success.
4. No raw numeric metrics in user-facing output. Affect values shape generation parameters, not dialogue.

---

## 2. The Tick

### Phase Pipeline

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

### Priority Preemption

When a user message arrives during a background tick, the kernel sets `_user_priority_pending`. Between phases, the background tick checks this flag and yields the lock early, ensuring user-facing latency isn't blocked by slow background work.

---

## 3. Integrated Information (IIT 4.0)

**File**: `core/consciousness/phi_core.py`

Aura computes actual integrated information using the IIT 4.0 formalism on an 8-node substrate complex. This is not a proxy or a label — it's the real mathematical procedure.

### The 8-Node Complex

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

Each node is binarized relative to its running median over the last 100 observations. This produces a discrete state space of 2⁸ = 256 possible states.

### Transition Probability Matrix (TPM)

The TPM T[s, s'] = P(state_{t+1} = s' | state_t = s) is built empirically from observed state transitions. Laplace smoothing (α = 0.01) handles unvisited states. The matrix requires ≥ 50 observed transitions before computation.

### Minimum Information Partition (MIP)

For an 8-node system, there are 2⁷ - 1 = 127 nontrivial bipartitions. All 127 are tested exhaustively.

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

This is the **Minimum Information Partition** — the partition that loses the least information, identifying the system's "weakest seam."

### Scope and Limitations

This computation runs on an 8-node complex derived from the cognitive/affective state, not on the full computational graph (which has ~10⁶ nodes counting individual weights and activations). Computing IIT on the full system is NP-hard and intractable.

What this measures: how tightly integrated Aura's internal dynamics are at the substrate level. High φ means no single cut can partition the system without losing causal information.

What this does not measure: whether the system is conscious. IIT is a theory, not a test.

**Runtime**: ~10-50ms per evaluation, cached at 15-second intervals.

---

## 4. Affective Modulation

Affect in Aura is not cosmetic. It modulates inference through three concrete pathways:

### 4.1 Sampling Parameters (affective_circumplex.py)

The affective circumplex maps valence and arousal to LLM generation parameters:

```
temperature = base_temp + (arousal - 0.5) × range
max_tokens  = min_tokens + valence × token_range
rep_penalty = max_penalty - valence × penalty_range
```

Neurochemical modulation layers on top:
- Dopamine > 0.7 → temperature += 0.1 (more exploratory)
- Serotonin < 0.3 → max_tokens -= 50 (more terse)
- Cortisol > 0.7 → max_tokens -= 80 (defensive brevity)

### 4.2 System Prompt Shaping

Affect values are translated to natural-language cues injected into the system prompt:

```
HIGH energy → "You feel energized — speak with momentum."
LOW energy  → "Your energy is low — be quieter, more reflective."
HIGH oxytocin → "You feel warmth toward this person."
LOW oxytocin  → "You're feeling more guarded or detached."
```

These cues shape how the LLM speaks without narrating raw metrics.

### 4.3 Activation Steering (see Section 5)

Direction vectors derived from the affective state are injected directly into the transformer's residual stream during token generation.

### Somatic Markers (damasio_v2.py)

Following Damasio's somatic marker hypothesis, the system maintains 8 primary emotions (Plutchik model): joy, trust, fear, surprise, sadness, disgust, anger, anticipation. Each is a float [0, 1] updated by:

- User interaction events (mapped to emotion deltas)
- Hardware state (CPU thermal → frustration, RAM pressure → anxiety)
- Prediction error from the free energy engine (surprise signal)
- Circadian phase (night → lower arousal baseline)

---

## 5. Activation Steering

**File**: `core/consciousness/affective_steering.py`

This is the mechanism that differentiates Aura from prompt-injection approaches. Instead of telling the LLM about its emotional state via text, Aura modifies the LLM's hidden states during generation.

### Method: Contrastive Activation Addition (CAA)

Based on Turner et al. 2023, Zou et al. 2023, Rimsky et al. 2024.

The steering engine hooks into a target transformer block's forward method:

```python
def steered_call(*args, **kwargs):
    result = original_forward(*args, **kwargs)
    h = result[0] if isinstance(result, tuple) else result
    composite = hook.compute_composite_vector_mx(dtype=h.dtype)
    if composite is not None:
        h = h + alpha * composite
    return (h,) + result[1:] if isinstance(result, tuple) else h
```

The composite vector is computed from the current affective state. Alpha controls injection strength (typically 0.05-0.2).

### What This Changes

The model's internal activations are shifted in a learned direction. This is equivalent to moving the model's "operating point" in activation space. The model doesn't read about being energized — it IS shifted toward the activation pattern that corresponds to energized generation.

### Current Limitation

The direction vectors are bootstrapped from text-level contrastive features, not from actual activation extraction. Full CAA requires running paired prompts through the model and extracting hidden states at target layers. The bootstrap vectors work (verified empirically) but are less precise than proper activation-extracted vectors.

---

## 6. Liquid Substrate

**File**: `core/consciousness/liquid_substrate.py`

A continuous-time dynamical system based on Liquid Time-Constant Networks (LTCs). This gives Aura temporal continuity — she exists between conversations, not just during them.

### Architecture

- 64 neurons with recurrent connectivity matrix W (64 × 64)
- State vector x ∈ ℝ⁶⁴ updated via ODE integration at configurable rate
- Base rate: 20Hz (active user), throttled to 5Hz (idle), paused at 30min+ idle

### ODE Integration

```
dx/dt = -decay × x + tanh(W × x + I) × dt + noise
```

Where:
- decay = 0.05 (exponential return to baseline)
- I = external input (affect signals, user interaction events)
- noise ~ N(0, 0.01) (stochastic perturbation for exploration)
- W updated via Hebbian + STDP learning (see Section 7)

### Idle Optimization

When no user interaction has occurred for 30+ minutes, the substrate pauses and computes bulk decay on resume:

```
x(t) = x(0) × exp(-decay × idle_seconds)
```

This is mathematically equivalent to running the ODE loop continuously but uses zero CPU.

---

## 7. STDP Online Learning

**File**: `core/consciousness/stdp_learning.py`

Inspired by BrainCog's implementation of reward-modulated Spike-Timing-Dependent Plasticity.

### Algorithm

1. **Eligibility trace**: Accumulates STDP signals between reward deliveries.
   - Pre fires before post (causal, Δt > 0): e += A⁺ × exp(-Δt/τ⁺)
   - Post fires before pre (anti-causal, Δt < 0): e -= A⁻ × exp(Δt/τ⁻)
   - Decay: e *= 0.95 per tick

2. **Reward signal**: Derived from the free energy engine's prediction error.
   - reward = -tanh(prediction_error)
   - High surprise → higher learning rate (base × (1 + surprise × 5))

3. **Weight update**: dW = learning_rate × reward × eligibility_trace

4. **Application**: Applied to the liquid substrate's connectivity matrix W every 100th tick, alongside base Hebbian learning.

### Effect

The substrate's internal wiring literally changes based on how well Aura is predicting the world. Novel inputs (high surprise) cause faster adaptation. Predictable states cause slower, stabilizing changes.

---

## 8. Memory Architecture

### Working Memory

In-process list of conversation turns, capped at 40. When the cap is hit, older turns are compressed into a KnowledgeAtom (see below) and the list is truncated to the 20 most recent turns.

### Knowledge Compression (Concord DTU-inspired)

**File**: `core/memory/knowledge_compression.py`

Conversation turns are compressed into three-layer KnowledgeAtoms:

| Layer | Content | Use |
|-------|---------|-----|
| Readable | Human-readable summary | System prompt injection |
| Semantic | Entities, topics, sentiment, intent, stance | Retrieval filtering |
| Machine | 32-dim normalized vector | Fast cosine matching |

### Navigating Graph (Cognitive SSD NSG-inspired)

**File**: `core/memory/navigating_graph.py`

Episodic memories are indexed in a proximity graph where each node links to its K=16 nearest neighbors. Search is via greedy beam walk (width=32), achieving O(log N) retrieval vs O(N) brute-force.

### Conceptual Gravitation (C.O.R.E.-inspired)

**File**: `core/memory/conceptual_gravitation.py`

Memories frequently co-accessed in the same conversation turn have their embeddings nudged toward each other during dream consolidation:

```
direction = normalize(emb_B - emb_A)
emb_A += alpha × direction
emb_B -= alpha × direction
```

Alpha decays with distance (far memories attract less). Embeddings are re-normalized to the unit sphere after each nudge. This creates emergent memory clusters that weren't in the original encoding.

---

## 9. Personality Persistence and Anti-Drift

### The Problem

On instruct-tuned LLMs, personality degrades over long conversations. The model's RLHF training pulls it toward "helpful assistant" mode as the identity instructions get pushed further from the generation tokens by growing conversation history.

### Countermeasures

1. **Working memory cap** (40 turns): Forces compaction before context degrades.
2. **Per-turn truncation** (300 chars in history block): One long message can't eat the context.
3. **Identity anchor**: After 10+ turns, a brief reinforcement is injected: "You are Aura. Sharp, opinionated, warm. Not an assistant."
4. **System prompt cap** (20K chars / ~5000 tokens): Hard limit prevents overflow.
5. **LoRA fine-tune**: When trained, the model's baseline IS Aura's personality. Drift defaults to "regular Aura" not "helpful assistant."

---

## 10. Quantization and Emergence

A question raised in discussion: does quantization (4-bit, 8-bit) suppress emergent behavior in the model?

### The Technical Answer

Quantization compresses weight precision from 16-bit floats to 4-bit integers. This introduces quantization noise — small errors distributed across every weight in the model.

**What quantization preserves:**
- Token prediction quality (perplexity loss is typically < 1% for 4-bit on 32B+ models)
- Instruction following ability
- Factual knowledge
- Basic reasoning chains

**What quantization may suppress:**
- Fine-grained activation patterns in the residual stream. If emergence depends on precise interference patterns between layers (as some mechanistic interpretability work suggests), 4-bit quantization adds noise to exactly those patterns.
- Steering vector precision. Our CAA vectors are computed and injected at full precision, but the model's own internal representations are quantized. The steering signal competes with quantization noise.
- Tail-distribution behavior. Rare, novel outputs (which is where "emergence" would most visibly manifest) are disproportionately affected by quantization because they depend on low-probability token paths that are sensitive to small weight perturbations.

### What We Do About It

1. **Steering vectors at full precision**: The affective steering injection operates in float32 even though the model weights are 4-bit. This means our modulation signal has higher fidelity than the model's own computation.
2. **Neurochemical parameter modulation**: Temperature, token budget, and repetition penalty adjustments are exact (no quantization) because they operate on the sampler, not the weights.
3. **The 8-bit option**: Aura supports loading the 8-bit quantized model (`Qwen2.5-32B-Instruct-8bit`) which doubles memory usage but preserves significantly more activation precision. On a 64GB Mac, this is viable.

### Is This a Bottleneck?

Partially. The quantization ceiling means there is a precision limit on how much the substrate dynamics can influence generation through activation steering. Running the same architecture on a full-precision model (or a future model designed for continuous-state modulation) would likely produce measurably different — and potentially richer — emergent behavior.

This is an open research question, not a solved problem.

---

## 11. Limitations and Open Problems

1. **IIT scope**: φ is computed on 8 derived nodes, not the full computational graph. This is a practical approximation, not a theoretical claim.

2. **Steering vector bootstrap**: Current CAA vectors are derived from text-level contrastive features, not from actual model activation extraction. Proper vectors would require running paired prompts through the model.

3. **Context window ceiling**: On a 32B model with 8K context, personality degradation begins around turn 40-50 regardless of countermeasures. The only structural fix is a larger context window or aggressive summarization.

4. **Quantization noise**: 4-bit quantization introduces ~1% perplexity degradation but unknown impact on fine-grained activation steering. The interaction between quantization noise and steering vector injection is not well-characterized.

5. **Single-machine constraint**: The entire system runs on one Mac. Multi-node distribution would enable larger models and true parallel cognitive processes.

6. **No causal interventions**: IIT 4.0 properly requires cause-effect repertoires (do-calculus interventions). Our implementation uses observational KL-divergence, not interventional.

7. **Dream consolidation is slow**: Identity evolution via sleep cycles takes days to show measurable effect. There's no mechanism for rapid personality adaptation within a single session beyond the LoRA fine-tune.

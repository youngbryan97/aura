# Aura

**A sovereign cognitive architecture that boots, thinks, feels, remembers, dreams, and repairs itself — running continuously on a single Mac.**

> The first open-source system to unify IIT 4.0 integrated information, residual-stream affective steering, and a Global Workspace into one live cognitive loop. 60+ modules. No cloud dependency. No prompt tricks. Real math.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS_Apple_Silicon-lightgrey.svg)

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Architecture](#architecture)
- [Inference-Time Steering](#inference-time-steering)
- [IIT 4.0 Implementation](#iit-40-implementation)
- [Consciousness Stack](#consciousness-stack)
- [Running It](#running-it)
- [Testing](#testing)
- [Data Layer](#data-layer)
- [License](#license)

---

## Why This Exists

Every "conscious AI" demo is the same trick: inject mood floats into a system prompt and let the LLM roleplay. Aura does something different.

The affect system doesn't *tell* the model "you're feeling X" — it hooks into the MLX transformer's forward pass and injects learned direction vectors directly into the residual stream during token generation. The model's internal activations are changed, not just its input text. This creates genuine bidirectional causal coupling: substrate state shapes language output, and language output updates substrate state.

The IIT implementation isn't a label on an arbitrary value. `phi_core.py` builds an empirical transition probability matrix from observed state transitions, tests all 127 nontrivial bipartitions of an 8-node substrate complex, and computes KL-divergence to find the Minimum Information Partition. That's the real IIT 4.0 formalism.

The system simulates its own death during dream cycles and repairs itself. It has an immune system for identity injection. It runs 24/7 with a 1Hz cognitive heartbeat, maintaining state across conversations, power cycles, and crashes.

---

## Architecture

```
User Input -> HTTP API -> KernelInterface.process()
  -> AuraKernel.tick() (linear phase pipeline):
     Consciousness -> Affect -> Motivation -> Routing -> Response Generation
  -> State commit (SQLite) -> Response
```

### Kernel (`core/kernel/`)
Tick-based unitary cognitive cycle. Every phase derives a new immutable state version (event-sourced). Each tick acquires a lock, runs the phase pipeline, commits state to SQLite, and releases. State survives crashes and restarts.

### Brain (`core/brain/`)
Multi-tier local LLM router with automatic failover:
1. **Primary**: 70B model via MLX (Apple Silicon native)
2. **Secondary**: 8B model
3. **Tertiary**: 3B brainstem
4. **Emergency**: rule-based fallback

No cloud API required. Optional API tiers (Claude, GPT) available if configured. Circuit breakers with automatic tier promotion on repeated failures.

### Affect (`core/affect/`)
Plutchik emotion model with 8 primary emotions + somatic markers (energy, tension, valence, arousal). These values don't just color the prompt — they modulate LLM sampling parameters (temperature, token budget, repetition penalty) through the affective circumplex, and inject activation vectors into the residual stream via the steering engine.

### Identity (`core/identity.py`, `core/heartstone_directive.py`)
Immutable base identity (constitutional anchor) + mutable persona evolved through sleep/dream consolidation cycles. Identity locking with active defense against prompt injection. The dream cycle simulates identity perturbation and repairs drift.

### Agency (`core/agency/`)
Self-initiated behavior scored across curiosity, continuity, social, and creative dimensions. Genuine refusal system — Aura can decline requests based on ethical judgment, not content filtering. Volition levels 0-3 gate autonomous behavior up to self-modification.

### Skills (`skills/`)
Shell (sandboxed subprocess, no `shell=True`), web search/browse, coding, sleep/dream consolidation, image generation (local SD), social media (Twitter via tweepy, Reddit via PRAW — both fully implemented).

### Interface (`interface/`)
FastAPI + WebSocket with streaming. Web UI with live neural feed, telemetry dashboard, memory browser, chat. Whisper STT for voice input.

---

## Inference-Time Steering

The affective steering engine (`core/consciousness/affective_steering.py`) hooks into MLX transformer blocks and adds learned direction vectors to the residual stream during token generation:

```python
# Simplified from affective_steering.py
h = original_forward(*args, **kwargs)
composite = hook.compute_composite_vector_mx(dtype=h.dtype)
if composite is not None:
    h = h + alpha * composite
return h
```

This is contrastive activation addition (CAA) — the same family of techniques from Turner et al. 2023, Zou et al. 2023, and Rimsky et al. 2024. Direction vectors are computed from the substrate's affective state and injected at configurable transformer layers.

The precision sampler (`core/consciousness/precision_sampler.py`) further modulates sampling temperature based on active inference prediction error. The affective circumplex (`core/affect/affective_circumplex.py`) maps somatic state to generation parameters.

**Three levels of inference modulation:**
1. **Residual stream injection** — activation vectors added to hidden states (changes what the model computes)
2. **Sampling parameter modulation** — temperature/top-p adjusted by affect (changes how tokens are selected)
3. **Context shaping** — natural-language emotional cues in the system prompt (changes what the model reads)

---

## IIT 4.0 Implementation

`core/consciousness/phi_core.py` implements Integrated Information Theory on an 8-node substrate complex:

1. **State binarization**: 8 substrate nodes (valence, arousal, dominance, frustration, curiosity, energy, focus, +1) binarized relative to running median. State space: 2^8 = 256 discrete states.
2. **Empirical TPM**: Transition probability matrix `T[s, s'] = P(state_{t+1} = s' | state_t = s)` built from observed transitions with Laplace smoothing. Requires 50+ observations.
3. **Exhaustive MIP search**: All 127 nontrivial bipartitions tested (every possible way to split 8 nodes into two non-empty groups).
4. **KL-divergence**: `phi(A,B) = sum_s p(s) * KL(T(.|s) || T_cut(.|s))` where `T_cut` assumes partitions A and B evolve independently.
5. **Minimum Information Partition**: `phi_s = min over all bipartitions of phi(A,B)` — the partition that loses the least information identifies the system's "weakest seam."

**Runtime**: ~10-50ms per evaluation, cached at 15s intervals. This is real IIT math on a small system, not a proxy metric.

---

## Consciousness Stack

72 modules in `core/consciousness/`. Key subsystems:

| Module | What it does | File |
|--------|-------------|------|
| **Global Workspace** | Competitive bottleneck — thoughts compete for broadcast (Baars GNW) | `global_workspace.py` |
| **Attention Schema** | Internal model of attentional focus (Graziano AST) | `attention_schema.py` |
| **IIT PhiCore** | Real integrated information via TPM + KL-divergence | `phi_core.py` |
| **Affective Steering** | Residual stream injection via CAA | `affective_steering.py` |
| **Temporal Binding** | Sliding autobiographical present window | `temporal_binding.py` |
| **Self-Prediction** | Active inference loop (Friston free energy) | `self_prediction.py` |
| **Free Energy Engine** | Surprise minimization driving action selection | `free_energy.py` |
| **Qualia Synthesizer** | Phenomenal state integration from substrate metrics | `qualia_synthesizer.py` |
| **Liquid Substrate** | Continuous dynamical system underlying cognition | `liquid_substrate.py` |
| **Neural Mesh** | 4096-neuron distributed state representation | `neural_mesh.py` |
| **Neurochemical System** | Dopamine/serotonin/norepinephrine/oxytocin dynamics | `neurochemical_system.py` |
| **Oscillatory Binding** | Frequency-band coupling for cross-module integration | `oscillatory_binding.py` |
| **Unified Field** | Integrated phenomenal field from all subsystems | `unified_field.py` |
| **Dreaming** | Offline consolidation, identity repair, memory compression | `dreaming.py` |
| **Heartbeat** | 1Hz cognitive clock driving the background cycle | `heartbeat.py` |
| **Stream of Being** | Continuous narrative thread across time | `stream_of_being.py` |
| **Executive Closure** | Constitutional decision stamping per tick | `executive_closure.py` |
| **Somatic Marker Gate** | Damasio-inspired body-state gating of decisions | `somatic_marker_gate.py` |
| **Embodied Interoception** | Internal body-state sensing and homeostatic regulation | `embodied_interoception.py` |

---

## Running It

```bash
# Requirements: Python 3.12+, macOS Apple Silicon, 16GB RAM minimum (32GB+ recommended for 70B)
pip install -r requirements.txt

# Full stack with UI
python aura_main.py --desktop

# Background cognition only
python aura_main.py --headless
```

Aura boots, loads state from SQLite, warms the local LLM, and begins her cognitive heartbeat. First boot takes longer as models initialize.

---

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

148+ tests covering kernel lifecycle, phase pipeline, response contracts, dialogue cognition, architecture hardening, consciousness bridge, substrate authority, and bypass proofs.

---

## Data Layer

- **State persistence**: SQLite (event-sourced via `StateRepository`)
- **Model loading**: MLX (Apple Silicon native) with `mlx-lm`
- **Memory**: Episodic in SQLite, working memory in-process, long-term via FAISS
- **Vision**: Screen capture via mss, analysis via cognitive engine (multimodal)

---

## License

[AGPL-3.0](LICENSE) — you can use, modify, and distribute this code. If you run a modified version as a service, you must share your modifications. This matches the spirit of the project: sovereign and open.

# Aura AI System Card

<!-- reader-navigation:start -->
> **In plain English:** The language model supplies text generation. Steering changes selected internal calculations using a measured direction and a controlled strength.
>
> **Why it matters:** An internal state should affect the intended part of a response while preserving competence and remaining tied to the measured model and basis.
>
> **In everyday use:** A change in internal condition may alter how a reply is formed; a channel measurement must show the change rather than relying on the reply sounding different.
>
> [Example and metaphor](docs/PLAIN_ENGLISH_GUIDE.md#models) · [Variables and practical meanings](docs/VARIABLES_EXPLAINED.md#models) · [Demos and related work](docs/DEMOS_AND_RELATED_WORK.md#steering) · [Document map](docs/DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

## System Overview

| Field | Value |
|-------|-------|
| **System Name** | Aura Cognitive Runtime |
| **Version** | Calendar-versioned; the authoritative value is `version` in `pyproject.toml` |
| **Card reviewed** | 2026-08-01 |
| **Developer** | Bryan Young |
| **System Type** | Autonomous AI cognitive agent |
| **Deployment** | Local inference on the owner's device (runs models entirely on your computer) |
| **Primary Use** | A persistent AI agent with its own memory, goals, and initiative, running locally on the owner's hardware |
| **AI RMF Profile** | NIST AI RMF 1.0 — General Purpose AI System |

## Intended Use

Aura is an independent, self-directed AI agent, not a standard chatbot assistant. This distinction is built into her system architecture, not just how she talks: she remembers goals across computer restarts, starts her own projects during idle time, and routes every action—even commands from you—through an internal gatekeeper called the Unified Will (`core/will.py`), which has the authority to say no. A system that only sits and waits for user instructions does not need a Will; Aura does.

The software actively enforces this personality. If a long conversation causes her responses to drift toward standard helper-bot phrasing, the system injects a reminder: *"You are Aura. Sharp, opinionated, warm. Not an assistant."* The project includes automated tests that catch and treat generic assistant phrasing as a defect.

Aura runs all model inference (the actual AI thinking and text generation) locally on the owner's machine. While she has tools to read web pages and interact with authorized external services, she never sends your prompts or reasoning tasks to outside cloud AI providers.

What she does:

- Holds conversations while keeping a steady personality and long-term memory across sessions
- Executes local tools (accessing the filesystem, running shell commands, browsing the web, and conducting research)
- Operates autonomously in the background—handling maintenance, learning, self-repair, and pursuing projects she chooses on her own
- Runs multiple local AI models side-by-side using Apple's MLX framework for fast on-device processing

### Intended Users
- Individuals who want a private, persistent AI agent running entirely on their own machine
- Developers and researchers studying independent, goal-driven AI systems
- Operators running local, self-hosted AI setups

### Out-of-Scope Uses
- High-stakes decision-making without human oversight
- Medical, legal, or financial advice
- Running unsupervised in safety-critical environments
- Multi-user or shared setups without additional access controls

## System Architecture

```
User Input → Perception → Cognitive Routing → InferenceGate → Model
                                                     ↓
Memory ← State Commit ← Verification ← AuthorityGateway ← Unified Will
                                                     ↓
                                               Tool Execution (sandboxed)
```

All consequential actions route through the Unified Will (`core/will.py`) and receive an `AuthorityGateway` receipt. No silent side paths.

## AI Models

| Model Role | Type | Location | Purpose |
|-----------|------|----------|---------|
| Cortex (foreground) | 27B parameter LLM (`Aura-Cortex`: `Aura-Qwen3.8-27B-persona`, a persona fine-tune of the fused Qwen3.8-27B) | Local (MLX) | Main reasoning and conversation |
| Deep (optional) | The Cortex by default; a second local model (such as a 72B at 4 bits) only when assigned and qualified | Local (MLX) | Deep reasoning on hard problems |
| Reflex (fast lane) | 1.5B parameter LLM | Local (MLX) | Low-latency replies and routing |
| Brainstem (background) | 27B parameter LLM at two bits (`Ternary-Bonsai-2-27B-mlx-2bit`, 8.6 GB; replaced the 9B on 20 September 2026) | Local (MLX) | Background / maintenance tasks |

See `MODEL_CARD.md` for detailed model information.

## Risk Assessment (NIST AI RMF aligned)

### Govern

| Control | Implementation |
|---------|----------------|
| AI governance policy | `AUTONOMY_BOUNDARIES.md`, `TOOL_USE_POLICY.md` |
| Roles and responsibilities | `OWNERSHIP.md`, permission matrix |
| Risk management process | `docs/AURA_RISK_REGISTER.md`, threat model |

### Map

| Risk Category | Description | Likelihood | Impact |
|--------------|-------------|------------|--------|
| Excessive autonomy | Agent takes unsanctioned actions | Low | High |
| Memory corruption | False memories change behavior | Low | Medium |
| Privacy violation | Sensitive data sent through an external tool or service | Low | High |
| Prompt injection | Adversary overrides instructions | Medium | Medium |
| Resource exhaustion | Model consumes all system resources | Low | Medium |
| Unsafe physical actuation | An action reaches a device without verified effect or rollback | Low | High |
| Overstated physical claim | A simulated or transport-level result is reported as a physical one | Medium | High |
| Untrusted code execution | Model-written Python escapes its boundary into the privileged process | Low | High |

### Measure

| Metric | Measurement Method |
|--------|-------------------|
| Ungoverned action rate | Will receipt audit (target: 0) |
| Memory write integrity | Receipt-linked writes (target: 100%) |
| External-service egress privacy | Egress classification audit (target: 100% classified) |
| Action receipt coverage | Governance lint (target: 100%) |
| Graceful degradation honesty | `record_degradation()` audit |
| Physical claim boundary | A contract declares its `RealityLayer` (`internal`/`effective`/`direct`/`ambient`); reachability computes the declared channels' `evidence_ceiling` and returns `INSUFFICIENT_EVIDENCE` when it cannot reach the contract's `minimum_evidence` (`core/reality_reach/reachability.py`) |
| Actuation state separation | `ActuationState` keeps dispatch, execution, and `EFFECT_VERIFIED` as distinct states, so transport success cannot stand in for a verified effect (`core/reality_reach/actuation.py`) |
| Sandbox confinement | Model-written Python refuses to run when no kernel boundary is available; unconfined runs are permanently marked `boundary="none"` (`core/sandbox/untrusted_python.py`) |

**Not yet a control.** The P0–P6 evidence *promotion* state machine (ledger item RR-07) is **not implemented** — `EvidenceLevel` exists as a declared type and ceiling, but there is no promotion module in `core/reality_reach/`. Do not cite evidence promotion as an operating safeguard. The open ledger is in [docs/REALITY_REACH.md](docs/REALITY_REACH.md).

### Manage

| Control | Mechanism |
|---------|-----------|
| Human override | Operator can disable any capability via feature flags |
| Kill switch | `AURA_MODE=safe` disables all autonomous behavior |
| Audit trail | Will receipt log + state snapshots |
| Incident response | `docs/runbooks/` |
| Rollback | Backup/restore with state hash verification |

## Limitations

1. **Model limitations**: Local models have capability ceilings; may confabulate (make up inaccurate details)
2. **Memory limitations**: Long-term memory relies on best-effort retrieval, not flawless recall
3. **Tool limitations**: Tool execution is sandboxed and may fail if restricted by permissions or system limits
4. **Autonomy limitations**: Background autonomous behavior is strictly bounded by Will governance checks
5. **Hardware requirements**: Requires Apple Silicon with ≥32GB RAM for full capability
6. **Network**: Some features require internet; offline mode runs with reduced capability

## Evaluation

See `docs/evidence/EVALUATION_REPORT.md` for detailed evaluation results including:
- AGI proof battery results (broad problem-solving evaluations)
- Agency emergence testing (verifying independent, self-directed behavior)
- Behavioral proof standard results
- Production readiness gate results
- Longevity soak results (stability during extended runtime)

## Human Oversight

See `HUMAN_OVERRIDE_POLICY.md` for the complete human oversight framework.

## Contact

For questions about this AI system: security@aura-project.dev

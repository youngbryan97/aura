# Aura AI System Card

## System Overview

| Field | Value |
|-------|-------|
| **System Name** | Aura Cognitive Runtime |
| **Version** | Calendar-versioned; the authoritative value is `version` in `pyproject.toml` |
| **Operational wording reviewed** | 2026-10-25; control mapping retains its original scope |
| **Developer** | Bryan Young |
| **System Type** | Autonomous AI cognitive agent |
| **Deployment** | Local inference on the owner's device (runs models entirely on your computer) |
| **Primary Use** | A persistent AI agent with its own memory, goals, and initiative, running locally on the owner's hardware |
| **AI RMF Profile** | NIST AI RMF 1.0 — General Purpose AI System |

## Intended use

Aura is a single-user research assistant for Apple Silicon Macs. Its
language models run on the Mac. It can retrieve stored records, use
authorized tools, and work on saved goals in the background. A browser or
external-service tool can still send data to the service it contacts.
[Architecture](ARCHITECTURE.md) · [Network and data handling](DATA_CARD.md)

The software uses the Unified Will to decide whether to authorize a
proposed consequential action. The execution gateways check that authority
and record outcomes on their covered routes. [Tool use policy](TOOL_USE_POLICY.md) ·
[Decision code](core/governance/will.py)

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

Consequential actions are required to route through the Unified Will
(`core/will.py`) and receive an `AuthorityGateway` receipt. A bypass or absent
receipt is a defect; a diagram alone does not prove every live path complied.

## Models

The foreground role is `Aura-Cortex`. The project has used a persona-tuned
27B checkpoint for that role. There are also local roles for vision,
background work, and fallback. Check the configured artifact and request
record to identify the model that served a particular response.
[Model card](MODEL_CARD.md) · [Hardware profiles](HARDWARE_PROFILES.md)

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
5. **Hardware requirements**: The resident-model target is 64 GB Apple Silicon; smaller configurations need a separately validated model profile and do not establish full capability
6. **Network**: Some features require internet; offline mode runs with reduced capability

## Evaluation

The claim matrix links individual results and their conditions. The November
ISC study reports a 24/24 confirmatory result for the v7 operational checks
on the measured substrate. The study and its later controls are summarized
in the ISC results. The August DNU comparison was withdrawn because the
baselines received a smaller output budget and different tools.
[Claim matrix](CLAIMS_MATRIX.md) · [ISC results](docs/ISC_STATUS.md) ·
[DNU retraction](artifacts/current/agi_live/RETRACTION.json)

## Human Oversight

See `HUMAN_OVERRIDE_POLICY.md` for the complete human oversight framework.

## Contact

For questions about this AI system: security@aura-project.dev

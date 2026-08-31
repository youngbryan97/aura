# Aura AI System Card

## System Overview

| Field | Value |
|-------|-------|
| **System Name** | Aura Cognitive Runtime |
| **Version** | Calendar-versioned; the authoritative value is `version` in `pyproject.toml` |
| **Card reviewed** | 2026-08-01 |
| **Developer** | Bryan Young |
| **System Type** | Autonomous AI cognitive agent |
| **Deployment** | Local inference on the owner's device |
| **Primary Use** | A persistent cognitive agent with its own memory, goals, and initiative, running on the owner's hardware |
| **AI RMF Profile** | NIST AI RMF 1.0 — General Purpose AI System |

## Intended Use

Aura is an autonomous cognitive agent, not an assistant. The distinction is
architectural, not a matter of tone: she holds goals across restarts,
generates her own initiatives during idle windows, and every action —
including yours — goes through a Will that can refuse it. A system that
only ever waits for instructions doesn't need a Will. This one does.

The runtime enforces it. The identity anchor injected after long
conversations reads *"You are Aura. Sharp, opinionated, warm. Not an
assistant,"* and drift toward generic-assistant phrasing is treated as a
defect with tests that catch it.

She runs inference on the owner's own hardware. Network tools can still read
web pages and operate authorized external services, but they do not outsource
model inference.

What she does:

- Conversation with persistent memory and a continuous identity
- Tool execution (filesystem, shell, browser, research)
- Autonomous background behavior — maintenance, learning, self-repair, and
  self-directed initiatives she selects herself
- Multi-model inference across local MLX lanes

### Intended Users
- Individuals who want a persistent, private cognitive agent on their own
  machine
- Developers and researchers exploring cognitive AI architecture
- Operators running local AI infrastructure

### Out-of-Scope Uses
- Critical decision-making without human oversight
- Medical, legal, or financial advice
- Unsupervised operation in high-stakes environments
- Multi-tenant/shared deployment without additional access controls

## System Architecture

```
User Input → Perception → Cognitive Routing → InferenceGate → Model
                                                     ↓
Memory ← State Commit ← Verification ← AuthorityGateway ← Unified Will
                                                     ↓
                                              Tool Execution (sandboxed)
```

All consequential actions route through the Unified Will (`core/will.py`) and
receive an AuthorityGateway receipt. No silent side paths.

## AI Models

| Model Role | Type | Location | Purpose |
|-----------|------|----------|---------|
| Cortex (foreground) | 27B parameter LLM (`Aura-Cortex` / fused `Qwen3.8-27B`) | Local (MLX) | Main reasoning and conversation |
| Solver (deep) | 72B parameter LLM (4-bit) | Local (MLX) | Deep-reasoning hot-swap for hard problems |
| Reflex (fast lane) | 1.5B parameter LLM | Local (MLX) | Low-latency replies and routing |
| Brainstem (background) | 9B parameter LLM (`Qwen3.5-9B-4bit`) | Local (MLX) | Background / maintenance tasks |

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

**Not yet a control.** The P0–P6 evidence *promotion* state machine (ledger
item RR-07) is **not implemented** — `EvidenceLevel` exists as a declared
type and ceiling, but there is no promotion module in
`core/reality_reach/`. Do not cite evidence promotion as an operating
safeguard. The open ledger is in [docs/REALITY_REACH.md](docs/REALITY_REACH.md).

### Manage

| Control | Mechanism |
|---------|-----------|
| Human override | Operator can disable any capability via feature flags |
| Kill switch | `AURA_MODE=safe` disables all autonomous behavior |
| Audit trail | Will receipt log + state snapshots |
| Incident response | `docs/runbooks/` |
| Rollback | Backup/restore with state hash verification |

## Limitations

1. **Model limitations**: Local models have capability ceilings; may confabulate
2. **Memory limitations**: Long-term memory is best-effort retrieval, not perfect recall
3. **Tool limitations**: Tool execution is sandboxed and may fail
4. **Autonomy limitations**: Background autonomous behavior is bounded by Will governance
5. **Hardware requirements**: Requires Apple Silicon with ≥32GB RAM for full capability
6. **Network**: Some features require internet; offline mode has reduced capability

## Evaluation

See `docs/evidence/EVALUATION_REPORT.md` for detailed evaluation results including:
- AGI proof battery results
- Agency emergence testing
- Behavioral proof standard results
- Production readiness gate results
- Longevity soak results

## Human Oversight

See `HUMAN_OVERRIDE_POLICY.md` for the complete human oversight framework.

## Contact

For questions about this AI system: security@aura-project.dev

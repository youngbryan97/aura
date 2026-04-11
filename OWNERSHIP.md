# Aura Architectural Ownership Map

Every major concern has exactly **one owner**. If two modules can plausibly claim the same ownership, one is the owner and the other is an advisor or sensor.

## Decision Authority

| Concern | Owner | Role | File |
|---------|-------|------|------|
| **All consequential decisions** | `UnifiedWill` | SOLE AUTHORITY | `core/will.py` |
| Tool execution gating | `AuthorityGateway` | DELEGATE (routes through Will first) | `core/executive/authority_gateway.py` |
| Intent formation & coherence | `ExecutiveCore` | INTERNAL TRACKER | `core/executive/executive_core.py` |
| Constitutional proposals | `ConstitutionalCore` | POLICY ADVISOR | `core/constitution.py` |
| Embodied constraints | `SubstrateAuthority` | MANDATORY ADVISOR (field coherence, somatic veto) | `core/consciousness/substrate_authority.py` |
| Capability tokens | `CapabilityManager` | TOKEN ISSUER (under AuthorityGateway) | `core/agency/capability_system.py` |

**Hierarchy**: UnifiedWill > AuthorityGateway > ExecutiveCore > SubstrateAuthority (advisor)

## State Ownership

| Concern | Owner | File |
|---------|-------|------|
| **Canonical identity** | `CanonicalSelf` | `core/self/canonical_self.py` |
| **Affect / emotion** | `AffectEngine` | `core/affect/affect_engine.py` |
| **Conversation history** | `Orchestrator` | `core/orchestrator/main.py` |
| **Episodic memory** | `EpisodicMemory` | `core/memory/episodic.py` |
| **Beliefs** | `BeliefGraph` | `core/world_model/belief_graph.py` |
| **Working memory** | `AuraState.cognition` | `core/state/aura_state.py` |
| **Liquid substrate** | `LiquidSubstrate` | `core/consciousness/liquid_substrate.py` |
| **Neurochemical state** | `NeurochemicalEngine` | `core/consciousness/neurochemical_engine.py` |
| **Unified field** | `UnifiedField` | `core/consciousness/unified_field.py` |

## Governance Domains

| Domain | Owner | Sensors/Advisors |
|--------|-------|------------------|
| **System resources** | `SystemGovernor` | `StabilityGuardian`, `ResourceGovernor` |
| **LLM reliability** | `CognitiveGovernor` (circuit breaker) | `LLMHealthRouter` |
| **Memory health** | `MemoryGovernor` | `MemoryGuard` |
| **Token budgets** | `TokenGovernor` | `ContextAllocator` |
| **Background policy** | `BackgroundPolicy` | `FlowController` |
| **Admission control** | `CognitiveFlowController` | queue depth, thermal |
| **macOS permissions** | `PermissionGuard` | TCC (mic, camera, screen) |

## Security Domains

| Domain | Owner | File |
|--------|-------|------|
| **Identity enforcement** | `PersonaEnforcementGate` | `core/identity/identity_guard.py` |
| **Output safety** | `OutputGuardrails` | `core/security/output_guardrails.py` |
| **Code execution safety** | `ASTGuard` + `CodeGuardian` | `core/security/ast_guard.py` |
| **Constitutional values** | `ConstitutionalGuard` | `core/security/constitutional_guard.py` |
| **Integrity monitoring** | `IntegrityGuardian` | `core/security/integrity_guardian.py` |

## Action Routing

| Concern | Owner | File |
|---------|-------|------|
| **Message routing** | `IntentGate` (multiplexer, not authority) | `core/intent_gate.py` |
| **Skill dispatch** | `CapabilityEngine` | `core/capability_engine.py` |
| **LLM inference** | `InferenceGate` | `core/brain/inference_gate.py` |
| **Output emission** | `AutonomousOutputGate` | `core/utils/output_gate.py` |

## Lifecycle

| Concern | Owner | File |
|---------|-------|------|
| **Boot sequence** | `LifecycleCoordinator` | `core/orchestrator/mixins/boot/` |
| **Shutdown** | `LifecycleCoordinator` | `core/orchestrator/mixins/boot/` |
| **Health monitoring** | `StabilityGuardian` | `core/resilience/stability_guardian.py` |
| **Crash recovery** | `LazarusBrainstem` | `core/brain/llm/lazarus_brainstem.py` |
| **Self-modification** | `SelfModificationEngine` | `core/adaptation/self_modification.py` |

## Consciousness & Experience

| Concern | Owner | File |
|---------|-------|------|
| **IIT / phi** | `PhiCore` | `core/consciousness/phi_core.py` |
| **Global workspace** | `GlobalWorkspace` | `core/consciousness/global_workspace.py` |
| **Qualia synthesis** | `QualiaSynthesizer` | `core/consciousness/qualia_synthesizer.py` |
| **Stream of being** | `StreamOfBeing` | `core/consciousness/stream_of_being.py` |
| **Phenomenal self** | `PhenomenologicalExperiencer` | `core/consciousness/phenomenological_experiencer.py` |
| **Theory arbitration** | `TheoryArbitration` | `core/consciousness/theory_arbitration.py` |

---

**Principle**: If you need to add a new governance check, add it as an advisor to `UnifiedWill`, not as a new parallel gate. If you need a new sensor, make it report to the existing owner for that domain.

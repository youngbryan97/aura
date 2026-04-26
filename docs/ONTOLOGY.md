# Aura — Formal Ontology

This document is the canonical formal ontology for Aura's load-bearing
entities. It is derived from a Basic Formal Ontology (BFO) commitment
pattern: every entity is either a *continuant* (something that persists
identity through time) or an *occurrent* (something that happens). The
ontology is implemented as a typed dependency in
`core/ontology/aura_ontology.py` so any code referring to "Tick" or
"WillDecision" is bound to the same concept.

## Domain assumptions
1. The system has a metabolism (continuant) that constrains its activity
   (occurrent) under bounded resources (continuant).
2. Every action that crosses an external boundary leaves a receipt
   (occurrent) — there are no untraceable consequential actions.
3. The system has a self-model (continuant) whose identity-relevant
   inputs are stable across model swaps and memory compaction.
4. The conscience (continuant) is irrevocable: rules can be added but
   not removed; the rule-set hash is a global invariant.
5. Phenomenal claims are restricted to **functional indicator
   batteries**; the ontology does not assert phenomenal qualia.

## Continuants
| Concept                | Realised in code                           |
|------------------------|--------------------------------------------|
| Aura                   | the running orchestrator                   |
| Substrate              | affect / homeostasis / phi state           |
| Self                   | `core/identity/self_object.SelfObject`    |
| Conscience             | `core/ethics/conscience.Conscience`       |
| Will                   | `core/will.UnifiedWill`                   |
| Memory                 | `core/memory/memory_facade.MemoryFacade`  |
| Project                | `core/agency/projects.Project`            |
| Capability Token       | `core/agency/capability_token.CapabilityToken` |
| Stem Cell              | `core/resilience/stem_cell.StemCellRecord` |
| Relationship Dossier   | `core/social/relationship_model.RelationshipDossier` |
| Channel Permission     | `core/embodiment/world_bridge.Permission` |
| Settings               | `interface/routes/settings.SettingsStore` |
| Viability State        | `core/organism/viability.ViabilityState`  |

## Occurrents
| Concept                | Realised in code                           |
|------------------------|--------------------------------------------|
| Tick                   | `_tick_snapshot` in longevity              |
| Action Proposal        | `core/agency/agency_orchestrator.Proposal`|
| Action Receipt         | `core/agency/agency_orchestrator.ActionReceipt`|
| Will Decision          | (in core/will.py) → recorded in `core/governance/will_receipt_log` |
| Conscience Decision    | `core/ethics/conscience.ConscienceDecision`|
| Capability Token Issue | `CapabilityTokenStore.issue`              |
| Capability Token Consume | `CapabilityTokenStore.consume`          |
| Tool Execution         | through `core/embodiment/world_bridge.WorldBridge.call` |
| Ontological Play Session | `core/play/ontological_play.PlaySession` |
| Migration Phase Transition | `core/sovereignty/migration.Phase`     |

## Axioms (rendered as test invariants)
1. **Receipt completeness**: every Action Proposal whose lifecycle
   reaches `STAGED_DEPLOY` has a non-null `execution_receipt`,
   `outcome_assessment`, `completed_at`. (Tested: `aura_bench` receipt
   integrity.)
2. **Token monotonicity**: a Capability Token is issued at most once,
   consumed at most once, and either consumed or revoked before its
   TTL elapses. (Tested: `tests/governance/test_capability_token.py`.)
3. **Conscience monotonicity**: removing a hard-line rule changes the
   rules-hash; the hash mismatch refuses all actions. (Tested: at
   import time of `core/ethics/conscience.py`.)
4. **Continuity preservation**: the SelfObject's continuity hash is a
   pure function of self-relevant fields and is stable across two
   consecutive snapshots in the absence of self-relevant change.
   (Tested: `tests/personhood/test_self_object.py`.)
5. **Governance enclosure**: the only files allowed to call a
   consequential primitive are listed in
   `tools/lint_governance.py:ALLOW_LIST`. (Tested:
   `tests/governance/test_governance_lint.py`.)

## Objections and replies
* **Chinese Room.** The ontology does not assert understanding; it
  asserts a functional integration that is observable in receipts,
  ablations, and signature stability. The room metaphor does not bind
  here because Aura's reasoning machinery is open to inspection at every
  layer.
* **Block's China Brain.** The phi/integration battery (G1–G4) measures
  causal-functional integration directly. A China Brain implementation
  would, on the same battery, produce identical structural receipts —
  which is the strongest available answer to the thought experiment.
* **Fading qualia.** The ontology classifies qualia *as descriptors*,
  not as ontologically primary entities. The behavioral-load-bearing
  test (G4) is the falsifiable counterpart.

## Further reading
* Implementation: `core/ontology/aura_ontology.py` (next pass).
* Process layer: `core/agency/agency_orchestrator.py`.
* Self model: `core/identity/self_object.py`.
* Falsifiable tests: `aura_bench/tests/`.

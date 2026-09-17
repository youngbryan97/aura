# Recursive Self-Improvement (RSI) Architecture

*Verified against code as of 2026-09-15. Claim ONLY what is implemented.*

This document describes Aura's Recursive Self-Improvement (RSI) pipeline, which allows Aura to diagnose, draft, validate, and propose improvements to its own source code and architecture.

## What RSI Is and Is Not

**What RSI IS in Aura:**
- A bounded, read-only AST/source self-model over architecture files.
- A controlled pipeline for local code repair and validation using a deterministic shadow runtime and formal verifier.
- A fail-closed, constitutionally governed mechanism that explicitly isolates self-modification execution from production runtime.

**What RSI IS NOT:**
- **It is NOT unbounded or uncontrolled.** The process is constrained by formal safety rules, constitutional invariants, and mutation tiers.
- **Autonomous source promotion is DISABLED by default.** Normal desktop/server Aura sessions can diagnose and draft repairs, but they **CANNOT** promote patches to disk. Source promotion is reserved for an explicit repair-lab profile with opt-in switches.
- **It does NOT remove governance or bypass safety.** Self-improvement that can rewrite its own constraints is considered an unbounded process; Aura explicitly prohibits this via Tier 3 sealed paths.

## Module Counts

As of 2026-09-15, the RSI architecture comprises **64** primary modules:
- `core/self_modification/`: ~41 files orchestrating the self-modification engine.
- `core/self_improvement/`: ~23 files defining the self-improvement lab and program DNA.

## Self-Modification Engine Architecture

The self-modification engine (`core/self_modification/self_modification_engine.py`) serves as the orchestrator for the entire self-improvement system. It integrates error intelligence, kernel refinement, structural mutation, and safety harnesses to evaluate and prepare patches safely.

### Mutation Constitution & Tiers

All modifications are subject to `mutation_constitution.py`, which consults `mutation_tiers.py` to classify every source path into risk tiers. Aura enforces four distinct tiers:
- **Tier 0 (Free Auto-Fix):** Low-risk test files, docs, generated tools, UI assets. Eligible for auto-apply.
- **Tier 1 (Shadow Validated Auto-Fix):** General core and skill code. Eligible for auto-apply only after shadow validation and regression checks.
- **Tier 2 (Propose-Only):** Consequential agency, memory, and substrate paths. Aura may draft the patch but cannot apply it without explicit human approval.
- **Tier 3 (Sealed):** Critical safety, governance, and self-modification roots. Completely **sealed** from runtime self-modification. Changes require external review, manual patches, and a cold restart.

### Shadow AST Healer & Shadow Runtime

Code modifications are strictly tested via the `ShadowASTHealer` and executed in a `ShadowRuntime`. This provides an isolated execution environment where potential patches are validated against test harnesses before they are ever considered for application or proposal. 

### Promotion Gate & Promotion Policy

Aura operates under a **fail-closed safety policy** (`promotion_policy.py`). Normal runtime environments cannot write patches back to the source tree. Autonomous source promotion is disabled unless operators explicitly enable a repair-lab profile by setting specific environment variables (e.g., `AURA_ALLOW_AUTONOMOUS_PATCH_PROMOTION`, `AURA_ALLOW_REPAIR_LAB_SOURCE_PROMOTION`).

The `LabPromotionGate` (`core/self_improvement/promotion_gate.py`) evaluates candidate patches for promotion. It assesses comparison reports, guardrail audits, hardcoding audits, and syntax validations. A patch is rejected outright if it violates governance bounds.

## Self-Improvement Lab & Program DNA

The self-improvement lab (`core/self_improvement/`) is a clean-room reimplementation and analysis engine.

### Program DNA & Interface Contracts

The `program_dna.py` module builds a lawful behavioral "DNA" profile from authorized sources (e.g., open source, user-owned, public observation). It reconstructs a program's blueprint without decompiling proprietary code. Any changes must adhere strictly to predefined `interface_contract.py`, ensuring that public surfaces are preserved and modifications behave predictably.

## Evidence, Consent, and Authorization

### Lineage Enclosure & Patch Genealogy
Aura maintains a hash-chained RSI generation lineage (`lineage_enclosure.py` and `patch_genealogy.py`). This guarantees deterministic accountability for any autonomous self-improvement generation, preserving the patch history for external review.

### Repair Calibration & Standing Authorization
The systems behind `repair_calibration.py` and `standing_authorization.py` ensure that Aura's self-improvement loops act continuously but only within explicitly authorized boundaries. They calibrate confidence against formal proof obligations rather than arbitrary thresholds.

### Consent Invariants
The RSI pipeline observes strict `consent_invariant.py` checks. It cannot perform actions on external internet scopes without explicit opt-in, rejecting out-of-bounds propagation fail-closed and writing the refusal to the audit manifest.

## Evidence Boundaries

- **PROVEN:** Bounded hash-chained lineage tracking, formal verifier preservation of public surfaces, deterministic shadow runtime evaluations.
- **SUPPORTED:** Bounded local process-pool scaling for evaluations, clean-room Program DNA reconstruction.
- **BOUNDED:** Autonomous source promotion (restricted to explicit repair-lab opt-in), BCI/neural-decode affect inputs (capped as advisory context).
- **CONJECTURE:** Unbounded self-editing without regression (Aura formally rejects this).
- **REFUTED:** Bypassing governance constraints to rewrite safety files (Blocked by Tier 3 Sealed paths).

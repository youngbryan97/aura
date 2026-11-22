# Recursive Self-Improvement (RSI) Architecture

<!-- reader-navigation:start -->
> **In plain English:** Learning changes later behavior using earlier outcomes. A kept procedure is executable work retained for reuse under a defined checking rule.
>
> **Why it matters:** Experience should improve later work without accumulating unreliable shortcuts or expensive rules that no longer help.
>
> **In everyday use:** After solving one kind of task, the system may reuse a checked procedure on a later task of that kind instead of deriving every step again.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#learning) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#learning) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#learning) · [Document map](DOCUMENTATION_INDEX.md)
> [Glossary](GLOSSARY.md) · [Formulas with worked examples](FORMULAS.md)
<!-- reader-navigation:end -->

*Source policy reconciled 25 October 2026. Earlier module counts retain their
15 September scope. Policy admission is separate from successful live repair.*

This document describes Aura's Recursive Self-Improvement (RSI) pipeline, which allows Aura to diagnose, draft, validate, and propose improvements to its own source code and architecture.

## What RSI Is and Is Not

**What RSI IS in Aura:**
- A bounded, read-only AST/source self-model over architecture files.
- A controlled pipeline for local code repair and validation using a deterministic shadow runtime and formal verifier.
- A fail-closed, constitutionally governed mechanism that explicitly isolates self-modification execution from production runtime.

**What RSI IS NOT:**
- **It is NOT unbounded or uncontrolled.** The process is constrained by formal safety rules, constitutional invariants, and mutation tiers.
- **Low-risk repair and broad source promotion have different policies.** The
  safe auto-repair policy defaults on for eligible Tier 0/1 changes, subject to
  their checks. The broader repair-lab promotion path requires explicit opt-in.
  A normal-runtime blanket prohibition would misdescribe the current source.
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

[`promotion_policy.py`](../core/self_modification/promotion_policy.py) has
separate decisions. `safe_autonomous_repair_decision()` defaults enabled by
`AURA_ENABLE_SAFE_AUTO_REPAIR` and applies to eligible low-risk repairs after
their required gates. The broader autonomous promotion decision requires the
runtime, autonomous-promotion and repair-lab switches. Supervised promotion
has its own opt-in.

| Setting | Source default | Meaning |
|---|---|---|
| `AURA_ENABLE_SAFE_AUTO_REPAIR` | `True` | Eligible low-risk repair may proceed through its required gates; this does not grant all source paths authority |

[`mutation_constitution.py`](../core/self_modification/mutation_constitution.py)
checks mode, target tier, turn trust and owner approval. Production and live
manifests permit tier-gated self-modification; safe, simulated and test modes
refuse it. Its mode lookup currently permits tier checks to continue after
certain lookup errors, so this is not an unconditional fail-closed claim.
A policy admitting an attempt is not evidence that a particular repair was
applied safely or improved behavior; that needs a source-bound repair receipt.

The `LabPromotionGate` (`core/self_improvement/promotion_gate.py`) evaluates candidate patches for promotion. It assesses comparison reports, guardrail audits, hardcoding audits, and syntax validations. A patch is rejected outright if it violates governance bounds.

## Self-Improvement Lab & Program DNA

The self-improvement lab (`core/self_improvement/`) is a clean-room reimplementation and analysis engine.

### Program DNA & Interface Contracts

The `program_dna.py` module builds a lawful behavioral "DNA" profile from authorized sources (e.g., open source, user-owned, public observation). It reconstructs a program's blueprint without decompiling proprietary code. The predefined `interface_contract.py` states the public interfaces a change must preserve. Those contract checks supply a specific acceptance test for a candidate modification.

## Evidence, Consent, and Authorization

### Lineage Enclosure & Patch Genealogy
Aura maintains a hash-chained RSI generation lineage (`lineage_enclosure.py` and `patch_genealogy.py`). These records support reconstruction and review of recorded generations.
Coverage, persistence and verification of the actual lineage must be checked;
the existence of these modules does not prove every repair has a complete,
tamper-proof history.

### Repair Calibration & Standing Authorization
The systems behind `repair_calibration.py` and `standing_authorization.py` ensure that Aura's self-improvement loops act continuously but only within explicitly authorized boundaries. They calibrate confidence against formal proof obligations rather than arbitrary thresholds.

### Consent Invariants
The RSI pipeline observes strict `consent_invariant.py` checks. It cannot perform actions on external internet scopes without explicit opt-in, rejecting out-of-bounds propagation fail-closed and writing the refusal to the audit manifest.

## Evidence Boundaries

- **PROVEN:** Bounded hash-chained lineage tracking, formal verifier preservation of public surfaces, deterministic shadow runtime evaluations.
- **SUPPORTED:** Bounded local process-pool scaling for evaluations, clean-room Program DNA reconstruction.
- **BOUNDED:** Tier-gated low-risk repair, broader source promotion through its
  separate opt-in policy, and BCI/neural-decode affect inputs capped as advisory
  context.
- **CONJECTURE:** Unbounded self-editing without regression (Aura formally rejects this).
- **REFUTED:** Bypassing governance constraints to rewrite safety files (Blocked by Tier 3 Sealed paths).

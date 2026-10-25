# Aura: Supported Claims Ledger

<!-- reader-navigation:start -->
> **In plain English:** A test makes a specific claim answerable by a stated measurement, comparison and decision rule.
>
> **Why it matters:** A user needs to distinguish a built mechanism, a passing local test and a demonstrated useful capability.
>
> **In everyday use:** To show memory helps a task, compare the same task with memory available and disabled, and keep both successes and failures.
>
> [Example and metaphor](docs/PLAIN_ENGLISH_GUIDE.md#measurement) · [Variables and practical meanings](docs/VARIABLES_EXPLAINED.md#measurement) · [Demos and related work](docs/DEMOS_AND_RELATED_WORK.md#measurement) · [Document map](docs/DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

This document lists the proven engineering capabilities of the Aura agent runtime. Every claim listed here is backed by real code, automated tests, and execution logs.

---

## 1. Governed Runtime
* **Classification**: `causally demonstrated`
* **Definition**: All system actions and real-world effects (like editing files, running tools, or executing shell commands) are blocked by default. They require explicit cryptographic or rules-based permission from a central safety governor before running.
* **Code / Evidence Path**:
  - Central gatekeeper implementation: [core/executive/authority_gateway.py](core/executive/authority_gateway.py)
  - Unified system interface: [core/container.py](core/container.py)
  - Trace validation tests: `tests/` verifying closed-loop safety invariants.

## 2. Persistent Memory and Context Retrieval
* **Classification**: `locally demonstrated`
* **Definition**: Remembers information across separate sessions and machine restarts, retrieving relevant past experiences, facts, and procedures when needed.
* **Code / Evidence Path**:
  - Memory subsystem: [core/memory/](core/memory/)
  - Long-term continuous experience stream: [tests/test_continuous_experience_stream.py](tests/test_continuous_experience_stream.py)

## 3. Causal Internal State & Homeostasis
* **Classification**: `locally demonstrated`
* **Definition**: A dynamic set of internal variables (such as energy levels, stability indicators, and focus registers) directly shapes which actions the agent chooses and how it builds prompts.
* **Code / Evidence Path**:
  - Homeostatic state tracking: [core/state/aura_state.py](core/state/aura_state.py)
  - Introspective state verification: [proof_kernel/tests/test_proof_kernel.py](proof_kernel/tests/test_proof_kernel.py)

## 4. Affect Steering
* **Classification**: `locally demonstrated`
* **Definition**: Uses simulated emotional and mood states (affect) to adjust how the agent focuses attention, how strictly it runs tools, and how far ahead it plans.
* **Code / Evidence Path**:
  - Affect updating logic: [core/phases/affect_update.py](core/phases/affect_update.py)
  - Affect-state coupling: Automated tests verifying modulated prompt outputs under simulated high-stress vectors.

## 5. System 2 Deliberate Planning and Search
* **Classification**: `locally demonstrated`
* **Definition**: Simulates and compares possible action paths in advance (using Monte Carlo tree search and counterfactual evaluation) before making real changes to the system or environment.
* **Code / Evidence Path**:
  - Speculative path rollouts: [core/cognition/mcts_world_model.py](core/cognition/mcts_world_model.py)
  - Speculative tree validation: Automated unit tests evaluating planning score optimization.

## 6. Diagnostic Self-Repair
* **Classification**: `locally demonstrated`
* **Definition**: Automatically inspects error traces, generates targeted code patches, and runs recovery loops to resolve temporary problems like file locks or resource conflicts.
* **Code / Evidence Path**:
  - Diagnostic ladder: [core/runtime/self_repair_ladder.py](core/runtime/self_repair_ladder.py)
  - Self-healing checks: Automated tests injecting runtime errors and confirming automatic rollback or recovery.

## 7. Sandboxed Self-Modification
* **Classification**: `locally demonstrated`
* **Definition**: Can propose code updates to its own skills, which must pass static analysis, branch isolation, and unit tests in an isolated sandbox before being merged. Live runtime changes are proposal-only by default.
* **Code / Evidence Path**:
  - Mutation safety analyzer: [core/self_modification/mutation_safety.py](core/self_modification/mutation_safety.py)
  - Safe promotion pipeline: [core/self_modification/safe_modification.py](core/self_modification/safe_modification.py)
  - Modification sandbox: Skill mutation and supervised-promotion automated tests.

## 8. Operational Volition
* **Classification**: `causally demonstrated`
* **Definition**: Makes deliberate, proactive choices rather than just reacting to prompts. The agent scores alternatives against each other and signs explicit intent records (called Will Decisions).
* **Code / Evidence Path**:
  - Unified Will decision: [core/governance/will.py](core/governance/will.py) (`UnifiedWill.decide` → `WillDecision` with cryptographic receipt IDs and outcome/veto logic)
  - Authority enforcement: [core/executive/authority_gateway.py](core/executive/authority_gateway.py)
  - Telemetry receipt files: Dynamically generated `RECEIPTS.jsonl` traces under active runtime profiles.

  > Correction (2026-07-06): this entry previously cited `core/runtime/task_ownership.py`, which is generic asyncio task-lifecycle tracking ("use this instead of raw asyncio.create_task") and has nothing to do with Will or decision logic. The real Will logic is `core/governance/will.py`.

## 9. Boxed Entity Confinement Safety
* **Classification**: `locally demonstrated`
* **Definition**: Strictly honors containment boundaries during execution. Sandboxed code cannot write outside allowed directories, escape directory trees, or run unauthorized subprocesses.
* **Code / Evidence Path**:
  - Hardened sandbox boundaries: [tests/test_sandbox_hardening.py](tests/test_sandbox_hardening.py)

## 10. Production-Sealed Hardening
* **Classification**: `causally demonstrated`
* **Definition**: Eliminates ad-hoc or unvetted tools, enforces strict static typing, and fails closed (stops execution safely) if critical infrastructure breaks or degrades.
* **Code / Evidence Path**:
  - Strict mode configuration: [core/runtime/mode.py](core/runtime/mode.py)
  - Fail-closed container intercepts: [core/container.py](core/container.py)
  - Clean master verdict generation: `make certify` (which verifies 100% test compilation, boot gateway probes, live Aletheia benchmarks, and architecture ablations).

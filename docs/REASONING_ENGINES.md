# Reasoning Engines & Execution Boundaries

This document catalogues what Aura has taken out of the model's hands.

Aura's core philosophy for cognition is to stop relying on prompts to make the model "act smart", "be careful", or "not hallucinate." Instead, we enforce correctness structurally by extracting verifiable computation to deterministic deduction engines, and isolating execution to OS-level sandboxes. 

The architecture moves the model from being a *black-box oracle* to a *proposer* in a verifiable search process.

## 1. The Reasoning Amplifier

The Reasoning Amplifier is the mandatory hard-task cognition layer. It ensures Aura almost never uses a simple `prompt → one generation → answer` pipeline for complex problems.

### Reasoning Amplifier v1: Verifier-Filtered Self-Consistency
Implemented in `core.brain.reasoning_amplifier`, v1 acts as the causal mechanism for lifting a base model's accuracy on hard tasks.
- **How it works:** It samples multiple reasoning paths, passes each through Aura's deterministic verification engines (like the symbolic bridge), and throws out paths that contain provable errors (e.g., non-sequiturs or arithmetic flaws).
- **Consensus:** It takes the answer most of the *verifier-clean* paths converge on. 
- **Verifiability:** A crash in a verifier yields `UNKNOWN`—which explicitly does not count as a pass. Absence of a check is not a passed check.

### Reasoning Amplifier v2: Execution Modes
Implemented in `core.brain.reasoning_amplifier_v2`, the amplifier adjusts the compute budget dynamically based on the cognitive integration phase (Φ thresholds: dormant, reactive, deliberate) and task stakes.
It defines 5 execution modes:
- `FAST`: 1 candidate + calibration (casual / low-stakes).
- `NORMAL`: 3 candidates + verifier + self-consistency.
- `DEEP`: Up to 9 candidates + verifier + courtroom judge.
- `EXTREME`: Courtroom + sandbox repair + memory (high stakes).
- `PROOF`: Refuses to answer unless the output is verifier-clean.

## 2. Adversarial Courtroom
Implemented in `core.brain.courtroom`, this module forces adversarial cognition over a single local model by isolating roles. Roles cannot see each other's drafts until they have spoken independently:
- **Solver**: Produces candidate answers and reasoning.
- **Skeptic**: Independently lists how answers usually go wrong, then explicitly attacks the Solver's answer.
- **Evidence Clerk**: Assembles grounding from supplied evidence.
- **Verifier**: Runs deterministic truth engines (not an LLM opinion).
- **Simplifier**: Reduces the winning answer to its minimal correct form.
- **Judge**: Rules on the assembled record, explicitly honoring the verifier (cannot certify a candidate that fails mechanical verification).

## 3. Native System-2 Reasoning
Implemented in `core.reasoning.native_system2`. This is not an LLM prompt wrapper. It is a genuine explicit search tree of latent plans.
- Maintains a search tree of latent plans (MCTS, BEAM, BEST_FIRST).
- Evaluates nodes via a value interface and simulates outcomes using a side-effect-free world model.
- Emits an auditable commitment receipt before a plan is executed.

## 4. Deterministic Deduction & Arithmetic
The model's math and logic are checked by trusted kernels, using the de Bruijn criterion (untrusted search/elaborator + trusted checker).

### Proof Kernel & Natural Deduction
Implemented in `core.reasoning.proof_kernel`.
- **Trusted Checker**: Re-verifies every produced proof term (tableau prover) node-by-node against fixed rule schemas.
- **Axiom Audit**: Computes exactly which premises a refutation touches.
- **Fail-Closed**: A theorem transitively resting on an admitted claim (`sorry`) is visibly tainted until discharged.

### Linear Arithmetic Solver
Implemented in `core.reasoning.linear_arithmetic`.
- **Search**: Fourier-Motzkin elimination (untrusted).
- **Checker**: Re-verifies Farkas certificates independently using exact `Fraction` arithmetic, summing and confirming cancellation without trusting the search order.

### Symbolic Bridge
Implemented in `core.reasoning.symbolic_bridge`.
- Serves as the neuro-symbolic bridge extracting claims from model output and auditing them for provable non-sequiturs and arithmetic bugs, feeding results directly into the Reasoning Amplifier.

## 5. Scientific Engine
Implemented in `core.cognition.scientific_engine`.
- Provides a real causal loop for hypothesis testing: `hypothesis → experiment → belief update`.
- **Mechanism**: Forming a hypothesis commits an *expected* observable. Running an experiment opens a receipt. Observing the result resolves the receipt, assigns credit, and drives a Bayesian-flavored confidence update.
- Persisted in SQLite so experiments can cross session boundaries.

## 6. Execution Sandboxing & Coding ACI
Executing model-authored code requires hard boundaries, not just prompt instructions.

### OS-Level Sandboxed Python Execution
Implemented in `core.sandbox.untrusted_python`.
- **The Boundary**: Uses kernel boundaries (`sandbox-exec`/Seatbelt on macOS, `bwrap` on Linux) to confine execution. Denies network outright. Limits filesystem to the interpreter's read paths and a scratch directory.
- **Fail-Closed Policy**: If no boundary is available, the system *refuses to run the code*. This explicitly fixes the vulnerabilities of AST blacklists and the `python -I` flag, which fail to isolate the process.

### Coding ACI & Code Repair Pipeline
Implemented in `core.self_modification.coding_aci`.
- **Repository Surface**: Exposes a typed, small interface for code edits rather than a shell. 
- **Bounded Views**: Navigation returns structure (AST outlines). Read sizes are bounded. Edits are structural (`replace_definition`), preventing regex collision bugs.
- **Test Selection**: Finds tests that touch the modified code dynamically, rather than guessing.

## 7. Trace Compiler & Procedure Execution
Implemented in `core.cognition.trace_compiler` and `core.cognition.procedure_execution`.
- Reads the event DAG of successfully completed tasks. 
- Computes minimal support, generalizes what varied across multiple runs, and emits a reusable `Procedure`.
- Refuses to compile anecdotes (defaults to a minimum of 3 occurrences) and measures exact deliberation cost saved.

## 8. CAD/Geometry/Materials Engine
Implemented in `core.engineering.geometry`.
- Parametric solids know their own exact volume, mass, and shape. 
- Math is exact and relies on material properties and dimensions rather than LLM estimation. Generates parametric wireframes and triangular meshes deterministically.

## 9. Evidence Boundaries: What These Engines DON'T Replace
While Aura enforces strict verification on logic, math, and execution, it is critical to note what these boundaries **do not** do:
- They **do not** replace the model's world knowledge. The verifiers check *consistency*, not external truth (unless explicitly grounded via Evidence Clerk).
- They **do not** guarantee an optimal search path, only that the *selected* path is free of provable structural/arithmetic errors.
- A passed check means "no provable error found", not "provably optimal."

*Note: Test counts and verification reliability metrics are continually measured by the internal verifier foundry.*

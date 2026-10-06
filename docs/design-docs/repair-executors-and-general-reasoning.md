# Repair executors and general reasoning

Architectural strengthening dated 2026-10-05. This records mechanisms and evidence boundaries, not a claim of parity with a frontier model.

## Research and engineering decisions

Agent interface design affects repository navigation, editing and execution performance. Aura already has a structural coding surface; this change preserves decorators and exact line endings, refuses ambiguous definitions, and protects undo from intervening edits. [SWE-agent paper](https://arxiv.org/abs/2405.15793).

Automatically evaluated candidate search can improve algorithms when objective evaluators exist. Aura's existing reasoning amplifier now receives typed independent function examples from the coding skill; examples are evaluated in the existing OS sandbox, outside the candidate's authority. The repair lane pins original examples before trying edits and preserves previously passing cases. This is a bounded repair application of evaluator feedback, not a reproduction of AlphaEvolve's scientific results. [AlphaEvolve paper](https://arxiv.org/abs/2506.13131).

Observable application state and mechanically enforced architectural boundaries support reliable execution. Repair observation now uses the same rendered-object machinery as live control, falling back to pixels when the drawing cannot be faithfully represented. Directed function experiments complement the visible run, and saved file effects have independent readback receipts. [OpenAI harness engineering](https://openai.com/index/harness-engineering/).

End-state grading and repeat trials distinguish reliable execution from a lucky successful attempt. The presentation proof must include an independently graded repaired file, the complete live attempt receipts, and a reset to a demonstrably broken original. Regression tests do not establish broad coding benchmark performance. [Anthropic agent evaluation methods](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

## Production connections

| Requirement | Existing machinery and added connection | Proof boundary |
| --- | --- | --- |
| Exact calculations | Bounded AST arithmetic compiles to the metered universal floor; rational final answers compare exactly | Supported closed integer expressions and exact numeric comparison; unsupported questions remain unknown |
| Logic | Existing symbolic bridge, proof kernel and countermodels | Only claims actually parsed and checked; no new general natural-language logic qualification |
| Python generation | Coding skill → existing reasoning strategies/amplifier → code verifier → OS sandbox function calls | Independent supplied examples can reject a draft; draft-written examples are explicitly labelled self-examples |
| Python repair | Original literal examples pinned → structural/model candidate edits → sandbox checks → preserve passing cases → verified save | Passing examples is evidence about those functions and inputs; no examples means behaviour remains unverified |
| Browser repair | Structural suspicions → fresh isolated function cases → before/after visible runs → verified file readback | Reflection is an inferred source contract; open edges are not universally walls |
| Live control | Rendered-object observations → tracked motion → supplied boundary contract → witnessed violation → bounded repair retry | No private application variables control play; diagnostic function inputs exist only in isolated copies |
| Partial failure | Full bounded audit payload plus a separate successful file-effect receipt | A failed overall task is still failed; observed changes remain reportable |

Reference material, the resident model, imagination and reasoning systems remain sources of hypotheses and context. They do not approve a repair or override a counterexample. Invoking every module on every task would add latency without establishing correctness; deterministic checks run first where an appropriate executor exists. No task-specific prompt changes are part of this work.

A missing visual track can be extrapolated beyond a wall even after the real object reflected. That prediction cannot establish a runtime defect. The motion monitor now requires an observed centre beyond a required edge with outward motion; missing-track predictions remain bounded, unconfirmed evidence. A complete renderer observation can expose a partially clipped body before it disappears. Pixel-only observations may leave that breach unmeasured, so isolated function checks and the visible run provide separate evidence. The boundary's meaning still comes from an explicit or source-inferred contract, with its provenance retained.

Repair narration also publishes factual progress to the fenced chat delivery owner. Reference excerpts stay in the repair receipt rather than interrupting work narration. A live skill registry can retain an old method after module reload; the presentation replay therefore uses a controlled full runtime restart at the tested revision. Revalidating a previously held input model as data protects schema compatibility when the current method is actually installed.

## Qualification beyond the presentation

An inconclusive control experiment no longer exhausts keyboard discovery for the rest of a run. After a bounded wait, the controller starts a fresh experiment and releases previous object exclusions and response calibration. The absence check allows enough time for those experiments; the task deadline still bounds execution. Per-attempt receipts distinguish successfully delivered key presses from pictures with an established controlled object. These measure input and attribution separately; neither alone proves a win.

To compare Aura with a current coding agent, run the same held-out repository tasks with unchanged model weights, clean initial state, a fixed time/token budget and independent fail-to-pass plus pass-to-pass tests. Report first-attempt success, all-trial consistency, unknown outcomes, p50/p95 latency, candidate count and executor cost. Pair an ablation of the added executors with the same task set and hardware. Separate regression fixtures from held-out capability measurements; renamed fixture tests establish interface transfer, not unseen repository reasoning.

The universal floor supplies exact, metered computation. It does not by itself establish correct interpretation of arbitrary user goals. A larger deterministic executor library expands what can be proved once a task is grounded; measured semantic grounding remains necessary.

## Behavior expressed as executable functions

A capability can be decomposed into operations with observable inputs, outputs and failure conditions. This permits implementation and diagnosis without changing a model's task prompts. The implementation must preserve the uncertainty that the measurements leave unresolved.

| Observed practice | Executable function | Current connection and limit |
| --- | --- | --- |
| Identify what an action controls | Repeated independent input experiments, candidate witnesses, evidence epochs and explicit unknown results | The reusable causal identification executor is connected to real-time control; it does not identify arbitrary natural-language concepts |
| Maintain an identity through changes | Preserve an established response; use shape and physically reachable motion for track continuity | A distant lookalike cannot inherit controls; absence or contradiction starts fresh experiments |
| Check a hypothesis | Run a bounded experiment whose result is independent of its author | Original examples, isolated boundary cases and sandbox checks reject unsupported repair candidates |
| Find an error that normal use misses | Choose directed cases near a structural boundary and preserve passing cases | JavaScript reflection checks and literal Python examples cover the supported contracts; arbitrary specification inference remains unqualified |
| Calculate precisely | Parse a supported expression, execute exact operations and compare exact values | The metered floor and rational comparison cover closed integer expressions and numeric answers |
| Explore alternatives | Generate candidates, simulate supported transitions and compare measured effects | Existing amplifier and imagination connections propose alternatives; no new frontier creativity qualification |
| Learn from execution | Retain counterexamples, provenance, versions and bounded decision receipts | Repair audits and causal attribution receipts expose failures; summaries alone cannot certify knowledge |

Improvement starts with the earliest incorrect operation in a failed trace. An observation failure calls for a better measurement; ambiguous identity calls for a distinguishing experiment; a wrong candidate calls for a counterexample; a failed effect calls for a stronger executor. Passing a later gate cannot repair an earlier interpretation error. Held-out, repeated evaluations are needed to measure the resulting breadth and consistency.

## Language profile extension

Aura already has a source registry and a content-addressed compiled-understanding library. Its concept digests and bridge index can retain definitions, constraints and cross-domain connections. The current digest grounding score measures word overlap; it does not prove a language rule or establish programming expertise. A summary about an API must retain its source and version, and executable checks must remain independent of that summary.

The extension should use typed language profiles containing grammar identity, compiler/runtime version, checker adapters, package and build metadata, official reference identifiers and content hashes. Tree-sitter supplies a common parsing interface and error nodes, rather than semantic correctness. [Tree-sitter documentation](https://tree-sitter.github.io/tree-sitter/index.html). Language checkers supply separate measurements: Pyright checks Python types and inference; Clippy provides Rust correctness and idiom lints. [Pyright documentation](https://github.com/microsoft/pyright/blob/main/docs/type-concepts.md), [Clippy documentation](https://doc.rust-lang.org/clippy/).

Each profile needs isolated compile/type/lint operations, independent examples and property checks, a dependency graph for affected-test selection, and receipts distinguishing passed, failed, unavailable and unmeasured. Resource limits and file custody belong to the executor. Candidate generation, analogy and imagination can propose alternatives; measured counterexamples decide which candidates survive. Verified cases can enter the reusable library with their assumptions and tool versions, and must be rechecked after an incompatible version change. This is an extension design, not installed support for every language.

Qualification requires held-out tasks for each profile, including ambiguous requirements, multi-file changes, concurrency, numerical limits and adverse dependency versions. Report language coverage, unknown rates and repeat-trial outcomes. Neither a parser catalogue nor a large reference collection is evidence of world-class performance.

## Worker isolation

Canvas decoding and chat journal operations previously used the default worker pool. Unrelated blocking work could therefore stop control observations and delay lease renewal past its expiry. Frame decoding now has two reserved interactive workers and a bounded wait; journal operations use the existing ordered durable receipt lane and retain their governance context. Pool saturation tests exercise the actual canvas decoder and admission, renewal, progress, finalization and replay reads while the default pool is occupied. A live stall's exact worker stack was not captured, so saturation is a reproduced failure mechanism, rather than a retrospectively proven cause of every delay in that run.

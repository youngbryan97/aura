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

# Research Request: Close Aura's General Reasoning Ledger

Copy this entire document into your research-capable AI. It is a snapshot of
the engineering record on September 20, 2026, not a claim that the open work
has succeeded. Local paths are navigation hints for reviewers given a source
bundle. If you cannot access them, do not claim to have inspected them.

---

## Assignment

Act as an independent research and engineering reviewer of Aura, a local,
systems-oriented AI assistant. Produce a rigorous, implementation-ready report
that addresses EVERY open item G03-G10 and G12 below, preserves the evidence
behind G01/G02/G11/G13, and identifies an efficient route to completion.

We need solutions, not reassurance, a list of fashionable techniques, or a
rewording of the problem. Develop competing approaches, derive their
assumptions, try to falsify them, implement and test tractable reference
mechanisms, and recommend a coherent integrated design. You may challenge
the architecture, experiment, optimization objective, or interpretation of
results. Recommend replacement or deletion where justified.

Do not assume that arbitrary-domain generalization, perfect accuracy, or
frontier performance follows from owning the code or having a computationally
universal interpreter. Conversely, do not dismiss feasible engineering as
impossible without a precise obstruction. Distinguish mathematical
impossibility, missing information, computational cost, implementation defects,
unmeasured hypotheses, and failures of the present design.

Use your available research, code execution, formal reasoning, and evaluation
tools. Provide checkable derivations, experiments, and conclusions, not private
chain-of-thought. Never invent citations, tool executions, measurements, code
access, contemporary model rankings, or proofs. If execution is unavailable,
label the implementation untested and give exact reproduction instructions.

## Goal And Constraints

Aura should interpret ordinary language, retain conversational meaning, use
knowledge and tools, compose reliable computations, reason and plan, learn
reusable abstractions, and express correct complete public answers. We want
measurable reasoning gains beyond its resident model, general transfer beyond
trained templates, and eventually credible frontier comparisons.

The architecture supplies much of the intelligence. The resident language
model is used for language, reasoning, and planning; Python modules and an
existing computational substrate supply objective execution. Deterministic
execution is only as sound as the specification, input grounding, arithmetic,
implementation, and external observations. Do not treat Python as an oracle.

Engineering constraints:

- Local serving, on a 64 GiB unified-memory Mac. Storage capacity is separate
  from RAM. No production dependency on a cloud model is desired.
- The resident artifact is locally named Aura-Qwen3.8-27B; historical metadata
  describes a Qwen3.5-derived hybrid backbone. Treat those as local artifact
  labels, not independently verified public model identities. Require the
  actual config, tensor, tokenizer, and fusion manifests before model-specific
  recommendations. Reported geometry is 64 layers, hidden width 5120, with 16
  attention and 48 gated-delta layers; confirm against the artifact.
- Persona tissue has a fusion receipt. Steering and recurrent tissue must be
  qualified separately for the current basis. Equal tensor shapes do not
  establish representational compatibility.
- Reuse the existing language substrate, universal computational floor,
  procedure registry, evidence system, connectome, memory, knowledge, and
  learning infrastructure. Do not propose a disconnected second architecture.
- No prompt engineering as the repair: no instructions telling the LLM to
  reason better, special benchmark wording, answer-specific prompts, keyword
  patches, or hidden expected-answer routing. Ordinary task communication is
  still necessary; the improvement must reside in an actual mechanism.
- No family-ID shortcuts, test-label leakage, oracle candidate selection,
  answer injection, or post hoc changes to make a score pass.
- Do not improve apparent reliability by rejecting every difficult request,
  deleting correct answers, concealing failures, or truncating reasoning.
  Measure coverage, correct completions, abstention, and latency separately.
- More compute is acceptable when it helps, but charge retries, verification,
  retrieval, search, and warm-up fairly. Do not presume unlimited resources.
- Runtime improvements must be integrated and actually exercised through the
  ordinary user path. A module, unit test, or health flag alone is insufficient.

## Every Ledger Obligation

The closed entries are historical, scoped closures, not blanket qualification
of subsequent code or artifacts. Preserve them and rerun affected checks.

| Item | Obligation | Recorded status |
| --- | --- | --- |
| G01 | Freeze baseline and exact mechanism/claim boundary | Closed, September 8; identity-bound historical evidence |
| G02 | Reconcile bounded 1.5B/32B/27B evidence and activation | Closed, September 12; negatives retained; composition shadow-only |
| G03 | Close learned semantic binding/composition failures on development tasks using existing substrates | OPEN; principal development bottleneck |
| G04 | Held-out construction, vocabulary, depth, and family transfer; distinguish neural computation from executable assistance | OPEN |
| G05 | Translate internal gains into correct freely decoded public answers | OPEN |
| G06 | Compare ordinary model, equal-compute alternatives, matched controls, causal lesions; count selection, retries, regressions | OPEN |
| G07 | Preregister powered fresh-task/seed replication and stopping rules | OPEN; planning machinery exists, candidate-specific confirmation does not |
| G08 | Independently verify artifacts, uncertainty, contamination, and cross-domain outcomes | OPEN |
| G09 | Establish broad reasoning gain beyond bounded synthetic success | OPEN |
| G10 | Qualify materialization/fusion on the current model: geometry, identity, rollback, canaries, no-regression behavior | OPEN |
| G11 | Prove the qualified mechanism serves eligible live requests | Closed, September 14, for a bounded mechanism and exact configuration; not G03's unqualified candidate |
| G12 | Evaluate frontier performance against named current baselines on independent broad tasks with fair resource/tool accounting | OPEN |
| G13 | Keep public RLC claims precise and aligned with measurements | Closed historical reconciliation; new results still require updates |

For EACH open item, supply its dependency set, current evidence, missing
mechanism or evidence, implementation package, acceptance experiment,
falsifier, likely failure modes, and completion criterion. Do not omit an item
because another item is difficult. Identify work that can proceed concurrently.

## Current G03 Mechanism

The learned front end compiles source-token IDs and frozen language-model
hidden features into typed semantic programs. It recovers source literals,
operation spans, argument references, and definition ownership. Learned
operation, pointer, and relation components score candidate graphs. Typed
argument optimization enforces structure; joint selection compares complete
graphs. The selected program executes through existing procedure/floor code.

Relevant components include:

- `core/learning/semantic_program_compositional_transducer.py`
- `core/learning/semantic_argument_chart.py`
- `core/learning/semantic_argument_optimization.py`
- `core/learning/semantic_joint_graph_learning.py`
- `core/learning/semantic_relation_graph_learning.py`
- `core/learning/semantic_operation_graph_learning.py`
- `core/learning/semantic_candidate_bank.py`
- `core/learning/semantic_failure_diagnosis.py`
- `core/learning/semantic_cohort_diagnosis.py`
- `core/learning/semantic_validation_checkpoint.py`
- `core/learning/semantic_program_runtime.py`

The important distinction: a well-typed, connected, acyclic, executable graph
may still express the wrong meaning. Improving its numerical training margin
does not ensure the runtime decoder selects the intended graph.

Existing diagnostics separate grounding, candidate reachability, semantic
selection, execution, and public emission. Candidate generation must be blind
to target annotations; diagnostic comparison happens afterward. If bounded
search fails to find the target without proving exhaustion, reachability is
UNKNOWN, not false. Semantic success does not imply public execution success.

## Measured Development Evidence

These are exposed development results, not independent fresh transfer. Source
and validation counts below name different cohorts; do not pool them.

1. Archived inventory: 1,764 examples, 656 schemas, eight families. The current
   audit covers all 764 training plus 500 development-validation observations.
   The remaining test split is excluded from this audit.
2. An earlier full-source candidate reached 764/764 training, but development
   validation fell from the parent's 488/500 to 472/500. It was rejected.
3. A later supervised conditional pilot fit five source examples with auxiliary
   constraints from all 764 training observations. Source decoding rose 2/5
   to 5/5; wrong/tied numerical comparisons fell 1341 -> 352 -> 76 -> 12.
   On a 100-case development slice: parent 99, unfit conditional model 98,
   trained candidate 96. Three regressions, no gains; rejected.
4. Component interventions on that pilot: applying only its operation-pointer
   update reproduced all three regressions. Restoring the parent's pointer
   repaired those regressions but lost one repaired source case. This identifies
   a causal tradeoff in that experiment, not a universal diagnosis.
5. A retain-existing conditional pilot used the same five source examples and
   764 auxiliary observations, two updates. Numerical wrong/tied comparisons
   fell 8 -> 0 then 4 -> 0; source redecoding reached 3/5 then 5/5. No retained
   positive inequality reversals were reported. Yet validation was 98/100
   against parent 99/100: one regression, no gains.
6. Without further training, that candidate reached 24/25 on an expanded
   diagnostic source cohort against parent 16/25. This is useful development
   improvement, not fresh transfer; one source error still remained.
7. Further scoring changes are frozen pending the full 1,264-observation
   diagnosis. At this brief's snapshot, 162 durable audit rows are semantically
   equivalent; the process stopped before completion and is being resumed.
   There is no final full-cohort result. Do not extrapolate from the prefix.

Versioned checkpointing retains real candidate states and numerical problem
archives. Model, source observations, implementation, and search allowances
are identity-bound. Failed candidates are not promoted. Historical results
are append-only.

Please address the recurring pattern directly: fitting known constraints
improves chosen examples yet perturbs global decoding elsewhere. We need a
cohort-level explanation and repair, not a special case for each remaining row.

## Public Answer And Causal Evidence

- Bounded typed-machine/processor results exist, including historical strong
  treatment/control differences on both old and current backbones. Some of
  the decisive tissue is model-independent. This does not prove arbitrary
  neural reasoning or cross-domain semantic interpretation.
- One natural-composition result was 21/48 versus ordinary controls 1/48 and
  2/48; it remained shadow-only.
- A public composition diagnostic had treatment 8/8 and controls 0/8, but
  omitted operation definitions from the ordinary interface, disabled native
  thinking, and repeated one construction. It demonstrates a narrower effect
  than general reasoning gain.
- A native coding diagnostic had ordinary native reasoning 6/6, treatment 6/6,
  controls 0/6. That is no accuracy gain over the ordinary model.
- A current-model public-shape canary reached 3/3 correct EOS answers. This
  verifies a component, not general RLC gain.
- A steering comparison retained 216 decodes and a negative result: one
  treatment target-score win versus four matched no-op wins, with regressions.
  The new steering generation was not qualified or published.
- Runtime evaluation shortcuts previously supplied canned answers before the
  normal pipeline. That route was removed for non-safety evaluation requests;
  broad evaluation still needs actual ordinary-runtime measurements.

Require correct public answers, complete termination, honest accounting of
unparsed/censored output, and independently justified semantic grading. Never
grade private reasoning as if the user received it. Distinguish useful
structured execution from merely making the answer easier to serialize.

## Existing Architecture To Reuse

The following is an inspected implementation map, not evidence that every
connection is useful or qualified:

| Purpose | Existing entry points |
| --- | --- |
| Language/work meaning | `core/language/semantic_work.py` |
| Computational floor | `core/cognition/the_floor_she_stands_on.py`, `core/learning/semantic_program_floor.py` |
| Shared procedures/types | `core/cognition/procedure.py`, `procedure_execution.py`, `core/learning/semantic_procedure_currency.py` |
| Knowledge/evidence | `core/evidence/packet.py`, `core/knowledge/atomspace.py`, episodic/other memory, offline corpus, search |
| Plans/effects | `core/cognition/tool_plan.py`, `an_action_she_composed.py` |
| Search allocation | `core/cognition/value_of_computation.py`, `how_far_the_search_reaches.py` |
| Connectivity | `core/connectome/integration.py`, cognitive contract health |
| Explicit/implicit/neural forms | `core/cognition/dual_knowledge.py` |
| Live semantic intervention | `core/brain/llm/compositional_semantic_shadow.py` |
| Ordinary cognition | `core/brain/cognitive_engine.py` |
| Outcome/retrieval experiments | `core/learning/integrated_reasoning_eval.py` |

The floor supports functions, recursion, and pairs, but the learned semantic
front end is narrower, principally integer/integer-sequence procedures.
Expressibility and learnable, affordable discovery are separate problems.
The shared action-conditioned world-rollout work exists; broad predictive
accuracy of its learned transitions is not established.

There is already a broad intrinsic-recurrence runner and a frontier comparison
runner: `tools/run_unified_recurrent_broad_canary.py` and
`tools/measure_frontier_gap.py`. Their protocols may be reused, but their
treatment identities differ from the semantic compiler. A diagnostic solver
using an amplifier is not automatically a test of the whole live system.

## Questions You Must Solve Or Bound Precisely

### A. Semantic Selection And Learnability

1. State a generative/statistical model of intended meanings, observations,
   annotations, candidate programs, and decoder decisions. Identify which
   variables Aura observes. Diagnose identifiability before optimization.
2. Separate ambiguous language, inadequate hidden features, corrupted labels,
   missing hypotheses, approximate search, and a miscalibrated selection
   objective. Design experiments that distinguish them with few runs.
3. Explain why satisfied sampled inequalities can coexist with changed runtime
   argmax decisions. Examine joint normalization, graph-length effects, latent
   assignments, new competitor graphs, hidden negative classes, and drift
   across coupled components. Derive a decoder-consistent objective.
4. Design semantic selection that can transfer across vocabulary, construction,
   depth, and genuinely new families. Compare structured prediction, latent
   variable objectives, equivariant representations, contrastive binding,
   program synthesis, abstraction induction, calibrated search and retrieval.
   These are candidates to investigate, not a mandated shopping list.
5. Specify when retaining the baseline answer prevents regressions and when it
   does not. A selector cannot know which answer is correct merely because one
   is the baseline. Derive feasible conditional guarantees and their costs.
6. Give a way to distinguish an objective defect from insufficient hypothesis
   capacity before committing to another expensive fit. Include exact small
   enumerations or constructive counterexamples where possible.

### B. Representation, Search, And Generality

7. Show how to compile knowledge, goals, constraints, procedures, and observations
   into a shared typed representation without pretending all tasks are numeric
   programs. Preserve provenance, uncertainty, reference, effects, and time.
8. Assess higher-order abstractions, typed holes, symbolic unknowns, partial
   values, quotient representations, abstract interpretation, and reusable
   learned macros. Specify denotational/operational semantics and a consumer.
   New symbols do not supply absent information; explain the actual benefit.
9. Make search over the existing floor/procedure library cheaper through learned
   proposals, indexing, compositional caching, abstraction, and verification.
   Compare worst-case complexity with measured expected cost; do not confuse
   a complete unbounded enumerator with a practical solver.
10. Develop a shared arbitration mechanism for model answers, tools, retrieval,
    explicit computation, and learned world models, grounded in calibrated
    evidence. Show integration with existing infrastructure and causal tests.
11. Handle changed facts, conflicts, partially correct answers, source errors,
    stale memories, corrections, and underspecified requests. Explain what
    revises knowledge, what remains unresolved, and how learning avoids turning
    self-generated plausibility into truth.

### C. Mathematics And Formal Guarantees

12. State and prove useful bounded theorems: compositional execution correctness,
    semantics-preserving transformations, graph-search completeness under
    declared bounds, and retention guarantees where achievable. Explicitly
    isolate assumptions about interpretation and external facts.
13. Write an end-to-end error model. A union bound over information, grounding,
    interpretation/search, execution, and emission errors needs no independence;
    estimate its terms or explain why they are unidentifiable. Do not assign
    arbitrary reassuring probabilities.
14. Derive sample complexity/power under a declared task population and paired
    design. Account for clustered templates, multiple domains, repeated model
    selection, sequential looks, stochastic decoding, and contamination.
15. Attempt to falsify every major proposed guarantee. Supply counterexamples,
    invalid assumptions, numerical stability cases, and an independent checker
    or property test. Separate a proof about your formal system from empirical
    evidence that natural-language tasks instantiate it.

### D. Broad Reasoning, Qualification, And Frontier Evaluation

16. Design G04-G08 as one coherent experiment, with explicit disjoint development
    and fresh splits, candidate freeze, arm order, compute/tool budgets, per-task
    outcomes, failures, regressions, uncertainty, and immutable artifacts.
17. Design G09 across math, executable coding, planning with observed effects,
    knowledge synthesis, multi-turn corrections, and unfamiliar compositions.
    Include adversarial and negative controls. Define what counts as broad
    without hiding failed domains in an aggregate.
18. Design G10-G11 qualification for the actual current backbone and combined
    compiler/steering/persona configuration. Specify adapters versus static
    fusion, representation alignment, rollback, qualification invalidation,
    serving alarms, and real causal live-path interventions.
19. Research current G12 baselines from dated primary sources. Name exact model
    versions, access dates, tool permissions, reasoning settings, resource
    accounting, task provenance, and reproducible scoring. Distinguish base-model
    comparisons from system-versus-system comparisons. Published benchmark
    scores on different protocols are context, not matched experimental results.
20. Explain what happens if the candidate loses. Specify decision rules for
    retaining the incumbent, revising the mechanism, gathering discriminating
    evidence, or abandoning a hypothesis without abandoning the user goal.

## Required Deliverables

1. An evidence table: supplied fact, independently checked fact, inference,
   hypothesis, or unknown; source and date for every material external claim.
2. A failure/dependency map covering every G item, including infrastructure
   dependencies. Separate blocking work from work that can proceed concurrently.
3. At least three genuinely competing complete designs, with failure analysis;
   a selection rationale; and a minimal integrated design that can ship first.
   Do not combine mutually inconsistent suggestions merely to include them all.
4. For each recommended mechanism: exact interface, state representation,
   equations/objective, pseudocode or code, integration owner, persistence,
   complexity and memory, training data requirements, observability, rollback,
   acceptance checks, and a cheap experiment that could refute it.
5. A reproducible reference package for the decisive new ideas. Include tests,
   seeds, versions, example inputs/outputs, measured failures, and an actual
   run log where tools permit. Label mocks and synthetic fixtures clearly.
6. A single experimental plan from small exhaustive checks through full
   development, fresh transfer, public decoding, live serving, and frontier
   evaluation. Identify exactly which tests can and cannot close each claim.
7. A resource-aware execution plan: parallel work packages, serial critical
   path, prerequisites, estimated work with assumptions, measured throughput
   where available, checkpoints, resumability, and stop/revise rules. Avoid
   promising calendar completion for an unproved scientific outcome.
8. A red-team section written against your own recommendation: strongest
   objections, counterexamples, rival explanations, contamination routes,
   failure under distribution shift, and what evidence would change your mind.
9. A final per-G-item implementation matrix: adopt/build/reuse/retire/experiment,
   precise reason, unresolved uncertainty, and next executable action. Include
   no unsupported checkmarks and no implied evidence from an unrun test.

Research primary papers, official implementations, and authoritative technical
documentation. Record publication and access dates. Investigate related
solutions beyond the named suggestions when they address the actual mechanism.
Compare against the simplest sufficient baseline. A smaller causal repair with
strong evidence is preferable to an elaborate framework with no discriminating
experiment.

If essential artifacts are missing, state the smallest exact request and still
complete the independent parts. Do not fabricate the missing result. The final
report should make it possible for an engineer to implement and falsify the
proposal without interpreting vague aspirations.

## Optional Artifact Requests

Ask for only the files needed to discriminate among your hypotheses:

- Current master G ledger and `docs/G03_SEMANTIC_CORRECTNESS_CONTRACT.md`.
- The modules named above plus their focused tests.
- Full cohort diagnosis when finished, with failure rows and target-blind bank
  receipts; do not request sealed test labels to tune a proposal.
- Parent/candidate identities, training configuration, component interventions,
  numerical constraints, and complete paired development outcomes.
- Resident model/config/tokenizer/fusion identity and steering qualification.
- Ordinary-runtime trace samples showing inputs, tool effects, public output,
  latency, and failure state, sanitized of credentials and private user data.

End with the five highest-value experiments in execution order, each with a
predicted outcome under competing hypotheses. The objective is to learn which
mechanism is wrong and repair it generally, not to repeat a costly benchmark
until a favorable score appears.

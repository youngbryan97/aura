# Semantic correctness and transfer contract

This is the engineering contract for G03-G12. It does not declare those items
complete or change their acceptance rules.

## What can be proved

For a task whose intended meaning is a program in Aura's supported typed
algebra, a correct result follows if all of these conditions hold:

1. The input grounding preserves the task's values and their identities.
2. The compiler recovers the intended operations and dependency graph.
3. Every reference binds to the intended definition; argument order survives.
4. The executor implements each primitive and composition rule correctly.
5. Public answer emission preserves the computed result.

These premises separate the problems. A correct executor cannot repair a
miscompiled request. A graph that satisfies every type and dependency
constraint can still express the wrong meaning.

For the typed algebra, primitive correctness and semantics-preserving
composition support an inductive correctness argument for larger programs.
That argument is conditional on the intended program being represented and
recovered. It is not a proof that a learned language compiler interprets every
unseen request correctly, or that the algebra covers every reasoning task.

## Mechanical obligations

| Boundary | Required property | Executable checks |
| --- | --- | --- |
| Representation | Values, types and identities survive serialization; unsupported structure is explicit | `tests/test_semantic_program_ir.py`, `tests/test_semantic_program_floor.py` |
| Supervision | Token-aligned labels preserve source text and executable meaning; sources remain split-bound | `tests/test_semantic_definition_attachment.py`, `tests/test_semantic_definition_pointer_refit.py` |
| Recognition | Predicted operation spans are calibrated in source order, independently of execution order | `tests/test_semantic_operation_chart_calibration.py`, `tests/test_semantic_operation_view_refit.py` |
| Definition ownership | The semantic graph carries learned attachment evidence, not only a local-window ownership assumption | `tests/test_semantic_definition_attachment.py` |
| Graph construction | Typed, consistent, connected acyclic bindings; each definition's evidence counted once; optimizer checked against enumeration | `tests/test_semantic_argument_optimization.py` |
| Procedure execution | Actual typed outputs cross backend boundaries; failed steps retain prior effects without claiming rollback or task correctness | `tests/test_procedure_execution.py`, `tests/test_semantic_procedure_currency.py`, `tests/test_semantic_program_runtime.py` |
| Selection | Full development rows, paired regressions, and coefficient/hidden controls retained | `tests/test_semantic_program_validation_selection.py`, `tests/test_semantic_program_compositional_verification.py` |
| Public answers | The decoded answer is independently graded; private reasoning is not a substitute | `tests/test_semantic_neural_composition_decode.py`, `tests/test_semantic_neural_composition_decode_canary.py` |
| Transfer | Frozen candidate, fresh construction/vocabulary/depth/family cases and independent adjudication | `tests/test_semantic_program_natural_transfer.py`, `tests/test_semantic_program_replication_verification.py` |

The presence of these suites does not establish that their associated live or
scientific obligations have passed. Unit contracts, development measurements,
fresh replication, runtime activation and frontier evaluation remain separate.

## Consolidated development pass

The current repair covers supervision, operation recognition, definition
ownership and graph scoring together. Frozen legacy artifacts remain readable.
New learned coefficients and policies receive new identities. Failed candidates
remain negative evidence; no parameter change can relabel them as qualified.

One combined candidate must be measured on every source-development cohort and
the exposed composition cohort, with per-family and paired regression reports.
Only after selection is frozen can fresh G04/G07 transfer runs test whether its
learned semantics extend beyond those development decisions.

G09 needs broad tasks beyond this bounded algebra. G12 additionally needs named
current baselines, independent tasks, fair resource/tool accounting, and actual
comparative measurements. Expanding the algebra, verifier, memory or planning
system can help meet those obligations; changing our score definitions cannot
establish them.

## End-to-end design for G09 and G12

The target pipeline is a request with retained conversational meaning, grounded
facts with source identities, candidate procedures, checked execution, a public
answer, and an observed outcome returned to learning. The existing modules below
are the implementation starting points. This map records inspected code, not a
claim that every connection is serving or qualified.

| Part | Existing code and connection | Work needed for the broad claim |
| --- | --- | --- |
| Linguistic meaning | `core/language/semantic_work.py` forms a work contract. `core/learning/semantic_program_runtime.py::execute_compositional_semantic_observation` recovers source literals and calls the learned compiler. | The learned compiler currently speaks integer and integer-sequence procedures. Cover references to retained facts, goals, constraints and procedures without routing by benchmark family or requiring every number in a request to be an operand. |
| Computation | `core/learning/semantic_program_floor.py` compiles the declared operation vocabulary into `core/cognition/the_floor_she_stands_on.py`. The floor also supports functions, recursion and pairs. | Extend the learned front end to the existing floor and registered procedures. General expressibility in the interpreter does not establish that the front end can find those programs. Add new operator semantics and type checks together, with independent execution comparisons. |
| Procedure reuse | `core/learning/semantic_procedure_currency.py` registers semantic programs in `core/cognition/procedure.py`. The resident shadow passes the shared registry into execution. `core/cognition/contract_health.py` installs adapters for the other learners. | Show a request retrieving, composing and executing procedures across backends, then recording its actual effect. A registry match or health count alone is insufficient. Test novel compositions and remove each reused procedure in a matched control. |
| Knowledge | `core/evidence/packet.py` carries source identities; `core/knowledge/atomspace.py` and the memory systems already hold knowledge. `core/learning/integrated_reasoning_eval.py` has retrieval/depth factorial tasks. | Preserve evidence across retrieval, interpretation and computation. Use the real retrieval consumer in end-to-end tests. The fixture retriever routes by task ID, and the old factorial grader uses answer containment; neither establishes broad retrieval quality or complete answer correctness. |
| Planning | `core/cognition/tool_plan.py` executes explicit tool plans. `core/cognition/an_action_she_composed.py` represents branching, repetition and recovery over an abstract world. | Connect task-grounded plan proposals to those existing executors. Measure actual state transitions, recovery and cancellation. Keep preview/simulation outcomes distinct from effects in the user's environment. |
| Search allocation | `core/cognition/value_of_computation.py` is used by agency-kind and multiple-drafts decisions; search reach is measured in `core/cognition/how_far_the_search_reaches.py`. | Tie continuation decisions to observed progress, uncertainty and cost for the same task. Verify that decisions improve completed answers and latency. A value-of-computation formula is not evidence of calibrated inputs or a closed learning loop. |
| Coordination | `core/connectome/integration.py` exposes topology, gates and telemetry; cognitive contract health reports traffic and handoffs. | Trace a broad task across the participating components and intervene on the relevant connection. Connectivity and activation do not by themselves prove useful reasoning. |
| Learning and fusion | `core/cognition/dual_knowledge.py` represents explicit, implicit and neural forms and measures conversion agreement. The weight-learning and model-identity paths remain the publication owners. | Learn from independently assessed outcomes, retain counterexamples and qualify conversion on fresh cases. Two failed executions must never count as successful agreement. Qualify the whole current model/steering/compiler configuration before live promotion. |
| Public result | `core/brain/llm/compositional_semantic_shadow.py` observes the resident sequence and executes the compiler, currently as a shadow mechanism. Existing decode canaries grade public output. | Bind the emitted answer to selected results and evidence without exposing private reasoning or truncating valid explanations. Reproduce through ordinary chat, not only an evaluation-specific shortcut. |

### Mathematical obligations

For a declared task population, let the failure events be missing required
information, an incorrect interpretation, unsuccessful search, incorrect
execution, and incorrect public emission. If these cover the ways the pipeline
can fail, then the probability of failure is at most the sum of their
probabilities. This union bound does not require independence. It is useful
only when the events and their rates are actually measured on that population.
Low interpreter error cannot compensate for an unknown interpretation error.

The executor proof is compositional: correct leaves and semantics-preserving
composition give a correct result for the represented program. The learning
problem is separate: recover that program from the available evidence. A graph
can be well typed, acyclic and executable while meaning the wrong thing.

Search needs both coverage and reach. A complete enumerator can include the
solution yet fail to reach it with available computation. Learned proposals,
reusable procedures, memoization and counterexample-guided refinement should
reduce measured search effort. Their benefit must survive fresh problems; a
shorter description or more expressive grammar alone does not demonstrate it.

### Implementation packages

1. **Close the shared compiler.** Fit attachment and graph decisions from
   source supervision; measure the complete development matrix and paired
   regressions. Freeze one candidate before fresh transfer. This is G03-G04,
   with G05-G08 checks prepared alongside it.
2. **Compose existing procedures across tasks.** Reuse `Procedure`, `Signature`
   and floor execution. Add task-grounded selection and an execution trace
   that preserves inputs, effects and evidence through backend crossings.
   Test a novel task requiring multiple existing procedures and a case where
   an inapplicable procedure must lose to an applicable one. This can be built
   without waiting for a perfect arithmetic compiler.
3. **Close the knowledge/planning outcome loop.** Connect real retrieval and
   effect observations to the same task representation. Reuse evidence packets
   and outcome ledgers; do not add a second knowledge store. Test source
   conflicts, missing information, changed goals and multi-turn corrections.
   Learning receives independently graded outcomes, not a response-length
   proxy for correctness.
4. **Run the broad measurement path.** Reuse the frontier task and artifact
   machinery, but supply the actual system under test through its ordinary
   runtime route. Freeze development choices and resource policy, then collect
   disjoint tasks across mathematical reasoning, code execution, planning,
   knowledge synthesis and transfer. Report domain-level gains, regressions,
   failures and uncertainty; a pooled score must not hide a failed domain.
5. **Qualify and compare.** G10-G11 require current identity, geometry,
   successful conversion, rollback and live causal use. G12 additionally
   requires named contemporary reference systems measured on the same tasks
   under declared tool and resource access. A positive comparison is an
   empirical result to earn, not an internal constant to set.

Packages two through four have work that does not depend on G03 closure. They
must not promote an unqualified compiler while developing those connections.

The first execution connection in package two now uses
`core/cognition/procedure_execution.py` to lower registered compositions into
the existing tool-plan executor. Caller-supplied backends produce explicit
typed values. The semantic runtime uses that path when given the shared
registry, and the RLC adapter executes the existing universal floor. This
does not yet supply task-grounded cross-backend selection or broad live proof.

### Measurement reuse and exclusions

`tools/run_unified_recurrent_broad_canary.py` measures a different mechanism,
the intrinsic recurrent controller. Its matched-arm and resumable-run patterns
are reusable; its treatment identity cannot label the semantic compiler.

`tools/measure_frontier_gap.py` already separates synthetic controls, unattested
diagnostics and signed worker measurements. Its diagnostic solver currently
uses `ReasoningAmplifierV2`; that is not automatically the whole live Aura
pipeline. A signed comparison must name what actually executed.

`core/brain/cognitive_engine.py::_structured_evaluation_thought` can return a
prompt-shaped floor before the cognitive pipeline for evaluation origins. Its
receipt says `pipeline_executed=False`. Such an answer must not count as proof
that broad reasoning or the live compiler ran. A general runtime comparison
must exercise the ordinary user path and preserve these execution distinctions.

No current frontier reference was selected by this design pass. No G09 or G12
measurement is implied by this document. The design changes internal mechanisms
while keeping task correctness and external comparisons independent.

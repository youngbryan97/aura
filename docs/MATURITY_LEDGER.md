# The maturity ledger — 202 findings, adjudicated

An external review compared Aura against eleven agent architectures — BabyAGI,
Generative Agents, Voyager, AutoGPT, OpenHands, LangGraph, AutoGen, Soar,
OpenCog AtomSpace, CrewAI and Letta Code — at commit 03f520ba0cf2, and
wrote 202 findings. Twenty-seven are P0, one hundred and seventy-four P1, one
P2.

Nothing here is accepted on sight. `tools/maturity/adjudicate.py` gives the
first cut mechanically — does the named anchor still exist, do the words of the
closure appear in the tree, is anything testing them — and the adjudication is
written by hand, because a grep hit is a reason to read something rather than a
verdict about it.

A row moves to DONE when a test that fails without the change passes with it,
and the commit is pushed. A row moves to REFUTED when the code already does the
thing and the check that says so exists. Neither is written on the strength of
an argument.

## The clusters

The 202 are not 202 pieces of work. They fall into about a dozen, and the P0s
name each cluster more than once:

| Cluster | Findings | What it comes to |
|---|---|---|
| A. One cancellation token | OpenHands #7, AutoGen #5 | A token in the execution context that model, tool, search, workflow and subagent calls all take |
| B. Typed events | OpenHands #17, CrewAI #1 | The topic is a transport address; the payload validates against a registered schema |
| C. The phase DAG, compiled | Generative Agents #1, LangGraph #6, Soar #1 | One machine-readable cognition order, compiled at boot, sealed, and refused when it does not resolve |
| D. State patches and reducers | LangGraph #1, #2 | One typed patch as the only cross-organ write, and a declared write mode per shared key |
| E. Deterministic laboratory mode | Soar #6 | Step one phase, freeze the background clocks, virtual time, inspect |
| F. Semantic identity | OpenCog #1, #2, #21 | One canonical id for an entity across every store, and dangling references detectable |
| G. Memory provenance | Letta #1, #11, #16 | Raw experience never mutates; summaries reference ranges; conflicts have a vocabulary |
| H. Conformance suites | LangGraph #9, #10, CrewAI #11 | Backend-independent suites every implementation passes unchanged |
| I. One working memory | Soar #8 | One canonical API; the other caches become projections |
| J. Runtime protocol and ownership | AutoGen #1, Soar #3 | One runtime object owns the authoritative services; no module singletons in core |
| K. Tool schemas | CrewAI #18 | JSON Schema in and out, a version, and a side-effect class, validated before authority |

## What the clusters cost, and what they found

Every cluster landed with a module, a suite, and a place in
`runtime_health_report()["integrity"]` so it can be read from a running
system. Several found defects that were not in the review:

* The sandbox's memory-bomb guard read `state.working_memory`, an attribute
  `AuraState` does not have, and passed a 5,000-item state.
* The compiled plan reported its own mode as a write mode, so every seal
  published before that was a digest of a plan mislabelling itself.
* The resource handoff woke a waiter and let it install itself, so a third
  caller arriving in between held the same resource.
* `schedule_relaunch` replayed `sys.argv`, which in a test run is pytest —
  5,205 chained processes in fifteen minutes.
* The compounding study's cost was constant across 824 operators, in three
  separate ways, none of which any test could see.


## Every finding

| Finding | Priority | Title | Closure (abridged) | State |
|---|---|---|---|---|
| BabyAGI #1 | P2 | One canonical extension unit | Define a CapabilityArtifact protocol with one stable ID, version, provenance, activation state, dependency set and execution contract. Make skills, sy | TODO |
| BabyAGI #2 | P1 | Single version-activation vocabulary | Create one ArtifactLifecycle enum and adapter layer: draft -> candidate -> qualified -> staged -> active -> retired/rolled_back. Preserve domain-speci | TODO |
| BabyAGI #3 | P1 | Tiny authoritative execution path | Add a capability-resolution proof API that returns the complete chosen chain (name -> canonical artifact -> authority -> policy -> executor), and a st | TODO |
| Generative Agents #1 | P0 | One explicit cognition sequence | Generate and enforce a machine-readable canonical phase DAG for foreground/background/degraded modes. Every phase must declare inputs, outputs, author | DONE |
| Generative Agents #2 | P1 | Single simulation clock | Introduce ClockDomain (wall, monotonic, subjective, conversation, simulation, model-budget) and require persisted timestamps/decays/deadlines to decla | TODO |
| Generative Agents #3 | P0 | Compact persona-state ownership | Complete the state-ownership registry so every durable semantic field has exactly one authoritative owner and all mirrors declare derivation/freshness | DONE |
| Generative Agents #4 | P1 | One obvious save/load shape for persona state | Create a signed WholeAuraSnapshot manifest that names every authoritative store, schema version, digest, model/adapters, memory roots and replay point | TODO |
| Generative Agents #5 | P1 | Small canonical memory taxonomy | Publish a canonical MemoryKind taxonomy and MemoryRecord envelope used by every durable memory system: identity, content/reference, temporal scope, so | TODO |
| Voyager #1 | P1 | Closed curriculum-practice-critic-skill loop | Create an ImprovementTransaction trace that starts at deficit selection and ends at measured post-change capability, with mandatory links for proposal | TODO |
| Voyager #2 | P1 | Externally grounded success gates skill promotion | Require every production capability promotion to carry an EvidenceGrade and at least one task-outcome receipt; distinguish structural validity from be | TODO |
| Voyager #3 | P1 | One bounded retry budget per task | Implement hierarchical BudgetContext: root turn/task budget -> child CPU/model/tool/retry budgets; every nested retry consumes the parent and cannot s | DONE |
| Voyager #4 | P1 | Four explicit learning roles | Create an authority matrix for improvement decisions: proposer, executor, verifier, promoter, activator. Enforce separation-of-duty for high-impact ch | TODO |
| Voyager #5 | P1 | Explicit reset after rollout failure | Add EnvironmentTransaction adapters for desktop/browser/sandbox tasks with capture, commit/abort, compensating action and post-abort equivalence check | TODO |
| Voyager #6 | P1 | Compact experiment resume state | Add ExperimentCapsule export/import: code SHA, config, RNG seeds, model digest, state snapshot manifest, active artifacts, task stream and receipts. O | TODO |
| Voyager #7 | P1 | Progress is updated from task outcomes | For developmental actions, require every credit update to identify the exact measured outcome and causal lag; add null/shuffled-credit controls for th | TODO |
| Voyager #8 | P1 | One promotion path from successful program to reusable skill | Unify skill/procedure/tool installation under CapabilityArtifact and one promotion ledger; permit multiple producers but exactly one activation servic | TODO |
| Voyager #9 | P1 | Learning claims are easy to lesion | Build OrganBoundary adapters with NullOrgan substitutes satisfying the same contract. Every major cognitive organ should support boot-time lesion with | DONE |
| AutoGPT #1 | P1 | Uniform AgentComponent lifecycle/order | Define Organ/Phase/Service lifecycle protocols with start/stop/health/dependencies/authority metadata and one lifecycle supervisor. | DONE |
| AutoGPT #2 | P1 | Generic typed configurable components | Create a typed Settings registry with per-component schema, source precedence, validation, reload policy and snapshot provenance; ban direct env reads | TODO |
| AutoGPT #3 | P1 | Explicit endpoint protocols | Define Protocols for major Aura service seams and migrate dynamic getattr/callback bridges behind typed adapters; mypy must prove implementations sati | TODO |
| AutoGPT #4 | P1 | Component-specific error hierarchy | Create canonical FailureScope + FailureKind typed exception/receipt taxonomy and map subsystem errors at boundaries. | TODO |
| AutoGPT #5 | P1 | Typed proposal/result lifecycle | Replace dict action boundaries with discriminated typed models; preserve extensibility with versioned unions. | TODO |
| AutoGPT #6 | P1 | Explicit root/subagent ExecutionContext | Introduce ExecutionContext as mandatory argument/context for all spawned cognitive/tool work: lineage, owner turn, budgets, authority, cancellation, s | DONE |
| AutoGPT #7 | P1 | Multiple reasoning strategies behind one interface | Define DeliberationStrategy protocol and strategy registry; all search modes return one typed DeliberationReceipt and obey the same budget/cancellatio | TODO |
| AutoGPT #8 | P1 | One pre-execution permission check | Prove a single ActionAuthority choke point for all side-effecting actions; every lower executor must require its unforgeable authorization receipt. | TODO |
| AutoGPT #9 | P1 | Central send/context token accounting | Add ContextBudget ledger with named allocations for identity/memory/interiority/tools/history/reasoning; all builders consume it and publish truncatio | TODO |
| AutoGPT #10 | P1 | Composable tool components | Standardize ToolProvider/ToolExecutor/ToolPolicy contracts and make every skill resolve through them; enforce no direct side effect below executor. | TODO |
| AutoGPT #11 | P1 | One action-history component | Create a canonical ActionEpisode projection sourced from the event spine, with adapters to memory/UI rather than parallel authoritative histories. | DONE |
| AutoGPT #12 | P1 | One MultiProvider completion API | Make ModelRuntimeProtocol the only Cortex-facing API: generate/stream/count/cancel/health/capabilities/receipts, with conformance tests for MLX/local/ | TODO |
| AutoGPT #13 | P1 | Component enabled/disabled reason is standardized | Add LifecycleStatus {state, reason, since, owner, dependencies} to every registered organ/service and one machine-queryable registry. | TODO |
| AutoGPT #14 | P1 | Dependency order declared by components | Require every phase/background job to declare requires/before/after and compile a DAG; reject cycles/unresolved dependencies at boot. | TODO |
| AutoGPT #15 | P1 | Explicit compatibility/deprecation path | Create DeprecatedSince/RemoveAfter metadata and CI that prevents new call sites of deprecated APIs and reports remaining users. | TODO |
| AutoGPT #16 | P1 | Parallel tool execution normalized at Agent boundary | Define ToolBatch with deterministic ordering, per-call cancellation, resource claims and aggregate failure semantics; route multi-tool plans through i | TODO |
| AutoGPT #17 | P1 | Subagent factory carries explicit parent/depth | Require all subagents/background autonomous workers to register parent lineage, authority inheritance, budget partition and termination policy in Exec | TODO |
| OpenHands #1 | P0 | Durable universal conversation event log | Make Aura event spine durable and mandatory for causally material events; state/action/memory projections may derive from it, but cannot bypass lineag | DONE |
| OpenHands #2 | P1 | Parent-linked branch tree | Index causal parents durably, reject dangling parents, detect cycles, support O(depth) path-to-root and branch replay across restarts. | DONE |
| OpenHands #3 | P1 | Duplicate event IDs rejected at append | Give each Aura event immutable EventId; append must reject duplicates and invalid parents, including across process restarts. | DONE |
| OpenHands #4 | P1 | Process/thread-safe event writes | Persist spine through a backend offering process-safe CAS/lock semantics; include multiprocess append stress tests. | DONE |
| OpenHands #5 | P1 | Bounded ancestry traversal | Maintain branch/parent indexes and benchmark bounded causal queries at million-event scale. | DONE |
| OpenHands #6 | P1 | Self-healing event index | For every durable event index, add generation markers, corruption detection, deterministic rebuild and verified no-loss recovery. | DONE |
| OpenHands #7 | P0 | First-class CancellationToken through agent/tool calls | Introduce AuraCancellationToken in ExecutionContext and require model, tool, search, workflow and subagent APIs to accept it; cancellation must propag | DONE |
| OpenHands #8 | P1 | Unfinished actions receive terminal outcomes on interrupt | On cancellation, synthesize a typed terminal receipt for every admitted but unfinished action/tool/model call; CI must detect orphaned starts. | TODO |
| OpenHands #9 | P1 | Pause is an explicit execution state | Adopt ExecutionState FSM (initializing, ready, running, waiting, paused, cancelling, completed, failed, degraded) across turn/workflow/subagent lifecy | TODO |
| OpenHands #10 | P0 | FIFO per-resource lock manager | Create ResourceClaimManager keyed by semantic resource, FIFO/fairness, timeout/cancel, reentrancy policy and observability; adapters wrap model lane,  | DONE |
| OpenHands #11 | P1 | Duplicate resource-key normalization | Define sorted multi-resource acquisition in the claim manager and property-test duplicate/permuted acquisition sets. | TODO |
| OpenHands #12 | P1 | Immutable prepared action batch | Define immutable ActionBatchPlan + mutable execution ledger separated by ID; no executor may mutate the admitted plan. | TODO |
| OpenHands #13 | P1 | Calls after FinishTool are discarded deterministically | Define terminal action semantics in the action schema; any subsequent actions in the same batch are invalid and recorded as rejected. | TODO |
| OpenHands #14 | P1 | Parallel execution with deterministic event emission order | Specify ordering semantics per batch: proposal order for causal log, completion timestamps separately; add randomized scheduling tests. | TODO |
| OpenHands #15 | P1 | Blocked tool calls become typed rejection events | Create universal ActionRejected event with action_id, authority, reason code, policy revision and remediation; all gates emit it. | TODO |
| OpenHands #16 | P1 | Small public ErrorClassification | Define PublicFailure {kind, action, retryability, user_remedy, correlation_id}; map internal errors at the boundary. | TODO |
| OpenHands #17 | P0 | Typed event classes across runtime | Generate a versioned Event union/schema registry. String topics remain transport addresses, but payload must validate against the registered event typ | DONE |
| OpenHands #18 | P1 | Cross-language event contract | Generate TS schemas/types from Aura Pydantic/dataclass event specs and run compatibility tests on every event version. | TODO |
| OpenHands #19 | P1 | Defensive prefix scan for event initialization | Declare TurnInitializationInvariant set and run it before foreground admission: identity loaded, system frame precedes user, state revision pinned, no | TODO |
| OpenHands #20 | P1 | Security analyzer defaults on | For each action class, publish the mandatory gate chain and boot-fail if a required gate is absent; no optional wiring for required safeguards. | TODO |
| OpenHands #21 | P1 | Tool execution is a first-class observability span | Use one TraceContext from turn -> deliberation -> action -> tool -> result -> learning; require trace_id/span_id on all receipts/events. | TODO |
| OpenHands #22 | P1 | FileStore abstraction below event log | Define EventStoreBackend and StateStoreBackend protocols with conformance suites; local SQLite is one implementation. | TODO |
| OpenHands #23 | P1 | Legacy parentless events get defined fallback semantics | Version all event envelopes and provide deterministic upcasters; replay old fixtures in CI through current projections. | DONE |
| OpenHands #24 | P1 | Event index gaps are diagnosed | Add sequence continuity/hash-chain verification to durable spine and fail replay past unexplained gaps. | DONE |
| OpenHands #25 | P1 | Pending concurrent executions are explicitly drainable/cancellable | Every spawned task must register in TaskRegistry with owner, cancel policy and drain deadline; shutdown asserts registry reaches zero or records orpha | TODO |
| OpenHands #26 | P1 | Event tree supports branch/rerun semantics natively | Represent workflow/turn forks as branches in the durable spine so replay, UI and learning share ancestry semantics. | TODO |
| OpenHands #27 | P1 | Event log can install a write guard | Make durable spine append require an AuthorityContext for protected event classes; guard is centralized and testable. | TODO |
| OpenHands #28 | P1 | Append path includes fast length marker optimization | Set event append latency/throughput SLOs and benchmark under multiwriter/replay loads; regressions gate CI for the canonical spine. | TODO |
| LangGraph #1 | P0 | One typed State -> Partial[State] contract | Define CognitiveStatePatch as the only cross-phase state update representation; direct field mutation remains internal to an owning organ, while inter | DONE |
| LangGraph #2 | P0 | Per-key reducer semantics declared in schema | For shared state, require each field to declare write mode: single-writer, last-value, additive, max/min, set-union, custom reducer. Reject unspecifie | DONE |
| LangGraph #3 | P1 | InvalidUpdateError for illegal state writes | Introduce StateConflictError with reason codes (stale revision, multiwriter, invalid reducer, ownership) and map all cross-organ state conflicts to it | DONE |
| LangGraph #4 | P1 | Bulk-synchronous superstep isolation | For parallel phase groups, create an explicit read-snapshot/write-buffer/commit barrier. Preserve asynchronous organs outside those groups. | TODO |
| LangGraph #5 | P1 | Immutable runtime context separate from mutable state | ExecutionContext must contain immutable run resources/authority/budgets; AuraState contains cognitive mutable state; CI rejects persisting process han | TODO |
| LangGraph #6 | P0 | Graph validated before execution | Compile the cognitive phase DAG at boot and emit a sealed plan hash; unresolved dependency/ownership/duplicate phase must prevent ready state. | DONE |
| LangGraph #7 | P1 | Uniform retry/cache/error/timeout policy | Define ExecutionPolicy for every phase/service call: timeout, retry, fallback, cache, idempotency, cancellation, criticality. Central executor applies | TODO |
| LangGraph #8 | P1 | Versioned checkpoint includes channel versions/versions_seen | Whole-state snapshot manifest must include per-owner revision vectors and consumed revisions, enabling causality-aware restore. | DONE |
| LangGraph #9 | P0 | One BaseCheckpointSaver backend interface | Create StoreBackend protocols for event/state/workflow/memory and a shared capability model; do not force identical data models, but standardize lifec | DONE |
| LangGraph #10 | P0 | Backend-independent checkpoint conformance suite | Build conformance packages for StateStore, EventStore, MemoryStore and ModelRuntime; every implementation must pass unchanged suites. | DONE |
| LangGraph #11 | P1 | Pending writes distinct from committed checkpoint | Create PendingEffect records with idempotency key, owner revision and terminal state; restart reconciles them before admitting new work. | DONE |
| LangGraph #12 | P1 | First-class checkpoint selection/replay | Expose one TimeTravel API over durable spine + state projection + workflow revisions: inspect, fork, replay, compare. | DONE |
| LangGraph #13 | P1 | First-class interrupt/resume protocol | Define SuspendedWork envelope with continuation token, required input schema, state revision and expiration; every pause/approval uses it. | DONE |
| LangGraph #14 | P1 | Delta-channel snapshot/prune rules | Every compactable store must expose materialize_before_prune() and prove state equivalence before/after compaction under randomized histories. | TODO |
| LangGraph #15 | P1 | Copy thread preserves semantic state | Make branch/fork a core spine operation producing new lineage ID with inherited state revision vector; workflows/conversations use it. | TODO |
| LangGraph #16 | P1 | Checkpointer serializer protocol | Create VersionedCodec registry for durable envelopes with schema ID, version, hash, upcaster and optional encryption; stores depend on codecs, not ad- | DONE |
| LangGraph #17 | P1 | Saver defines sync and async API pairs | For core storage/model/tool protocols, require native async plus documented sync facade, with equivalence tests and no event-loop blocking. | DONE |
| LangGraph #18 | P1 | Parent/subgraph namespace is explicit | Put parent_execution_id and namespace on ExecutionContext/Event/StatePatch, and derive all subagent/workflow lineage from them. | TODO |
| LangGraph #19 | P1 | Internal channel namespace is explicit | Create central NamespaceRegistry for event types, telemetry IDs, state keys, capability IDs and service names; CI allocates/checks ranges before runti | TODO |
| LangGraph #20 | P1 | StateGraph exposes precise generic type parameters | Ratchet Any/getattr at architectural seams: typed generics for state patches, services, events, tools and model responses; allow Any only inside adapt | TODO |
| LangGraph #21 | P1 | Versioned deprecation warnings in public API | Machine-register every compatibility shim with deprecated_since/remove_after/owner; CI forbids new consumers and reports expiry. | TODO |
| LangGraph #22 | P1 | Compiled graph is inspectable/renderable | Emit CognitiveTopology.json at boot with phases, services, schedulers, event subscriptions, state ownership and authority edges; diff it in CI. | DONE |
| LangGraph #23 | P1 | Concurrent fanout aggregation is defined by channels | Require every fanout group to declare join policy, failure policy, ordering and state merge; property-test scheduler permutations. | TODO |
| LangGraph #24 | P1 | Input and output schemas distinct from internal state | Each major cognition pipeline gets narrow Input/Output models; no phase should receive full AuraState unless it is explicitly an integrator. | TODO |
| LangGraph #25 | P1 | Managed values forbidden in graph I/O | Mark fields Durable/Derived/Ephemeral/ProcessLocal and validate serialization and API exposure automatically. | TODO |
| LangGraph #26 | P1 | Unknown interrupt targets rejected before run | All scheduler/hot-reload changes compile in shadow, run dependency/ownership checks, then atomically swap the plan. | TODO |
| LangGraph #27 | P1 | Error handlers are nodes with explicit recursion rule | ExecutionPolicy must state whether fallback errors recurse, escalate or terminate; default is escalate and is testable. | TODO |
| LangGraph #28 | P1 | Cache policy is explicit per node | Add CachePolicy metadata to any phase/tool/model call that caches: key inputs, invalidation revision, safety class, determinism. Include cache-hit rec | TODO |
| LangGraph #29 | P1 | Timeout policy is a typed node policy | Centralize timeouts into ExecutionPolicy and BudgetContext; ban naked wait_for timeout values in core except adapters, enforced by AST lint. | TODO |
| LangGraph #30 | P1 | Different channel type for same key is rejected | Build SemanticFieldRegistry with canonical name/type/unit/owner/version and detect duplicates across dataclasses/Pydantic/telemetry/event projections. | DONE |
| AutoGen #1 | P0 | One runtime_checkable AgentRuntime protocol | Define AuraRuntime Protocol exposing only stable lifecycle/message/state/capability/subscription operations; kernel/orchestrator implement it. | DONE |
| AutoGen #2 | P1 | AgentId separates logical identity from object instance | Represent cognitive workers/subagents/services by stable RuntimeAddress; direct object access is private implementation detail. | TODO |
| AutoGen #3 | P1 | Same runtime API supports remote agents | Add optional remote RuntimeAddress transport behind AuraRuntime while preserving local default; serialize typed messages and authority context. | TODO |
| AutoGen #4 | P1 | Direct and publish message semantics are distinct | Define CommandMessage (one recipient, reply) vs EventMessage (pub/sub, no reply); prohibit using broadcast bus for commands. | DONE |
| AutoGen #5 | P0 | Cancellation token is part of message API | ExecutionContext carries CancellationToken required by all async runtime calls; adapter bridges asyncio task cancellation into it. | DONE |
| AutoGen #6 | P1 | Message identity is part of runtime API | Use one MessageId type across commands/events/tool/model calls with parent/causal IDs in TraceContext. | DONE |
| AutoGen #7 | P1 | Agent factory has expected_class validation | ServiceRegistry.register requires Protocol/schema descriptor and factory shadow-instantiation; boot rejects type/contract mismatch. | TODO |
| AutoGen #8 | P1 | Agent type/instance uniqueness defined | Central NamespaceRegistry defines uniqueness domains for services, agents, events, telemetry and capabilities; all registries delegate to it. | TODO |
| AutoGen #9 | P1 | save_state/load_state on runtime protocol | Implement WholeAuraSnapshot through AuraRuntime and require restore equivalence checks over authoritative state and active artifact digests. | TODO |
| AutoGen #10 | P1 | agent_save_state/agent_load_state | Every autonomous child/subagent gets a StateCapsule protocol or explicitly declares stateless; parent snapshot enumerates them. | TODO |
| AutoGen #11 | P1 | add/remove_subscription protocol | Create SubscriptionSpec {id,event_type,filter,consumer,delivery,backpressure}; EventBus uses it and registry can audit dangling subscriptions. | DONE |
| AutoGen #12 | P1 | MessageSerializer registry | Create MessageCodecRegistry with schema ID/version/content type/upcaster; Redis/websocket/durable spine use the same codecs. | DONE |
| AutoGen #13 | P1 | Direct underlying-agent access is explicitly discouraged | Restrict direct service instance resolution to composition root/adapters; cross-organ interaction goes through Protocol methods/runtime messages. | TODO |
| AutoGen #14 | P0 | Handlers receive typed MessageContext | All event/command handlers receive immutable HandlerContext with sender, authority, trace, cancellation, state revision, timestamp. | DONE |
| AutoGen #15 | P1 | Intervention/middleware handler protocol | Define RuntimeMiddleware protocol with pre/post command/event/model/tool stages, ordering and failure semantics; governance can implement it. | TODO |
| AutoGen #16 | P1 | gRPC/distributed runtime preserves agent semantics | Add loopback/process-isolated AuraRuntime backend first; remote support optional. Use it to prove no hidden in-process coupling. | TODO |
| AutoGen #17 | P1 | Serializable component model across ecosystem | Every organ/service publishes ComponentDescriptor {type,version,config_schema,deps,factory,capabilities}; boot manifest is serializable. | TODO |
| AutoGen #18 | P1 | Team save/load state contract | Any composite cognitive subsystem with children must implement snapshot/restore or declare all child state external/derived. | TODO |
| AutoGen #19 | P1 | Tool/workbench lifecycle abstraction | Create Workbench protocol for tool environments with start/stop/reset/snapshot/resource claims; desktop/browser/sandbox implement it. | TODO |
| AutoGen #20 | P1 | Runtime interfaces tested independent of agent app | For every Protocol, ship a contract test kit; fakes must implement the same typed receipt shapes as production and be run against production adapters. | TODO |
| AutoGen #21 | P1 | Undeliverable/CantHandle semantics are API-level | Define DeliveryError taxonomy: UnknownRecipient, CantHandle, Rejected, Undeliverable, TimedOut, Cancelled; all runtime messages use it. | TODO |
| AutoGen #22 | P1 | Runtime get(... lazy=True) semantics | ComponentDescriptor declares activation mode eager/lazy/on-demand; runtime owns activation state and readiness, callers never probe implementation fla | TODO |
| Soar #1 | P0 | Canonical top-level phase state machine | Compile Aura’s scheduler into an explicit state machine/DAG with one advance API per clock domain; external schedulers submit work instead of owning c | DONE |
| Soar #2 | P1 | Stable named phase vocabulary | Version Aura phase contracts; each phase ID has immutable responsibility/inputs/outputs and deprecation path. | TODO |
| Soar #3 | P0 | One central agent_struct owns core managers | Make AuraRuntime instance own every authoritative service; eliminate module singletons/global resolvers from core through dependency injection or runt | DONE |
| Soar #4 | P1 | Exact decision/phase/firing counters | Define CognitiveAccounting with standard tick/phase/action/model/tool/state-write counters and causal correlation; all phases update centrally. | TODO |
| Soar #5 | P1 | Precise timer accounting by phase and callbacks | Use nested span accounting with exclusivity: wall, exclusive CPU, child time, queue time, model/tool time; validate totals mathematically. | TODO |
| Soar #6 | P0 | Deterministic run-for-N APIs | Add deterministic LaboratoryMode: step one phase/tick/decision, freeze background clocks, inject event, inspect state; use virtual time and determinis | DONE |
| Soar #7 | P1 | Explicit halt and reinitialize semantics | Define reset levels (turn, cognition, runtime, organism) with exact preserved/cleared state matrix and equivalence tests. | TODO |
| Soar #8 | P0 | Integrated working-memory manager | Choose one canonical WorkingMemory API/state model; other caches become projections. Enforce capacity, activation and provenance uniformly. | DONE |
| Soar #9 | P1 | Central symbol manager | Introduce SemanticIdentity service for canonical entity/concept IDs and aliasing; graphs/memories refer to IDs rather than re-inventing string identit | TODO |
| Soar #10 | P1 | Incremental rule matching engine | For shared symbolic constraints/rules, implement an incremental match network or indexed dependency engine; benchmark against rescanning at Aura scale | TODO |
| Soar #11 | P1 | One semantic-memory manager | Define one SemanticMemory contract and migrate specialized stores behind it or explicitly classify them as non-semantic projections. | TODO |
| Soar #12 | P1 | One episodic-memory manager | Define canonical EpisodeRecord sourced from durable event spine; conversation/action memories index it rather than independently own episodes. | TODO |
| Soar #13 | P1 | Reinforcement learning integrated into kernel | Create LearningAuthority service that receives typed Outcome/Credit events and dispatches learning algorithms; all learned policy updates register thr | TODO |
| Soar #14 | P1 | Explanation-based chunking is integrated | Unify proceduralization lifecycle with provenance graph from traces/evidence to learned artifact and one promotion authority. | TODO |
| Soar #15 | P1 | Canonical production representation/firing counts | Define ExecutableKnowledge interface with identity, antecedents, effects, confidence, source, firing metrics and retirement; adapters cover heuristics | TODO |
| Soar #16 | P1 | Stable callback/event API around phases | Publish versioned lifecycle events for every canonical phase transition; no extension may monkey-patch phase internals. | TODO |
| Soar #17 | P1 | Native memory manager/pools for hot kernel | Profile and move only proven hot coordination structures to optimized arrays/native extensions; establish per-tick orchestration CPU/allocation budget | TODO |
| Soar #18 | P1 | Small set of core managers defines cognition | Keep organ count, but compress authority: one runtime, one scheduler, one event spine, one state gateway, one action authority, one learning authority | TODO |
| Soar #19 | P1 | Uniform introspection commands/statistics | Expose a single `aura inspect` API/CLI over topology, state owners, active phases, counters, queues, resources and degradations with machine JSON outp | DONE |
| Soar #20 | P1 | Statistics reset semantics are explicit | Every metric declares lifetime domain (turn/session/boot/lifetime); reset APIs operate by domain and never silently mix them. | DONE |
| Soar #21 | P1 | Dedicated decision consistency machinery | Build WholeMindInvariantEngine consuming topology/state/authority/event-spine audits and run at boot + sampled ticks; findings have stable IDs/severit | TODO |
| Soar #22 | P1 | Explicit input/output phase boundary | For each sensor/action channel declare consistency mode: sampled-at-tick, streaming, transactional; state snapshots must record which observation fron | DONE |
| Soar #23 | P1 | Timing can be compiled/enabled explicitly | Classify instrumentation as mandatory/minimal/debug/scientific; benchmark overhead and allow reproducible modes while never disabling safety receipts. | TODO |
| Soar #24 | P1 | Kernel data/phase semantics have decades of continuity | Freeze a Kernel ABI v1 for phase IDs, event envelopes, state patch, action receipt and runtime protocol; changes require versioning/upcasters, not sil | TODO |
| OpenCog AtomSpace #1 | P0 | One canonical AtomSpace | Define a canonical SemanticGraph interface/ID layer shared by knowledge systems; specialized stores can remain, but cross-system references use canoni | DONE |
| OpenCog AtomSpace #2 | P0 | Atoms are globally unique/interned in an AtomSpace | SemanticIdentity service canonicalizes entities/concepts/relations and stores alias/equivalence; duplicate detection tests span memories/graphs. | DONE |
| OpenCog AtomSpace #3 | P1 | Node/Link representation is universal | Create minimal canonical relation algebra (EntityId, RelationType, edge/value metadata) and adapters from richer domain models. | TODO |
| OpenCog AtomSpace #4 | P1 | Dedicated TypeIndex | SemanticGraph maintains canonical type/relation/source/time indexes with invariants and performance benchmarks. | TODO |
| OpenCog AtomSpace #5 | P1 | Layered AtomSpace environments | Implement OverlayState/OverlayKnowledge abstraction with read-through + isolated writes + merge/discard, reusable by imagination, shadow evaluation an | TODO |
| OpenCog AtomSpace #6 | P1 | Read-only knowledge space is first-class | Add immutable snapshot/read-only modes to canonical graph/state stores enforced by storage layer, not caller convention. | TODO |
| OpenCog AtomSpace #7 | P1 | COW scratch spaces | Use shared overlay/COW primitive for world-model simulation, self-mod shadow runs, hypothesis spaces and counterfactuals. | TODO |
| OpenCog AtomSpace #8 | P1 | Explicit localize operation | Any mutation of inherited overlay state must materialize a local version carrying origin revision; enforce in overlay API. | TODO |
| OpenCog AtomSpace #9 | P1 | Read/write synchronization barrier | Add `await runtime.barrier(scope)` across event/state/memory queues with documented happens-before guarantees. | TODO |
| OpenCog AtomSpace #10 | P1 | Explicit recursive extraction semantics | Canonical semantic/memory deletion must check inbound references; caller chooses reject, detach or cascade and receives impact set. | TODO |
| OpenCog AtomSpace #11 | P1 | Reference-counted stable handles | Use typed IDs/handles with resolver and tombstone semantics; never pass raw mutable object as durable cross-organ reference. | TODO |
| OpenCog AtomSpace #12 | P1 | Content comparison is a testable core operation | Define semantic_equivalence(snapshotA,B, tolerances) for Aura state/knowledge; use in restore, compaction, migration and shadow tests. | DONE |
| OpenCog AtomSpace #13 | P1 | Persistence/remote storage behind StorageNode | Canonical graph/store protocol separates semantic operations from SQLite/vector/file backend; conformance suite validates each backend. | DONE |
| OpenCog AtomSpace #14 | P1 | Typed Values associated with atoms | Define versioned SemanticValue union for common confidence/time/source/vector/numeric/text/provenance values; domain extras remain namespaced. | TODO |
| OpenCog AtomSpace #15 | P1 | Generic query mechanisms operate on same representation | Expose one cross-memory semantic query layer over canonical IDs/relations, delegating to specialized stores but returning normalized results. | TODO |
| OpenCog AtomSpace #16 | P1 | GroundedSchema/Runner bridges symbolic and executable | Represent capabilities as canonical semantic entities linked to preconditions/effects/evidence and resolver; execution remains governed by ActionAutho | TODO |
| OpenCog AtomSpace #17 | P1 | Guile/Python extension bridges around same core | Define JSON/Protobuf schemas for runtime/event/state/capability contracts so native helpers/JS clients do not duplicate semantics. | TODO |
| OpenCog AtomSpace #18 | P1 | C++ indexed atom storage | Benchmark actual graph workloads; if overhead is material, implement native/Rust/C++ backend behind the same SemanticGraph contract, not premature rew | TODO |
| OpenCog AtomSpace #19 | P1 | AtomTable/StateLink unit suites target substrate invariants | Create canonical SemanticGraph conformance/invariant suite: identity, type index, links, overlays, COW, deletion, persistence, concurrency. | TODO |
| OpenCog AtomSpace #20 | P1 | Parallel/ExecuteThreaded links are first-class atoms | ActionPlan schema explicitly represents sequence/parallel/race/barrier/retry nodes, with deterministic execution semantics. | TODO |
| OpenCog AtomSpace #21 | P0 | Incoming/outgoing sets enforce graph relationships | Canonical semantic IDs maintain inbound-reference index across stores or a reconciliation job; dangling references are detectable and policy-controlle | DONE |
| OpenCog AtomSpace #22 | P1 | Type/name servers centralize type registration | Unify registries under NamespaceRegistry with typed domains and schema descriptors, preventing collisions like the telemetry issue class at source. | TODO |
| CrewAI #1 | P0 | Event bus dispatches BaseEvent classes | Add typed EventSpec registry and generated validators; transport topic is not the payload schema. | DONE |
| CrewAI #2 | P1 | Dependency-aware handler execution graph | SubscriptionSpec declares depends_on; bus compiles handler DAG and rejects cycles. | TODO |
| CrewAI #3 | P1 | Event handler execution plan cached/invalidation-aware | Compile subscription graph; cache by registry revision; expose plan in inspect API. | TODO |
| CrewAI #4 | P1 | ContextVar marks replayed event dispatch | HandlerContext includes replay_mode; side-effect handlers must declare replay policy and CI tests ensure replays do not duplicate external effects. | TODO |
| CrewAI #5 | P1 | RWLock protects event handler registry | After typed subscription graph, use immutable registry snapshots or RW lock with benchmark; dispatch should not block on unrelated registration. | TODO |
| CrewAI #6 | P1 | Distinct sync and async handler sets | SubscriptionSpec declares execution kind/queue policy; runtime owns workers and shutdown rather than every consumer hand-rolling tasks. | TODO |
| CrewAI #7 | P1 | Bus tracks pending futures and graceful shutdown | Bus-created delivery/handler tasks register with TaskRegistry and shutdown reports/drains all owned futures. | TODO |
| CrewAI #8 | P1 | Agent config is uniformly typed/validated | Move core runtime/organ settings into versioned Pydantic schemas under SettingsRegistry and generate docs/default dumps. | TODO |
| CrewAI #9 | P1 | Deprecated fields are machine-visible | Deprecation registry + CI new-use ban + migration adapters, as above. | TODO |
| CrewAI #10 | P1 | LLM references have validators and serializers | Create ModelRef typed URI + ModelDescriptor digest; config always serializes ModelRef, runtime resolves to provider/model artifact. | TODO |
| CrewAI #11 | P0 | Provider-specific message/tool normalization | ModelRuntime conformance suite with golden message/tool/stream/error/token cases for every adapter; provider quirks cannot leak upward. | DONE |
| CrewAI #12 | P1 | Planning behavior has one typed config | Create DeliberationConfig with strategy, budget, model tier, search limits, uncertainty threshold and fallback; include in receipts. | TODO |
| CrewAI #13 | P1 | Output guardrail protocol and retry budget | Define OutputValidator protocol returning typed violation/remedy; response pipeline composes validators under one retry budget. | DONE |
| CrewAI #14 | P1 | Executor class validation/serialization + deprecation | For each legacy execution path, publish replacement map, compatibility period and test that no new direct call sites appear. | TODO |
| CrewAI #15 | P1 | Typed A2A client/server configuration | Define AgentDelegation protocol/schema independent of transport; local subagents implement it first, optional remote adapter later. | TODO |
| CrewAI #16 | P1 | max_execution_time/max_retry_limit are top-level agent contracts | Expose TurnBudget/TaskBudget summary at admission; all child budgets draw from it and UI/receipts show remaining budget. | TODO |
| CrewAI #17 | P1 | respect_context_window is explicit agent behavior | ContextBudget plus declared truncation priority/semantic preservation policies; no hidden contributor may exceed allocation. | TODO |
| CrewAI #18 | P0 | Tools use structured schemas and normalization | All Aura tools expose JSON Schema/Pydantic input/output, semantic version and side-effect class; executor validates before authority check. | DONE |
| CrewAI #19 | P1 | Starting/ending event pairs tracked as scopes | TraceContext manages structured scopes; starting event creates span ID, terminal event required, dangling scope fails invariant audit. | TODO |
| CrewAI #20 | P1 | ContextVar RuntimeState/entity registration | HandlerContext snapshot captures runtime identity/state revision at emission, not lookup-at-consume time. | TODO |
| CrewAI #21 | P1 | Deliberate stops excluded from generic retry | Central RetryClassifier marks NeverRetry classes (cancel, denied, invariant, invalid input, unsafe) and all retry loops use it. | TODO |
| CrewAI #22 | P1 | Dedicated streaming output wrappers | Define GenerationStream protocol with chunks, metadata, cancellation, completion receipt and error terminal event; all model routes adapt to it. | TODO |
| Letta Code #1 | P0 | Agent memory is a git-backed filesystem | Project a canonical subset of Aura long-term semantic/identity memory into a versioned MemoryFS with commit IDs linked to event-spine receipts; keep d | DONE |
| Letta Code #2 | P1 | Memory edits apply on recompile, not current turn | Classify self-model/memory mutations by activation frontier: immediate, next phase, next turn, next boot. Persist frontier in receipt and test no earl | TODO |
| Letta Code #3 | P1 | Pre-commit hook validates memory metadata/protected files | MemoryRecord schema + transaction validator runs before commit for identity/protected memories; malformed/prohibited writes never reach durable store. | TODO |
| Letta Code #4 | P1 | Writes check repository cleanliness | Canonical memory transaction requires base revision; stale base returns conflict with three-way merge data rather than last-writer behavior. | TODO |
| Letta Code #5 | P1 | Memory tool commits with agent authorship | Every high-level memory mutation emits MemoryChangeSet containing before/after digest, semantic diff, actor, evidence, activation frontier and rollbac | TODO |
| Letta Code #6 | P1 | Background reflection uses isolated worktrees then integrates | Run consolidation/reflection in MemoryOverlay branches; foreground sees base until merge passes conflict/evidence checks. | TODO |
| Letta Code #7 | P1 | Git config mutation has a dedicated nonblocking lock | Route runtime/settings/config writes through ConfigStore CAS/lock API with revision and atomic replace; ban direct writes. | TODO |
| Letta Code #8 | P1 | Memory sync has explicit dirty/conflict/push status | Expose MemoryDurabilityStatus to cognition/UI: committed, pending, conflict, unsynced, failed; no memory may self-report “remembered” without committe | TODO |
| Letta Code #9 | P1 | Memory belongs to agent ID across environments/models | WholeAuraSnapshot + ModelRef separation: identity/memory can restore with a compatible different Cortex; run continuity invariants across model swap. | TODO |
| Letta Code #10 | P1 | One agent can have multiple conversation IDs | Define ConversationId as branch under EntityId with per-conversation context but shared durable self/memory; isolate turn state and test concurrent co | TODO |
| Letta Code #11 | P0 | Recall history is automatically stored and non-mutating | Maintain immutable ExperienceLog in durable spine; summaries/memories reference event ranges and may change, raw experience never mutates. | DONE |
| Letta Code #12 | P1 | System/in-context vs external memory is explicit | Canonical MemoryKind includes in_context, recall, semantic, procedural, external; each declares default retrieval/retention/context cost. | TODO |
| Letta Code #13 | P1 | CLI can report memory token size | Expose per-memory-class token footprint and marginal prompt cost in ContextBudget report; alert on growth slopes, not arbitrary fixed limits. | TODO |
| Letta Code #14 | P1 | Read-only memory files enforced by hook | Memory schema supports immutable/read_only/operator_only classifications enforced by MemoryStore, independent of tool prompt/policy. | TODO |
| Letta Code #15 | P1 | Frontmatter schema required for memory files | All MemoryRecords require type, description, provenance, confidence, privacy, retention and owner; conformance rejects incomplete entries. | TODO |
| Letta Code #16 | P0 | Git gives explicit merge/conflict semantics | For semantic memories, implement three-way merge with conflict objects and evidence-aware resolution; never silently overwrite divergent identity/self | DONE |
| Letta Code #17 | P1 | Reflection/memory subagents have explicit worktree/parent integration | Default subagent state/memory writes to branch overlays; parent promotes accepted changes through a merge receipt. | TODO |
| Letta Code #18 | P1 | Memory vs skills vs mods have distinct semantics | Adopt ArtifactKind decision table in developmental planner: belief/memory, procedure/skill, deterministic mechanism/code, model weight, configuration; | TODO |
| Letta Code #19 | P1 | Agent identity/context designed to survive model changes | Define CompatibilityProfile for model swap and a continuity test suite ensuring self/memory/goals remain invariant except documented model-dependent l | TODO |
| Letta Code #20 | P1 | Memory evolution is readable with ordinary git tooling | Provide `aura history memory/self/capability` commands that render semantic diffs and provenance without querying raw SQLite/JSON; optionally export t | TODO |

## Second wave — a source-level comparison, 2026-09-06

A second external review read the current implementations rather than the
commit stream: Aura's event spine, checkpoint providers, lifecycle supervisor,
hierarchical budget, interrupt/resume, answer guardrails and semantic state
comparison, against the corresponding runtime code in LangGraph, AutoGen,
OpenHands, AutoGPT, CrewAI and BabyAGI. It placed Aura at 3.3–3.7 of 5 on a
five-stage engineering ladder, above BabyAGI wholesale and below LangGraph's
durable execution semantics, OpenHands' agent lifecycle discipline, AutoGen's
message-runtime abstraction, CrewAI's provider coherence and AutoGPT's
distributed operations.

Its central finding is not that abstractions are missing. It is that the
standard is ahead of adoption: a 150-line module reads as a system-wide
invariant and is not one. The rows below are the specific gaps it named in
code, and each is scored the same way as every other row here — a test that
fails without the change and passes with it.

| Row | Pri | Finding | Closure | Status |
| --- | --- | --- | --- | --- |
| Wave2 #1 | P0 | The interruption registry was a module-level dict | Make it durable; an interruption recorded in one process resumes in another. | DONE |
| Wave2 #2 | P0 | A broken guardrail failed open with no declared failure mode | Each rail declares carry-on, refuse, abstain or escalate; an undeclared rail refuses. | DONE |
| Wave2 #3 | P0 | The organ audit had never looked at 42 namespace packages | Count every directory under core/ holding .py files. 162 organs, 42 answer all four. | DONE |
| Wave2 #4 | P0 | "What do you promise" was answered by a string, so the number moved by adding markers | A promise needs a sentence, a test node id that exists, and where the breach goes. | DONE |
| Wave2 #5 | P1 | A checkpoint can become durable before the writes that produced it | Drain pending write futures before the next checkpoint is made durable, as LangGraph's PregelLoop does. | TODO |
| Wave2 #6 | P1 | Checkpoints have no branch or parent lineage | Add branch directories and parent ids to both providers, plus prune and checkpoint-id extraction, as CrewAI's BaseProvider has. | TODO |
| Wave2 #7 | P1 | The hierarchical budget is a well-designed primitive with incomplete adoption | Thread it through every significant operation rather than leaving it an in-memory object callers may pass. | TODO |
| Wave2 #8 | P1 | 118 of 162 organs say nothing about what they promise | Declare checkable promises per package. Six declare twenty. | TODO |
| Wave2 #9 | P1 | 28 of 162 organs say nothing about how failure propagates | Route failure through record_degradation, or declare that the organ cannot fail and check it. | TODO |
| Wave2 #10 | P1 | 69 declared services, 6 lesionable, 0 carrying an intervention verdict | Instrumentation exists before coverage; run do(X=present) against do(X=null) for the major organs. | TODO |
| Wave2 #11 | P0 | Generated Python was delivered successfully and was semantically wrong | Queued job coroutines never awaited; a deadlock waiting on queue data while holding a producer lock; direct deque mutation bypassing put(); four never-awaited coroutine warnings; a false claim that cancellation releases an asyncio.Lock. | TODO |
| Wave2 #12 | P1 | Cancellation is not linked to the future representing the call | Tie a cancellation token to the awaiting future, as AutoGen's runtime does. | TODO |
| Wave2 #13 | P1 | A prepared action batch is mutable and unordered | Immutable ActionBatch, blocked and executable separated, results joined by action id and emitted in the original order, with a tool-concurrency limit. | TODO |
| Wave2 #14 | P1 | Runtime conceptual compactness scored 4/10 and decomposability 5/10 | Continue the decomposition; chat.py is 24,185 lines down to 18,734 and the tree is 30,303 over its oversize budget. | TODO |
| Wave2 #15 | P1 | RLC generation ends at the token limit with the private channel still open | 3,201 tokens of private reasoning, none of it past the public-answer boundary, so the parent correctly saw no public answer and the feature still failed. | TODO |

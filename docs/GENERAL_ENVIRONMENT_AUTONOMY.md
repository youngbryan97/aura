# General Environment Autonomy

This is the current, code-grounded checklist for an Aura architecture that can
handle NetHack-scale complexity without becoming NetHack-specific. NetHack is a
stress adapter; the reusable system is the environment OS.

## Required Capabilities

The executable audit lives in `core/environment/capability_matrix.py`.

| Capability | Implemented by |
| --- | --- |
| Bounded adapter contract | `core/environment/adapter.py` |
| Observation normalization | `core/environment/state_compiler.py`, adapter compilers |
| Shared ontology | `core/environment/ontology.py`, `ParsedState` |
| Persistent belief graph | `core/environment/belief_graph.py` |
| Canonical spatial/topological memory | `EnvironmentBeliefGraph.spatial`, `upsert_spatial()` |
| Resource homeostasis | `core/environment/homeostasis.py` |
| Modal state machine | `core/environment/modal.py` |
| Startup prompt policy | `core/environment/startup_policy.py` |
| Candidate generation | `core/environment/policy/candidate_generator.py` |
| Tactical simulation | `core/environment/simulation.py` |
| Action ranking | `core/environment/policy/action_ranker.py` |
| Strategic HTN planning | `core/environment/strategy/htn_planner.py` |
| Semantic command compilation | `core/environment/command.py`, environment compilers |
| Closed-loop action semantics | `core/environment/action_semantics.py` |
| Semantic action budgets | `core/environment/action_budget.py` via `EnvironmentKernel` |
| Effect gate | `core/environment/action_gateway.py` |
| Governance bridge | `core/environment/governance_bridge.py` |
| Receipt chain | `core/environment/receipt_chain.py` |
| Semantic outcome diff | `core/environment/outcome/semantic_diff.py` |
| Outcome attribution | `core/environment/outcome_attribution.py` |
| Cross-run outcome ledger | `core/environment/outcome/ledger.py` |
| Procedural memory connection | `core/memory/procedural/store.py` via `EnvironmentKernel` |
| Competence tracking | `EnvironmentKernel.competence_tracker` |
| Hindsight replay / causal rules | `core/environment/experience_replay.py` |
| Autonomous abstraction discovery | `core/environment/abstraction_discovery.py` |
| Open-ended curriculum tasks | `core/environment/curriculum.py` |
| Black-box trace + replay | `core/environment/blackbox.py`, `core/environment/replay.py` |
| Run lifecycle/postmortem | `core/environment/run_manager.py`, `core/environment/postmortem.py` |
| Benchmark mode separation | `core/environment/benchmark_runner.py`, `BoundaryGuard` |
| External task proof gate | `core/environment/external_validation.py` |
| Runtime activation proof bridge | `core/runtime/activation_audit.py`, `core/runtime/proof_kernel_bridge.py` |
| Async/concurrency health | `core/runtime/concurrency_health.py`, `core/utils/task_tracker.py`, `core/resilience/lock_watchdog.py`, `core/dead_letter_queue.py` |
| Corrective grounding | `core/brain/grounding_guard.py`, `core/self_evaluator.py` |
| Robust context budgeting | `core/brain/llm/context_gate.py` |
| Structured knowledge distillation | `core/learning/formalizer.py` |

## Current Integration State

- `EnvironmentKernel` is the canonical loop for autonomous environment runs.
- The kernel starts a durable run record, seeds the same HTN planner used by
  policy, observes through the adapter, compiles state, updates belief/spatial
  memory, selects an intent, simulates it, routes governance, gates effects,
  compiles a command, validates closed-loop action semantics, records action
  budget pressure, executes, diffs pre/post state, attributes outcome, updates
  replay/abstraction/curriculum/ledger/procedural stores, emits trace rows,
  and closes the run with a postmortem when terminal failure/success is
  detected.
- Generic policies emit semantic intents. Environment-specific code is limited
  to adapters, parsers/compilers, and command compilers.
- Authority-required actions now fail closed if no authority gateway is
  connected.
- Spatial memory is metadata-rich but remains backward-compatible with legacy
  kind-string lookups.
- External proof now distinguishes strict-real, simulated-canary, and fixture
  evidence, and refuses placeholder/stub adapters as capability proof.
- Activation audit now starts and samples the live proof-kernel bridge,
  lock watchdog, and concurrency health monitor. The proof bridge reports an
  explicit claim scope and does not treat engineered proxies as subjective
  consciousness proof.
- The generic terminal-grid compiler parses ASCII/ANSI-like screens into the
  shared ontology. NetHack-specific code remains an adapter/parser/command
  translation layer, not shared strategy.
- Repository repair scripts are archived under `archive/repair_scripts/` and
  are no longer root-level canonical runtime artifacts.

## NetHack Stress Adapter

Run the strict real stress loop:

```bash
python challenges/nethack_challenge.py --mode strict_real --steps 5000
```

Run the deterministic safe canary:

```bash
python challenges/nethack_challenge.py --mode simulated --steps 100
```

NetHack-specific modules are not strategy organs. They provide terminal-grid
observation and command translation so the general kernel can be tested in a
harsh, long-horizon environment.

## Verification

Current focused suite:

```bash
python -m pytest \
  tests/test_final_general_hardening.py \
  tests/test_environment_general_integration.py \
  tests/environment/final_blockers \
  tests/environments/terminal_grid/test_nethack_audit_comprehensive.py \
  tests/environments/terminal_grid/test_terminal_grid_live_canary.py \
  tests/environments/terminal_grid/test_nethack_adapter_preflight.py \
  tests/environments/terminal_grid/test_terminal_grid_contract.py \
  tests/architecture \
  tests/test_embodied_cognition_runtime.py \
  tests/test_runtime_stability_edges.py \
  tests/test_runtime_service_access.py \
  tests/nethack_crucible.py -q
```

Focused final-pass result: `271 passed, 1 subtests passed`.

Stress canary:

```bash
python challenges/nethack_challenge.py --mode simulated --steps 20 \
  --trace artifacts/test_nethack_kernel_trace.jsonl --log-level ERROR
```

Result: passed, with 40 hash-chained trace rows emitted to
`artifacts/test_nethack_kernel_trace.jsonl`.

Strict-real smoke:

```bash
python challenges/nethack_challenge.py --mode strict_real --steps 40 \
  --trace /tmp/aura_strict_probe_after_threat_response.jsonl --log-level INFO
```

Result: reached a live `dlvl_1` run, resolved startup modals, opened a door,
moved, handled information modals, and stayed alive through 40 steps. This is
not an ascension and is not counted as task mastery.

No successful strict-real ascension is recorded in this document.

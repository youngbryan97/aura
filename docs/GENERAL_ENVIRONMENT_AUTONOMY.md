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
| Candidate generation | `core/environment/policy/candidate_generator.py` |
| Tactical simulation | `core/environment/simulation.py` |
| Action ranking | `core/environment/policy/action_ranker.py` |
| Strategic HTN planning | `core/environment/strategy/htn_planner.py` |
| Semantic command compilation | `core/environment/command.py`, environment compilers |
| Effect gate | `core/environment/action_gateway.py` |
| Governance bridge | `core/environment/governance_bridge.py` |
| Receipt chain | `core/environment/receipt_chain.py` |
| Semantic outcome diff | `core/environment/outcome/semantic_diff.py` |
| Outcome attribution | `core/environment/outcome_attribution.py` |
| Cross-run outcome ledger | `core/environment/outcome/ledger.py` |
| Procedural memory connection | `core/memory/procedural/store.py` via `EnvironmentKernel` |
| Competence tracking | `EnvironmentKernel.competence_tracker` |
| Black-box trace + replay | `core/environment/blackbox.py`, `core/environment/replay.py` |
| Run lifecycle/postmortem | `core/environment/run_manager.py`, `core/environment/postmortem.py` |
| Benchmark mode separation | `core/environment/benchmark_runner.py`, `BoundaryGuard` |

## Current Integration State

- `EnvironmentKernel` is the canonical loop for autonomous environment runs.
- The kernel starts a durable run record, seeds the same HTN planner used by
  policy, observes through the adapter, compiles state, updates belief/spatial
  memory, selects an intent, simulates it, routes governance, gates effects,
  compiles a command, executes, diffs pre/post state, attributes outcome,
  updates learning stores, emits trace rows, and closes the run with a
  postmortem when terminal failure/success is detected.
- Generic policies emit semantic intents. Environment-specific code is limited
  to adapters, parsers/compilers, and command compilers.
- Authority-required actions now fail closed if no authority gateway is
  connected.
- Spatial memory is metadata-rich but remains backward-compatible with legacy
  kind-string lookups.

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
  tests/test_environment_general_integration.py \
  tests/environment/final_blockers \
  tests/environments/terminal_grid \
  tests/architecture \
  tests/test_embodied_cognition_runtime.py \
  tests/nethack_crucible.py -q
```

Focused result: `210 passed`.

Full repository result after this patch set:

```bash
python -m pytest -q
```

Result: `4333 passed, 7 skipped, 7 warnings, 1 subtests passed`.

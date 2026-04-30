# Autonomous Architecture Governor

Aura's Autonomous Architecture Governor (ASA) is an audit-first software
architect subsystem. It builds a live architecture graph, detects structural
and governance smells, creates staged refactor plans, runs candidates in a
local shadow workspace, verifies proof obligations, promotes only eligible
low-risk changes, monitors after promotion, and restores from rollback packets
when delayed regression appears.

The implementation lives in `core/architect/` and is registered as the optional
`architecture_governor` / `autonomous_architecture_governor` service. Boot mode
is conservative: audit-only unless configuration enables autonomous promotion.
CLI `auto` can still run the full loop in test or temp repositories.

## Mutation Tiers

| Tier | Meaning | Autonomous promotion |
|---|---|---|
| `T0_SYNTAX_STYLE` | Comments, formatting, docstring-only changes when docstrings are not runtime contracts, and proven-safe import cleanup | Allowed after syntax/static checks |
| `T1_CLEANUP` | Dead-code quarantine, unused import cleanup, duplicate import cleanup, trivial extraction | Allowed after graph proof, shadow import, rollback packet |
| `T2_REFACTOR` | Structure-preserving refactors, file splits, helper extraction, caller migration, duplicate consolidation | Allowed only when full proof obligations pass and config permits `T2` |
| `T3_BEHAVIORAL_IMPROVEMENT` | Intentional behavior improvements outside protected surfaces | Allowed only with declared improvement target, full proof, extended monitor |
| `T4_GOVERNANCE_SENSITIVE` | Authority, constitution, capability issuance, memory/state gateways, model routing, boot ordering, self-mod machinery, identity surfaces | Proposal and proof only |
| `T5_SEALED` | ASA/proof/rollback/classifier machinery and root constitutional invariants | Proposal only; no autonomous live mutation |

The classifier uses the maximum tier across touched paths, symbols, semantic
surfaces, callers, callees, and dynamic dependencies. Uncertainty upgrades risk.

## Architecture Graph

`python -m core.architect.cli graph` writes:

- `.aura_architect/architecture_graph.json`
- `.aura_architect/architecture_graph.jsonl`
- `.aura_architect/architecture_graph.sqlite`
- `.aura_architect/reports/architecture-graph-*.json`

The graph records:

- Files, imports, classes, functions, async functions, constants, decorators,
  calls, inheritance, tests mapped to modules, service-container get/register
  patterns, event-bus publish/subscribe patterns, dynamic import and string
  dispatch patterns.
- Effect annotations for file writes, database writes, subprocess/shell,
  network calls, LLM calls, tool execution, authority calls, capability token
  usage, background task creation, direct state-like mutation, and broad
  exception handling.
- Semantic surfaces such as authority/governance, memory/state, boot/runtime,
  consciousness, LLM routing, identity, self-modification, tests/proofs, UI/API,
  training, and utility/peripheral.
- Ownership rows parsed from `OWNERSHIP.md`.
- Runtime receipt ingestion for current and future JSONL receipts.

## Smells

`python -m core.architect.cli smells` emits findings with evidence, graph
references, severity, suggested tier, proof obligations, and auto-fixability.
Detected classes include god files/classes, fan-in/fan-out, import cycles,
duplicate ownership, duplicate responsibility, duplicate AST implementations,
broad exception clusters, critical `ServiceContainer.get(... default=None)`,
state/memory/tool bypasses, subprocess hazards, dynamic import hazards,
compatibility surfaces without receipts, dead symbol candidates, missing
tests/receipts, source-string tests, boot monkey patches, untracked background
tasks, critical-path markers, and sealed edit attempts.

## Shadow And Ghost Boot

Shadow workspaces are local temp copies. They exclude `.git`, virtualenvs,
bytecode, caches, model weights, DBs, logs, and training artifacts. Candidate
files are written only inside the shadow root first, with `PYTHONPATH` pointing
at that shadow root for proof commands. Artifacts are persisted under:

`.aura_architect/shadow_runs/<run_id>/`

Ghost boot attempts, in order:

1. AST parse and `py_compile` over changed files.
2. Import smoke tests for changed modules.
3. Architecture graph rebuild in shadow.
4. Existing architecture quality/hardening tests when present.
5. Relevant pytest subset by changed path when discoverable.
6. Critical runtime/self-modification subset when present.
7. Optional broader pytest when configured.
8. Safe boot command if `AURA_ASA_SAFE_BOOT_COMMAND` is configured.
9. Minimal live harness if a safe harness exists.

If a full boot harness is unavailable, the proof records
`BOOT_HARNESS_UNAVAILABLE`. That is tolerated for `T0/T1` cleanup, but higher
tiers must prove more.

## Proof Obligations

Universal obligations include syntax success, changed-file scope checks, no
sealed autonomous edit, rollback packet creation, rollback dry-run success,
shadow artifacts saved, graph rebuild success, changed-module import, proof
receipt generation, and signed/hash-stamped decision data.

`T1` adds static cleanup proof and minimal ghost/import proof.

`T2` adds relevant tests, behavior fingerprint equivalence, service graph
validity, public API compatibility, no new authority/memory/state/tool bypasses,
and rollback dry-run.

`T3` adds declared improvement proof and monitor requirements.

`T4/T5` produce proposal and proof artifacts only.

Proof receipts are written to:

`.aura_architect/shadow_runs/<run_id>/proof_receipt.json`

## Rollback

Rollback packets are created before promotion under:

`.aura_architect/rollback/<run_id>/`

Each packet contains original files, candidate files, hashes, a repo-root hash
summary for changed files, dry-run status, and restoration verification fields.
Restoration uses Aura's atomic writer.

CLI rollback:

```bash
python -m core.architect.cli rollback --run <run_id>
```

## Post-Promotion Monitor

Promotion arms an observation record under:

`.aura_architect/observations/`

The lightweight monitor recompiles changed files and can restore the rollback
packet automatically if syntax or invariant checks regress after promotion.
CLI status:

```bash
python -m core.architect.cli monitor
python -m core.architect.cli monitor --run <run_id>
```

## CLI

```bash
python -m core.architect.cli audit
python -m core.architect.cli smells
python -m core.architect.cli graph
python -m core.architect.cli plan --target <path-or-smell-id>
python -m core.architect.cli shadow-run --plan <plan_id>
python -m core.architect.cli promote --run <run_id>
python -m core.architect.cli monitor
python -m core.architect.cli rollback --run <run_id>
python -m core.architect.cli auto --tier-max T1
python -m core.architect.cli auto --tier-max T2
python -m core.architect.cli proposal --target <path-or-smell-id>
```

Use `--repo <path>` to point the CLI at a temp repository for destructive-loop
testing.

## MVP

The production MVP is autonomous unused-import cleanup and uncertain dead-code
quarantine planning. `auto --tier-max T1` builds the graph, detects cleanup
candidates, creates a staged plan, applies it in shadow, runs ghost proof,
creates and dry-runs rollback, promotes when proof passes, records a receipt,
and immediately performs a lightweight monitor pass.

## Configuration

- `AURA_ASA_ENABLED`: register/run background audit mode.
- `AURA_ASA_AUTOPROMOTE`: allow boot/background autonomous promotion when a
  caller uses the governor service.
- `AURA_ASA_MAX_TIER`: maximum autonomous tier (`T1` by default).
- `AURA_ASA_SHADOW_TIMEOUT`: bounded subprocess timeout.
- `AURA_ASA_OBSERVATION_WINDOW`: monitor observation window hint.
- `AURA_ASA_PROTECTED_PATHS`: additional protected path patterns separated by
  the platform path separator.
- `AURA_ASA_SAFE_BOOT_COMMAND`: explicit safe boot command for ghost boot.

## Failure Modes

ASA fails closed. Missing rollback, missing proof, sealed edits, failed imports,
failed graph rebuilds, behavior regressions, and weakened protected invariants
reject promotion. Protected and sealed surfaces can still receive proposal
artifacts for human review, but the autonomous path cannot promote them.

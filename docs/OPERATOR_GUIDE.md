# Aura — Operator Guide

Running Aura on your own hardware, and what to do when it goes wrong.

*Last reviewed against the tree: 2026-08-01.*

## Requirements
- macOS on Apple Silicon. The tracked target is M5-class.
- 64 GB+ RAM. That's the reference target for the 27B Cortex plus sustained
  background loops — not a comfort margin.
- 50 GB+ free disk for models and data.
- Python 3.12.

Before anything else, know where the incident docs are.
[KNOWN_FAILURE_MODES.md](../KNOWN_FAILURE_MODES.md) catalogues 19 ways this
runtime fails, each with a runbook in [runbooks/](runbooks/). Five of those
— F15 through F19 — are not hypothetical. They happened on the live desktop
and the forensics are still on disk. One of them, F16, is **not fully
fixed**, and its runbook says so instead of implying otherwise.

## Install + boot
```bash
git clone https://github.com/youngbryan97/aura
cd aura
make setup-prod   # fail-closed install: no fallbacks, a missing dep fails the install
make quality      # the full scrutiny sweep (see below)
make production-gate
make provenance   # writes artifacts/provenance/{sbom,provenance}.json
make run          # foreground launch
```

`make quality` is the aggregate gate. It runs, in order: `source-hygiene`,
`enterprise-gate`, `enterprise-collect`, `production-gate`,
`frontend-contract`, `cognitive-gate-audit`, `skill-catalog-audit`,
`model-load-audit`, `resource-observation-audit`, `integration-liveness`,
`architecture-map`, `compile`, `lint`, `governance-lint`, `security`,
`typecheck`, `smoke`.

Two gates worth knowing separately:

- `make layering` — the DEPS include-rule gate. `core/runtime` and
  `core/observability` carry `DEPS` files and may not import cognition or
  agency. The grandfathered baseline in `config/layering_baseline.json` only
  ever shrinks.
- `make test` — the full offline suite, run as 6 bounded process chunks via
  `tools/run_test_chunks.py`. As of 2026-08-21 the tree collects **40,139
  tests across 2,697 files**. A single pytest process over the whole suite
  gets OOM-killed around 83%; always use the chunk runner.

## Backup & restore
- Backup: `tar czf aura-backup.tar.gz ~/.aura/data ~/.aura/live-source`.
- Restore: `tar xzf aura-backup.tar.gz -C ~/`.

## Diagnostics
- `aura doctor`             — pre-boot self-check (python, sqlite, mlx,
  data dir, atomic writer round-trip)
- `aura doctor --bundle [--bundle-path PATH]` — assembles a redacted
  tarball (health, config, metrics, tasks, models, memory, gateway,
  receipts, audit chain export, recent logs) for incident triage. The
  bundle is what every runbook in `docs/runbooks/` references.
- `aura conformance`        — schema + integrity sweep
- `aura verify-state`       — cross-subsystem state coherence
- `aura verify-memory`      — memory facade integrity
- `aura rebuild-index`      — vector index rebuild
- `aura chaos`              — fault injection smoke
- Dashboard: open `http://localhost:<port>/api/dashboard/snapshot` for a
  raw JSON view of every live subsystem.

## General environment stress runs

The general environment OS is documented in
[`docs/GENERAL_ENVIRONMENT_AUTONOMY.md`](GENERAL_ENVIRONMENT_AUTONOMY.md).
Run the deterministic canary before any strict-real long run:

```bash
python challenges/nethack_challenge.py --mode simulated --steps 100
```

Run NetHack as a strict real stress adapter:

```bash
python challenges/nethack_challenge.py --mode strict_real --steps 5000
```

The trace defaults to `~/.aura/logs/nethack/kernel_trace.jsonl`.

## Platform posture
The deliberate platform decisions — RBAC, SSO, tenant isolation, DR,
plugin signing — are declared in [`docs/PLATFORM_POSTURE.md`](PLATFORM_POSTURE.md)
along with what enforces each one in code.

## Production readiness
The non-longevity production bar is
[`docs/PRODUCTION_READINESS_STANDARD.md`](PRODUCTION_READINESS_STANDARD.md).
It covers clean-clone install, compile, collection, full tests, quality,
governance bypass sweeps, proof bundle regeneration, signed release,
SBOM/provenance, privacy, incident response, rollback, model/provider failure,
and replayable memory/state writes.

## Privacy and retention
Retention/deletion rules are in
[`docs/DATA_RETENTION_DELETION_POLICY.md`](DATA_RETENTION_DELETION_POLICY.md).
Continuous experience frames enforce private and standard retention windows and
redact private exports by default.

## Service-level objectives
The contract operators can hold Aura to lives in [`docs/SLO.md`](SLO.md).
Numbers are measured by `python -m slo.measure` and gated in CI
(`.github/workflows/slo-gate.yml`); a regression past tolerance or a
hard-limit breach fails the release gate.

## Runbooks
Every documented incident class has a runbook under
[`docs/runbooks/`](runbooks/) with concrete symptoms tied to fields the
diagnostics bundle emits, plus diagnosis, mitigation, rollback, and
verification steps.

## Self-improving research core
Aura ships her own hybrid attention/SSM/MoE/world-head model and an
autonomous research substrate that drives capability evaluation,
algorithm discovery, semantic verification, and unknown-unknown test
generation under a statistical promotion gate. The core registers
itself in the `ServiceContainer` as `research_core` and runs cycles
in process. Inspect via `aura doctor --bundle` — the bundle includes
`research_core.json` with iteration count, last cycle time, model
parameter count, vault size, and the most recent five cycle reports.

## Tamper-evident audit trail
Every receipt the runtime emits is appended to a hash-chained ledger at
`~/.aura/receipts/_chain.jsonl`. To verify the chain after an incident:
`python -c "from core.runtime.receipts import get_receipt_store;
print(get_receipt_store().verify_chain())"`. The diagnostics bundle
includes a portable export at `audit_chain/chain.jsonl` plus a
`MANIFEST.txt` with the head hash and length.

## Self-modification quarantine
When Aura proposes a code mutation, the typed evaluator in
`core/self_modification/mutation_safety.py` runs it in a subprocess
with rlimits and emits one of seven outcomes: `passed`, `compile_fail`,
`import_fail`, `runtime_exception`, `assertion_fail`, `timeout`, `oom`.
Any non-`passed` outcome is written to
`~/.aura/data/mutation_quarantine/<id>/` with the source, optional
test source, stdout, stderr, and a structured `result.json`. A
malformed mutation cannot crash the parent process.

## Reading logs
- Live tail: `tail -f ~/.aura/data/logs/aura.log`
- Receipt log: `~/.aura/data/agency_receipts/agency_receipts.jsonl`
- Will receipts: `~/.aura/data/will_receipts/receipts.jsonl`
- Stem cells: `~/.aura/data/stem_cells/`
- Migration ledger: `~/.aura/data/migration/ledger.jsonl`

## Service lifecycle
- macOS launchd: `launchctl load ~/Library/LaunchAgents/aura.plist`
- Linux systemd: `systemctl --user start aura`
- Stop: SIGTERM is graceful (drains receipts, revokes capability tokens).

## Model configuration
- `AURA_MODEL`        — primary model name (default: `Aura-Cortex` / fused Qwen3.8-27B)
- `AURA_DEEP_MODEL`   — heavy lane for solver tier
- `AURA_LLM__MLX_DEEP_MODEL_PATH` — explicit on-disk path
- There is no cloud fallback and no setting for one. Every lane the router
  can reach is local, and `allow_cloud_fallback` is coerced to `False` in
  `core/brain/request_contract.py` whatever a caller passes — see
  [`docs/runbooks/local-inference-boundary.md`](runbooks/local-inference-boundary.md).
- Failure policy: [`docs/MODEL_PROVIDER_FAILURE_POLICY.md`](MODEL_PROVIDER_FAILURE_POLICY.md).

### Fully local frontier-reasoning solver lane

Aura can fetch an optional local reasoning solver without using an external
inference server. This does not replace the Aura Cortex/personality lane; it
adds a governed solver model for hard reasoning/tool-validation calls.

```bash
python scripts/fetch_models.py --reasoning-solver r1-qwen32b --status --print-env
python scripts/fetch_models.py --reasoning-solver r1-qwen32b
```

Supported aliases:

- `r1-qwen32b` → `DeepSeek-R1-Distill-Qwen-32B-4bit`
- `r1-qwen32b-8bit` → `DeepSeek-R1-Distill-Qwen-32B-8bit`
- `qwq32b` → `QwQ-32B-4bit`

After download, use the exports printed by `--print-env`, for example:

```bash
export AURA_DEEP_MODEL=DeepSeek-R1-Distill-Qwen-32B-4bit
export AURA_LLM__MLX_DEEP_MODEL_PATH=/Users/bryan/Desktop/aura/models/DeepSeek-R1-Distill-Qwen-32B-4bit
```

Keep this separate from the primary 27B desktop Cortex unless a live proof run
shows the alternate lane preserves Aura's conversation identity, RAM envelope,
and full-mind route.

## Performance tuning
- There is no Performance settings group. Lane concurrency is not an operator
  setting: one model loads at a time through the GPU semaphore, and the
  memory monitor decides what stays warm.
- Memory monitor lowers max_tokens under RAM pressure and triggers a VRAM
  purge as pressure climbs. The ceilings are set by
  `AURA_PROCESS_RSS_LIMIT_GB` (main process), `AURA_MLX_MEMORY_LIMIT_GB`
  (MLX allocator), `AURA_MLX_WORKER_RSS_LIMIT_GB` (inference worker), and
  `AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB` (refuse a heavy Cortex load below this much
  free memory; env var name retained for compatibility). There is no `AURA_MEM_THRESHOLDS` variable.
- Under sustained pressure the OOM shed ladder drops load bottom-up
  starting with the prompt KV cache; the current shed order is reported in
  `runtime_health_report()["integrity"]`.

## Security settings
- Conscience: hard-line rules at `~/.aura/data/conscience/rules.sha256`
  — tampering refuses all actions until the file is restored.
- World bridge: per-channel permissions live at
  `~/.aura/data/world/permissions.json`.
- Capability tokens are bound to PID + thread. Restart revokes all live
  tokens.

## Governance lint
`make governance-lint` fails the build if any code makes a direct consequential
call exceeding the ratchet baseline enforced by `tools/lint_governance.py`.

## Physical actuation and Reality Reach

Physical requests do not go straight to a device. `core/reality_reach/`
compiles a requested observable into a typed contract, proves reachability
against the host's declared sensor/actuator channels, and returns a
limitation certificate when the request cannot be met — rather than an
optimistic simulation or a verbal success claim. Evidence is layered
`internal` / `effective` / `direct` / `ambient`, and transport success is
never treated as effect verification.

Operationally that means:

- Registered hardware is dispatched through `HardwareManager` and
  `BaseHardwareDevice.safe_execute`; a robotics or environment action cannot
  fall through to an unrelated AppleScript handler.
- Command acceptance, transport completion, actuator execution, observed
  local effect, and promoted evidence are separate monotonic receipt states.
  No earlier state stands in for a later one.
- Boot registers the Reality Reach service during cognitive/sensory
  initialization and refreshes the host inventory off the event loop.
  Readiness means at least one currently usable declared channel plus a
  healthy refresh loop.

Invariants, runtime ownership, the open implementation ledger, and an
explicit statement of what is *not* claimed are in
[`docs/REALITY_REACH.md`](REALITY_REACH.md). Read the "Current Evidence"
section before repeating any physical claim from this system.

## Debugging entry points

- `AURA_PASS_BISECT_LIMIT=N` runs only the first N cognitive phases — binary
  search N to find which phase ruined an answer. `AURA_PASS_TRACE=1`
  announces each phase as it runs.
- `runtime_health_report()["integrity"]` carries taint, lockdep splats, PSI,
  the OOM shed order, sanitizer findings, the last verifier report, telemetry
  limit violations, and unsupported claims.
- `get_bus_recorder().dump()` writes the event-bus ring for replay;
  `get_tracer().write()` writes a Perfetto-loadable trace;
  `get_memory_infra().diff(a, b).narrative()` names what grew.
- Crash forensics when the runtime dies: `data/error_logs/crash/`
  (faulthandler, loop-wedge and memory-spike stacks),
  `data/error_logs/stalls/`, `data/error_logs/memory/` (sentinel ring,
  tombstones, death syslogs), and `~/.aura/logs/desktop-launch.log` for the
  live stdout stream.
- Set `AURA_LOG_DIR` for anything test-like so you never write into the live
  instance's logs.

The full map is [docs/ENGINEERING_ADOPTION.md](ENGINEERING_ADOPTION.md).

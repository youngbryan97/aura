# Aura — Operator Guide

How to run Aura on your own hardware, configure system settings, and troubleshoot operational issues.

*Last reviewed against the tree: 2026-08-01.*

## Requirements
- macOS on Apple Silicon (M-series chips; development is tracked on M5-class hardware).
- 64 GB or more of unified memory (RAM). This is the required minimum to run the 27-billion parameter language model (27B Cortex) alongside continuous background processing tasks — not an optional buffer.
- 50 GB or more of free disk space for AI models and operational data.
- Python 3.12.

Before starting, review the incident response documentation. [KNOWN_FAILURE_MODES.md](../KNOWN_FAILURE_MODES.md) catalogs 19 specific failure modes for this runtime, each linked to a step-by-step troubleshooting guide in [runbooks/](runbooks/). Five of these (F15 through F19) occurred in actual production testing, and their diagnostic forensic logs are preserved in the repository. Failure mode F16 is **not yet fully resolved**, and its runbook explicitly documents its current status and workarounds.

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

`make quality` is the primary verification suite. It runs the following automated checks in order: `source-hygiene`, `enterprise-gate`, `enterprise-collect`, `production-gate`, `frontend-contract`, `cognitive-gate-audit`, `skill-catalog-audit`, `model-load-audit`, `resource-observation-audit`, `integration-liveness`, `architecture-map`, `compile`, `lint`, `governance-lint`, `security`, `typecheck`, and `smoke`.

Two specific checks are especially important:

- `make layering` — Enforces architectural boundaries defined in `DEPS` files. Low-level components (`core/runtime` and `core/observability`) are strictly forbidden from importing high-level cognitive or autonomous agent modules. Any legacy exceptions listed in `config/layering_baseline.json` can only be removed over time, never added.
- `make test` — Runs the full test suite split into 6 separate worker processes via `tools/run_test_chunks.py`. As recorded in `config/test_inventory.json` on 2026-09-30, the test suite contains **60,348 tests across 4,457 files**. Running all tests in a single `pytest` process runs out of system memory (OOM crash) at about 83% completion, so you must always use the chunk runner.

## Backup & restore
- Backup: `tar czf aura-backup.tar.gz ~/.aura/data ~/.aura/live-source`
- Restore: `tar xzf aura-backup.tar.gz -C ~/`

## Diagnostics
- `aura doctor` — Pre-flight check before booting (verifies Python environment, SQLite database, Apple MLX framework, storage directories, and file writing permissions).
- `aura doctor --bundle [--bundle-path PATH]` — Packages a redacted diagnostic archive (containing system health, active configuration, runtime metrics, background tasks, model status, memory state, API gateway info, execution receipts, audit logs, and recent console output) for troubleshooting. The runbooks in `docs/runbooks/` reference the fields in this bundle.
- `aura conformance` — Validates database schemas and data integrity.
- `aura verify-state` — Checks that state is consistent across all subsystems.
- `aura verify-memory` — Verifies the internal integrity of the memory storage system.
- `aura rebuild-index` — Reconstructs the semantic vector search index from stored memories.
- `aura chaos` — Runs basic fault-injection smoke tests to verify error handling.
- Dashboard: Open `http://localhost:<port>/api/dashboard/snapshot` in a web browser for a raw JSON view of all active subsystems.

## General environment stress runs

Documentation for operating in interactive general environments is in [`docs/GENERAL_ENVIRONMENT_AUTONOMY.md`](GENERAL_ENVIRONMENT_AUTONOMY.md).
Run the deterministic canary test before any extended run in a real environment:

```bash
python challenges/nethack_challenge.py --mode simulated --steps 100
```

Run NetHack as an extended stress test in a real environment:

```bash
python challenges/nethack_challenge.py --mode strict_real --steps 5000
```

The trace log defaults to `~/.aura/logs/nethack/kernel_trace.jsonl`.

## Platform posture
Key architectural decisions — such as intentionally avoiding role-based access control (RBAC), omitting single sign-on (SSO), using single-tenant deployment, relying on manual disaster recovery (DR), and requiring cryptographic hash allowlists for plugins — are detailed in [`docs/PLATFORM_POSTURE.md`](PLATFORM_POSTURE.md), along with the specific code mechanisms that enforce each choice.

## Production readiness
The baseline standard for production deployment is defined in [`docs/PRODUCTION_READINESS_STANDARD.md`](PRODUCTION_READINESS_STANDARD.md). It covers fresh installation from a clean clone, compilation, test collection, the full test suite, quality checks, governance bypass scans, regeneration of proof bundles, signed releases, software bill of materials (SBOM) and build provenance, privacy requirements, incident response procedures, automated rollbacks, handling model and API provider failures, and deterministic, replayable state and memory storage.

## Privacy and retention
Data retention and deletion policies are defined in [`docs/DATA_RETENTION_DELETION_POLICY.md`](DATA_RETENTION_DELETION_POLICY.md). Continuous experience records adhere to distinct private and standard retention windows, and sensitive private data is redacted by default in diagnostic exports.

## Service-level objectives
Operational performance targets (Service-Level Objectives, or SLOs) are documented in [`docs/SLO.md`](SLO.md). Metrics are gathered using `python -m slo.measure` and automatically validated in continuous integration (`.github/workflows/slo-gate.yml`). Any performance degradation exceeding defined tolerances or breaching a hard limit automatically blocks deployment.

## Runbooks
Every known incident scenario has a dedicated runbook under [`docs/runbooks/`](runbooks/). Each guide maps observable system symptoms to specific diagnostic fields from `aura doctor --bundle`, followed by concrete steps for diagnosis, temporary mitigation, system rollback, and post-fix verification.

## Self-improving research core
Aura includes an experimental neural architecture (combining attention mechanisms, state-space models [SSM], and mixture-of-experts [MoE] layers) alongside an autonomous research engine. This engine runs iterative cycles to evaluate capabilities, explore algorithmic improvements, verify logic, and generate stress tests, promoting new changes only when they meet statistical quality thresholds.

The research engine registers as `research_core` in the application `ServiceContainer` and runs in the background within the main process. You can inspect its status using `aura doctor --bundle`; the resulting archive includes `research_core.json`, which details the total iteration count, timestamp of the last cycle, model parameter count, storage vault size, and summary reports from the last five research cycles.

## Tamper-evident audit trail
Every action receipt produced by the runtime is appended to a cryptographically linked log (a hash chain where each entry includes the hash of the preceding one) at `~/.aura/receipts/_chain.jsonl`. To verify the integrity of the audit log after an incident:
`python -c "from core.runtime.receipts import get_receipt_store; print(get_receipt_store().verify_chain())"`. The diagnostics bundle includes an export of this log at `audit_chain/chain.jsonl`, accompanied by a `MANIFEST.txt` file listing the latest hash and total chain length.

## Self-modification quarantine
When Aura proposes an automated code change (mutation), the safety evaluator in `core/self_modification/mutation_safety.py` tests it in an isolated subprocess with strict resource limits (CPU and memory limits). The evaluation produces one of seven outcomes: `passed`, `compile_fail`, `import_fail`, `runtime_exception`, `assertion_fail`, `timeout`, or `oom` (out of memory).

If a proposed change results in anything other than `passed`, it is quarantined in `~/.aura/data/mutation_quarantine/<id>/`. This folder preserves the generated code, optional test files, standard output, error messages, and a structured `result.json` file. Because execution is sandboxed in a separate process, broken code cannot crash the parent application process.

## Reading logs
- Live log output: `tail -f ~/.aura/data/logs/aura.log`
- Action receipts: `~/.aura/data/agency_receipts/agency_receipts.jsonl`
- Decision and intent receipts: `~/.aura/data/will_receipts/receipts.jsonl`
- Dynamic code generation records (stem cells): `~/.aura/data/stem_cells/`
- Database migration ledger: `~/.aura/data/migration/ledger.jsonl`

## Service lifecycle
- macOS background service (launchd): `launchctl load ~/Library/LaunchAgents/aura.plist`
- Linux user service (systemd): `systemctl --user start aura`
- Stop: Sending a standard termination signal (`SIGTERM`) triggers a clean, graceful shutdown — pending receipts are written to disk and active authorization tokens are revoked before exiting.

## Model configuration
- `AURA_MODEL` — Name of the primary conversational model (default: `Aura-Cortex`, based on a quantized 27-billion parameter Qwen model).
- `AURA_DEEP_MODEL` — Identifier for an optional heavy reasoning model used for difficult analytical problems.
- `AURA_LLM__MLX_DEEP_MODEL_PATH` — File path to the heavy reasoning model on local disk.
- There is no cloud fallback and no setting for one. All model routing stays entirely on your local machine. In code, `allow_cloud_fallback` is hard-coded to `False` in `core/brain/request_contract.py` regardless of any parameter passed by callers — see [`docs/runbooks/local-inference-boundary.md`](runbooks/local-inference-boundary.md).
- Failure policy: Guidelines for handling model loading or inference errors are documented in [`docs/MODEL_PROVIDER_FAILURE_POLICY.md`](MODEL_PROVIDER_FAILURE_POLICY.md).

### Fully local frontier-reasoning solver lane

Aura can fetch an optional local reasoning solver without using an external inference server. This does not replace the Aura Cortex conversational model; instead, it provides a dedicated secondary solver model for complex mathematical, logic, and tool-validation queries.

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

Keep this separate from the primary 27B desktop Cortex unless a live proof run shows the alternate lane preserves Aura's conversational personality, fits within system RAM limits, and supports all internal cognitive routing.

Setting these variables does not admit the model on its own. A second deep model is used only after its artifact has been measured and an attested qualification certificate (`AURA_DEEP_SPECIALIST_CERTIFICATE`) verifies against a pinned key (`AURA_DEEP_SPECIALIST_TRUST_ROOT`); until then deep work stays on the Cortex. The rule is in `core/brain/llm/model_registry_deep_solver.py`.

## Performance tuning
- There is no Performance settings group. You do not need to configure model parallelism or concurrency settings: one model loads at a time through the GPU semaphore, and the memory monitor decides what stays cached in memory.
- The memory monitor automatically lowers generation length (`max_tokens`) under RAM pressure and purges GPU memory (VRAM) as pressure climbs. The memory ceilings are set by:
  - `AURA_PROCESS_RSS_LIMIT_GB` (main Python process)
  - `AURA_MLX_MEMORY_LIMIT_GB` (Apple MLX allocator)
  - `AURA_MLX_WORKER_RSS_LIMIT_GB` (background inference worker)
  - `AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB` (refuses loading a heavy model when free memory falls below this threshold; environment variable name retained for compatibility)
  - Note: There is no `AURA_MEM_THRESHOLDS` variable.
- Under sustained memory pressure, an automatic load shedding ladder drops non-essential data bottom-up, starting with the prompt key-value (KV) attention cache. The current shedding order is reported in `runtime_health_report()["integrity"]`.

## Security settings
- Safety rules (Conscience): Core behavioral safety rules are protected by a cryptographic hash at `~/.aura/data/conscience/rules.sha256`. If the rule file is modified or tampered with, Aura refuses all actions until the file is restored.
- External permissions (World bridge): Granular permissions for external tools and communication channels live at `~/.aura/data/world/permissions.json`.
- Capability tokens: Security tokens granting access to system capabilities are bound to the specific process ID (PID) and operating thread. Restarting the application revokes all active tokens.

## Governance lint
Running `make governance-lint` fails the build if any code attempts to bypass safety checks or makes high-impact system calls that violate the baseline rules enforced by `tools/lint_governance.py`.

## Physical actuation and Reality Reach

When Aura is asked to interact with physical hardware or sensors in the real world, the request does not send raw commands directly to a device. Instead, `core/reality_reach/` translates the requested real-world outcome into a verifiable contract. It checks whether the machine actually has the necessary sensor or motor controls (reachability), and issues an explicit limitation certificate if the hardware cannot fulfill the request — rather than guessing, simulating, or falsely claiming success. Evidence of real-world outcomes is categorized into tiers (`internal`, `effective`, `direct`, and `ambient`), and successfully sending a command over a network or port is never treated as proof that the physical action succeeded.

Operationally that means:

- Registered hardware is dispatched strictly through `HardwareManager` and `BaseHardwareDevice.safe_execute`. A robotic or environmental action cannot fall through to an unrelated desktop automation or AppleScript handler.
- Command acceptance, transport completion, actuator execution, observed local effect, and promoted evidence are separate, non-reversible receipt states. No earlier state stands in for a later one.
- System startup registers the Reality Reach service during cognitive and sensory initialization, and refreshes the host device inventory off the event loop. System readiness requires at least one currently usable declared channel plus a healthy refresh loop.

Invariants, runtime ownership, the open implementation ledger, and an explicit statement of what is *not* claimed are documented in [`docs/REALITY_REACH.md`](REALITY_REACH.md). Read the "Current Evidence" section before repeating any physical claim from this system.

## Debugging entry points

- `AURA_PASS_BISECT_LIMIT=N` runs only the first N cognitive phases — binary search N to find which phase degraded an answer. `AURA_PASS_TRACE=1` announces each phase in the console as it runs.
- `runtime_health_report()["integrity"]` aggregates system health diagnostics, including security taint flags, deadlock warnings (lockdep splats), pressure stall information (PSI), the memory shed order, memory and thread sanitizer findings, the last verifier report, telemetry limit violations, and unsupported claims.
- Tracing and replay tools:
  - `get_bus_recorder().dump()` writes recent event-bus messages for step-by-step replay.
  - `get_tracer().write()` generates a performance trace viewable in the [Perfetto](https://ui.perfetto.dev/) trace viewer.
  - `get_memory_infra().diff(a, b).narrative()` compares two memory snapshots (`a` and `b`) and explains what grew.
- Crash diagnostics when the runtime dies:
  - `data/error_logs/crash/` (Python fault handler stack traces, event-loop stalls, and memory-spike traces).
  - `data/error_logs/stalls/` (thread stall logs).
  - `data/error_logs/memory/` (memory monitor ring buffer, process termination records [tombstones], and system memory logs).
  - `~/.aura/logs/desktop-launch.log` for the live console standard output stream.
- Set `AURA_LOG_DIR` for test runs and experiments so you never write into the live instance's logs.

The full map is [docs/ENGINEERING_ADOPTION.md](ENGINEERING_ADOPTION.md).

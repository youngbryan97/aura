# Aura Risk Register

## Active Risks

| ID | Risk | Impact | Likelihood | Current Mitigation | Planned Action |
|---|---|---|---|---|---|
| R-001 | Missing mission-critical spec docs from repo | High | High | Record explicitly in tracker | Reconstruct execution docs in `docs/` and continue from live code/docs |
| R-002 | False `ok: true` in skill/result paths | Critical | Medium | Patched in both BaseSkill paths and capability engine; regressions added | Expand verification across additional skill/router surfaces |
| R-003 | Sync execution blocks event loop | Critical | Medium | Legacy infrastructure BaseSkill now offloads sync `execute()` via `asyncio.to_thread`; regression added | Continue auditing sync-in-async hotspots beyond BaseSkill |
| R-004 | Filesystem existence fast path bypasses governance | Critical | Low | Direct shortcut disabled for user-facing requests; regression added | Reintroduce only via governed `file_operation`/receipt path if needed |
| R-005 | `LocalPipeBus` loop/bootstrap hazards deadlock IPC | Critical | Medium | Orphan-loop creation removed, per-bus executors added, supervised actors now use split read/write pipe pairs, legacy shared single-connection compatibility removed, regressions added | Sweep for any remaining out-of-band IPC construction outside `ActorBus` / supervisor-managed pairs |
| R-006 | `StateVault` readiness non-fatal in strict runtime | Critical | Low | Strict runtime now aborts boot on failed vault handshake; regression added | Extend strict fail-closed behavior to other critical services |
| R-007 | `GracefulShutdown` async `sys.exit(0)` bypasses cleanup | High | Low | Async shutdown no longer exits the process directly; regression added | Audit remaining direct exit paths in launcher/runtime surfaces |
| R-008 | `agency_test` exception in production file tool | High | Low | Exception removed; regression added | Sweep for any other test-only production exceptions |
| R-009 | Unowned `asyncio.create_task` spread across codebase | Critical | Medium | SensoryGate and StateVault background tasks now have owned task sets; launcher/bootstrap/watchdog/sovereign-watchdog hot paths, websocket disconnect scheduling, graceful-shutdown signal scheduling, bus-internal telemetry/reader loops, `StateRepository` consumer startup/repair, `AuraEventBus` Redis listener, `Scheduler` loops, `ContinuousCognitionLoop`, `SessionGuardian`, `SystemGovernor`, actor-local `StateVaultActor` / `SensoryGateActor` loops, lifecycle startup/signal-stop paths in `LifecycleCoordinator`, coordinator-side reply/reflection/thought/TTS paths in `AutonomousConversationLoop`, `MessageCoordinator`, `MetabolicCoordinator`, and `CognitiveCoordinator`, plus metabolic subscription/hook/save/drive-update/impulse/memory-hygiene/archive tasks now use lifecycle ownership | Continue the raw-task ownership sweep in adjacent orchestrator mixins and then move into the TaskSupervisor milestone |
| R-010 | Multiple launch/boot surfaces break runtime singularity | Critical | High | Watchdog is now supervision-only, launcher-only owners spawn the reaper, 3D launcher uses runtime lock detection instead of stale timestamps, strict runtime now fails closed for critical `ResilientBoot` stages, and CLI/server/desktop now share a canonical runtime boot helper | Finish canonical boot/service-manifest ownership across any remaining launch surfaces and runtime entrypoints |
| R-011 | Supervisor heartbeat polling can falsely kill healthy actors | High | Medium | `ActorHealthGate` now counts distinct missed heartbeat windows instead of every poll tick; regression added | Extend heartbeat proof coverage to additional actor classes and supervisor restart flows |
| R-012 | Reaper manifest path can diverge across launcher contexts on macOS | High | Medium | Canonical `AURA_REAPER_MANIFEST` path now drives both parent and reaper sides; regression added | Validate the manifest contract across more subprocess entry surfaces |
| R-013 | Strict runtime can still permit critical boot stages to degrade silently | Critical | Medium | `ResilientBoot` now fails closed for strict-mode failures in `State Repository`, `LLM Infrastructure`, `Cognitive Core`, and `Kernel Interface`; regressions added | Extend strict readiness proof coverage to additional critical services and API boot gates |
| R-014 | Import-time asyncio primitives can bind runtime services to stale/non-running loops | High | Medium | `Scheduler` now lazy-initializes `asyncio.Lock`/`Event` inside runtime execution paths; regressions added | Sweep remaining singleton/runtime services for import-time asyncio primitive construction |

## Acceptance Rule

No critical risk can be considered mitigated until:

- the relevant code path is patched,
- at least one regression test exists,
- the verification command is recorded in the tracker,
- unresolved follow-up work is documented if the fix is partial.

## Phase C-O Risks Mitigated This Session

| ID | Risk | Impact | Likelihood | Current Mitigation | Planned Action |
|---|---|---|---|---|---|
| R-015 | Multiple model/memory/state owners drift after boot | Critical | Medium | `core/runtime/service_manifest.py` + `_enforce_service_manifest` after `lock_registration`; strict-mode aborts on critical violation | Wire concrete `MemoryWriteGateway` / `StateGateway` adapters |
| R-016 | Shutdown order leaves uncommitted state behind | Critical | Medium | `core/runtime/shutdown_coordinator.py` ordered phases (output→memory→state→actors→model→bus→tasks); strict-mode logs failures | Hook into existing graceful shutdown so all subsystems register handlers |
| R-017 | Consequential action commits without governance receipt | Critical | Medium | `core/runtime/will_transaction.py` async context manager; fail-closed; strict-mode logs missing-result | Apply WillTransaction across memory/tool/output paths |
| R-018 | Persistent writes leave torn state | Critical | Medium | `core/runtime/atomic_writer.py` (temp+fsync+rename, parent dir fsync, cleanup on failure, schema-versioned envelopes) | Route remaining JSON state writes through it |
| R-019 | Self-repair patch lands on AST parse alone | Critical | Medium | `core/runtime/self_repair_ladder.py` 8-rung validator; banned-import AST scan; `patch_is_acceptable` requires every rung | Wire into existing self-modification proposal flow |
| R-020 | Flagship modules ship below their named depth | High | High | `core/runtime/depth_audit.py` Tier 0-5 + `enforce_depth_audit` strict-mode block | Have each flagship module emit `report(...)` |
| R-021 | Skill returns success without verifier coverage | Critical | Medium | `core/runtime/skill_contract.py` registry auto-tags un-verified skills as `success_unverified` | Register concrete verifiers per skill |
| R-022 | Sensors activated without governance receipt | Critical | Medium | `core/perception/perception_runtime.py` denies capability without governance and refuses sensor start without token | Wire real platform driver registrations |
| R-023 | Tool / browser action escapes workspace sandbox | Critical | Medium | `core/runtime/security.py` deny-by-default for terminal/network.post/credentials/self.modify; protected-path patterns; browser file:// blocked | Apply policy across browser + file tools |
| R-024 | Stable release ships without abuse / migration / rollback proof | High | Medium | `core/runtime/release_channels.py` enforces gate set; `evaluate_release` rejects partial submissions | Hook into actual release pipeline once it exists |
| R-025 | Formal protocols drift from spec | High | Low | `core/runtime/formal_models.py` state machines + invariant checks for all 7 dangerous protocols | Extend with hypothesis-based property tests |
| R-026 | SLI/SLO targets undocumented | Medium | High | `core/runtime/telemetry_sli.py` SLO_CATALOG covering availability, latency, durability, governance coverage, recovery, actor health, self-mod safety, checkpoints | Wire to OpenTelemetry/Prometheus exporters |
| R-027 | Per-actor resource exhaustion | High | Medium | `core/runtime/memory_guard.py` quotas per actor (memory, threads, fds, subprocess, browser ctx, queue depth, CPU) | Wire enforcement to supervisor health gate |
| R-028 | Conversational turn-taking absent in voice/movie modes | High | Medium | `core/social/turn_taking.py` four-mode engine + `SilencePolicy` | Wire into output gate + perception event stream |

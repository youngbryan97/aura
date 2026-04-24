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
| R-009 | Unowned `asyncio.create_task` spread across codebase | Critical | Medium | SensoryGate and StateVault background tasks now have owned task sets; launcher/bootstrap/watchdog/sovereign-watchdog hot paths, websocket disconnect scheduling, graceful-shutdown signal scheduling, bus-internal telemetry/reader loops, `StateRepository` consumer startup/repair, `AuraEventBus` Redis listener, `Scheduler` loops, `ContinuousCognitionLoop`, `SessionGuardian`, and `SystemGovernor` now use lifecycle ownership | Address remaining actor/process/service-loop hotspots in the next B3 slice, then continue into the TaskSupervisor milestone |
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

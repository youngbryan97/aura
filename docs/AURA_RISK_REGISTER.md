# Aura Risk Register

## Active Risks

| ID | Risk | Impact | Likelihood | Current Mitigation | Planned Action |
|---|---|---|---|---|---|
| R-001 | Missing mission-critical spec docs from repo | High | High | Record explicitly in tracker | Reconstruct execution docs in `docs/` and continue from live code/docs |
| R-002 | False `ok: true` in skill/result paths | Critical | Medium | Patched in both BaseSkill paths and capability engine; regressions added | Expand verification across additional skill/router surfaces |
| R-003 | Sync execution blocks event loop | Critical | Medium | Legacy infrastructure BaseSkill now offloads sync `execute()` via `asyncio.to_thread`; regression added | Continue auditing sync-in-async hotspots beyond BaseSkill |
| R-004 | Filesystem existence fast path bypasses governance | Critical | Low | Direct shortcut disabled for user-facing requests; regression added | Reintroduce only via governed `file_operation`/receipt path if needed |
| R-005 | `LocalPipeBus` loop/bootstrap hazards deadlock IPC | Critical | Medium | Orphan-loop creation removed, per-bus executors added, supervised actors now use split read/write pipe pairs, regressions added | Finish removing remaining legacy shared-connection call paths and tighten supervisor invariants |
| R-006 | `StateVault` readiness non-fatal in strict runtime | Critical | Low | Strict runtime now aborts boot on failed vault handshake; regression added | Extend strict fail-closed behavior to other critical services |
| R-007 | `GracefulShutdown` async `sys.exit(0)` bypasses cleanup | High | Low | Async shutdown no longer exits the process directly; regression added | Audit remaining direct exit paths in launcher/runtime surfaces |
| R-008 | `agency_test` exception in production file tool | High | Low | Exception removed; regression added | Sweep for any other test-only production exceptions |
| R-009 | Unowned `asyncio.create_task` spread across codebase | Critical | Medium | SensoryGate and StateVault background tasks now have owned task sets | Address remaining create-task hotspots in TaskSupervisor milestone |
| R-010 | Multiple launch/boot surfaces break runtime singularity | Critical | High | None | Address in runtime singularity milestone |

## Acceptance Rule

No critical risk can be considered mitigated until:

- the relevant code path is patched,
- at least one regression test exists,
- the verification command is recorded in the tracker,
- unresolved follow-up work is documented if the fix is partial.

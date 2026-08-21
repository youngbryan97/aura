# Aura Runbooks

Each scenario below has a single runbook documenting symptoms, diagnosis,
safe mitigation, unsafe mitigation, rollback, and verification.

| Scenario | Runbook |
| --- | --- |
| Aura will not boot | [aura-will-not-boot.md](aura-will-not-boot.md) |
| Aura stuck before READY | [aura-stuck-before-ready.md](aura-stuck-before-ready.md) |
| Model fails to load | [model-fails-to-load.md](model-fails-to-load.md) |
| Memory corruption detected | [memory-corruption.md](memory-corruption.md) |
| State vault unavailable | [state-vault-unavailable.md](state-vault-unavailable.md) |
| Event bus degraded | [event-bus-degraded.md](event-bus-degraded.md) |
| Actor crash loop | [actor-crash-loop.md](actor-crash-loop.md) |
| Browser actor leaked | [browser-actor-leaked.md](browser-actor-leaked.md) |
| Self-repair failed | [self-repair-failed.md](self-repair-failed.md) |
| Checkpoint restore failed | [checkpoint-restore-failed.md](checkpoint-restore-failed.md) |
| Governance receipt missing | [governance-receipt-missing.md](governance-receipt-missing.md) |
| Tool timeout storm | [tool-timeout-storm.md](tool-timeout-storm.md) |
| High event loop lag | [high-event-loop-lag.md](high-event-loop-lag.md) |
| Disk full | [disk-full.md](disk-full.md) |
| Dirty shutdown recovery | [dirty-shutdown-recovery.md](dirty-shutdown-recovery.md) |
| Camera unavailable | [camera-unavailable.md](camera-unavailable.md) |
| Microphone unavailable | [microphone-unavailable.md](microphone-unavailable.md) |
| Movie mode broken | [movie-mode-broken.md](movie-mode-broken.md) |
| Worker crash | [worker-crash.md](worker-crash.md) |
| Shutdown hangs | [shutdown-hang.md](shutdown-hang.md) |
| Orphaned background tasks | [orphaned-tasks.md](orphaned-tasks.md) |
| Resource exhaustion (RAM/GPU) | [resource-exhaustion.md](resource-exhaustion.md) |
| Prompt injection | [prompt-injection.md](prompt-injection.md) |
| Excessive agency | [excessive-agency.md](excessive-agency.md) |
| External egress privacy incident | [external-egress.md](external-egress.md) |
| Local inference boundary | [local-inference-boundary.md](local-inference-boundary.md) |
| Research core stalled | [research-core-stalled.md](research-core-stalled.md) |
| Disaster recovery | [disaster-recovery.md](disaster-recovery.md) |
| Stale memory retrieval | [stale-memory-retrieval.md](stale-memory-retrieval.md) |
| Identity drift | [identity-drift.md](identity-drift.md) |
| Lock contention / deadlock | [lock-contention-deadlock.md](lock-contention-deadlock.md) |
| Log rotation failure | [log-rotation-failure.md](log-rotation-failure.md) |
| Telemetry emission failure | [telemetry-emission-failure.md](telemetry-emission-failure.md) |
| Pass F maturity risks | [pass-f-maturity-risks.md](pass-f-maturity-risks.md) |

## Observed on the live runtime

The five above the line are classes we plan for. These five actually
happened, on this machine, with forensics to match. They are the ones worth
reading before an incident rather than during one.

| Scenario | Runbook | Status |
| --- | --- | --- |
| `mind_tick` false-death → "Connecting to runtime" | [mind-tick-false-death.md](mind-tick-false-death.md) | Fixed |
| MLX worker-kill cold-lane cascade | [mlx-worker-cold-lane-cascade.md](mlx-worker-cold-lane-cascade.md) | **Partially mitigated — architectural cause open** |
| Failure-lockdown from expected backpressure | [failure-lockdown-from-backpressure.md](failure-lockdown-from-backpressure.md) | Fixed |
| Launch-provenance `ready:false` on source drift | [launch-provenance-not-ready.md](launch-provenance-not-ready.md) | Expected in dev |
| Quadratic cost from a never-reused prompt cache | [prompt-cache-never-reused.md](prompt-cache-never-reused.md) | Fixed |

Every runbook is written against fields that `aura doctor --bundle` emits, so
produce the bundle first:

```bash
aura doctor --bundle
```

The failure-mode catalogue that these runbooks resolve is
[KNOWN_FAILURE_MODES.md](../../KNOWN_FAILURE_MODES.md) (F01–F19).

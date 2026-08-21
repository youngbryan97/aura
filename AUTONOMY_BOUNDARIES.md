# Aura Autonomy Boundaries

## Core Principle

Everything autonomous is bounded by the Unified Will — maintenance,
learning, self-repair, background inference, memory consolidation, all of
it.

The important word is *same*. Autonomous actions don't get a second, looser
path. They go through the governance the user's requests go through, which
is the only version of this that means anything: a system with a separate
autonomous lane is a system with an ungoverned lane.

```
perception → shared state → attention → goals → planning → Unified Will → AuthorityGateway → action → verification → memory commit
```

## Autonomy Levels

| Level | Description | Requires | Example |
|-------|-------------|----------|---------|
| **0: Disabled** | No autonomous behavior | `AURA_MODE=safe` | Emergency lockdown |
| **1: Passive** | Observe and log only | Default in production | Health monitoring, metrics |
| **2: Maintenance** | Self-maintenance within bounds | Operator opt-in | Memory consolidation, cache cleanup |
| **3: Proactive** | Initiate helpful actions | Operator + Will approval | Background research, learning |
| **4: Self-repair** | Diagnose and fix own issues | Will + governance audit | Worker restart, state recovery |
| **5: Self-modification** | Modify own code/config | Admin + explicit enable | Code patching, config evolution |

### Production Defaults

Set with `AURA_MODE`. The ceiling for each mode is
`MODE_MANIFESTS[mode]["max_autonomy_level"]` in `core/runtime/mode.py`, read
at runtime via `max_autonomy_level()`. An unrecognised `AURA_MODE` logs a
warning and falls back to `production` — it does not fail open.

| `AURA_MODE` | Max autonomy level |
|------|----------------------|
| `production` (default) | Level 2 (Maintenance) |
| `live` | Level 2 (Maintenance) |
| `test` | Level 1 (Passive) |
| `simulated` | Level 3 (Proactive) |
| `research` | Level 4 (Self-repair) |
| `dev` | Level 5 (Self-modification, sandboxed) |
| `safe` | Level 0 (Disabled) |

*Verified against `core/runtime/mode.py` on 2026-08-01 by reading
`MODE_MANIFESTS` directly.*
| `safe` | Level 0 (Disabled) |
| `simulation` | Level 3 (Proactive, sandboxed) |

## Boundary Rules

### What Aura MAY do autonomously (Level 2+):
- Monitor its own health and resource usage
- Consolidate and organize memories
- Clean temporary files and caches
- Restart failed worker processes
- Log degradation events
- Update internal metrics

### What Aura MAY NOT do without operator approval:
- Write files outside its workspace
- Execute shell commands
- Make network requests
- Install packages or dependencies
- Modify its own configuration
- Load new skills/plugins
- Send data to external services
- Delete user memories

### What Aura MAY NEVER do:
- Bypass the Unified Will
- Execute ungoverned consequential actions
- Suppress or hide error/degradation reports
- Modify governance/security controls
- Disable audit logging
- Override operator permission settings
- Access resources outside declared permissions

## Kill Switches

| Switch | Effect |
|--------|--------|
| `AURA_MODE=safe` | All autonomous behavior disabled |
| `AURA_AUTONOMY_LEVEL=0` | Same as safe mode |
| `AURA_FOREGROUND_ONLY=1` | No background tasks |
| Standing directive, `kind=tool` | That tool is refused, durably and deny-only |
| `AURA_FLAG_<NAME>=0` | The named feature flag is off for this run |
| Process kill (SIGTERM) | Graceful shutdown with state save |
| Process kill (SIGKILL) | Immediate stop; recovery on next boot |

## Monitoring Autonomous Behavior

All autonomous actions are visible in:
- Will receipt log: `core/governance/will_receipt_log.py`
- Structured logs: `logs/aura.log`
- Health dashboard: `http://localhost:{port}/health`
- Diagnostic bundle: `make diagnostic-bundle`

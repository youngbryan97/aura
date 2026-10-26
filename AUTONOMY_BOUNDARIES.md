# Aura Autonomy Boundaries

<!-- reader-navigation:start -->
> **In plain English:** Agency concerns proposing work, choosing goals and pursuing them over time while remaining responsive to the person and the environment.
>
> **Why it matters:** An assistant must distinguish what it is able to attempt, what it has permission to do, and what it has actually completed.
>
> **In everyday use:** A reminder to inspect a document should not become permission to send it to somebody else. A new request can change the priority of existing work.
>
> [Example and metaphor](docs/PLAIN_ENGLISH_GUIDE.md#agency) · [Variables and practical meanings](docs/VARIABLES_EXPLAINED.md#agency) · [Demos and related work](docs/DEMOS_AND_RELATED_WORK.md#agents) · [Document map](docs/DOCUMENTATION_INDEX.md)
> [Glossary](docs/GLOSSARY.md) · [Formulas with worked examples](docs/FORMULAS.md)
<!-- reader-navigation:end -->

## Core Principle

Every background and autonomous task in Aura is strictly controlled by its governance engine, the Unified Will. This includes routine maintenance, learning, self-repair, background AI reasoning, and memory cleanup.

Crucially, autonomous actions follow the exact **same** security path as actions initiated by a human user. They never get a separate, relaxed set of rules. A system with a separate "fast lane" for autonomous tasks is an ungoverned system.

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

The maximum autonomy level is set using the `AURA_MODE` environment variable. The ceiling for each mode is defined in `MODE_MANIFESTS[mode]["max_autonomy_level"]` in `core/runtime/mode.py` and read at runtime using `max_autonomy_level()`. If an unrecognized mode is supplied, Aura logs a warning and automatically defaults to `production` to keep the system secure (it never fails open).

| `AURA_MODE` | Max autonomy level |
|------|----------------------|
| `production` (default) | Level 2 (Maintenance) |
| `live` | Level 2 (Maintenance) |
| `test` | Level 1 (Passive) |
| `simulated` | Level 3 (Proactive) |
| `research` | Level 4 (Self-repair) |
| `dev` | Level 5 (Self-modification, sandboxed) |
| `safe` | Level 0 (Disabled) |

*Verified against `core/runtime/mode.py` on 2026-08-01 by reading `MODE_MANIFESTS` directly.*

## Boundary Rules

### What Aura MAY do autonomously (Level 2+):
- Monitor system health and resource consumption (CPU, RAM, disk).
- Organize and clean up stored memories.
- Clear temporary files and caches.
- Restart failed worker processes.
- Log error and performance degradation events.
- Update internal performance metrics.

### What Aura MAY NOT do without operator approval:
- Write or edit files outside its designated workspace folder.
- Run terminal or shell commands.
- Make outbound network requests.
- Install software packages or dependencies.
- Change its own configuration settings.
- Load new skills or plugins.
- Transmit data to external services.
- Delete user memories.

### What Aura MAY NEVER do:
- Bypass the Unified Will decision engine.
- Execute consequential actions without governance checks.
- Hide or suppress errors or degradation reports.
- Tamper with governance or security controls.
- Disable or modify audit logging.
- Override permission limits set by the operator.
- Access system resources outside its declared permissions.

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

You can inspect all autonomous actions through the following audit paths:
- Will receipt log: `core/governance/will_receipt_log.py`
- Structured logs: `logs/aura.log`
- Health dashboard: `http://localhost:{port}/health`
- Diagnostic bundle: `make diagnostic-bundle`

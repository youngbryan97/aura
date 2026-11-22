# Aura Autonomy Boundaries

<!-- reader-navigation:start -->
> Aura can propose tasks, keep goals, and return to unfinished work. The user and the current situation can change which task takes priority.
>
> The useful questions are what it tried, who allowed it, and whether it finished. Permission to inspect a document covers reading it. Sending that document is a separate action that needs its own authorization.
>
> [Plain-language guide](docs/PLAIN_ENGLISH_GUIDE.md#agency) · [Terms and examples](docs/VARIABLES_EXPLAINED.md#agency) · [Evidence and related work](docs/DEMOS_AND_RELATED_WORK.md#agents)
> [All docs](docs/DOCUMENTATION_INDEX.md) · [Glossary](docs/GLOSSARY.md) · [Worked formulas](docs/FORMULAS.md)
<!-- reader-navigation:end -->

## Core Principle

An assistant can suggest work on its own. Before a proposed action runs,
the same permission path should apply whether the user requested it or a
background job suggested it. On governed routes, the Unified Will checks the
proposal and the authority gateway checks the tool action. The design covers
maintenance, learning, self-repair, background reasoning, and memory cleanup.
The [claim ledger](CLAIMS_NOT_SUPPORTED.md#10-process-level-non-bypassable-governance)
records the limits of enforcement.

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

The maximum autonomy level is set using the `AURA_MODE` environment variable. The ceiling for each mode is defined in `MODE_MANIFESTS[mode]["max_autonomy_level"]` in `core/runtime/mode.py` and read at runtime using `max_autonomy_level()`. If an unrecognized mode is supplied, Aura logs a warning and defaults to `production`. That is a level-2 ceiling, not the level-0 safe
mode. A misspelled request for lockdown therefore does not establish lockdown;
verify the resolved mode. Tier admission and individual tool policies still
apply.

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

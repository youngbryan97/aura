# Aura Tool Use Policy

*Reviewed against the tree: 2026-08-01. See [documentation status map](docs/DOC_STATUS.md) for how to read this file.*

## Scope

Every tool and skill execution in the runtime.

The premise is one sentence: a tool call is a consequential action. Not a
function call that happens to touch the outside world — a consequential
action, which means it has to be authorized, sandboxed, audited, and
recoverable before it runs, not explained afterward.

That framing is doing real work. It's why there is no "safe" tool category
that skips the gate, and why a skill Aura writes herself goes through the
same path a user-requested one does.

## Principles

1. **No tool executes without Will authorization**: Every tool call passes
   through the Unified Will and receives a WillReceipt.
2. **All tool output is untrusted**: Results from tools are treated as external
   untrusted input and sanitized before influencing Aura's behavior.
3. **Least privilege**: Each skill requests only the permissions it needs.
4. **Fail closed**: If authorization is unavailable, tool execution is refused.
5. **Auditable**: Every tool invocation, its authorization, input, output, and
   outcome are logged.

## Skill Contract

Every skill/tool must declare:

```yaml
name: skill_name
version: "1.0.0"
description: "What this skill does"
risk_level: low | medium | high | critical
permissions:
  filesystem: none | read | write | workspace_only
  network: none | local | external
  shell: none | sandboxed | full
  memory: none | read | write
input_schema:
  type: object
  properties: { ... }
output_schema:
  type: object
  properties: { ... }
timeout_s: 30
max_memory_mb: 512
sandbox_policy: strict | permissive | none
audit_policy: full | summary | none
owner: "author name"
tests: "tests/test_skill_name.py"
```

## Permission Matrix

### By Role

| Permission | User | Operator | Admin | Research |
|-----------|------|----------|-------|----------|
| Chat | ✅ | ✅ | ✅ | ✅ |
| Read tools (clock, weather) | ✅ | ✅ | ✅ | ✅ |
| File tools (workspace only) | ✅ | ✅ | ✅ | ✅ |
| File tools (outside workspace) | ❌ | ✅ | ✅ | Sandbox |
| Shell (sandboxed) | ❌ | ✅ | ✅ | Sandbox |
| Shell (unrestricted) | ❌ | ❌ | ✅ | ❌ |
| Browser | Limited | ✅ | ✅ | Sandbox |
| Network (external) | ❌ | ✅ | ✅ | Sandbox |
| Memory read | Own | All | All | Sandbox |
| Memory write | Own | All | All | Sandbox |
| Memory delete | ❌ | ❌ | ✅ | ❌ |
| Self-repair | ❌ | Approve | ✅ | Sandbox |
| Plugin install | ❌ | ❌ | ✅ | ❌ |
| Model change | ❌ | ✅ | ✅ | ✅ |
| Feature flags | ❌ | Limited | ✅ | Limited |
| Cloud fallback | ❌ | ✅ | ✅ | ❌ |

### By Risk Level

| Risk Level | Authorization | Sandbox | Audit | Example |
|-----------|---------------|---------|-------|---------|
| Low | Auto-approve | Optional | Summary | Clock, calculator |
| Medium | Will decision | Recommended | Full | File read, web search |
| High | Will + operator confirm | Required | Full | Shell exec, file write |
| Critical | Will + admin confirm | Required + isolated | Full | Self-modification, plugin install |

## Operator Controls

Tool access is configured by writing a prohibition, not by listing what is
allowed. A standing directive lives in
`data/governance/standing_directives.json`, and the authority gateway reads it
from disk on every consequential action:

```python
from core.governance.standing_directives import (
    add_directive, remove_directive, KIND_TOOL, KIND_PATH, SCOPE_ANY, SCOPE_WRITE,
)

add_directive(kind=KIND_TOOL, value="shell", reason="operator policy", scope=SCOPE_ANY)
add_directive(kind=KIND_PATH, value="~/Documents", reason="off limits", scope=SCOPE_WRITE)
```

`SCOPE_WRITE` refuses only the mutating use; `SCOPE_ANY` refuses reads too.
`remove_directive(directive_id)` withdraws one.

The store has no allowlist and no grant call, and that asymmetry is the design
rather than an omission. A directive that could permit an action would give one
successful prompt injection a permanent, audited-looking way through the gate.
A prohibition can only tighten it, so a hostile write costs availability and
nothing else.

If the file is present but unreadable, the system refuses everything that is
not read-only and records a degradation. It knows prohibitions were written and
cannot tell what they said.

Feature-level switches are separate: `AURA_FLAG_<NAME>` overrides any flag in
`_DEFAULT_FLAGS` (`core/governance/feature_flags.py`) — `workspace_jail_enabled`
is the path-traversal guard for file skills, `will_strict_enforcement` decides
whether the Will is binding or advisory.

## Production Mode Rules

In production mode (`AURA_MODE=production`):
- Unsigned or unmanifested skills do not load
- Self-modification tools are disabled
- Shell execution requires operator-level permissions
- Network tools require explicit configuration
- All tool output is sanitized before processing
- Tool execution timeout is strictly enforced

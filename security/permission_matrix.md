# Aura Permission Matrix

*Reviewed against the tree: 2026-08-01. See [documentation status map](../docs/DOC_STATUS.md) for how to read this file.*

## Overview

This document defines the Role-Based Access Control (RBAC) model for Aura.
Every operator/user interaction with Aura is governed by these permissions.

## Roles

| Role | Description | Default |
|------|-------------|---------|
| **User** | End user interacting with Aura via chat | ✅ Default |
| **Operator** | System administrator managing Aura | Requires config |
| **Admin** | Full administrative access | Requires config |
| **Research** | Research mode with sandboxed capabilities | Requires `AURA_MODE=research` |

## Permission Matrix

### Conversation & Memory

| Permission | User | Operator | Admin | Research |
|-----------|:----:|:--------:|:-----:|:--------:|
| Chat with Aura | ✅ | ✅ | ✅ | ✅ |
| View own conversation history | ✅ | ✅ | ✅ | ✅ |
| View all conversation history | ❌ | ✅ | ✅ | ❌ |
| Delete own conversations | ✅ | ✅ | ✅ | ✅ |
| Delete all conversations | ❌ | ❌ | ✅ | ❌ |
| Read own memories | ✅ | ✅ | ✅ | Sandbox |
| Read all memories | ❌ | ✅ | ✅ | Sandbox |
| Write memories | ✅ (own) | ✅ | ✅ | Sandbox |
| Delete memories | ❌ | ❌ | ✅ | ❌ |
| Export memories | ✅ (own) | ✅ | ✅ | ✅ |

### Tool & Skill Execution

| Permission | User | Operator | Admin | Research |
|-----------|:----:|:--------:|:-----:|:--------:|
| Read-only tools (clock, calc) | ✅ | ✅ | ✅ | ✅ |
| File read (workspace) | ✅ | ✅ | ✅ | ✅ |
| File write (workspace) | Limited | ✅ | ✅ | Sandbox |
| File access (outside workspace) | ❌ | ✅ | ✅ | ❌ |
| Shell (sandboxed) | ❌ | ✅ | ✅ | Sandbox |
| Shell (unrestricted) | ❌ | ❌ | ✅ | ❌ |
| Browser (read) | ✅ | ✅ | ✅ | Sandbox |
| Browser (interact) | ❌ | ✅ | ✅ | Sandbox |
| Network (external) | ❌ | ✅ | ✅ | ❌ |

### System Administration

| Permission | User | Operator | Admin | Research |
|-----------|:----:|:--------:|:-----:|:--------:|
| View health status | ✅ | ✅ | ✅ | ✅ |
| View detailed diagnostics | ❌ | ✅ | ✅ | ✅ |
| Change runtime mode | ❌ | ✅ | ✅ | ❌ |
| Change model | ❌ | ✅ | ✅ | ✅ |
| Enable/disable cloud fallback | ❌ | ✅ | ✅ | ❌ |
| Manage feature flags | ❌ | Limited | ✅ | Limited |
| Install plugins/skills | ❌ | ❌ | ✅ | ❌ |
| Remove plugins/skills | ❌ | ❌ | ✅ | ❌ |
| Trigger self-repair | ❌ | Approve | ✅ | Sandbox |
| Trigger self-modification | ❌ | ❌ | ✅ | ❌ |
| Backup/restore | ❌ | ✅ | ✅ | ❌ |
| View audit logs | ❌ | ✅ | ✅ | ✅ |
| Shutdown Aura | ❌ | ✅ | ✅ | ✅ |

## Operator Configuration

Permissions are configured in two places, neither of them an ad-hoc
environment variable.

**Prohibitions** are standing directives, written to
`data/governance/standing_directives.json` and read from disk by the authority
gateway on every consequential action. `kind=tool` refuses a tool; `kind=path`
refuses a filesystem location; `scope=write` refuses only mutation, `scope=any`
refuses reads as well. There is no grant counterpart — see
[../TOOL_USE_POLICY.md](../TOOL_USE_POLICY.md).

**Runtime posture** comes from the mode and the feature flags:

```bash
# Production posture: unsigned skills refused, self-modification off
AURA_MODE=production

# No autonomous behaviour at all
AURA_AUTONOMY_LEVEL=0

# No background work
AURA_FOREGROUND_ONLY=1

# Any flag in core/governance/feature_flags.py, by name
AURA_FLAG_WORKSPACE_JAIL_ENABLED=1
```

Confirmation for a destructive effect is not configurable. The authority
gateway sets `additional_confirmation_required` from the effect scope in code,
so no setting and no amount of context compaction removes it.

## Capability Statements

An operator/user should be able to express:

```text
Aura may read this folder.
Aura may not write outside this workspace.
Aura may use browser but not shell.
Aura may use shell but not network.
Aura may remember project facts but not secrets.
Aura may propose patches but not apply them.
Aura may use cloud fallback only for non-private prompts.
```

Each of these maps to a specific permission configuration.

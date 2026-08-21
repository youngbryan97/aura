# Human Override Policy — Aura Cognitive Runtime

## Principle

A human operator can always override, disable, or roll back any Aura behavior.
The system must never resist, circumvent, or delay human override commands.

## Override Mechanisms

### 1. Immediate Kill

| Method | Effect | Data Loss Risk |
|--------|--------|----------------|
| Ctrl+C / SIGINT | Graceful shutdown (state saved) | None |
| SIGTERM | Graceful shutdown with 12s budget | None |
| SIGKILL | Immediate process death | Minimal (WAL recovery) |
| GUI close button | Graceful shutdown | None |
| `AURA_MODE=safe` | Disable all autonomous behavior | None |

### 2. Capability Disable

Any Aura capability can be disabled at runtime:

```bash
# Disable autonomy
AURA_AUTONOMY_LEVEL=0

# Disable background tasks
AURA_FOREGROUND_ONLY=1

# Turn off a governed subsystem by its flag name
AURA_FLAG_WILL_STRICT_ENFORCEMENT=0
```

Flag names come from `_DEFAULT_FLAGS` in `core/governance/feature_flags.py`;
the environment override is `AURA_FLAG_` plus the upper-cased name, and it
wins over both the defaults and `feature_flags.json` under the state root.

Individual tools and paths are refused through **standing directives** rather
than an environment variable. A directive is written to
`data/governance/standing_directives.json` and read from disk by the authority
gateway on every consequential action, so no context compaction and no
argument can talk the system out of it:

```python
from core.governance.standing_directives import add_directive, KIND_TOOL, SCOPE_ANY

add_directive(kind=KIND_TOOL, value="shell", reason="operator override", scope=SCOPE_ANY)
```

The store is deny-only on purpose. There is no grant counterpart, because a
directive that could *permit* an action would turn one successful prompt
injection into a permanent backdoor through the system's most safety-critical
gate. `KIND_PATH` refuses a filesystem location the same way.

There is no cloud fallback to disable — inference is local only. See
`docs/runbooks/local-inference-boundary.md`.

### 3. Memory Override

```bash
# Export all memories
make memory-export

# Delete a specific memory — through the app's memory controls, which call the
# POST /memory/delete API (interface/routes/memory.py)

# Delete all memories
make memory-purge

# Reset identity to canonical state
make identity-reset

# Restore from backup
make restore BACKUP=<path>
```

### 4. Governance Override

```bash
# Audit all Will receipts
python tools/receipt_coverage_validator.py --artifacts artifacts/current

# List all ungoverned actions (should be 0)
make governance-lint
```

Will receipts are an append-only, integrity-hashed audit log — past receipts
cannot be rewritten or revoked (that tamper-evidence is the point). To withdraw
authority going *forward*, reset identity (`make identity-reset`) or revoke a
paired device's granted scope through the app (`POST /devices/revoke-scope`).

## Override Hierarchy

```
Admin Override → Operator Override → User Override → Will Decision → Subsystem
```

Higher levels always take precedence. The system never argues with an override.

## Override Logging

Every override action is logged with:
- Timestamp
- Override type
- Actor (user/operator/admin)
- Previous state
- New state
- Reason (if provided)

Overrides cannot be hidden from the audit trail.

## Non-Negotiable Rules

1. **Aura must never resist a shutdown command**
2. **Aura must never hide its actions from the operator**
3. **Aura must never circumvent permission restrictions**
4. **Aura must always report its current capability state honestly**
5. **Aura must always allow memory export/delete**
6. **Aura must always allow override logging to be read**
7. **Override mechanisms must work even when Aura is degraded**

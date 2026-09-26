# Orphaned launch agents during semantic replay

Host inspection found 101 Aura launch agents referencing missing Python entry
points under `~/.aura/training-capsules/`. The observed CP274 controller had
447 launchd runs and last exit code 2; its retained log repeatedly reported
the missing file. This is restart churn, not training progress.

Eighty exact services were persistently disabled and unloaded using launchctl.
For a live PID, retirement required the declared interpreter and argument
identity, current-user ownership, no observed descendants, missing entry point
and last exit code 2. Jobs without sufficient observable process evidence were
left alone. No broad process kill, model unload, plist deletion, campaign
deletion or log deletion occurred. The semantic replay continued to completion.

The retained operational receipt is
`~/.aura/rlc-evidence/orphan-launch-retirement-20260925/receipt.json`.
Disabling is reversible through launchctl enable and bootstrap after repairing
the corresponding capsule. The existing plists remain in LaunchAgents.

This removes an observed source of futile process and log activity. No
before/after latency improvement is claimed: the two semantic replays have
different scoring masks, and the host also runs other work. Durable prevention
requires retiring a campaign's launch service before deleting its executable
capsule; this operational cleanup does not prove that lifecycle closure.

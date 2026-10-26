# Human override policy — Aura

Status: Policy · Mechanisms and limitations reconciled 25 October 2026

The operator must be able to stop effects, withdraw permissions and retain
control of their data. This is a required contract. It is not a guarantee that
every override works instantly during every host or runtime failure.

## Stop and containment

A graceful stop asks the supported lifecycle to drain work and save state.
Its success must be checked. SIGKILL ends the selected process without that
cleanup and can lose in-flight data; a write-ahead log does not make it
risk-free. Closing a GUI window may differ from stopping its backend: confirm
the actual runtime process rather than inferring termination from the window.

For agent work on Bryan’s shared host, follow [AGENTS](AGENTS.md): the live
instance cannot be restarted or killed without current-task authorisation.
Identify training/campaign ownership and use one controlled lifecycle change.

## Launch settings and live settings differ

`AURA_MODE=safe`, `AURA_AUTONOMY_LEVEL=0` and `AURA_FOREGROUND_ONLY=1`
are process settings. Assigning or exporting them in another shell does not
change an already-running process. A supported live settings endpoint must
apply the particular setting; otherwise it takes effect on a later launch.
Do not weaken Will enforcement as a way to contain an unsafe capability.

Standing directives in [the governance module](core/governance/standing_directives.py)
can deny tool or path use on routes that read them. Use the supported owner
control and inspect the resulting directive/receipt. Their existence is not
proof that every possible effect path reads the file, and executing a Python
snippet in a separate process does not automatically exercise the resident
application’s container or settings.

[Paired-device routes](interface/routes/devices.py) provide explicit device
and scope revocation. Device credentials, model-lane capability leases and
identity state are different authorities: resetting one is not a general
revocation of all the others.

## Memory and identity controls

Use the app’s implemented memory controls and the actual response from
[the memory routes](interface/routes/memory.py). Export/delete scope is
store-specific; consult [the data card](DATA_CARD.md). Verify the selected
records were exported or removed rather than trusting a status message.

**Known command limitation:** the current Makefile’s `memory-export` target
prints availability metadata, `data-export` creates a directory, and
`memory-purge`, `data-purge` and `identity-reset` print completion messages
without implementing the advertised operation. These targets must not be
cited as completed exports, deletion or identity reset. This documentation
correction does not implement them.

For backup and recovery, use [the recovery runbook](docs/runbooks/disaster-recovery.md)
and verify the actual archive, included stores and required keys before
restoring during an authorised window. Restoring state does not undo an
external action that has already happened.

## Auditability

Governed decisions are required to have authorisation and outcome records.
Hash links can expose covered modifications to retained records; they do not
make a disk immutable or guarantee all effects were captured. Intentional
privacy deletion and retention have their own record-handling rules.

`make governance-lint` checks named source patterns, not every historical live
action. `tools/receipt_coverage_validator.py` checks supplied artifacts; missing
or wrong-process artifacts cannot establish receipt coverage for the resident
app. Keep the source revision, process and observation window with each result.

## Required behavior and acceptance

Shutdown, revocation and memory controls must remain usable during degradation,
with explicit failures when an operation cannot complete. Test each control’s
normal, denied, unavailable and interrupted cases. Verify both that the
forbidden next effect is refused and that permitted recovery remains possible.
A policy statement or a printed success message is not acceptance evidence.

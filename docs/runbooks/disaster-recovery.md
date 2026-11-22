# Runbook: disaster recovery

<!-- reader-navigation:start -->
> **In plain English:** Reliability work detects failed or stalled components, preserves useful state and recovers through a controlled lifecycle.
>
> **Why it matters:** A user should not lose work because a slow component was mistaken for a dead one, or because a restart created another large process.
>
> **In everyday use:** If generation is progressing slowly, the system needs a different response from the one appropriate to a worker that has stopped making progress.
>
> [Example and metaphor](../PLAIN_ENGLISH_GUIDE.md#reliability) · [Variables and practical meanings](../VARIABLES_EXPLAINED.md#reliability) · [Demos and related work](../DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](../DOCUMENTATION_INDEX.md)
> [Glossary](../GLOSSARY.md) · [Formulas with worked examples](../FORMULAS.md)
<!-- reader-navigation:end -->

Status: Runbook · Commands and recovery limits reconciled 25 October 2026

Recover a known state without destroying the evidence needed to understand
what failed. A command returning success does not establish that every store
or external effect was restored.

## Identify the owner before changing anything

Record the installed source revision, process identity, active model artifact,
last completed work and relevant logs. Check whether a campaign or another
operator owns the resident model. `make release-status` reports the release
train’s recorded state; it is not a complete live-process health check.

Do not launch `make run` while another Aura instance owns the runtime. Do not
restart, restore over live stores or kill a process from an unverified PID file.
Use an authorised maintenance window, graceful shutdown and proof that the
old owner has exited before starting one replacement through the supported
launcher. See [the operational guide](../../CLAUDE.md).

## Preserve the affected stores

Keep the database, its `-wal` and `-shm` files, and the failure logs together.
Work on a scratch copy when possible. SQLite WAL recovery can replay valid
committed transactions; it cannot guarantee recovery from arbitrary corruption.
`PRAGMA wal_checkpoint(TRUNCATE)` is a checkpoint operation, not a repair
that reconstructs missing data. Do not use it as a substitute for a backup.

The real database path comes from the installed configuration and store owner;
`data/aura_memory.db` is not a universal path for all memory. Open a known
scratch copy explicitly when testing integrity. Connecting to a guessed path
can create an empty database and make a meaningless integrity check pass.

## Choose the backup family and verify its coverage

There are two different backup paths in source:

| Entry point | Owner | Scope to inspect |
|---|---|---|
| `make backup`, `make backup-verify`, `make restore BACKUP=<archive>` | [tools/state_backup.py](../../tools/state_backup.py) | Repository roots plus the resolved live data root, with exclusions, per-file bounds and a manifest |
| `aura backup`, `aura restore` | [core/runtime/backup_restore.py](../../core/runtime/backup_restore.py) | The declared directories under its resolved Aura home |

Do not assume their archives, verification rules or included stores are
interchangeable. Read the archive’s manifest, omitted files, creation source
and verification result. The state-backup tool skips oversized and excluded
files; a successful archive is not automatically a complete copy of everything
at `~/.aura/`.

During an authorised restore window, select an explicit verified archive
rather than blindly choosing the newest file. Confirm compatible source,
schema, keys and model custody; restore into a scratch location first when
supported. A copy without the required encryption key cannot restore its
encrypted contents. Keep the pre-restore state until acceptance is complete.

## Handle the diagnosed failure

- **Dirty shutdown:** retain flight/crash records and check the actual stores.
  WAL recovery and a clean next boot are observations to verify, not promises.
- **Corrupt memory:** preserve the original and test a verified backup or
  store-specific recovery on a copy. Do not truncate WAL or delete the vault
  to silence the error.
- **Incomplete model artifact:** inspect the artifact manifest and missing
  files before a separately budgeted download. A structural fingerprint is
  not a full weight-file checksum. Do not load another model to test a path.
- **Failed repair:** retain the proposed patch, quarantine result and lineage.
  Disable its promotion through the owning control. Deleting the patch
  registry destroys evidence and does not establish a repaired source tree.
- **Full disk:** inspect actual disk consumers and retention rules. Preserve
  recent incident logs and verified backups. `make log-purge` only targets its
  declared repository `logs/` patterns; it is not the complete live-log policy.
- **Source rollback:** use a known source/state pair and the release lifecycle.
  A bare checkout of an older commit does not roll back schemas, installed
  packages, model pointers or external effects.

## Verify the replacement

Record what was checked and what remained unavailable:

- Exactly one intended runtime loaded the expected source/configuration/model.
- The affected stores open and return known retained content.
- Required services become ready and an ordinary request finishes correctly.
- New receipts name the replacement process and expected action path.
- Resource use stays inside the declared observation budget.
- No new error or missing-data condition is being hidden as healthy.

`make doctor` checks its environment. A CLI diagnostic bundle can contain
collector-local state rather than the running app’s service container; bind
it to the right process before citing it. A successful smoke test supplies
offline contract evidence, not proof of a complete live recovery.

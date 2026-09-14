# Q05 persistence, migration, corruption recovery, backups, rollback — 2026-09-13

## Backups

Three ways there was no backup while a green target said otherwise, all
found on 2026-09-13 and all fixed in 1e525d284:

1. `make backup` raised `ModuleNotFoundError: No module named 'core'` on
   every run since 2026-08-05, when `tools/state_backup.py` gained an import
   from `core` and no path to find it on. The newest archive in the ring was
   from July 12.
2. The archive it would have made held the repository's `data/` — what
   campaigns write with the repository as their working directory — and not
   `~/.aura/data`, where the desktop keeps its conversations, memory and
   state. No backup had ever contained the live stores.
3. `make restore-test` printed "Simulating state corruption..." and
   simulated nothing: it restored an archive over a tree that was already
   correct and said the drill passed.

Now: the tool stages the configured live data directory under `aura_data/`
beside the repository roots, snapshots every SQLite store through the backup
API, bounds each file at 2GB and names what it skipped in the manifest (the
46GB orphan store, `aura_data/state/aura_state.db`, is the one skip), writes a
sha256 manifest line, and prunes its own ring. `verify` re-hashes and
`quick_check`s every store; one the runtime itself quarantined as damaged is
carried as-is and reported, not counted.

Measured on the live ring, 2026-09-13:

```
backup aura_state_20260913_105005-64335.tar.gz: 41540 files, 74 consistent DB
snapshots, 3016MB archive; live data from /Users/bryan/.aura/data
skipped aura_data/state/aura_state.db (49.4GB, over the 2GB per-file bound)
verify OK: hash matches manifest, 73 SQLite stores pass quick_check
```

## Restore, and a drill that can fail

`make restore` was a bare `tar xzf` in the repository, which would have put a
copy of the live data directory under the repository and left the runtime
reading what it had. `tools/state_backup.py restore` puts `aura_data/` back
where the runtime reads it and the repository roots under the repository,
and refuses under a live instance on :8000 without `--force`.

`tools/state_backup.py drill` extracts the archive into a scratch root,
damages two stores and one file the way disks and crashes do (a page
overwritten in the middle; a truncation), requires the damage to be visible
(hashes change, `quick_check` fails), restores over the scratch root the way
`make restore` does, and requires every file to hash back and every store to
pass. `make restore-test` is `create` then `drill`.

```
drill on aura_state_20260913_105005-64335.tar.gz: 41540 files, 73 SQLite
stores; damaged 3 (live_data/adaptive_mood.sqlite3: sqlite page overwritten,
live_data/audit.db: sqlite page overwritten, live_data/.continuity_hmac.key:
truncated to half); 2 store(s) failed quick_check while damaged; restored:
41540/41540 files hash back, 73/73 stores pass quick_check
restore drill passed: damage was visible and the restore repaired it
```

Held by `tests/test_a_restore_drill_that_can_fail.py` (5) and
`tests/test_state_backup.py` (11).

## The runtime's own backup

`core/ops/backup.py` registered a daily backup with `last_run=now` at every
boot, so the first backup was a day after boot; the desktop restarted before
that every time. Seventy-seven boots in `desktop-launch.log`, not one
"Creating backup" line. It also zipped the data directory raw — hot WAL stores
copied byte for byte. It now runs the verified tool through the subprocess
gateway and is due by the age of the newest archive on disk, after a
fifteen-minute boot grace. `tests/test_a_backup_is_due_by_the_ring_not_the_boot.py` (3).

## Rollback

`tools/release_train.py rollback` resets to the last recorded update point
and refuses a target that is not an ancestor of `origin/main`. It has no
recorded point on this checkout, because this checkout is updated by pushes,
not by `release_train update`; `--to COMMIT` is the path here and it says so.
`update` set aside uncommitted work with `git stash push` and, on success,
left it there labelled — uncommitted work made to disappear; on failure it
`stash pop`ped, which on a checkout other sessions push to takes whatever is on
top. The stash is found by its label, applied by its SHA, dropped by re-finding
the label, and given back after a successful update. `tests/test_release_train.py` (10).

## Corruption recovery and migration

A store that will not parse is moved aside, never overwritten:
`core/persistence/a_versioned_store.py` (major version unknown → refuse;
minor ahead → readable; `tests/test_a_versioned_store.py`, 30), and the
long-term memory engine quarantines a corrupt store beside itself
(`tests/test_long_term_memory_engine_hardening.py`). The live data directory
carries one such quarantine from May 2026 —
`memory/quarantine/cognitive_ledger.db.corrupt.1779836906` — which is how the
drill found it.

## Left as it is

The 46GB orphan store is the owner's to remove; it is in the Q02 record.

# Aura data card

Aura keeps application state on the Mac, usually under `~/.aura/`.
Conversation records, semantic memories, goals, and receipts have separate
stores. Close the app and verify the state location before handling its
files. [State paths](core/runtime/state_ownership.py) · [Operator guide](docs/OPERATOR_GUIDE.md)

## Export or delete a memory

Open **Backup & Export** in the memory panel to download a JSON file of
records exposed by the semantic memory service. To remove a supported
record, select **Delete** on that record and check the result. Conversation
transcripts, audit receipts, backups, and exports have their own lifetimes.
[Memory panel](interface/static/memory_panel.html) · [API handlers](interface/routes/memory.py)

The `make memory-export`, `make memory-purge`, `make data-export`, and
`make data-purge` targets are placeholders. The export targets print status
or create an output directory. The purge targets wait and print messages
without deleting records. This corrects the export and deletion entry in
the earlier closeout. [Makefile](Makefile) · [Earlier closeout](docs/evidence/CLOSEOUT.md)

## Retention and backups

The experience stream prunes private frames after 24 hours and standard
frames after 30 days when its retention job runs. Those defaults apply
to that stream. Audit receipts, exports, and archived copies are governed
separately. [Experience stream](core/consciousness/continuous_experience.py) · [Retention policy](docs/DATA_RETENTION_DELETION_POLICY.md)

`make backup` archives the configured state trees and available live data.
It keeps up to seven archives by default. Check the manifest for skipped files
and use `make backup-verify` to check the archive before relying on it.
[Backup implementation](tools/state_backup.py)

## Network access

Model inference runs locally. Authorized browsing and external services
can send data over the network. If configured, the error-alert webhook can
send log messages to its destination. The local API has its own access
rules. [Model request contract](core/brain/request_contract.py) ·
[Egress checks](core/security/egress_privacy.py) ·
[Logging](core/utils/aura_logging.py) · [Local API](docs/LOCAL_API_TRUST_BOUNDARY.md)

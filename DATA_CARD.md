# Aura Data Card

<!-- reader-navigation:start -->
> **In plain English:** Memory keeps records beyond the immediate request and selects relevant records for the current turn.
>
> **Why it matters:** Useful continuity depends on recalling the right information, retaining its source and correcting or deleting it when needed.
>
> **In everyday use:** If you correct a fact about a project, later answers should use the correction rather than retrieving the older claim as equally reliable.
>
> [Example and metaphor](docs/PLAIN_ENGLISH_GUIDE.md#memory) · [Variables and practical meanings](docs/VARIABLES_EXPLAINED.md#memory) · [Demos and related work](docs/DEMOS_AND_RELATED_WORK.md#memory) · [Document map](docs/DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

This document details every piece of data Aura stores, where that data is kept on your computer, and how you can export or delete it.

Because Aura runs entirely on your local machine, your data is stored in files on your own hard drive rather than on remote cloud servers. This eliminates external tracking and third-party data sharing. However, it means your data security depends on your local computer security: enabling full-disk encryption, maintaining backups, and managing who has access to your machine.

## Memory Systems

| Store | Type | Persistence | Encryption | Privacy |
|-------|------|-------------|------------|---------|
| Working Memory | In-process | Session only | N/A | Cleared on shutdown |
| Conversation History | SQLite | Durable | Available (vault) | User-deletable |
| Semantic Memory (RAG) | Vector DB | Durable | At-rest available | User-deletable |
| ColdStore (Long-term) | SQLite | Durable | Available (vault) | User-deletable |
| State Snapshots | JSON/SQLite | Durable | Available (vault) | User-exportable |
| Will Receipt Log | Append-only log | Durable | Integrity-hashed | Audit-readable |

## Data Retention

| Data Type | Default Retention | User Control |
|-----------|-------------------|--------------|
| Conversation history | Indefinite | Delete any/all |
| Semantic memories | Indefinite | Delete any/all |
| Long-term memories | Indefinite | Delete any/all |
| Will receipts | Indefinite (audit) | Export only |
| Logs | 30 days (rotation) | Export/delete |
| Metrics | 7 days | Export/delete |
| Backups | 3 most recent | Delete any/all |

## Data Flow

```
User Input → Sanitizer → Working Memory → Model Context
                                              ↓
                                         Model Output
                                              ↓
                                    Integrity Check → User
                                              ↓
                               Will Decision → Memory Write
                                              ↓
                                    State Snapshot → Backup
```

## Privacy Controls

| Control | Mechanism |
|---------|-----------|
| No remote inference | All AI model processing runs locally; `allow_cloud_fallback` is pinned to `False` in `core/brain/request_contract.py` |
| Outbound body inspection | `core/security/egress_privacy.py` inspects any outgoing network request before it leaves your machine |
| Memory export | `make memory-export` |
| Memory delete | `make memory-purge` (delete all memories) / app memory panel `POST /api/memory/delete` (delete specific memories) |
| Log purge | `make log-purge` |
| Full data export | `make data-export` (complete export of all your stored data) |
| Full data delete | `make data-purge` (deletes all application data) |

## No External Data Collection

In its default configuration, Aura:
- Sends no data to external services or cloud providers.
- Has no telemetry, tracking, or "phone-home" behavior.
- Does not collect product analytics or user telemetry.
- Does not send crash reports to external servers.
- Stores all files, logs, and databases locally on your computer.

If cloud fallback is explicitly enabled by the operator, Aura sends only prompt sections that have been scanned and classified as safe.

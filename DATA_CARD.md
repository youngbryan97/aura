# Aura Data Card

<!-- reader-navigation:start -->
> **In plain English:** Memory keeps records beyond the immediate request and selects relevant records for the current turn.
>
> **Why it matters:** Useful continuity depends on recalling the right information, retaining its source and correcting or deleting it when needed.
>
> **In everyday use:** If you correct a fact about a project, later answers should use the correction rather than retrieving the older claim as equally reliable.
>
> [Example and metaphor](docs/PLAIN_ENGLISH_GUIDE.md#memory) · [Variables and practical meanings](docs/VARIABLES_EXPLAINED.md#memory) · [Demos and related work](docs/DEMOS_AND_RELATED_WORK.md#memory) · [Document map](docs/DOCUMENTATION_INDEX.md)
> [Glossary](docs/GLOSSARY.md) · [Formulas with worked examples](docs/FORMULAS.md)
<!-- reader-navigation:end -->

This card maps the declared application stores and their controls. It is not
an exhaustive inventory of every experiment artifact, model cache, external
service record or operator-created backup.

Application memory is stored locally. Local storage does not by itself
eliminate off-device exposure: authorised browser and external-service tools
make network requests, and those requests need their own data-handling
controls. Protect the local files, credentials, backups and exported copies
according to their actual contents and use.

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
| Outbound body inspection | `core/security/egress_privacy.py` inspects requests routed through its controls; this is not an operating-system boundary covering every possible call |
| Memory export | `make memory-export` |
| Memory delete | `make memory-purge` (delete all memories) / app memory panel `POST /api/memory/delete` (delete specific memories) |
| Log purge | `make log-purge` |
| Full data export | `make data-export` (inspect which configured stores the export includes) |
| Full data delete | `make data-purge` (covered application stores; separate artifacts and backups require their own handling) |

## No External Data Collection

In its default configuration, Aura:
- Uses local model inference. Authorised browser and external-service tools
  can still send their required requests off-device.
- Has no telemetry, tracking, or "phone-home" behavior.
- Does not collect product analytics or user telemetry.
- Does not send crash reports to external servers.
- Stores all files, logs, and databases locally on your computer.

The retired cloud-model fallback is not a supported inference mode. Legacy
`api_fast`/`api_deep` labels resolve to local lanes; a remote-only model request
is refused. This does not make browser/tool requests local. See
[model routing](docs/MODEL_ROSTER.md) and [the API trust boundary](docs/LOCAL_API_TRUST_BOUNDARY.md).

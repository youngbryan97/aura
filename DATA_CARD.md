# Aura Data Card

<!-- reader-navigation:start -->
> Aura keeps records of events, facts, preferences, and procedures. It looks for records relevant to the question or task in front of it.
>
> A memory is useful when you can find its source, correct it, or remove it. If you correct the date of a project, a later reply should use the correction and remember where it came from.
>
> [Plain-language guide](docs/PLAIN_ENGLISH_GUIDE.md#memory) · [Terms and examples](docs/VARIABLES_EXPLAINED.md#memory) · [Evidence and related work](docs/DEMOS_AND_RELATED_WORK.md#memory)
> [All docs](docs/DOCUMENTATION_INDEX.md) · [Glossary](docs/GLOSSARY.md) · [Worked formulas](docs/FORMULAS.md)
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

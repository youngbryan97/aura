# Aura Platform Posture

<!-- reader-navigation:start -->
> Aura runs on an Apple Silicon Mac. The desktop app opens a conversation, and local software handles speech, memory, tools, and background work.
>
> Start with the parts you need and check what the machine can run. The setup guides give the hardware requirements and link to measured capabilities. To work on a document, Aura needs a model that has finished loading, permission to read the file, and enough memory to handle the request.
>
> [Plain-language guide](PLAIN_ENGLISH_GUIDE.md#getting-started) · [Terms and examples](VARIABLES_EXPLAINED.md#getting-started) · [Evidence and related work](DEMOS_AND_RELATED_WORK.md#runtime)
> [All docs](DOCUMENTATION_INDEX.md) · [Glossary](GLOSSARY.md) · [Worked formulas](FORMULAS.md)
<!-- reader-navigation:end -->

*Reviewed against the tree: 2026-08-01. See [documentation status map](DOC_STATUS.md) for how to read this file.*

Aura is designed for one operator per installation. It uses API tokens,
keeps each installation in one tenant boundary, relies on the operator to
make backups, and checks plugins against a list of permitted hashes. The
table explains how the source enforces those choices, including the scope of
role-based access control (RBAC) and single sign-on (SSO).

| Decision area      | Posture today                       | Enforced by                                                                                       |
|--------------------|-------------------------------------|---------------------------------------------------------------------------------------------------|
| RBAC               | **Single user (no RBAC)**           | one operator per install; capability tokens in `core/security/` are the only authorisation gate    |
| SSO                | **API tokens only**                 | the existing capability-token system; no OIDC/SAML wiring; capability leases are process-scoped; paired-device credentials have separate persistence and revocation       |
| Tenant isolation   | **Single-tenant per install**       | `core/runtime/tenant_boundary.py` (stamp + `assert_owned()`); `AURA_TENANT_ID` + `AURA_HOME` env vars |
| Disaster recovery  | **Manual backups + tested drill**   | `aura backup` / `aura restore` + `core/runtime/restore_drill.py` round-trip with fingerprint diff |
| Plugin signing     | **SHA-256 hash allowlist**          | `core/security/plugin_allowlist.py` (`is_allowed()` raises on unlisted/revoked/drift)             |

The five decisions are **deliberate**.  They match Aura's local-first,
sovereign-one-operator posture.  Anything more elaborate (multi-org
RBAC, OIDC, multi-tenant, off-host replication, sigstore-signed
plugins) is a future migration, not the current model.

## What each decision means in practice

### RBAC: Single user
There are no roles.  The operator running the process is the only
principal.  Tools that an LLM tries to call go through the existing
`UnifiedWill` (governance), not through a per-role permission matrix.
A future migration to "admin / operator / viewer" would require a
`Role` enum, a per-tool policy file, and a session/user identity
binding — none of those exist today.

### SSO: API tokens only
Separate API authentication from execution authority. Internal capability
leases in `core/capability_engine.py` bind their declared process/context.
[Paired-device credentials](../interface/routes/devices.py) have separate
storage, scopes and revocation; a runtime restart is not a general claim
that every API credential has been revoked.  There is
no OIDC discovery, no SAML, no IdP integration.  Operators wanting
hosted-deployment-style SSO should treat that as a separate platform
project.

### Tenant isolation: Single-tenant per install
One Aura install owns one tenant's data.  The boundary check is
`TenantBoundary.assert_owned()`: every data-dir touch verifies the
on-disk `tenant.json` matches `AURA_TENANT_ID`.  A foreign tenant's
data dir refuses to mount.  Two installs on the same machine must
set distinct `AURA_TENANT_ID` *and* `AURA_HOME`.

### DR: Manual backups + tested restore drill
`aura backup` writes a tar.gz of `state/`, `memory/`, `receipts/`,
`workflows/`, `data/`.  `aura restore` swaps the live tree from a
snapshot.  `core/runtime/restore_drill.perform_drill()` runs the
backup → wipe → restore → fingerprint-diff cycle as a self-test;
its CI job catches regressions in either path.

The platform does **not** ship: scheduled backups, off-host
replication, point-in-time recovery, RPO/RTO numbers.  Add those
when the deployment surface justifies them.

### Plugin signing: SHA-256 hash allowlist
Every plugin file's SHA-256 lives in `~/.aura/plugins/allowlist.json`,
along with `approved_by` / `approved_at` / `reason`.  Loading a
plugin computes its hash and matches against the allowlist.  Edits
produce a fresh hash that isn't listed; loading is refused until the
operator re-approves.  Revoked entries refuse with a distinct
`hash_revoked` reason so operators can tell forgotten approvals from
edited files.

The trust decision is the operator’s recorded approval of a file digest.
A SHA-256 allowlist is not a publisher’s digital signature and does not
by itself authenticate who wrote the plugin. No external CA is required.  Future migrations to sigstore/cosign
would replace `compute_sha256` with cryptographic verification but
keep the same `is_allowed()` interface.

## When to revisit

* **Multi-user requested** → add roles + per-role permissions; revisit
  RBAC + SSO together.
* **Hosted deployment** → tenant isolation must move from "stamp
  check" to namespaced storage; SSO becomes mandatory.
* **Compliance audit** → DR posture needs RPO/RTO numbers and an
  off-host replication story.
* **Untrusted plugin authors** → plugin signing migrates from hash
  allowlist to sigstore-signed-by-trusted-publisher.

Each migration is a real platform project, not a config flip.  The
current postures are listed here so the surface area to reason about
stays small until one of those triggers fires.

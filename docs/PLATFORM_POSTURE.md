# Aura Platform Posture

This document declares the deliberate platform decisions Aura runs
under today.  Every entry below is a load-bearing choice — operators
can change them, but they should know what they are choosing.

| Decision area      | Posture today                       | Enforced by                                                                                       |
|--------------------|-------------------------------------|---------------------------------------------------------------------------------------------------|
| RBAC               | **Single user (no RBAC)**           | one operator per install; capability tokens in `core/security/` are the only authorisation gate    |
| SSO                | **API tokens only**                 | the existing capability-token system; no OIDC/SAML wiring; tokens revoke on process restart       |
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
Authentication is bearer-token style.  Tokens are bound to a process
+ thread (see `capability_engine`) and revoke on restart.  There is
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

The trust root is the operator's signature on each file.  No
external CA, works offline.  Future migrations to sigstore/cosign
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

# Security Policy — Aura Cognitive Runtime

## Supported Versions

Aura is calendar-versioned; the authoritative string is `version` in
`pyproject.toml`. There is no 1.0 release line, and an earlier revision of
this file implying otherwise was wrong.

| Version | Supported |
|---------|-----------|
| Current `main` | ✅ Active — this is the only supported line |
| Anything older | ❌ Research only, no security patches |

Practically: this is single-owner research software with one deployment.
"Supported" means the current tip gets fixes. Nothing is backported.

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Email: security@aura-project.dev

(There is no published PGP key at present. Earlier revisions of this file
pointed at `security/pgp-public-key.asc`, which has never existed in the
tree — do not wait on it to report something.)

Response SLA:
- Acknowledgement: 48 hours
- Triage: 5 business days
- Fix (critical): 7 calendar days
- Fix (high): 14 calendar days
- Fix (medium/low): next release cycle

## Security Architecture

Aura runs locally, so the threat model doesn't look like cloud SaaS. The
primary trust boundary is the machine itself.

That's the easy part. The harder part is that the agent surfaces — tool use,
memory, model routing, autonomous action — are attack surface that most
local software doesn't have. A local text editor can't be talked into
running something. This can.

### Trust Boundaries

```
┌─────────────────────────────────────────────────┐
│  Local Machine (Operator Trust Boundary)        │
│  ┌───────────────────────────────────────────┐  │
│  │  Aura Runtime Process                     │  │
│  │  ┌─────────────┐  ┌──────────────────┐   │  │
│  │  │ User Input  │  │ Tool/Skill Exec  │   │  │
│  │  │ (untrusted) │  │ (sandboxed)      │   │  │
│  │  └──────┬──────┘  └────────┬─────────┘   │  │
│  │         │                  │              │  │
│  │  ┌──────▼──────────────────▼──────────┐  │  │
│  │  │  Unified Will / AuthorityGateway   │  │  │
│  │  │  (all consequential actions gated) │  │  │
│  │  └──────┬─────────────────────────────┘  │  │
│  │         │                                │  │
│  │  ┌──────▼──────┐  ┌──────────────────┐  │  │
│  │  │ Memory/State│  │ Model Inference  │  │  │
│  │  │ (encrypted) │  │ (local/cloud)    │  │  │
│  │  └─────────────┘  └──────────────────┘  │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  ┌───────────────┐  ┌───────────────────────┐  │
│  │ Filesystem    │  │ Network (optional)    │  │
│  │ (workspace)   │  │ (cloud fallback/API)  │  │
│  └───────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Executing model-written code

Code the model writes is treated as untrusted input, not as trusted program
text. `core/sandbox/untrusted_python.py` runs it behind a kernel boundary —
`sandbox-exec` (Seatbelt) on macOS, `bwrap` on Linux — deny-by-default, with
the interpreter's own read paths and a single scratch directory re-allowed
and network denied outright.

Two defences this deliberately replaced, because both are broken in the same
way:

- **AST screening is a denylist.** `__import__` reached through
  `().__class__.__mro__[1].__subclasses__()` never appears in the import
  table, and a screen that reads source text cannot see what
  `getattr(mod, name)` resolves to at runtime.
- **`python -I` is not isolation.** It ignores `PYTHON*` environment
  variables and drops the script directory from `sys.path`. That is import
  hygiene. The child keeps the parent's filesystem, network, process, and
  signal access in full — including the user's home directory, the live
  runtime's sockets, and the machine's keychain.

The load-bearing property is the refusal: **when no boundary is available,
the code does not run.** `AURA_SANDBOX_ALLOW_UNCONFINED=1` exists for
platforms with no boundary and is deliberately awkward — the outcome carries
`boundary="none"` and `sandboxed=False` permanently, so a caller that records
results cannot later claim they were confined. Live escape attempts against
this boundary are covered by tests.

### Physical actuation

Reaching a physical device is a governed path, not a direct call. Registered
hardware dispatches through `HardwareManager` and
`BaseHardwareDevice.safe_execute`; `core/reality_reach/` proves reachability
against declared channels before execution and keeps dispatch, execution, and
`EFFECT_VERIFIED` as separate `ActuationState` values so a successful send is
never recorded as a verified effect. See
[docs/REALITY_REACH.md](docs/REALITY_REACH.md) for the invariants and the
open items.

### Security Controls Summary

| Control | Status | Implementation |
|---------|--------|----------------|
| Action governance (Will/Authority) | ✅ Enforced | `core/will.py`, `core/governance/will_gate.py` |
| Tool sandboxing | ✅ Enforced | `security/code_sandbox.py`, `security/sandbox.py` |
| Input sanitization | ✅ Enforced | `security/sanitizer.py` |
| Memory encryption at rest | ✅ Available | `core/state/vault.py` |
| Secret scanning (CI) | ✅ CI gate | `.github/workflows/enterprise-gate.yml` |
| Dependency vulnerability scanning | ✅ CI gate | `pip-audit` in CI |
| SBOM generation | ✅ Available | `tools/build_provenance.py` |
| Governance bypass detection | ✅ Enforced | `tools/lint_governance.py` |
| Prompt injection defenses | ✅ Multi-layer | Input sanitizer + integrity checks |
| Outbound egress privacy | ✅ Enforced | `core/security/egress_privacy.py`, `core/runtime/network_gateway.py` |

### Secure Defaults

Aura ships with these defaults in production mode (`AURA_MODE=production`):

- All tool/skill execution is sandboxed
- Self-modification is disabled
- Cloud fallback requires explicit opt-in
- Memory writes require Will receipts
- Unsigned/unmanifested skills do not load
- Broad filesystem access is denied by default
- Network access requires explicit skill permission
- Debug/research endpoints are disabled
- Verbose logging is disabled (no secret leakage)

## Dependency Management

- Hash-pinned installs come from `requirements_lock.txt` (3,625 `--hash`
  entries), used with `pip install --require-hashes -r requirements_lock.txt`.
  `requirements/core.txt` is the human-edited runtime list and is **not**
  hash-pinned — install from the lock file when supply-chain integrity
  matters. Regenerate the lock with
  `pip-compile --allow-unsafe --generate-hashes --output-file=requirements_lock.txt requirements.txt`.
- `requirements_hardened.txt` is the additional hardened pin set.
- Automated vulnerability scanning via `pip-audit` and OSV
- SBOM generated per release via `make provenance`
  (writes `artifacts/provenance/{sbom,provenance}.json`)
- `make setup-prod` is the fail-closed install path: no fallback dependency
  installation, so a missing dependency fails the install rather than
  silently degrading it. Plain `make setup` does fall back and is for
  development only.

## Incident Response

See `docs/runbooks/` for operational incident response procedures.

## Compliance Mappings

| Framework | Document |
|-----------|----------|
| NIST SSDF | `security/NIST_SSDF_MAPPING.md` |
| OWASP ASVS | `security/OWASP_ASVS_MAPPING.md` |
| OWASP LLM Top 10 | `security/OWASP_LLM_MAPPING.md` |
| SLSA | `security/SLSA_PROVENANCE.md` |
| MITRE ATLAS | `security/MITRE_ATLAS_MAPPING.md` |

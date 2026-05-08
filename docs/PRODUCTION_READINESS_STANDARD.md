# Aura Production Readiness Standard

This is the non-longevity release bar. The 24h, 72h, and 30-day soaks remain
separate evidence, but no build is considered production-ready without passing
these repeatable controls from a clean clone.

## Required Gates

- Install: `make setup`
- Compile: `make compile`
- Fast collection: `make enterprise-collect`
- Full tests: `make test`
- Quality: `make quality`
- Enterprise ratchet: `make enterprise-gate`
- Governance bypass sweep: `make governance-lint`
- Security scan: `make security`
- Proof bundle: `make proof-bundle`
- Production contract: `make production-gate`
- Provenance/SBOM: `make provenance`

## Runtime Controls

- Every consequential action routes through Will and Authority receipts.
- Memory and state writes use the gateway or `core/runtime/atomic_writer.py`.
- Continuous experience frames are hash-chained and replay-validated.
- Repeated harm, high surprise, or fragmented Unity switches mode to
  observe/stabilize/replay before more action.
- Failures must call `record_degradation(...)` or return an explicit refusal;
  silent success is not allowed.

## Security Controls

- Runtime API authentication fails closed when no token is configured outside
  internal-only mode.
- Authorization covers tool execution, memory writes, state mutation,
  initiative, expression, and response paths.
- Secrets resolve through `core/zenith_secrets.py` with environment/Keychain
  priority and no value logging.
- Secret-like literals are scanned by `tools/security_scan.py`.
- Sandbox escape coverage lives in `tests/test_sandbox_hardening.py` and
  `tests/test_local_sandbox_hardening.py`.
- Release tags require signing credentials, codesign, notarization, stapling,
  and uploaded provenance/SBOM artifacts.

## Operating Controls

- Incident response is runbook-driven under `docs/runbooks/`.
- Privacy retention and deletion are governed by
  `docs/DATA_RETENTION_DELETION_POLICY.md`.
- Model/provider failure behavior is governed by
  `docs/MODEL_PROVIDER_FAILURE_POLICY.md`.
- Load/performance evidence lives in `tests/test_load_stress.py`,
  `tests/performance/locustfile.py`, and SLO CI.

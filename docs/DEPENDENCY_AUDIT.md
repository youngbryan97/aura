# Dependency Audit Policy

`make audit-deps` runs `pip-audit` over the installed environment and fails
on any known vulnerability that is not explicitly waived. Waivers live in
the `AUDIT_WAIVERS` variable in the Makefile and each one must be recorded
here with its justification and revisit condition. The Makefile still
names the waiver below; a historical rationale is not a fresh vulnerability
audit. This documentation review did not update the shared Python environment
or run a dependency scan beside the active campaign. A waiver without a
written justification is a policy violation, not a configuration choice.

`tools/runtime_preflight.py` separately reports drift between installed
package versions and `requirements_lock.txt` pins, so the audited
environment and the declared environment cannot silently diverge.

## Active waivers

### CVE-2025-3000 — torch `torch.jit.script` memory corruption

- **Waived:** 2026-07-12
- **Recorded 12 July environment:** torch 2.11.0. This is not a current
  installed-version or upstream-fix assertion; recheck the advisory and
  installed environment before relying on the waiver.
- **Why waived:** the vulnerable function is `torch.jit.script`; a repo-wide
  grep (enforced inline in the `audit-deps` target — the waiver self-revokes
  if a call site ever appears) shows zero call sites. Attack vector is
  local-host only; torch here processes Aura's own tensors (plasticity /
  lattice / vector-memory math), never untrusted serialized models.
- **Revisit when:** a fixed torch release ships (then upgrade and drop the
  waiver), or any code starts using `torch.jit` (the gate fails closed).

## Resolved findings (2026-07-12 sweep)

| Package | Vulnerable | Fixed-to | Advisory |
| --- | --- | --- | --- |
| msgpack | 1.1.2 | 1.2.1 | [GHSA-6v7p-g79w-8964](https://github.com/msgpack/msgpack-python/security/advisories/GHSA-6v7p-g79w-8964) |
| pydantic-settings | 2.14.0 | 2.14.2 | [GHSA-4xgf-cpjx-pc3j](https://github.com/advisories/GHSA-4xgf-cpjx-pc3j) (lock pin updated with official PyPI digests) |
| soupsieve | 2.8.3 | 2.8.4 | CVE-2026-49476, CVE-2026-49477 |
| tiktoken (drift, not CVE) | 0.12.0 installed vs 0.13.0 locked | 0.13.0 | caught by preflight lock-drift on its first run |

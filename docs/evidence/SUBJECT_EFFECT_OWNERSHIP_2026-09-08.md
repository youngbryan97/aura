# Subject evidence effect ownership repair

Date: 2026-09-08. Source basis: `43b32d714`.

The six governance-lint findings encountered during R07 were investigated and
repaired, not grandfathered:

- `subject.provenance._git` now uses SubprocessGateway with an explicit
  read-only, nonaccelerator source declaration. A nonzero exit no longer
  returns partial stdout as successful provenance.
- `next_run_directory` creates its parent through FileWriteGateway inside a
  governed internal scope.
- The archive writer uses the gateway factory directly, eliminating the
  opaque factory wrapper and redundant directory creation. The atomic writer
  already creates the parent.
- Three calls to a local `write_text` function were incorrectly classified as
  Path mutations. The scanner now resolves unambiguous module-local function
  definitions while still scanning their bodies. Reassignment, parameters,
  imports, star imports, nested definitions, decorators, deletion and class
  shadowing retain conservative classification.

The inventory diff contains only three declared gateway call sites and its
new digest. It adds no raw-effect allowance and changes no canonical-owner
classification. This does not make every inherited effect ActionExecutor-owned.

## Verification

- Scanner and subject effect tests: 31 passed, two whole-repository lint
  subprocess tests deselected. The repository lint was run directly below.
- `make smoke`: 164 passed, one skipped in 34.02 seconds.
- `make governance-lint`: passed, exact inventory matched.
- `make layering`: passed at the unchanged 37-entry layering baseline.
- Ruff and whitespace checks: passed.
- Real read-only Git probe through SubprocessGateway: returned the expected
  40-character source commit `43b32d714764af1627cd9f147a47657681a1d0ea`.
- Real archive writes preserve separate run outputs and round-trip JSON.

## Remaining obligation discovered

`tests/test_effect_ownership_tiers.py::test_total_debt_only_falls` fails:
the recorded aggregate is 1900 against its 1840 ceiling. Before this repair it
was already 1897; the three now-declared gateway sites account for the change.
The raw tier remains 833, below its 854 ceiling. Neither ceiling was raised.
Q06 retains the aggregate reduction obligation; this receipt closes the six
encountered findings, not all architecture/governance debt or R07 live proof.

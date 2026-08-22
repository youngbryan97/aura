# Security

Aura runs with the reach of the person using it. It executes commands, drives
the computer, holds persistent personal context, listens on a local port,
modifies its own source, and starts actions nobody asked for in that moment.
Every one of those is a capability somebody would want to borrow.

## Reporting a vulnerability

Open a private advisory:
<https://github.com/youngbryan97/aura/security/advisories/new>

Do not open a public issue for a vulnerability. Include what you ran, what
happened, and what you expected; a proof of concept is worth more than a
description, and a failing test in the shape of `tests/security/` is worth
more than either.

Expect a first reply within seven days. There is one maintainer and no
bounty programme.

## Scope

In scope, and the parts worth attacking first:

- the local HTTP and WebSocket API, its authentication, and its browser trust
  boundary — see [docs/LOCAL_API_TRUST_BOUNDARY.md](docs/LOCAL_API_TRUST_BOUNDARY.md);
- the execution surfaces: the shell skill, the terminal skill, MCP servers,
  host automation, the terminal motor;
- the sandbox — `security/sandbox.py` and its seatbelt profile;
- path containment: the workspace jail, the file write gateway, the skills
  that take a path from a caller;
- the governance chain: `core/security/execution_authority.py`, the capability
  tokens, and the standing directives;
- persistent state: the memory stores, the identity record, the migration
  ledger;
- anything that turns text from outside into an action inside.

Out of scope: the model's outputs considered as text, denial of service
against your own machine, and findings that require an attacker who already
has your login session.

## What the threat model says

[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) lists the assets, the trust
boundaries, and each class of attack with the control that answers it and the
test that runs it. Every control named there is attacked from the attacker's
side in `tests/security/`, so a control that stops working fails a test rather
than continuing to be documented. `tools/check_threat_model.py` fails when the
document names a test that does not exist.

The same document is explicit about what has not been done: no independent
security engineer has attacked this system. That is a gap in the practice, not
a gap in the writing, and no amount of self-review closes it.

## Handling of your data

Everything stays on the machine Aura runs on. There is no telemetry endpoint
and no account. `docs/DATA_RETENTION_DELETION_POLICY.md` says what is written
where and how to remove it; `make data-purge` and `make memory-purge` do it.

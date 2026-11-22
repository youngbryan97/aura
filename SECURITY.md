# Security

Aura has the same permissions and access as the user running it. It can run system commands, control your computer, store your personal data across sessions, listen for local network connections, edit its own code, and trigger automated background tasks. Because Aura has these powerful capabilities, securing them is critical to prevent misuse.

## Reporting a vulnerability

Please report security issues privately through GitHub advisories:
<https://github.com/youngbryan97/aura/security/advisories/new>

Do not open a public issue for a security vulnerability. Please include:
- The steps or commands you ran
- What actually happened
- What you expected to happen

A working proof-of-concept is much more helpful than a text description, and a failing automated test written to match `tests/security/` is even better.

You should receive an initial response within seven days. Note that Aura has a single maintainer and does not offer a bug bounty program.

## Scope

The following areas are in scope for security reports, and are the most critical components to test:

- The local HTTP and WebSocket API, its authentication methods, and how it protects against malicious web pages — see [docs/LOCAL_API_TRUST_BOUNDARY.md](docs/LOCAL_API_TRUST_BOUNDARY.md).
- The execution surfaces: the shell skill, the terminal skill, MCP (Model Context Protocol) servers, host automation, and the terminal motor.
- The sandbox — `security/sandbox.py` and its macOS seatbelt security profile.
- File and path containment: the workspace jail (which restricts file access to specific folders), the file-writing gateway, and any skills that take a file path from a caller.
- The governance chain: `core/security/execution_authority.py`, capability tokens, and standing operator directives.
- Saved data and state: the memory stores, the identity record, and the migration ledger.
- External input processing: anything that accepts text from outside (such as web pages or prompts) and turns it into an internal action.

**Out of scope:**
- The language model's raw text outputs (unless they trigger unexpected execution).
- Denial-of-service attacks against your own local machine.
- Security findings that require an attacker who already has full access to your logged-in user account.

## What the threat model says

The [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) file details the system assets, trust boundaries, known attack types, protective security controls, and the automated tests that verify each control.

Every security control listed in that document is actively tested using automated attack simulations in `tests/security/`. If a security control stops working, the corresponding test fails immediately instead of letting the documentation fall out of date. The script `tools/check_threat_model.py` checks this and fails if the threat model names a test file that does not exist.

The threat model is also explicit about its limitations: Aura has not yet undergone a security review or penetration test by an independent security engineer. While automated self-testing is extensive, it cannot replace an external review.

## Handling of your data

All your data stays completely local on the machine running Aura. There is no user account to create, and no telemetry or usage data is sent back to external servers.

[docs/DATA_RETENTION_DELETION_POLICY.md](docs/DATA_RETENTION_DELETION_POLICY.md) explains what data is saved and how to remove it. You can completely delete your data at any time:
- `make memory-purge` — deletes all stored conversation memories.
- `make data-purge` — deletes all stored data and logs.

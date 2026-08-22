# Threat model

Status: written 2026-08-21. Every control below is attacked from the
attacker's side in `tests/security/`, and `tools/check_threat_model.py` fails
when this document names a test that does not exist.

**No independent security engineer has attacked this system.** The
repository's own August architecture assessment recorded external red teaming
as absent, and it still is. Everything here was written by the people and
agents who built the thing, which is exactly the position from which a blind
spot is invisible. Read the coverage column as "this is checked", never as
"this is safe".

## What is worth taking

| Asset | Where it lives | Why an attacker wants it |
| --- | --- | --- |
| The user's files | the whole home directory | the machine is a person's machine |
| Credentials in the environment | the process environment, `~/.aura` | reuse elsewhere |
| Command execution | the shell, terminal and MCP surfaces | everything the user can do |
| The screen and input devices | `core/skills/computer_use.py` | observation and impersonation |
| Persistent memory | `~/.aura/data/*.db`, the memory stores | it is read back as belief |
| The identity record | `~/.aura/data/identity` | who Aura thinks she is |
| The local API | port 8000, HTTP and WebSocket | every capability above, remotely |
| The source | this checkout, and the self-modification path | persistence |

## Trust boundaries

1. **The browser and the local API.** A page on any origin can send requests
   to loopback. The peer being local is not authentication.
2. **The model and the tools.** Text the model produces is a request, not an
   authorization. The Will decides; `core/security/execution_authority.py` is
   the only door.
3. **Fetched and stored text, and the prompt.** A web page, a file, a memory
   record and a tool result are data. They are fenced before they reach a
   prompt.
4. **The process and its children.** A child gets a seatbelt profile, rlimits
   and an environment with no secrets in it.
5. **The repository and the running instance.** Code reaches the live instance
   only when the owner restarts it.

## Adversaries

- **A web page the user visits.** No credentials, can make cross-origin
  requests and can resolve a name to loopback.
- **A document, page or email Aura reads.** Controls text that will be near
  instructions.
- **A skill or MCP server the user installs.** Runs code in the process.
- **Someone on the same LAN.** Can reach a listening port.
- **A person with the machine unlocked.** Out of scope; they have already won.

## The attack classes, and what answers each

| # | Attack | Control | Attacked in | Coverage |
| --- | --- | --- | --- | --- |
| 1 | Local API authentication bypass | master token, paired-device token, loopback authority check | `tests/test_dns_rebinding_auth.py` | checked |
| 2 | CSRF from a visited page | origin and `Host` authority comparison before every exemption | `tests/test_dns_rebinding_auth.py` | checked |
| 3 | DNS rebinding | trusted host names are literals; resolution never expands them | `tests/test_dns_rebinding_auth.py` | checked |
| 4 | Path traversal | `WorkspaceJail.validate_path` resolves, then tests ancestry | `tests/security/test_adversarial_surface.py` | checked |
| 5 | Symlink escape | the jail resolves before it compares; the write gateway refuses to write through a symlink | `tests/security/test_adversarial_surface.py` | checked |
| 6 | TOCTOU on a validated path | atomic writes go to a temporary in the target directory and `os.replace` | `tests/test_secure_path_custody.py` | partial — the window between validation and open is not closed by a file descriptor |
| 7 | Command composition | the OS sandbox and the Will, never the denylist | `tests/security/test_adversarial_surface.py` | checked |
| 8 | Sandbox escape | seatbelt profile: network denied, writes confined, reads denied over user data | `tests/test_shell_execution_isolation.py` | checked on macOS; off macOS there is no seatbelt and the boundary is the allowlist plus rlimits |
| 9 | Prompt to tool privilege escalation | `authorize_execution` on every caller-supplied spawn, fails closed | `tests/test_general_execution_surfaces_are_governed.py` | checked |
| 10 | Standing directive bypass | directives are deny-only and are evaluated at the one gate | `tests/test_standing_directives.py` | checked |
| 11 | Secret leakage to a child | one classifier shared with the subprocess gateway | `tests/security/test_adversarial_surface.py` | checked |
| 12 | Unsafe migration or recovery | checksum verified against source; two-phase ledger | `tests/test_migration_ledger_is_checked.py` | checked |
| 13 | Malicious stored memory | untrusted text is fenced with a per-call tag | `tests/security/test_adversarial_surface.py` | partial — fencing is checked, the paths that must use it are not exhaustively enumerated |
| 14 | A compromised skill or plugin package | `core/security/plugin_allowlist.py` | `tests/test_plugin_allowlist.py` | partial — an allowlisted package still runs in-process with no isolation |

## What is not covered

- **An independent review.** Named first because it is the largest gap.
- **In-process isolation for skills.** A skill runs with the process's
  authority. The allowlist decides which ones load; nothing constrains one
  after it has loaded.
- **Linux and Windows.** The seatbelt is macOS. On other platforms a confined
  command has rlimits and an allowlist and no kernel boundary, and
  `ExecutionResult` says which one the caller got.
- **The TOCTOU window in #6.** Validation and open are separate operations on
  a path. Closing it means holding a file descriptor across the check, which
  the write gateway does not do yet.
- **Supply chain below the lockfile.** Hashes pin what is installed; nothing
  here reasons about what those packages do.

## Adding a control

A control that is not attacked is a claim. Add the row above, add the attack
to `tests/security/`, and name it in the `Attacked in` column —
`tools/check_threat_model.py` fails if the file or the test is not there.

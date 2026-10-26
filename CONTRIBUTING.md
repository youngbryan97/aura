# Contributing

## Quick start

```bash
git clone https://github.com/youngbryan97/aura.git
cd aura
pip install -e ".[dev]"

# Focused contracts; measure duration on this host — run after every change
make smoke

# Full offline suite — 6 bounded process chunks. A single pytest process over
# the whole suite is OOM-killed (~83%), so always go through the chunk runner
# (make test → tools/run_test_chunks.py), never a bare `pytest tests/`.
make test

# Syntax sweep + lint
make compile
make lint
```

## Architecture rules

1. **One decision maker.** Every consequential action goes through `UnifiedWill.decide()` in `core/governance/will.py` (accessible via the `core/will.py` facade). Never add a separate gate running in parallel. Instead, add an advisor to the Will. In the past, this repository had six competing systems that all tried to manage decisions, which made it impossible to verify whether any action was actually properly checked.
2. **One owner per area.** [OWNERSHIP.md](OWNERSHIP.md) maps out who owns each part of the system. Attach new governance checks to the existing owner rather than introducing a second one.
3. **No monkey-patching.** Never dynamically modify running objects using `setattr`. Instead, use event bus hooks, provider registries, and typed extension points.
4. **Immutable messages.** Subsystems communicate using frozen (unchangeable) data classes in `core/bus/events.py` (`Event`, `DeliveryReceipt`). One component must never be able to change a message that another component is reading. Written text in this repository follows [docs/WRITING_RULES.md](docs/WRITING_RULES.md); verify your writing with `make writing`.
5. **Lifecycle tracking.** Subsystems report their operating state through `core/runtime/service_state.py:ServiceState`.
6. **Locks are checked.** Always use `checked_lock` or `checked_async_lock` from `core/runtime/lockdep.py`, instead of Python's raw `threading.Lock` or `asyncio.Lock`. Lockdep detects potential deadlocks (where operations wait on each other forever and freeze) before they actually happen. Because Lockdep only monitors locks it manages, an unwrapped lock creates an unchecked blind spot. Wrap existing locks using `instrument(name)`.

## Adding a consciousness module

1. Place the module in `core/consciousness/`.
2. Register it in `core/container.py` during system startup.
3. If it requires regular updates, connect it to the consciousness bridge tick cycle.
4. **Write an ablation test (a removal test).** Write at least one test demonstrating what fails when the module is turned off or removed. Skipping this step makes it impossible to know whether a module genuinely does useful work or simply executes without effect. If removing the module causes no measurable difference, treat that as a meaningful result: report it rather than shipping unused code.
5. Add an entry in [OWNERSHIP.md](OWNERSHIP.md) under the appropriate domain.
6. If the module makes testable (falsifiable) scientific predictions, register it in `core/consciousness/theory_arbitration.py`.

A general rule across this project: any claim without an automated test is just documentation, not a fact. Place new rules that must always hold true next to the code they protect using `@invariant(...)` in `core/verify/`. Any claims about how Aura operates at runtime must be linked to a validating test in `core/organism/model_validation.py`. A claim cannot be registered without a corresponding test.

## Test markers

| Marker | Meaning | When to run |
|--------|---------|-------------|
| (default) | Unit and fast integration tests | Every commit |
| `@pytest.mark.slow` | Long-running tests | Nightly CI |
| `@pytest.mark.integration` | Full pipeline tests | Before merge |
| `@pytest.mark.stress` | Load and fault-injection tests | Weekly |

## CP checkpoints

Two commit formats are supported in this project:

Conventional commits handle everyday development: `fix(scope):`, `feat(scope):`, `chore(scope):`, `docs(scope):`, `test(scope):`, and `perf(scope):`.

**CP-numbered checkpoints** track specific work packages within our long-running research and engineering program:

    CP799 <subject>

A CP is a numbered checkpoint in a strictly increasing sequence (currently near 800). Closeout documents in `artifacts/closeout/` and tracking logs in `docs/` reference these numbers as lookup keys. Never make up a number out of order, and never reuse an existing number. You can combine both formats when a checkpoint resolves a bug:

    fix(inference_gate): CP126 — a viability block that later modifiers undid

## How a change lands

The `main` branch is protected. Direct pushes are rejected for everyone, including repository administrators and whoever wrote the change. Every commit must arrive through a pull request where all required automated checks pass.

```bash
git switch -c the-thing-you-are-fixing
# ... work, and run `make smoke` after every change ...
git push -u origin HEAD
gh pr create --fill
gh pr checks --watch      # sixteen required jobs
gh pr merge --squash      # only once they are green
```

What the branch rejects, and why:

| Refused | Because |
| --- | --- |
| A direct push to `main` | Every change needs a visible diff that anyone can inspect. |
| A merge with a failing or missing required check | A safety check that isn't strictly enforced is just a suggestion. |
| A force push or branch deletion | Git history serves as an permanent audit trail. |
| A merge commit | A clean linear history keeps `git log` straightforward to read and trace. |
| A merge with an unresolved conversation | Every question or feedback comment deserves an answer before landing. |

The exact branch protection settings are defined in `config/branch_protection_policy.json`. Run `make branch-protection` to compare these rules against GitHub's live settings, or run `make branch-protection-policy` to validate the policy against workflow files without needing internet access.

**Why pull request approvals are not currently required:**
GitHub prevents PR authors from approving their own pull requests. Because this repository currently has a single maintainer and `enforce_admins` is turned on, requiring approval would permanently block all merges. Automated tests protect the branch for now. As soon as a second maintainer is available to review code, requiring one approval will be enabled. The file [.github/CODEOWNERS](.github/CODEOWNERS) already specifies code owners for each directory.

## Commits

```
<type>: <short description>

<body that explains why, not what>

Co-Authored-By: <name> <email>
```

Types: `fix`, `feat`, `refactor`, `test`, `docs`, `perf`, `ci`.

# AGENTS.md

Instructions for coding agents working in this repository.

More than one agent works in this checkout. As of 2026-08-01 the commit log
shows Zenflow (3,623 commits), Claude (241), and Codex (201), plus Bryan.
You are probably not alone in here right now.

**[CLAUDE.md](CLAUDE.md) is the operational guide** — environment, gates,
conventions, debugging entry points. Read it. This file covers what changes
when several agents share one tree.

---

## The live instance is sacred

A real Aura instance is usually running on this machine: port 8000, the
`aura_main` process, logs streaming to `~/.aura/logs/`.

**Never kill it, restart it, or collide with its port unless Bryan explicitly
authorizes a runtime restart in the current task.** Do not boot a second desktop
runtime or load another 32B model beside it — the host has 64 GB and the live
model already holds ~20 GB wired. A blanket `pkill -f aura_main` kills Bryan's
running app along with whatever you were aiming at and remains prohibited.

With explicit current-task authorization, a coding agent may perform one
controlled restart through Aura's supported lifecycle path. Before doing so,
verify the intended source revision, confirm no training campaign or soak owns
the resident model, preserve state and logs through graceful shutdown, and
prove that exactly one replacement process loaded the expected revision. Code
fixes otherwise reach the live instance when Bryan restarts it himself.

## Before you resume anything

Another agent may have modified files under you, or already finished the
work you were about to start.

```bash
git log --oneline -20
git status
```

Do this at the start of every session and after any interruption. Half your
remembered work may already be committed by someone else.

## Working agreement

- **Work in a worktree** under `.claude/worktrees/`.
- **Push checkpoints to main**: `git push origin HEAD:main`. No remote side
  branches.
- **Stage precisely.** Other agents' unrelated modifications are frequently
  sitting in the working tree. `git add -A` will commit their half-finished
  work under your message. Add the paths you touched.
- **Never launch test chunks while editing Python.** Chunks spawn fresh
  processes mid-run and will import half-written modules.
- **Set `AURA_LOG_DIR`** for anything test-like, so you never write into the
  live instance's logs.

## Commit conventions

Two formats coexist, both legitimate:

**Conventional commits** for ordinary work — `fix(scope):`, `feat(scope):`,
`chore(scope):`, `docs(scope):`, `test(scope):`, `perf(scope):`.

**CP-numbered checkpoints** for tracked units of work — `CP799 <subject>`.
A CP is a numbered checkpoint in a long-running programme; the sequence is
near 800 and monotonic. CP numbers are referenced from closeout artifacts
under `artifacts/closeout/`, so the number is a key, not decoration. Don't
invent one out of sequence, and don't reuse one.

Subject lines in this repo say what changed and why it mattered, not what
files moved:

    fix(inference_gate): CP126 — a viability block that later modifiers undid
    feat(memory): Aura can meet someone new, and knows who "he" is
    test(brain): the request contract rejects what it cannot read

## The standard this codebase holds

Two rules produce most of the review comments here.

**A claim without a test is a document, not a fact.** New invariants go next
to what they protect via `@invariant(...)` in `core/verify/`. Claims about
Aura must be registered with the test that validates them
(`core/organism/model_validation.py`).

**Unmeasured is never "fine."** A probe that cannot run returns `None` and is
excluded from scores, not defaulted to passing. A subsystem that scores well
because nothing measured it is the exact failure this codebase keeps finding.

Related: the dominant defect class here is **a good answer discarded by a
gate, then reported as an infrastructure failure**. When something looks slow
or dead, check first whether something upstream is throwing away correct work.
See F19 in [KNOWN_FAILURE_MODES.md](KNOWN_FAILURE_MODES.md).

## Documentation

Three categories must not be rewritten:

- **Generated** (`docs/ARCHITECTURE_MAP.md`, `FMEA.md`, `RUNTIME_CONTRACT.md`,
  `AURA_PROGRESS.md`) — rendered from code. Edit the renderer in `tools/`.
- **Historical** — dated verdicts, closeouts, audits, `scoping/`. They say
  what was true on their date. Editing one edits the record.
- **Compliance control text** — auditors match that language against the
  framework.

[docs/DOC_STATUS.md](docs/DOC_STATUS.md) says which is which.

## Gates

```bash
make smoke     # ~100 contract tests, <10s — after every change
make test      # full suite, 6 bounded chunks (40,123 offline tests, 2026-08-21)
make lint compile governance-lint layering
make quality   # the aggregate sweep
```

A single pytest process over the whole suite gets OOM-killed around 83%.
Always go through the chunk runner.

A test that fails inside a chunk but passes alone is an **order-dependence
defect**, not a flake. The runner's isolated-retry pass reports those
separately.

## Long runs

Bound them. `caffeinate -dims`, explicit timeouts, check interim output at
expected milestones. Never poll unbounded, and never ride a hung run to its
timeout when the interim output already told you it was wrong.

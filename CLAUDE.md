# CLAUDE.md — Agent guide for the Aura codebase

Operational facts an agent needs before touching anything. Architecture
rules live in [CONTRIBUTING.md](CONTRIBUTING.md); the deep spec is
[ARCHITECTURE.md](ARCHITECTURE.md).

## The live instance is sacred

A real Aura instance is usually running on this machine (port 8000,
`aura_main` process, logs streaming to `~/.aura/logs/`). **Never kill,
restart, or port-collide with it.** Do not boot a second full desktop
runtime or load another 32B model beside it — the host has 64GB and the
live model already holds ~20GB wired. Code fixes reach the live instance
only when the user restarts it themselves.

## Environment

- Python: use the repo venv — `/Users/bryan/.aura/live-source/.venv/bin/python`
  (Python 3.12). The Homebrew `python3` is 3.14 and is NOT the runtime.
- Makefile gates accept it: `PYTHON=/Users/bryan/.aura/live-source/.venv/bin/python make <target>`.
- Worktrees share this venv; there is no per-worktree venv.

## Build / test / gates

```bash
make compile      # syntax sweep (core + tests)
make lint         # ruff, three passes (surface E9, critical F-codes, curated files)
make smoke        # ~100 contract tests, <10s — run after every change
make test         # FULL offline suite (40,123 tests) in 6 bounded process chunks
make governance-lint  security  enterprise-gate  # scrutiny gates
make layering     # DEPS include-rule gate; baseline in config/ only shrinks
```

- Full suite: `tools/run_test_chunks.py --chunks 6 --marker "not live and not network and not external"`.
  Use `--continue-on-failure` to collect everything, `--only-chunks 5,6` to
  resume a partial run. One pytest process on the whole suite gets
  OOM-killed (~83%); always use the chunk runner.
- **Chunk count is a memory budget, not a constant.** 6 chunks (≈353 files
  per pytest process) is right on an idle host and gets the *runner itself*
  killed when something else holds ~18GB — a resident 32B, a training sweep.
  The symptom is a log containing only the chunk header, because
  `capture_output=True` buffers the chunk's output in a parent that is then
  gone. Check `free` first; with a 32B up, use `--chunks 40` (≈54 files,
  ~70s each). `/tmp/aura_test_chunks_progress.log` names the chunk that was
  in flight, and `--min-free-gb N` refuses rather than gambles.
- A test failing in-chunk but passing alone is an ORDER-DEPENDENCE defect —
  the runner's isolated-retry pass reports these separately.
- Never launch test chunks while editing Python files: chunks spawn fresh
  processes mid-run and will import half-written modules.
- Long runs: bound them (`caffeinate -dims`, explicit timeouts), check
  interim output at expected milestones, never poll unbounded.

## Writing

Bryan writes about AI for a living and can spot machine-written prose
instantly. Everything you write here — docs, commit messages, code comments,
replies — is checked against [docs/WRITING_RULES.md](docs/WRITING_RULES.md).

- Follow **Zinsser's four principles**: 1. Simplicity 2. Brevity 3. Clarity
  4. Humanity. A controlled-English standard gives you the first three and
  loses the fourth; put it back.
- Use **ASD-STE100** for procedures, runbooks, gates, and API docs. Do not use
  it for anything with a voice — it flattens.
- The eighteen forbidden patterns, short version: no "That's not X, that's Y"
  (or its comma-spliced twin, "not just X, it's Y"); no stapled one-word
  sentences; no twin images without advice; **no clapping for your own point**;
  no analogy that assumes the reader knows both referents; no warming up before
  the sentence that matters; no reflexive triads; no ranges where a measurement
  belongs; no ending that recaps what was just read; no participle that
  restates its own sentence; no hedging before a fact; no unsourced "studies
  show"; no rhetorical question you then answer; no "let's dive in"; no stock
  opening; no long word where the short one was exact.
- `make writing` is the gate, and it covers docstrings and comments as well as
  the guides. The baselines in `config/ai_writing_baseline.json` only go down.
- A new rule needs three edits: the section in WRITING_RULES.md, the regex in
  `tools/lint_ai_writing.py`, and a worked example in
  `tests/test_ai_writing_rules.py`. That suite fails if any rule has no
  example, because a rule that cannot match reports green forever.
- **Append-only records are exempt and must not be restyled** — the execution
  tracker, the RLC ledger, `docs/evidence/`, dated verdicts. Editing those is
  falsifying a record.

## Conventions that will bite you

- **All consequential file writes go through `core/runtime/file_write_gateway.py`.**
  From async code use the `*_async` methods (or `async_atomic_*` in
  `core/runtime/atomic_writer.py`) — an on-loop fsync once froze the live
  event loop for 20 minutes. `tests/test_async_write_lane_ratchet.py`
  fails on new sync writes inside `async def`; its allowlist only shrinks.
- Internal maintenance writes need `local_internal_governed_scope(...)`
  (core/governance_context.py) or the live runtime refuses them as
  governance violations.
- Log through `logging`/structlog; the file sink JSON-wraps and redacts
  everything. Set `AURA_LOG_DIR` for anything test-like so you never write
  into the live `~/.aura/logs/`.
- Degradations: `record_degradation(subsystem, exc, action=...)` — never a
  silent `except: pass`. Modules on the fail-closed list (see
  `core/config.py`) escalate warning+ records to CRITICAL; for expected
  backpressure (timeouts under load), log at info and only record a
  degradation when the condition is persistent/total.
- ServiceContainer keys are the spine (`core/service_names.py`); health
  contract lives in `core/runtime/health_contract.py`.
- **Locks:** use `checked_lock` / `checked_async_lock`
  (`core/runtime/lockdep.py`) rather than raw `threading.Lock` /
  `asyncio.Lock`. Lockdep finds ABBA deadlocks without the deadlock
  happening, and it only sees locks it wraps. Adopt an existing lock with
  `instrument(name)`.
- **Layering:** every package under `core/`, plus `interface/`, `skills/`,
  `security/`, `llm/` and `executors/`, carries a `DEPS` file. Seven are
  hand-written and say what a foundation package may NOT reach for
  (`core/runtime`, `core/observability`, `core/verify`, `core/fsw`,
  `core/health`, `core/persistence`, `core/utils`); the rest are generated
  from the import graph by `tools/generate_deps.py` and allow exactly what
  the package imports today, so a new cross-package edge is an edit to a
  DEPS file. `make layering` is the gate, `make deps-check` catches a DEPS
  that no longer matches the graph, and `make deps-generate` rewrites them.
  The grandfathered baseline (`config/layering_baseline.json`) only shrinks.
- **New invariants** go next to what they protect, via
  `@invariant(name, scope=..., owner=...)` in `core/verify/`. A check that
  raises counts as a violation.
- **New telemetry** is a declared channel with an id, a unit, and limits
  (`core/fsw/telemetry_dictionary.py`). Ids are a contract; never reuse one.
- **Claims about Aura** must be registered with the test that validates
  them (`core/organism/model_validation.py`). A claim with no test cannot
  be registered.

## Debugging entry points worth knowing

- `AURA_PASS_BISECT_LIMIT=N` runs only the first N cognitive phases of each
  turn — binary-search N to find which phase ruined an answer.
  `AURA_PASS_TRACE=1` announces each one. Numbering restarts per turn, so N
  means the same thing on turn 40 as on turn 1, and both phase loops honour
  it: the legacy pipeline in `core/brain/cognitive_engine.py` (which serves
  chat) and `AuraKernel.tick`. Records from both land in the one
  `get_instrumentation().report()`, prefixed `legacy_pipeline/` or
  `kernel_tick/`.
- `runtime_health_report()["integrity"]` carries taint, lockdep splats,
  PSI, the OOM shed order, sanitizer findings, the last verifier report,
  telemetry limit violations, and unsupported claims.
- `get_bus_recorder().dump()` writes the event-bus ring for replay;
  `get_tracer().write()` writes a Perfetto-loadable trace;
  `get_memory_infra().diff(a, b).narrative()` names what grew.
- Full map: [docs/ENGINEERING_ADOPTION.md](docs/ENGINEERING_ADOPTION.md).

## Session mechanics for this repo

- Work in a worktree under `.claude/worktrees/`; push checkpoints with
  `git push origin HEAD:main` (no remote side branches).
- A parallel agent (Zencoder, commits as "Zenflow") shares this checkout
  and may modify files under you. `git log`/`git status` before resuming
  anything — your half-remembered work may already be committed.
- Crash forensics when the runtime dies: `data/error_logs/crash/`
  (faulthandler + loop-wedge + memory-spike stacks), `data/error_logs/stalls/`,
  `data/error_logs/memory/` (sentinel ring, tombstones, death syslogs),
  plus `~/.aura/logs/desktop-launch.log` for the live stdout stream.

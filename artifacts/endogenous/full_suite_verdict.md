# The full offline suite, after this work

Run on 2026-09-01 in `.claude/worktrees/endogenous-substrate`, twelve chunks,
marker `not live and not network and not external`, with the isolated-retry
pass the runner does for order dependence.

```
❌ 87 real failures — fail in-chunk AND alone
⚠️  15 order-dependence findings — fail in-chunk, pass alone
   chunks 7, 9 and 10 timed out at 2400s, so their coverage is partial
```

**None of the forty files carrying a real failure is a file this work created
or changed**, and none of the fifteen order-dependence findings either. The
list is in the log; the files are screen tasks, inference tiering, governance
lint, god-object and source-inspection ratchets, state ownership, and the
browsing lane.

Seven of the failures were reproduced at `ab58a159c` — the audited commit,
before any of this — in a detached worktree, and that transcript is in
[failures_that_predate_this_work.md](failures_that_predate_this_work.md). The
`governance-lint` one names `core/knowledge/atomspace_persistence.py` and
arrived with the atomspace refactor.

## What this run does not cover

The test files created after the run started are not in its chunk lists:

```
tests/test_a_head_that_refers_to_itself.py
tests/test_a_rule_with_no_shape.py
tests/test_she_writes_a_better_order.py
```

Those were run directly, repeatedly, and in randomised order together with
every neighbouring suite — 293 tests, one failure, which was a campaign
assertion decided by a wall clock and is now budgeted in candidates instead.
Saying so here rather than letting "the full suite is green in my area" cover
a gap it does not cover.

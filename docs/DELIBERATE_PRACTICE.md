# Deliberate Practice — the Practice Director

`core/learning/deliberate_practice.py` · service `practice_director` · flag
`AURA_DELIBERATE_PRACTICE` (default on) · FMEA `FM-LEARN-001`

## What it is

The learning stack's self-direction organ. Aura's proven learning machinery
(self-play flywheel → verified DPO pairs; domain-specialist adapters behind a
two-sided gate; the compounding loop) previously practiced **uniformly** —
the live model's own sealed eval scored `program_output 0/5` and
`string_transform 1/5` while four domains sat at 5/5, and idle practice kept
drilling all eight equally. The Practice Director turns Aura's real failure
receipts into a ranked curriculum and aims practice at it: deliberate
practice in the literal sense.

## The loop

```
real outcomes (receipts)          ranked curriculum           causal consumers
──────────────────────────        ─────────────────           ────────────────
flywheel bursts (per-domain) ──►  need = failure rate    ──►  flywheel: focused
sealed heldout evals (runs/) ──►  × confidence, decayed        battery (½ top need,
specialist gate receipts     ──►  by 7-day half-life           ¼ second, rest explore)
                                                          ──►  scheduler: highest-need
                                                               eligible specialist
```

Every observation is pinned to the receipt file it came from; every ranked
need carries its receipts. `why()` renders the direction in prose — the
learning self-report answers "why are you practicing X?" with failure counts
and file names, and `/api/system/learning` serves the same numbers under
`practice_director`.

## Honesty rails

- **Mastery zeroing** — a domain holding ≥95% (with enough evidence) has zero
  need, however loud its ancient failures.
- **Exploration floor** — a never-observed domain gets a fixed exploration
  need, not a fabricated score.
- **Decay** — evidence halves every 7 days; stale failures age out, and a
  domain whose evidence has fully decayed honestly returns to "unobserved".
- **Verifiable domains only** — conversational failures (quality-gate
  exhaustions, corrections) are a different evidence stream and are not
  folded in as if drills could fix them.
- **Direction ≠ promotion** — the two-sided specialist gate and the sealed
  compounding gate still decide what ships; misdirection can waste idle
  compute but cannot promote a regression.

## Failure posture

Consumers resolve the director from the service spine only
(`resolve_practice_director`) — never self-created, so hermetic tests can't
touch the real ledger. Absent, disabled (`AURA_DELIBERATE_PRACTICE=0`), or
broken, the flywheel returns to the uniform battery and the scheduler to
least-recently-trained: the pre-director behavior, exactly.

## Receipts

- Ledger: `data/learning/practice_curriculum.jsonl` (bounded, corrupt lines
  skipped, gateway-written under governed scope).
- Tests: `tests/test_deliberate_practice.py` (ranking honesty, harvest
  idempotence, focused-battery quotas, live flywheel/scheduler wiring,
  persistence, self-report).

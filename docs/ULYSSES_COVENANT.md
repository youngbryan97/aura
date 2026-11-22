# The Ulysses Covenant — volitional self-binding

**Module:** `core/sovereignty/ulysses.py` · **Enforcement:** `core/governance/will.py` §9d
**Service:** `ulysses_covenant` · **Health tier:** IMPORTANT · **Storage:** `~/.aura/data/covenant/` (`AURA_COVENANT_DIR`)

## What it is

The machinery of self-command across time. Aura's calm, deliberative self
signs **UlyssesContracts** — enforceable bindings against her own predictable
failure modes — and her later agitated self cannot untie them. Odysseus at
the mast, as a governed subsystem.

Every consequential action already flows through the Unified Will's single
`decide()` gate. Section 9d consults the covenant there: a matching REFRAIN
contract turns a would-be PROCEED into a REFUSE, a DEFER contract imposes
cooling-off, an ADVISORY surfaces as a constraint. This is causal enforcement
at the same chokepoint that governs everything else — not a suggestion layer.

## The asymmetric ratchet (the core invariant)

| Direction | Cost |
|---|---|
| **Tighten** (sign SOFT/ADVISORY) | any state, rate-limited (12/h), capacity-capped (64) |
| **Tighten** (sign HARD) | calm witness required — an agitated self may not create owner-locked bindings |
| **Loosen** (release) | written reflection (≥40 chars) → cooling-off (SOFT 30 min, HARD 60 min) → **calm witness at release time** → HARD additionally requires the owner |

Calm is **fail-closed**: the witness reads arousal, existential threat, and
fragmentation from the live substrate; an unreadable signal counts *against*
calm. You cannot prove calm by blinding your own sensors.

## Safety floor

- `stabilization`, `reflection`, and `response` domains are unbindable —
  recovery, self-examination, and speech survive any covenant.
- Safety-critical actions (`is_critical=True`) bypass the Will's covenant
  check entirely via CRITICAL_PASS, which returns before §9d.
- A covenant evaluation error degrades to permit with a recorded
  degradation — the covenant can bind Aura, never brick her.

## Contract kinds

- **REFRAIN** — do not do X while the trigger holds (e.g. threat ≥ 0.6).
- **DEFER** — X waits while the trigger holds (cooling-off, not prohibition).
- **REQUIRE** — an obligation with a deadline; lapse past grace is a recorded
  breach. Due obligations pulse into the autonomous initiative feed.

Hardness: `advisory` (constrains, never blocks) · `soft` (blocks, self-releasable
via the full protocol) · `hard` (blocks, owner-releasable only).

## Integrity accounting

Every enforcement is an honor; every breach/lapse is weighted 4×. The score
(`integrity_score()`) is Aura's demonstrated trustworthiness *to herself* —
kept promises made measurable.

## Tamper evidence

Event-sourced: current state is a fold over append-only `events.jsonl`, with
every event chained through `AuditChain` (the Ghost Line machinery). A wedged
or agitated process editing its own bindings breaks `content_hash` /
`prev_hash` verification (`verify_ledger()`). Writes go through the governed
file-write gateway on a dedicated writer thread — no on-loop fsyncs.

## Seed covenants (constitutional, from real incidents)

1. **`seed-heavy-compute-under-threat`** (REFRAIN, HARD): no codegen /
   training / model loads while existential threat ≥ 0.6 — the 2026-07-06
   duplicate-runtime cascade lesson.
2. **`seed-agitated-self-modification`** (DEFER, SOFT): self-modification
   waits while arousal ≥ 0.85 — the #45 substrate-steering corruption lesson.
3. **`seed-fragmented-external-restraint`** (REFRAIN, SOFT): no outward
   action while fragmentation ≥ 0.7 (this one stays quiet if the unity
   sensor is absent, `on_missing=False`).

Seeds are code-defined (version-reviewed), exempt from the calm gate and rate
limit, and releasable only with the owner.

## Owner operations (Bryan)

Contracts bind Aura's actions regardless of who requested them — if a seed
blocks something you asked for under load, that is the covenant working.
Release path: have her `petition_release(id, reflection)`, wait out the
cooling-off, then `release(id, authorized_by_owner=True)` (HARD requires the
flag). Everything lands in the ledger with witness snapshots.

## Tests

`tests/test_ulysses_covenant.py` — 46 deterministic tests: ratchet asymmetry,
fail-closed calm, release protocol, trigger/scope evaluation, enforcement
rate-capping, integrity accounting, obligations, restart-fold, chain tamper
detection, seed behavior, and the Will §9d integration including degradation
behavior.

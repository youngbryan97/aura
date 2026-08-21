# Reviewer packet

*Reviewed against the tree: 2026-08-01. Every path and target below was
verified to exist.*

## What you'll need
- A Mac (Apple Silicon), 50 GB+ free disk. 64 GB unified memory is the
  tracked target and what the 32B Cortex lane assumes; at 32 GB you can run
  the system but must downshift to the 7B/1.5B lanes rather than expecting
  32B latency.
- Python 3.12.
- 30 minutes for the install + bench pass; 7 days for the long-run.

## Steps
```bash
git clone https://github.com/youngbryan97/aura
cd aura
make setup
make quality
make bench
make courtroom
make baselines
make longevity   # optional, 24h
make chaos       # optional, single random fault
```

## Files to inspect
- ``docs/DOC_STATUS.md`` — **read first.** Which documents are current,
  which are dated records, and which are generated from code.
- ``CLAIMS_NOT_SUPPORTED.md`` — what is deliberately *not* claimed. The most
  useful page in the repo for a skeptic.
- ``docs/REALITY_REACH.md`` — the physical-claim boundary and its open
  ledger. Read "Current Evidence" before crediting any physical result.
- ``ROADMAP.md`` — code-grounded map of every claim and where it lives.
  Note: its letter grades predate the July–August work and are due a
  re-score.
- ``docs/TERMINOLOGY.md`` — sober ↔ poetic label mapping.
- ``aura_bench/runner.py`` — pre-registration discipline.
- ``aura_bench/courtroom/courtroom.py`` — adversarial 5-system bench.
- ``core/agency/agency_orchestrator.py`` — canonical life-loop.
- ``core/ethics/conscience.py`` — hard-line rule floor.
- ``core/brain/latent_bridge.py`` — substrate→sampling modulation.

## What you can verify yourself
1. **Governance fence** — `make governance-lint` returns clean. Try
   inserting a forbidden call into a non-allow-listed file and re-run
   the lint; it must reject.
2. **Capability token replay rejection** — `pytest tests/governance/test_capability_token.py`.
3. **Phenomenal error map** — `pytest tests/governance/test_phenomenal_error_map.py`.
4. **Conscience hard-lines** — `pytest tests/governance/test_conscience.py`.
5. **Self-object calibration** — `pytest tests/personhood/test_self_object.py`.
6. **Belief court provenance** — `pytest tests/belief_court/`.
7. **Bench harness pre-registration** — read `aura_bench/runner.py` and
   note that ``run_one`` requires a `Registration` produced by
   ``test.declare()`` *before* it accepts a verdict.

8. **Full test surface** — `pytest tests/ --collect-only -q`. As of
   2026-08-21 this collects **40,139 tests across 2,697 files**. Run them
   with `make test` (6 bounded chunks); a single pytest process over the
   whole suite gets OOM-killed around 83%.

## What you cannot verify in this packet
- Phenomenal consciousness (no codebase can).
- A 30-day live run (this packet ships the runner; the run itself is
  yours to perform).
- Real on-chain spending (the wallet adapter is in-memory by default).
- **Any physical effect on the world.** `core/reality_reach/` supplies the
  contract, channel declarations, and reachability proof; the RR-10
  acceptance battery is entirely open and the P0–P6 evidence promotion
  state machine is not implemented. Infrastructure existing is not a
  result. See `docs/REALITY_REACH.md`.

Open issues on the GitHub project for discrepancies; the ROADMAP file is
the single source of truth for claim ↔ code linkage.

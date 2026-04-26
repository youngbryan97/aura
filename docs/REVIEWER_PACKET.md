# Reviewer packet

## What you'll need
- A Mac (Apple Silicon) with 32 GB+ RAM, 50 GB+ free disk.
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
- ``ROADMAP.md`` — code-grounded map of every claim and where it lives.
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

## What you cannot verify in this packet
- Phenomenal consciousness (no codebase can).
- A 30-day live run (this packet ships the runner; the run itself is
  yours to perform).
- Real on-chain spending (the wallet adapter is in-memory by default).

Open issues on the GitHub project for discrepancies; the ROADMAP file is
the single source of truth for claim ↔ code linkage.

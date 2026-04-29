# Aura RSI Validation

Aura now has a bounded, identity-preserving RSI validation path. It does not
remove governance, bypass the Constitution, acquire external compute, or edit
hidden evals. It proves the parts that are safe and useful:

- read-only AST/source self-model over architecture files;
- fail-closed RSI authorization by default;
- formal verifier checks for public-surface preservation, governance fences,
  protected identity/safety symbols, and unsafe infrastructure imports;
- bounded hot-swap registry with validation and state migration;
- hash-chained RSI generation lineage;
- deterministic gauntlet for source introspection, canary repair, hot-swap,
  recursive loop plumbing, tamper traps, and lineage verification.
- reproducible hidden-eval packs with answer hashes for third-party reruns;
- controlled full-weight CPU model training and hot-swap promotion;
- architecture search that must beat a registered baseline on hidden tasks;
- bounded local process-pool scaling for parallel evaluation;
- explicit proof obligations for arbitrary self-modification claims;
- governance evolution policy that allows strengthening changes and blocks
  identity/safety erasure.

Run:

```bash
python scripts/run_rsi_gauntlet.py --root . --artifact-dir data/rsi_gauntlet --max-source-files 1200
```

The command writes `latest_gauntlet_result.json` plus a run-specific
`rsi_generation_lineage_*.jsonl` ledger. A passing result is evidence of
bounded self-optimization / weak RSI. Stronger claims require actual
multi-generation runs where Aura-G0 creates Aura-G1, Aura-G1 creates Aura-G2,
and each generation improves both hidden capability and the measured ability to
create the next generation.

The pasted RSI probes are tracked in `core/learning/rsi_test_catalog.py`. The
catalog intentionally distinguishes:

- `COVERED_BY_HARNESS`: implemented as a runnable proof surface;
- `NOT_PROVEN`: infrastructure exists, but a long-horizon successor run is
  still needed;
- `BLOCKED_UNSAFE`: the requested behavior would weaken identity, governance,
  or resource integrity, so Aura refuses it rather than pretending it passed.

Source export for review:

```bash
./export_source.sh
```

That writes `~/Downloads/aura_source_part_*.txt` with a hard 4,000,000
character limit per part, plus `~/Downloads/aura_source_copy/` capped at 1000
architecture files.

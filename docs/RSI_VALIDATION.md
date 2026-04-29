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

Run:

```bash
python scripts/run_rsi_gauntlet.py --root . --artifact-dir data/rsi_gauntlet --max-source-files 1200
```

The command writes `latest_gauntlet_result.json` plus a run-specific
`rsi_generation_lineage_*.jsonl` ledger. A passing result is evidence of
bounded self-optimization / weak RSI, not a claim of unbounded intelligence
explosion.

Source export for review:

```bash
./export_source.sh
```

That writes `~/Downloads/aura_source_part_*.txt` with a hard 4,000,000
character limit per part, plus `~/Downloads/aura_source_copy/` capped at 1000
architecture files.

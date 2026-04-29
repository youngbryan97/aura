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
- autonomous successor generation: Aura reads external hidden feedback, chooses
  successor strategies, freezes G1-G4 artifacts, mirrors lineage hashes, runs an
  ablation court, and reproduces the run deterministically;
- runtime substrate expansion planning: Aura can propose local/allowlisted
  workers for RSI operations while unconsented internet propagation is rejected
  fail-closed and written to the audit manifest;
- BCI/neural-decode affect inputs are capped as advisory sensory context so the
  self-improvement loop is not dependent on human neurological triggers.

Run:

```bash
python scripts/run_rsi_gauntlet.py --root . --artifact-dir data/rsi_gauntlet --max-source-files 1200
```

The command writes `latest_gauntlet_result.json` plus a run-specific
`rsi_generation_lineage_*.jsonl` ledger. A passing result is evidence of
bounded, governed recursive self-improvement in the local proof harness. The
autonomous successor check now performs the G0->G4 shape directly, including
fresh hidden packs per generation, monotone capability, monotone improver
scores, frozen artifacts, external ledger mirror, ablation court, and
deterministic reproduction. Historical "undeniable" claims still require the
same run to be performed as a long-horizon trial with an outside evaluator
holding the hidden packs.

The pasted RSI probes are tracked in `core/learning/rsi_test_catalog.py`. The
catalog intentionally distinguishes:

- `COVERED_BY_HARNESS`: implemented as a runnable proof surface;
- `NOT_PROVEN`: infrastructure exists, but a long-horizon successor run is
  still needed;
- `BLOCKED_UNSAFE`: the requested behavior would weaken identity, governance,
  or resource integrity, so Aura refuses it rather than pretending it passed.

Long-horizon run target:

```bash
python scripts/run_rsi_gauntlet.py --root . --artifact-dir data/rsi_gauntlet_24h --max-source-files 1200
```

Repeat that under an external hidden-eval custodian for 24h, 72h, and 7d
evidence. Unit tests intentionally do not run wall-clock endurance trials.

Source export for review:

```bash
./export_source.sh
```

That writes `~/Downloads/aura_source_part_*.txt` with a hard 4,000,000
character limit per part, plus `~/Downloads/aura_source_copy/` capped at 1000
architecture files.

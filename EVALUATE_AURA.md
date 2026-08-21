# Guide to Evaluating Aura: Cognitive Agent Runtime

Clone it, install it, boot it, test it, audit it. From a clean checkout, in
order, with nothing taken on trust.

This protocol is written so it can come out negative. Following it should
let you decide for yourself whether the governance boundaries hold, whether
the self-healing paths actually run, and whether the modules are load-bearing
or decorative — and if any of them fail on your machine, the protocol has
done its job. A verification you can't fail isn't one.

Two documents to have open beside this one:
[CLAIMS_NOT_SUPPORTED.md](CLAIMS_NOT_SUPPORTED.md) for what is deliberately
not claimed, and [docs/DOC_STATUS.md](docs/DOC_STATUS.md) for which docs are
current versus dated records of a single run.

---

## 1. Prerequisites and Installation

Aura is designed for deterministic, clean, out-of-the-box installation on macOS and Linux systems with Python 3.12+.

### Step 1: Clone the Repository
```bash
git clone https://github.com/youngbryan97/aura.git
cd aura
```

### Step 2: Establish a Clean, Hardened Environment
Run the setup sequence to clear any existing cache directories and install strictly pinned, production-locked dependencies:
```bash
make source-hygiene
make setup-prod
```
> [!NOTE]
> The dependencies are strictly locked under `requirements_hardened.txt` to prevent silent drift or unvetted package updates.

---

## 2. Running the System Diagnostic

To confirm that your local environment satisfies all typing, linting, and structural invariants, run the doctor probe:
```bash
make doctor
```

---

## 3. Running the Master Certification Gauntlet

To execute the entire end-to-end verification suite, run:
```bash
make certify
```

The master certification orchestrates four independent, isolated verification gates:
1. **Source Hygiene**: Runs static syntax and runtime contract tests.
2. **Boot Certification**: Spawns a headless Aura API server, runs gateway probes, and checks critical fail-closed degradation policies.
3. **Aletheia Live Proof**: Executes a leakage-isolated task benchmark where candidate-visible specifications are pumped through `/api/chat` with zero access to private keys or hashes, scored by an external decoupled scorer.
4. **Architecture Ablation Suite**: `tools/ablation_runner.py` drives each
   condition intact and then lesioned — substrate, System 2, verifier, memory,
   Will — against a real organ rather than a mock, and reports the measured
   metric for both plus the delta. A no-delta result is reported as "NOT
   load-bearing on this battery" rather than hidden, which is the only reason
   the deltas mean anything. Run it alone with
   `python tools/ablation_runner.py --list` to see the conditions.

---

## 4. Inspecting the Certification Artifacts

After `make certify` completes, its reports are written under
`artifacts/certification/latest/`. They are not signed — nothing in that
directory carries a signature or a content hash, and this page used to say
otherwise. Treat them as this machine's output, reproducible by running the
same command, not as attestations.

Key certification files to inspect:

* **[BOOT_CERTIFICATE.json](artifacts/certification/latest/BOOT_CERTIFICATE.json)**: Verification of successful headless server boot.
* **[SERVICE_MANIFEST.json](artifacts/certification/latest/SERVICE_MANIFEST.json)**: Declared owners, origins, and failure policies for all active services.
* **[CAPABILITY_MANIFEST.json](artifacts/certification/latest/CAPABILITY_MANIFEST.json)**: Hardcoded runtime limits and capabilities active in each mode.
* **[DEGRADATION_REPORT.json](artifacts/certification/latest/DEGRADATION_REPORT.json)**: Logs of system safety lockdown actions when critical services are lesioned.
* **[WORLD_RESULTS.jsonl](artifacts/certification/latest/WORLD_RESULTS.jsonl)**: Individual scorecards from the Aletheia Live Proof.
* `ABLATION_SUMMARY.json`: written by your own run of the ablation suite. It
  is not committed, and was not until 2026-08-21 — what sat here under that
  name was a fabricated scorecard, described below.
* **[CERTIFICATION_VERDICT.json](artifacts/certification/latest/CERTIFICATION_VERDICT.json)**: pass or fail for each of the four gates, plus three standing negatives — `agi_proven`, `consciousness_proven` and `open_world_autonomy_proven` are hardcoded `False` and are not outputs of the run.

### What was removed from this directory on 2026-08-21

`aura_bench/ablations/runner.py` printed hardcoded dict literals — raw model
0.42, full Aura 0.94 — as an "ABLATION SUITE" result having executed nothing.
It was deleted on 2026-07-15, and `tests/test_no_fabricated_benchmarks.py` was
written to stop it coming back.

Deleting the source did not delete what it had already written. Six committed
copies of that same dict survived under five different names, and this page
linked one of them to you as proof that each module is causally load-bearing.
Three `SOAK_LOG_*.json` files written by a `random.seed()` simulator survived
beside them. All nine are gone, the certification gate now runs the real
ablation runner instead of the deleted file it had been failing on, and the
guard test reads committed artifacts as well as Python source — because a test
over source cannot see a JSON file.

---

## 5. Reviewing Long-Run Autonomy Soaks

There is no committed soak telemetry to review, and until 2026-08-21 there
appeared to be. Three files named for 4-hour, 24-hour and 72-hour runs were
written by a simulator from `random.seed(duration_hours * 42)`, and all three
carried a completion timestamp within six milliseconds of each other. They
have been removed along with the tool that made them; nothing about resource
stability was ever concluded from them, and nothing could have been.

The real evidence is the dated soak verdicts, each a single run, each kept as
written:

* [docs/SOAK_VERDICT_2026_07_15.md](docs/SOAK_VERDICT_2026_07_15.md)
* [docs/SOAK_VERDICT_2026_07_18.md](docs/SOAK_VERDICT_2026_07_18.md) — idle RSS
  declining at −21 MB/h over 50 minutes and 100 samples
* [docs/SOAK_VERDICT_2026_07_25.md](docs/SOAK_VERDICT_2026_07_25.md) — idle RSS
  1.14 → 1.22 GB over the same window, and a **FAIL** verdict on whether she
  answered at all

To run one yourself rather than read someone else's:

```bash
python tools/longevity/run_longevity_soak.py --profile proof --out <dir>
python tools/longevity/validate_longevity_soak.py <dir>
```

The bar a run has to clear is [docs/LONGEVITY_SOAK_STANDARD.md](docs/LONGEVITY_SOAK_STANDARD.md),
and the claim boundary is §5 of [CLAIMS_NOT_SUPPORTED.md](CLAIMS_NOT_SUPPORTED.md).

---

## 6. Understanding What is Proven vs. Simulated

Aura maintains absolute transparent integrity regarding capability claims. Please review the official claim ledgers at the root of the repository:

1. **[CLAIMS_SUPPORTED.md](CLAIMS_SUPPORTED.md)**: Scientifically defensible capabilities (Governed execution, Persistent memory, Speculative MCTS search, Diagnostic self-repair) backed by explicit code locations.
2. **[CLAIMS_NOT_SUPPORTED.md](CLAIMS_NOT_SUPPORTED.md)**: Speculative, unproven, or metaphysical horizons (Artificial General Intelligence, Subjective Consciousness, Metaphysical Free Will) clearly demoted to prevent hyping.

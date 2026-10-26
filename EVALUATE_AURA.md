# Guide to Evaluating Aura: Cognitive Agent Runtime

<!-- reader-navigation:start -->
> **In plain English:** A test makes a specific claim answerable by a stated measurement, comparison and decision rule.
>
> **Why it matters:** A user needs to distinguish a built mechanism, a passing local test and a demonstrated useful capability.
>
> **In everyday use:** To show memory helps a task, compare the same task with memory available and disabled, and keep both successes and failures.
>
> [Example and metaphor](docs/PLAIN_ENGLISH_GUIDE.md#measurement) · [Variables and practical meanings](docs/VARIABLES_EXPLAINED.md#measurement) · [Demos and related work](docs/DEMOS_AND_RELATED_WORK.md#measurement) · [Document map](docs/DOCUMENTATION_INDEX.md)
> [Glossary](docs/GLOSSARY.md) · [Formulas with worked examples](docs/FORMULAS.md)
<!-- reader-navigation:end -->

Clone it, install it, boot it, test it, audit it. From a clean checkout, in order, with nothing taken on trust.

These instructions are designed so they can fail. Following them lets you see for yourself whether safety boundaries hold, whether self-healing features actually run, and whether the system's components genuinely matter or are just for show. If any test fails on your machine, the test has done its job. A test you can't fail isn't a real test.

Two documents to have open beside this one:
[CLAIMS_NOT_SUPPORTED.md](CLAIMS_NOT_SUPPORTED.md) for what is deliberately not claimed, and [docs/DOC_STATUS.md](docs/DOC_STATUS.md) for which docs are current versus dated records of a single run.

---

## 1. Prerequisites and Installation

Aura works on macOS and Linux systems running Python 3.12 or newer. The setup process is designed to be clean and repeatable.

### Step 1: Clone the Repository
```bash
git clone https://github.com/youngbryan97/aura.git
cd aura
```

### Step 2: Establish a Clean, Hardened Environment
Run the setup sequence to clear any existing cache files and install the exact, pinned package dependencies:
```bash
make source-hygiene
make setup-prod
```
> [!NOTE]
> Dependencies are strictly locked under `requirements_hardened.txt` to prevent unexpected package updates from altering how Aura behaves.

---

## 2. Running the System Diagnostic

To confirm that your local environment satisfies all typing, code quality, and structural checks, run the doctor diagnostic:
```bash
make doctor
```

---

## 3. Running the Master Certification Gauntlet

To run the complete end-to-end verification suite, run:
```bash
make certify
```

The master certification orchestrates four separate verification gates:
1. **Source Hygiene**: Checks code formatting, types, and basic runtime contracts.
2. **Boot Certification**: Starts a headless Aura API server, tests the API endpoints, and verifies that the system safely limits features or shuts down if critical parts fail (failing closed).
3. **Aletheia Live Proof**: Runs a benchmark task through `/api/chat`. The benchmark prompt cannot access private keys or internal hashes, and results are scored by an independent external scorer.
4. **Architecture Ablation Suite**: An ablation test disables parts of a system one by one to verify whether each part actually improves performance. `tools/ablation_runner.py` tests each component intact and then disabled (lesioned) — substrate, System 2 reasoning, verifier, memory, and Will — against real components rather than simulated mocks. It reports the measured score for both conditions plus the difference (delta). If disabling a component makes no difference, the report plainly states "NOT load-bearing on this battery" rather than hiding it. You can run it on its own with `python tools/ablation_runner.py --list` to see all test conditions.

---

## 4. Inspecting the Certification Artifacts

After `make certify` completes, its reports are written to `artifacts/certification/latest/`. These files are not cryptographically signed — nothing in that directory carries a digital signature or content hash (an earlier version of this documentation mistakenly claimed they did). Treat them as local test outputs that you can reproduce on your own machine, not as third-party certified proofs.

Key certification files to inspect:

* **[BOOT_CERTIFICATE.json](artifacts/certification/latest/BOOT_CERTIFICATE.json)**: Verifies that the headless server booted successfully.
* **[SERVICE_MANIFEST.json](artifacts/certification/latest/SERVICE_MANIFEST.json)**: Lists all active services, their owners, where they run, and their failure policies.
* **[CAPABILITY_MANIFEST.json](artifacts/certification/latest/CAPABILITY_MANIFEST.json)**: Built-in operational limits and capabilities enabled in each mode.
* **[DEGRADATION_REPORT.json](artifacts/certification/latest/DEGRADATION_REPORT.json)**: Logs showing how safety mechanisms lock down the system when critical services are disabled.
* **[WORLD_RESULTS.jsonl](artifacts/certification/latest/WORLD_RESULTS.jsonl)**: Individual scorecards from the Aletheia Live Proof benchmark.
* `ABLATION_SUMMARY.json`: Generated when you run the ablation suite yourself. This file is not committed to git (and has not been since August 21, 2026; what sat here previously was a fabricated scorecard, described below).
* **[CERTIFICATION_VERDICT.json](artifacts/certification/latest/CERTIFICATION_VERDICT.json)**: Shows pass or fail for each of the four verification gates. It also includes three standing negatives — `agi_proven`, `consciousness_proven`, and `open_world_autonomy_proven` are permanently hardcoded to `False` and are not claims made by the test run.

### What was removed from this directory on 2026-08-21

An earlier script (`aura_bench/ablations/runner.py`) printed hardcoded numbers — claiming a raw model scored 0.42 and full Aura scored 0.94 — without actually executing any tests. That script was deleted on July 15, 2026, and an automated test (`tests/test_no_fabricated_benchmarks.py`) was written to ensure it is never restored.

However, deleting the script did not remove the data files it had already generated. Six copies of those fake scores remained in the repository under five different filenames, and this page previously linked to one of them as proof that each module improves performance. In addition, three fake soak test logs (`SOAK_LOG_*.json`) generated by a random number generator (`random.seed()`) were also committed.

All nine fake files have been removed. The certification gate now runs the real ablation benchmark instead of the deleted file it had been failing on, and the guard test checks committed files as well as Python source code. It rejects the fabricated benchmark patterns the test defines.

---

## 5. Reviewing Long-Run Autonomy Soaks

A soak test runs software over many hours to check for memory leaks, crashes, or performance slowdowns. There are no committed soak test logs to review, though until August 21, 2026, it appeared that there were. Three files named for 4-hour, 24-hour, and 72-hour runs were actually produced by a simulator using `random.seed(duration_hours * 42)`, and all three carried completion timestamps within six milliseconds of each other. They have been removed along with the tool that created them; they proved nothing about resource stability.

The real evidence comes from dated test logs of individual runs, each preserved exactly as recorded:

* [docs/SOAK_VERDICT_2026_07_15.md](docs/SOAK_VERDICT_2026_07_15.md)
* [docs/SOAK_VERDICT_2026_07_18.md](docs/SOAK_VERDICT_2026_07_18.md) — idle memory usage (RSS) declined at −21 MB/h over 50 minutes and 100 samples.
* [docs/SOAK_VERDICT_2026_07_25.md](docs/SOAK_VERDICT_2026_07_25.md) — idle memory grew from 1.14 GB to 1.22 GB over the same window, and received a **FAIL** verdict because the system stopped responding.

To run a longevity soak test yourself rather than reading past logs:

```bash
python tools/longevity/run_longevity_soak.py --profile proof --out <dir>
python tools/longevity/validate_longevity_soak.py <dir>
```

The criteria a run must pass are set in [docs/LONGEVITY_SOAK_STANDARD.md](docs/LONGEVITY_SOAK_STANDARD.md), and the limits of our stability claims are detailed in §5 of [CLAIMS_NOT_SUPPORTED.md](CLAIMS_NOT_SUPPORTED.md).

---

## 6. Understanding What is Proven vs. Simulated

Aura maintains complete transparency about its capabilities. Please review the official claim ledgers at the root of the repository:

1. **[CLAIMS_SUPPORTED.md](CLAIMS_SUPPORTED.md)**: Proven capabilities (governed execution, persistent memory, Monte Carlo Tree Search planning, and diagnostic self-repair) backed by specific code locations.
2. **[CLAIMS_NOT_SUPPORTED.md](CLAIMS_NOT_SUPPORTED.md)**: Speculative or unproven claims (Artificial General Intelligence, subjective consciousness, and metaphysical free will) explicitly disclaimed to prevent hype.

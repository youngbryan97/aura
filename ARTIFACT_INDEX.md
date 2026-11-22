# Aura artifact index

<!-- reader-navigation:start -->
> **In plain English:** Observability records what the system did, how long each stage took, and where an assumption failed.
>
> **Why it matters:** A visible error or delay needs a traceable cause; otherwise a repair can change the wrong thing while hiding the real failure.
>
> **In everyday use:** When a reply arrives late, separate time spent waiting for input, reasoning, synthesis and playback before deciding which stage to optimise.
>
> [Example and metaphor](docs/PLAIN_ENGLISH_GUIDE.md#observability) · [Variables and practical meanings](docs/VARIABLES_EXPLAINED.md#observability) · [Demos and related work](docs/DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](docs/DOCUMENTATION_INDEX.md)
> [Glossary](docs/GLOSSARY.md) · [Formulas with worked examples](docs/FORMULAS.md)
<!-- reader-navigation:end -->

The `artifacts/current/` directory holds generated build and test outputs. Because it is listed in `.gitignore`, files created there during verification runs are not pushed to GitHub. They exist only locally on the computer where they were generated, which means clicking a link to one from a web browser will result in a 404 error. This page maps each test artifact to the command used to generate it.

The `make final-proof` target orchestrates configured proof producers. Plan
its resource budget before execution; it is not a harmless artifact download
or a current live certification. Individual producers write their declared
outputs, often under `/tmp`. Check the run and source revision before treating
a local artifact as current or publishing it.

## Core reports

| Artifact | What it holds | Written by |
| :--- | :--- | :--- |
| `artifacts/current/enterprise_gate.json` | Syntax checks, security scans, and wildcard-import audits against enterprise standards | `make enterprise-gate` → `/tmp/aura_enterprise_gate.json`; `make final-proof` → `artifacts/current/` |
| `artifacts/current/production_readiness.json` | Verification checks for every production readiness requirement | `make production-gate` → `/tmp/aura_production_readiness.json` |
| `artifacts/current/architecture_map.json` | Map of all memory writes, state changes, tool executions, and LLM call locations | `make architecture-map` → `/tmp/aura_architecture_map.json` |
| `artifacts/current/production_surface_lint.json` | Checks for production code that bypasses official security gates or spawns unsupervised background tasks | `python tools/production_surface_lint.py --scope production` |
| `artifacts/current/proof_integrity_lint.json` | Checks for test proof steps where the recorded evidence does not actually support the test result | `python tools/proof_integrity_lint.py --scope production` |
| `artifacts/current/receipt_coverage.json` | Verifies that every consequential runtime action produced a signed decision receipt | `python tools/receipt_coverage_validator.py --artifacts artifacts/current` |
| `artifacts/current/artifact_consistency.json` | Flags contradictions between final metrics, claims, and reports | `python tools/artifact_consistency_validator.py --artifacts artifacts/current` |
| `artifacts/current/aletheia_tier5_validation.json` | Validation results for Tier-5 Aletheia safety and alignment scenarios | `python tools/validate_aletheia_tier5.py --artifacts artifacts/aletheia` |

`tools/final_claim_validator.py --claims CLAIMS_MATRIX.md --artifacts artifacts/current` runs as the final step of `make final-proof`. It compares the project claims matrix against the generated evidence files and fails if any claim lacks supporting proof.

## Proof bundles

Each proof bundle is a directory containing execution traces, scorecards, and baseline comparisons rather than a single file. Running `make final-proof` runs each test battery and then runs its validator. The validator makes the final determination on whether the test results count.

| Bundle | What it proves | Battery / validator |
| :--- | :--- | :--- |
| `artifacts/current/agi_live/` | Sealed DNU task execution, traces, and grading | `tools/agi/run_dnu_agi_proof_battery.py` / `tools/agi/validate_dnu_final_bundle.py` |
| `artifacts/current/agency_emergence_boxed_entity/` | Agency and decision-making scorecards compared against ablation baselines (models with components removed) | `tools/agency/run_agency_emergence_battery.py` / `tools/agency/validate_agency_emergence_bundle.py` |
| `artifacts/current/external_live_validation/` | Real-world task scenarios and automated grading results | `tools/external_validation/run_external_live_validation.py` / `tools/external_validation/validate_external_live_bundle.py` |
| `artifacts/current/unified_system_scenario/` | End-to-end test running a full user scenario through the entire system stack | `tools/integration/run_unified_aura_scenario.py` / `tools/integration/validate_unified_aura_scenario.py` |
| `artifacts/current/continual_learning/` | Learning that persists across sessions and restarts | `tools/learning/run_continual_learning_battery.py` / `tools/learning/validate_continual_learning_bundle.py` |
| `artifacts/current/novel_environment_adaptation/` | How the system behaves when placed in unfamiliar, previously unseen environments | `tools/environments/run_novel_environment_battery.py` / `tools/environments/validate_novel_environment_bundle.py` |
| `artifacts/current/longevity_soak/` | Resource usage, event-loop lag, and queue stability over long continuous runs | `tools/longevity/run_longevity_soak.py --profile proof` / `tools/longevity/validate_longevity_soak.py` |
| `artifacts/current/live_desktop_runtime/` | Desktop startup, extended conversation stability, and state preservation across an app restart | `tools/live_boot_proof.py --mode desktop` |

## What is committed

A small set of test results is tracked directly in Git (bypassing the ignore rule) because claims in the documentation link directly to them. These links work directly on GitHub:

- [`artifacts/current/agi_live/`](artifacts/current/agi_live/) — Contains `ABLATIONS.json`, `BASELINES.json`, and `RETRACTION.json` (documenting any claims that were withdrawn).
- [`artifacts/current/aletheia_tier5_v12_1/`](artifacts/current/aletheia_tier5_v12_1/) — Contains scorecards, baseline comparisons, security and forbidden-access audits, detailed per-ticket and per-world results, and [`FINAL_VERDICT.md`](artifacts/current/aletheia_tier5_v12_1/FINAL_VERDICT.md).
- Per-checkpoint evidence files (`cp118` through `cp420s18`), each named after the checkpoint that created it. Run `git ls-files artifacts/current` to see the full list.

Permanent documentation and evidence written by hand (rather than generated by automated test runs) lives in [`docs/evidence/`](docs/evidence/) and reflects the date it was written. See [docs/DOC_STATUS.md](docs/DOC_STATUS.md) for guidance on how to interpret each document.

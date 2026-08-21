# Aura artifact index

`artifacts/current/` is a build output directory. It is listed in
`.gitignore`, so nothing written there by a proof run reaches GitHub — the
files below exist only on the machine that produced them, and a link to one
from a browser is a 404. This page is the map from an artifact to the command
that writes it.

`make final-proof` writes the whole set in one pass. Individual gates write the
same report to `/tmp` when run alone, which is the faster loop while working.

## Core reports

| Artifact | What it holds | Written by |
| :--- | :--- | :--- |
| `artifacts/current/enterprise_gate.json` | Syntax, security, and wildcard-import findings against the enterprise baseline | `make enterprise-gate` → `/tmp/aura_enterprise_gate.json`; `make final-proof` → `artifacts/current/` |
| `artifacts/current/production_readiness.json` | Verification records for every production-readiness control | `make production-gate` → `/tmp/aura_production_readiness.json` |
| `artifacts/current/architecture_map.json` | Memory writes, state mutations, tool execution, and LLM call sites | `make architecture-map` → `/tmp/aura_architecture_map.json` |
| `artifacts/current/production_surface_lint.json` | Production code that bypasses a canonical gateway or spawns an unsupervised task | `python tools/production_surface_lint.py --scope production` |
| `artifacts/current/proof_integrity_lint.json` | Proof steps whose evidence does not support the step | `python tools/proof_integrity_lint.py --scope production` |
| `artifacts/current/receipt_coverage.json` | Whether every consequential runtime action produced a signed decision receipt | `python tools/receipt_coverage_validator.py --artifacts artifacts/current` |
| `artifacts/current/artifact_consistency.json` | Contradictions between final metrics, claims, and reports | `python tools/artifact_consistency_validator.py --artifacts artifacts/current` |
| `artifacts/current/aletheia_tier5_validation.json` | Tier-5 Aletheia scenario validation | `python tools/validate_aletheia_tier5.py --artifacts artifacts/aletheia` |

`tools/final_claim_validator.py --claims CLAIMS_MATRIX.md --artifacts
artifacts/current` is the last step of `make final-proof`. It reads the claims
matrix against the bundles above and fails on a claim no artifact supports.

## Proof bundles

Each is a directory of traces, scorecards, and baselines rather than one file.
`make final-proof` runs the battery and then its validator; the validator is
what decides whether the bundle counts.

| Bundle | What it proves | Battery / validator |
| :--- | :--- | :--- |
| `artifacts/current/agi_live/` | Sealed DNU task execution, traces, grading | `tools/agi/run_dnu_agi_proof_battery.py` / `tools/agi/validate_dnu_final_bundle.py` |
| `artifacts/current/agency_emergence_boxed_entity/` | Agency and volition scorecards against ablation baselines | `tools/agency/run_agency_emergence_battery.py` / `tools/agency/validate_agency_emergence_bundle.py` |
| `artifacts/current/external_live_validation/` | Real-world task scenarios and grader results | `tools/external_validation/run_external_live_validation.py` / `tools/external_validation/validate_external_live_bundle.py` |
| `artifacts/current/unified_system_scenario/` | One scenario driven through the whole stack | `tools/integration/run_unified_aura_scenario.py` / `tools/integration/validate_unified_aura_scenario.py` |
| `artifacts/current/continual_learning/` | Learning that persists across sessions | `tools/learning/run_continual_learning_battery.py` / `tools/learning/validate_continual_learning_bundle.py` |
| `artifacts/current/novel_environment_adaptation/` | Behaviour in an environment the system has not seen | `tools/environments/run_novel_environment_battery.py` / `tools/environments/validate_novel_environment_bundle.py` |
| `artifacts/current/longevity_soak/` | Resource use, event-loop lag, and queue stability over a long run | `tools/longevity/run_longevity_soak.py --profile proof` / `tools/longevity/validate_longevity_soak.py` |
| `artifacts/current/live_desktop_runtime/` | A desktop boot, a conversation soak, and continuity across a restart | `tools/live_boot_proof.py --mode desktop` |

## What is committed

A small set of results is force-added past the ignore rule, because a claim
elsewhere in the repository cites it. These are the artifact links that work
from GitHub:

- [`artifacts/current/agi_live/`](artifacts/current/agi_live/) — `ABLATIONS.json`,
  `BASELINES.json`, and a `RETRACTION.json` recording what was withdrawn.
- [`artifacts/current/aletheia_tier5_v12_1/`](artifacts/current/aletheia_tier5_v12_1/) —
  scorecard, baseline comparison, policy and forbidden-access audits, per-ticket
  and per-world results, and [`FINAL_VERDICT.md`](artifacts/current/aletheia_tier5_v12_1/FINAL_VERDICT.md).
- Per-checkpoint evidence files, `cp118`–`cp420s18`, each named for the
  checkpoint that produced it. `git ls-files artifacts/current` lists them.

Evidence written for the record rather than by a proof run lives in
[`docs/evidence/`](docs/evidence/) and is read as of its date. See
[docs/DOC_STATUS.md](docs/DOC_STATUS.md) for how to read which document.

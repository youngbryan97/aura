# Third-Party Validation Run - 2026-05-05

This document records the local validation pass for the independent Aura proof
suite request. It separates what was actually run locally from long-horizon,
hardware, sealed-eval, and human-judged tests that cannot honestly be completed
inside a single local coding turn.

## Implemented Fixes

- Added Aura-native governed System 2 search in `core/reasoning/native_system2.py`.
- Registered System 2 as `native_system2` / `system2_search` in the cognitive provider.
- Wired System 2 into:
  - `DeliberationController` candidate choice.
  - `Planner` candidate rescoring.
  - `CounterfactualEngine` autonomous alternative ranking.
- Added explicit System 2 receipts with algorithm, budget, seed, best path,
  runner-up paths, value scores, rejected branches, uncertainty, and Will receipt.
- Hardened paraconsistent contradiction detection for common semantic antonym
  contradictions such as cooperative/competitive and stance pairs like
  should/should-not.
- Labeled vision cues as `rough_attention_indicator` and added prompt guidance
  that camera-derived gaze/head pose are not emotional certainty.
- Added steering-vector provenance metadata so cached/runtime/fallback vectors
  cannot be silently represented as stronger extracted CAA evidence.
- Removed hard-coded contact/recovery email literals from prime directives and
  moved those values to local environment/secret-backed configuration.
- Preserved earlier validation fixes: recurrent-depth compatibility,
  context token estimation/trimming, hierarchical phi time budget, Volition
  Will receipts, proof-obligation sandbox rooting, safe harness subprocess
  behavior, and root `aura_cleanup.py` shim.

## Local Commands Run

```bash
python -m pytest tests/test_recurrent_depth.py -q --tb=short
python -m pytest tests/test_hierarchical_phi.py -q --tb=short
python -m pytest tests/architect/test_autonomous_architect.py::test_cli_auto_t1_runs_in_temp_repo \
  tests/test_context_attentional_gate.py::test_estimate_tokens_reasonable \
  tests/test_context_attentional_gate.py::test_compact_truncates \
  tests/test_hierarchical_phi.py::test_compute_under_time_budget \
  tests/test_launcher_polish_contract.py::test_launcher_cleanup_shim_exists_at_repo_root \
  tests/test_volition.py::test_check_soul_drives_connection -q --tb=short
python -m pytest tests/test_consciousness_expansion_gauntlet.py -q --tb=short
python -m pytest tests/system2/test_native_system2.py tests/test_validation_hardening.py -q --tb=short
python -m pytest tests/test_consciousness_depth.py tests/test_cognitive_systems.py \
  tests/test_interaction_signals.py tests/test_steering_ab.py tests/test_unified_will.py \
  tests/test_volition.py::test_check_soul_drives_connection -q --tb=short
python -m pytest tests/test_hierarchical_phi.py tests/test_context_attentional_gate.py \
  tests/architect/test_autonomous_architect.py::test_cli_auto_t1_runs_in_temp_repo \
  tests/test_launcher_polish_contract.py::test_launcher_cleanup_shim_exists_at_repo_root \
  tests/test_consciousness_expansion_gauntlet.py -q --tb=short
python tools/security_scan.py
python tools/behavioral_proof_smoke.py \
  --output artifacts/proof_bundle/2026-05-05-ivs/behavioral_proof_latest.json \
  --receipt-root artifacts/proof_bundle/2026-05-05-ivs/behavioral_receipts
python tools/proof_bundle.py --output-dir artifacts/proof_bundle/2026-05-05-ivs/proof_bundle
python -m pytest tests/ -q --tb=no \
  --junitxml=artifacts/proof_bundle/2026-05-05-ivs/pytest_full_final2.xml
```

## Final Local Results

- Full suite before final System 2 changes: `4359 passed, 8 skipped, 7 warnings, 1 subtests passed`.
- Consciousness expansion gauntlet: `10 passed`.
- Native System 2 + validation hardening slice: `22 passed`.
- Cognitive/interaction/will compatibility slice: `157 passed`.
- Prior-failure/gauntlet compatibility slice: `36 passed`.
- Security scan after contact-literal scrub: `passed: true`, `findings: []`.
- Behavioral proof smoke: `passed: true`.
- Proof bundle generation: `passed: true`, `all_files_generated: true`.
- Final full suite after all code/docs changes:
  `4381 passed, 8 skipped, 7 warnings, 1 subtests passed in 415.40s`.

Final full-suite rerun artifact:

```text
artifacts/proof_bundle/2026-05-05-ivs/pytest_full_final2.xml
```

Proof artifacts:

```text
artifacts/proof_bundle/2026-05-05-ivs/MANIFEST.json
artifacts/proof_bundle/2026-05-05-ivs/IVS_CASE_MANIFEST.json
artifacts/proof_bundle/2026-05-05-ivs/behavioral_proof_latest.json
artifacts/proof_bundle/2026-05-05-ivs/proof_bundle/MANIFEST.json
```

## Scope Notes

The pasted independent suite contains tests that are intentionally not local
unit tests: 4-hour, 24-hour, 48-hour, 72-hour and 1-week runs; hardware spoofing
on a second Apple Silicon Mac; live camera/mic human scoring; sealed hidden eval
packs; external reviewer validation; public benchmark runs; and independent
human/blind-judge studies. Those are tracked in
`artifacts/proof_bundle/2026-05-05-ivs/IVS_CASE_MANIFEST.json` as
`pending_external_or_long_run`, not counted as local passes.

The honest standard is: runnable local tests must pass; nonlocal proof items
must remain explicit pending obligations with blockers and required artifacts.

The proof-bundle readiness check still marks `CAA_32B_RESULTS.json` as not
fully ready because no live production-32B behavioral A/B results were supplied.
The vector artifacts and geometry checks are present, but the strongest claim
that hidden-state steering beats rich textual controls on held-out live tasks
remains a separately tracked proof obligation. This is intentionally not
converted into a local pass.

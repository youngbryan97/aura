# Runbook: Research Core Stalled

## Symptoms
- `research_core.json[available]` is `false`, OR `status.iteration` has not advanced for >24 hours.
- `receipts.json[counts][governance]` shows no new `checkpoint_promotion` entries despite uptime.
- `audit_chain/info.json[verify].ok` is `true` but the chain length is flat over the same window — promotion gate is not running.
- `recent_cycles` in `research_core.json` empty or missing.
- `logs/` may show `tenant_boundary mismatch` or `research_core` initialization errors.

## Diagnosis
- Verify the core is registered: `python -c "from core.container import ServiceContainer; print(ServiceContainer.get('research_core', default=None))"`
- Confirm tenant stamp matches: `cat ~/.aura/research_core/tenant.json` (or whatever workdir is configured).
- Check the prediction ledger advanced recently: `python -c "from core.runtime.prediction_ledger import get_prediction_ledger; print(get_prediction_ledger().count())"`
- Run a single cycle manually: `python -c "from core.research_core.registry import register_research_core; r = register_research_core(); print(r.run_cycle().to_dict())"`
- Inspect the holdout vault: `python -c "from core.research_core.registry import register_research_core; r = register_research_core(); print(r.vault.size())"`

## Safe mitigation
- Re-trigger one cycle manually using the snippet above. A successful cycle re-establishes baseline.
- If the gate is rejecting every candidate, widen `cfg.max_regression` or set the `critical_metrics` to a smaller subset until the model recovers.
- If the LatticeLM is producing NaN losses, restart from the last known-good checkpoint via `LatticeTrainer.load_checkpoint(...)`.

## Unsafe mitigation (last resort)
- Wipe the holdout vault: only after explicit user confirmation. Loses the contamination history.
- Reset the novelty archive: `core.novelty.reset()`. Future unknown-task generation will start over.
- Force-stamp the workdir under a different tenant: `TenantBoundary(...).stamp(force=True)`. Loses the install identity.

## Rollback
- Reload the previous checkpoint via `LatticeTrainer.load_checkpoint(path)`.
- Reverse the most recent `GovernanceReceipt` of kind `checkpoint_promotion` by manually setting `gate.baseline = previous_baseline` from the receipt's metadata.
- Re-emit a `state_mutation` receipt noting the rollback so the audit chain reflects the manual step.

## Verification
- `aura doctor --bundle` and inspect `bundle_manifest.json` plus the fields named in Symptoms above.
- New cycles emit fresh `governance` receipts with `domain=checkpoint_promotion` and the audit chain head advances.
- `research_core.json[recent_cycles]` shows fresh entries with `started_at > pre-incident timestamp`.

## Postmortem checklist
- Add a regression test under `tests/test_research_core.py` covering the failure mode.
- Update `docs/AURA_RISK_REGISTER.md` with the failure class.
- If the cause was a metric regression, consider whether the metric belongs in `critical_metrics` or should move to non-critical.

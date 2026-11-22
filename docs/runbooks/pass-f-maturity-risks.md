# Runbook: Pass F Maturity Risks

## Symptoms
- An action returns success but lacks durable effect evidence, acceptance criteria,
  or a user-visible result that matches the request.
- Health, readiness, or proof status is green while chat, desktop, browser, or
  visible demo paths remain blocked.
- Resource pressure causes repeated model, worker, repair, or planner loops
  instead of admission control, backoff, or explicit degradation.
- Old objectives, memories, or proof artifacts keep steering current work.
- Internal event streams are noisy enough that blockers, receipts, and next
  actions are hard to identify.

## Diagnosis
- Run `python -m pytest tests/test_action_depth_honesty.py -q`.
- Run `python -m pytest tests/test_reliability_hardening.py -q -k "pass_f or FMEA or Traceability"`.
- Inspect `get_fmea_registry().faults_above_rpn(30)` and confirm Pass F entries
  remain visible.
- Compare live-path readiness to proof readiness before treating a green health
  result as demo-ready.
- Check the latest closeout tracker entry for explicit blockers, not just
  mechanical codebase hash or line-count results.

## Safe Mitigation
- Downgrade shallow successes to `success_unverified`, `partial_success`, or
  `failed_recoverable` and rerun with effect receipts.
- Quarantine contaminated proof artifacts and rerun the proof step through
  `tools/run_proof_step.py`.
- Prefer admission control, resource governor degradation, and circuit breakers
  over spawning more work under pressure.
- Surface browser/desktop permission blockers explicitly and do not substitute
  fixture proof for visible proof.
- Record semantic review gaps in the closeout ledger instead of treating the
  mechanical audit as completion.

## Unsafe Mitigation
- Marking a task complete because a tool fired once.
- Treating `proof_readiness_healthy` as chat or browser demo readiness.
- Reusing prior proof artifacts without fresh run metadata.
- Suppressing noisy streams without preserving the actionable state summary.

## Rollback
- Revert the change that introduced the shallow success, proof contamination,
  or loop, then rerun the focused test named in Diagnosis.
- If rollback is not local to one patch, restore the last pushed checkpoint and
  repeat production and enterprise gates before continuing closeout work.

## Verification
- `python -m pytest tests/test_reliability_hardening.py -q -k "pass_f or Traceability"`
- `python -m pytest tests/test_action_depth_honesty.py -q`
- `make production-gate`
- `make enterprise-gate`

## Postmortem Checklist
- Add or update the FMEA row, mitigation path, and runbook when a new maturity
  risk is found.
- Add a regression test that proves the failure is detected before user impact.
- Update `docs/AURA_EXECUTION_TRACKER.md` with the blocker, mitigation, and
  validation evidence.

# R02 Successor Identity

## Repair

The reboot waiter inherited AURA_RUNTIME_SOURCE_* from its predecessor.
The successor reused those values, incorrectly comparing its newly loaded
source against the previous boot. Commit ddca74a61 captures the successor
snapshot after the predecessor exits and before exec. Signed app-build
provenance is preserved. Ordinary child processes still reuse their parent's
snapshot; only an explicit new launch recaptures it.

## Live Evidence

On 2026-09-06, authenticated POST /api/reboot returned replacing_pid=96426
and waiter_pid=99362. The predecessor was idle and ready before shutdown.
The successor became the sole port-8000 listener, with PPID 1, PGID 99362.

Three observations at runtime ages 68.3, 80.8, and 88.5 seconds all showed:

- ready=true, process_id=99362, source_current=true.
- Expected and actual commit ddca74a61db8e95f7c459d11b34b36d0570fbd30.
- Matching workspace and shell hashes; revision issues empty.
- Same model path before and after reboot:
  Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15.
- runtime_identity_ok=true, no conversation readiness blockers.

Boot logs record identity and substrate restoration, value graph restoration,
four interrupted goals restored, and CanonicalSelf v192193 restored with
20 deltas. The prior spoon-conductivity chat remains in the durable delivery
journal as completed, terminal timestamp 1788741863.082516, response hash
3249a7060e23404d39f796ac5020b3923d8a4d64fce97a29db027395ab73aeda.
The persisted response still contains the spoon explanation.

This direct launch does not claim a signed app-build certificate:
verified=false, required=false, source_verified=true. No signature gate was
weakened to obtain readiness. Later source edits can legitimately report drift
until the next restart; these measurements apply to the named revision.

## Checks and Boundaries

43 focused provenance/reboot tests passed; smoke 164 passed, 1 skipped;
Ruff passed on the changed Python files. Tests cover capture ordering,
snapshot reuse, new-launch refresh, and preservation of app provenance.

This is bounded launch/update/restart proof, not an endurance soak or proof
that every capability is correct. R03-R11 remain separately tracked, including
streaming, semantic answer quality, health policy, and other on-loop writes.

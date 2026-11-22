# Runbook: stale memory retrieval

Covers [F09](../../KNOWN_FAILURE_MODES.md).

Nothing errors here. She just answers using context that stopped being
relevant, which is harder to notice than a crash.

## Symptoms
- Retrieved context is irrelevant or out of date.
- Recall quality metrics drop while every health check stays green.
- User-visible: she brings up something settled weeks ago.

## Diagnosis
1. `aura verify-memory` — memory facade integrity.
2. Compare recall quality against the faculty model's memory metric. Per the
   self-model, memory measures **recall, not whether the machinery is up** —
   a healthy subsystem with poor recall is exactly this failure.
3. Check for index drift: embeddings written under an older model than the
   one currently loaded.

## Safe mitigation
- `aura rebuild-index` — rebuild the vector index.
- Run a consolidation cycle.

## Unsafe mitigation (last resort)
- Do not purge memory to "start clean." You lose the record and keep the
  drift, since the next writes go through the same path.

## Rollback
If a rebuild makes recall worse, restore from backup (`aura restore`) and
capture the before/after so the regression is attributable.

## Verification
- Recall@k back above its declared floor in the faculty model.
- Spot-check retrieval on a topic you know the history of.

## Postmortem checklist
- If no probe could have caught this, the gap is instrumentation, not memory.
  An unmeasured faculty reports as a blind spot rather than as healthy.

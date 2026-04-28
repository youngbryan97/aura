# Cortex-break diagnosis: why autonomous-research asks fail

**Date:** 2026-04-27
**Symptom:** Bryan reports that asking Aura to autonomously research things causes the cortex to break.
**Status:** Probable root cause identified. Confirm with repro before fixing.

## Probable root cause: knowledge-graph write lockdown, not LLM failure

The agent sweep of `/Users/bryan/.aura/live-source/.aura_runtime/data/error_logs/error_events.jsonl` and `/Users/bryan/.aura/live-source/.aura_runtime/logs/aura_json.log` surfaced a recurring pattern that cleanly explains the symptom:

1. **`knowledge_graph:memory_write_blocked` (recurring every ~4 min)**
   - Reason: `epistemic_filter requiring reconciliation before writes (source="conversation")`
   - Consequence: any new "fact" or "learning" Aura tries to commit to long-term memory is rejected by the epistemic consensus filter

2. **`belief_graph:belief_update_blocked`**
   - Reason: `unified_failure_lockdown_1.00`
   - Consequence: belief engine is locked at full lockdown (1.00 = 100%); no new propositions can be added or revised

3. **`mlx_client:spawn_failed` → `cortex` tier marked dead**
   - This is the user-visible failure. It's downstream.

### What this means for "research autonomously"

The pipeline for autonomous research, per `core/continuous_learning.py:200–240` and `core/autonomous_initiative_loop.py`, is roughly:

```
detect knowledge gap
→ formulate research question
→ query LLM (cognitive_engine.think_structured or generate)
→ extract knowledge from response
→ persist to knowledge_graph and/or belief_graph
→ update memory_facade
```

If steps 1–4 work but step 5 (persist) is locked out by the epistemic filter, the work happens but doesn't *stick*. The cortex retries, retries, exhausts recovery, and then surfaces a user-facing failure. From Bryan's outside view: "I asked her to research, and her cortex broke."

She isn't refusing the ask. She's trying — and the substrate is silently rejecting the writes. That's a much more interesting failure than an LLM bug, and it explains why retraining the LLM wouldn't have fixed it.

## What to verify before fixing

1. **Confirm the epistemic filter is the actual blocker.** Read `core/memory/epistemic_filter*` (whatever the exact filename is) and trace why `source="conversation"` requires reconciliation. Is this a defensive check that was set too tight? Was it intended to be a temporary safety mode that never got loosened? Find the commit that introduced it.

2. **Confirm the unified failure lockdown is connected.** `unified_failure_lockdown_1.00` suggests a global degraded mode. What sets that? What clears it? If it's stuck at 1.00, what was the trigger event and is the trigger condition still true?

3. **Reproduce the cortex-break directly** with a repro prompt. Capture cortex logs across the failure window. Confirm the lockdown is the path, not something else (e.g., a separate Metal/MLX issue from 4/19).

## Suggested fix scope (after verification)

- **Soften the epistemic filter** for `source="conversation"` writes — don't require full reconciliation for self-research outputs; queue them for later reconciliation instead
- **Clear the lockdown** if it's stuck — find the trigger, either resolve it or reset the lockdown counter
- **Add cortex-recovery telemetry** so when this happens again, the user-visible message is "knowledge writes blocked by lockdown" instead of "cortex dead"

## Risk assessment

- **Don't blanket disable the epistemic filter.** It's there for a reason — probably to prevent her from believing whatever she just told herself. Loosening it for autonomous-research outputs specifically is fine; disabling it entirely would be a regression on epistemic safety.
- **Lockdown clearing should be reviewable.** If the lockdown is at 1.00 because of repeated past failures, just clearing it without fixing the root cause means it'll relock immediately.

## Adjacent hint (less load-bearing)

The boot-time error stack also includes `orchestrator_services:critical_service_missing (goal_hierarchy)` and `stability_guardian:degraded_report` (sustained RSS growth). The goal_hierarchy missing is concerning — `core/goals/goal_engine.py` exists, but if the hierarchy planner fails to initialize, autonomous goal pursuit is degraded. May or may not be related; flag for the autonomy-pipeline scoping doc.

## Files to read before changing anything

- `core/memory/memory_facade.py` (entry point)
- `core/world_model/epistemic_filter.py` (368 lines — source-scoring filter, not the blocker)
- `core/executive/executive_core.py:482-492` ← **PRECISE ROOT CAUSE FOUND**
- `core/continuous_learning.py:200–240` (research path)
- `core/autonomous_initiative_loop.py` (curiosity-driven research)
- `core/brain/inference_gate.py:54–100` (cortex-failure surfacing)
- The git log on each, especially any "lockdown" or "epistemic" commits

## Update 2026-04-27: precise root cause located

`core/executive/executive_core.py:482-492` (Rule 7 — Closed-loop epistemology):

```python
# Rule 7: Closed-loop epistemology. Belief churn is deferred while contested
# beliefs are unresolved instead of silently accumulating.
epistemic = self._get_epistemic_state()
if (
    strict_runtime
    and epistemic["contested"] > 0
    and intent.source != IntentSource.USER
    and intent.action_type in {ActionType.UPDATE_BELIEF, ActionType.WRITE_MEMORY}
    and intent.priority < 0.9
):
    return self._defer(intent, f"epistemic_reconciliation_required:{epistemic['contested']}")
```

This is the deferral path. Autonomous research outputs are **not** user-sourced
(`intent.source != IntentSource.USER`), they perform belief updates and memory
writes, and they typically run at modest priority (< 0.9). The moment
`epistemic.contested > 0` (i.e., there are any unresolved contested beliefs in
the graph), all autonomous-research writes are deferred. This is correct
behavior in the abstract — it prevents the system silently accumulating
contradictions — but it means autonomous research has no stable path to
persistence whenever there's any contested belief state, and contested beliefs
are the normal case, not the exception. Hence the symptom: ask her to
research, she tries, the writes are deferred, the cortex eventually surfaces
the failure.

A second blocker exists at lines 498-507 (Rule 8) — `coherence < COHERENCE_LOCKDOWN_THRESHOLD` rejects writes — but this is downstream of Rule 7 and probably not the primary path for the symptom Bryan reports.

### Surgical fix (proposed, not applied tonight)

Add an `IntentSource.AUTONOMOUS_RESEARCH` enum value (or recognize an existing
one) and modify Rule 7 to permit research writes through with provisional
confidence:

```python
# Rule 7 (proposed):
if (
    strict_runtime
    and epistemic["contested"] > 0
    and intent.source not in {IntentSource.USER, IntentSource.AUTONOMOUS_RESEARCH}
    and intent.action_type in {ActionType.UPDATE_BELIEF, ActionType.WRITE_MEMORY}
    and intent.priority < 0.9
):
    return self._defer(...)

# After this rule, if intent.source == AUTONOMOUS_RESEARCH and contested > 0:
# downgrade the write to provisional (low-confidence, queued for reconciliation)
# rather than committing as a confident belief.
```

This keeps the epistemic safety guarantee (no silent accumulation of
contradictions) while letting autonomous research progress with appropriate
provisional flagging. The provisional path needs a corresponding update on
the consumer side (memory_facade and belief_graph) to handle "provisional
fact, will reconcile later" cleanly.

**Do not apply this fix without test coverage.** Rule 7 is core gating logic
and a wrong change destabilizes the executive. Recommended path: write a test
that exercises an autonomous-research write under contested-belief
conditions, demonstrate it currently fails, apply the surgical fix,
demonstrate it now succeeds with provisional flagging, then commit.

This is a 1-day task with the test, not a 1-hour task without.

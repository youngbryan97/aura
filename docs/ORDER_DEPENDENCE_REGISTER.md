# Order-dependence register — triage and roots

Snapshot of the chunked-runner register (`tools/run_test_chunks.py` isolated-
retry pass) as of 2026-07-14, with root causes. Every entry here **fails
in-chunk, passes alone** — the defect is shared process state, not the
test's assertion.

**Count roots, not victims.** Two consecutive full runs produced different
victim sets from the same disease pool (run 1: gateway_enforcement ×7 +
local_agent_client ×3 + retrieval ×2 + 3 singles; run 2 after those roots
were fixed: llm_routing_tiering ×5, logging_integrity ×3, and others —
previously green tests). The register samples the pool per composition;
progress is measured by roots eliminated and by reproducible victims
staying gone across runs, not by one run's victim count.

The recurring disease has one shape: **a process-global singleton captures
state some earlier test mutated, and nothing re-heals on access.** Fixes that
worked: re-register/rebuild on access (autonomy_latitude, personality_engine),
expire *all* leaked tokens instead of one (desktop posture), stub ambient
governance out of unit tests whose subject is something else (FTS ranking).

## Fixed

| Victim(s) | Root | Fix |
|---|---|---|
| `test_autonomy_latitude::test_singleton_and_registration` | accessor registered only at first construction; container reset desynced it forever | re-heal registration on every access (8d7656bb) |
| `test_desktop_agency::test_automatic_posture_reversion` | expired only the first presence token in the shared authority gateway | expire all active tokens (8d7656bb) |
| `test_live_runtime_surface_regressions` ×4 | committed importer, uncommitted module (`core.runtime.resource_observation`) | module committed (46b997c4) |
| `test_local_agent_client_recovery` ×3 | `get_personality_engine()` promoted a container-registered test stub (SimpleNamespace) into the permanent module global | container is read-through, never cached; module singleton only caches what it constructs |
| `test_memory_retrieval_backbone::TestKnowledgeSearchIsRanked` ×2 | `add_knowledge` consults the live constitutional core; an earlier test left AuraNow present-state policy in `aura_now_defer` ("stabilization first"), so unit-test writes were denied and searches scanned an empty table | ranking tests stub `_approve_memory_write` (governance has its own tests) |
| `tests/test_gateway_enforcement.py` ×7 | same root as the personality family: the captured SimpleNamespace persona corrupted a downstream runtime path that ended in the global receipt store being closed mid-chunk (`emit()` → "receipt store is closed") | healed by the personality read-through fix — with it, chunk 3 runs 2115/2115 green and the store is never closed (probe plugin, zero transitions across two prior deterministic failures) |
| `test_sota_hardeners::test_extract_and_validate_args_resilience` | once an earlier test leaves `capability_engine` registered fail-closed (a real container boot does), the coercion-failure and malformed-JSON degradations in `capability_engine.extract_and_validate_args` escalated WARNING→CRITICAL and **raised** under production governance — so `limit="not_an_int"` crashed instead of returning `_error` | the two sites now pass `enforce_failure_policy=False` (the July-2026 pattern; sibling site already had it). Also a live pathology: bad skill params would mint a CRITICAL incident + existential-threat spike |

## Resolved upstream — `test_unified_will.py` (3, surfaced and cleared 2026-07-14)

Verified gone: a fixture-shaped-Will probe run over the full chunk-6 prefix
passed green with zero poisoning transitions, and a subsequent full chunk-6
certification passed 2001/2001 with an empty register. The pollution was
healed by intervening parallel-agent commits (runtime maintenance/agency
boundary hardening) rather than by a fix from this side. The probe method
below remains the recurrence playbook.

## (Historical) — `test_unified_will.py` (3, surfaced 2026-07-14 chunk 6)

`TestWillState::test_counters_track`,
`TestAllDomains::test_aura_now_defer_allows_read_only_observation_tool`,
`TestAllDomains::test_aura_now_defer_allows_explicit_user_memory_observation`.
Pass alone; fail in chunk 6. **Not** the §9d Ulysses covenant: proven by
forcing the global covenant + all seeds active — the failing decisions still
`proceed` with zero covenant constraints, and `test_counters_track` fails on
a **RESPONSE** decision, which `_consult_ulysses_covenant` skips by
construction (non-consequential domain). The register had **zero**
`test_unified_will` entries before the covenant work; these appeared only
after chunk 6's file composition changed (parallel-agent tests added).

Mechanism (partially localized): a benign `RESPONSE("good", source="user")`
decision comes back CONSTRAINED (`proceeds=0, constrains=1`) in-chunk. The
`will` fixture builds a *fresh* `UnifiedWill` with `_sample_aura_now_evidence`
neutralized, reading the **real global ServiceContainer** for scars, affect,
unity, welfare, and phenomenological modulation — so some earlier test leaves
one of those globals in a state that constrains even a neutral response. A
probe using the *global* `get_will()` did NOT reproduce it (different
instance, real aura_now), so the repro needs a fixture-shaped Will (fresh +
neutral aura_now). Next step for a focused session: a file-based teardown
probe that builds a fixture-shaped Will and records the first test after which
`RESPONSE("good")` constrains. Repro:
`tools/run_test_chunks.py --chunks 6 --only-chunks 6`.

## Open pool (sampled 2026-07-14 certification run)

Confirmed-fixed families stayed gone; the run surfaced a fresh sample:
`test_llm_routing_tiering` ×5, `test_logging_integrity::TestQueueHandlerOverflow`
×3, `test_live_runtime_surface_regressions::test_file_operation_write_creates_nested_live_runtime_directory`,
and `test_memory_facade_runtime::test_memory_ops_core_append_writes_to_block`.
The recurrent `test_sota_hardeners` victim — the only one present in both
runs — is now **fixed** (see table above). Remaining cluster shapes suggest
shared roots per file (routing: lane/endpoint state; logging: handler/alarm
throttle globals). These live in subsystems under active parallel
development; fixes need the owning context, not drive-by patches.

## Receipt-store family (7 victims) — how it was pinned

`tests/test_gateway_enforcement.py` failed in-chunk with
`RuntimeError: receipt store is closed` (`core/runtime/receipts.py` `emit()`)
in two consecutive full-chunk runs, while every partial reproduction passed:
files 0..105 executed alone, and full collection with only gateway selected.
A per-test probe plugin (report the exact test after which the global store
turns up closed) came back with ZERO transitions once the personality
read-through fix was in — same files, same order, 2115/2115 green.

Standing notes for any future receipt-store work:
- `get_receipt_store()` hands back the global even when closed; only
  `reset_receipt_store()` nulls it (heals). `close_receipt_store()` closes
  and deliberately keeps it — terminal shutdown semantics (f1c228e5,
  13b93fba). Preserve that: in-flight runtime shutdown keeps failing loud
  after close; only a fresh accessor call in a new logical run may rebuild.
- Collection leaves the store `None` (no import-time construction).
- Lesson: a "who closed X" symptom can have a root in an unrelated
  singleton — chase the earliest corrupted state, not the loudest error.

## Sampled 2026-09-10, while repairing the subject-core harness

Three victims found in selections run for other reasons, all with the shape
this register is about: green alone, red in company. Recorded here rather than
fixed in passing, because the root is in shared process state somewhere in the
selection and a drive-by patch to the victim would hide it.

| Victim | Selection that reproduces it | What is different in company |
|---|---|---|
| `test_core_affect_models::test_narrative_thread_start_seeds_snapshot_and_falls_back_task_tracker` | `pytest tests/ -k affect` | a degradation receipt is recorded where the test asserts none |
| `test_core_affect_models::test_narrative_thread_refresh_failure_writes_degraded_snapshot` | `pytest tests/ -k affect` | `NarrativeThread.get_current_snapshot()` returns the pending fallback, so `_current_narrative` is None where the refresh loop should have written a degraded snapshot |
| `test_cognitive_routing_runtime::test_substrate_handoff_failure_records_keyword_fallback` | `pytest tests/ -k "workspace or substrate or orchestrator_boot or consciousness_system"` | `_should_allow_deep_handoff` never calls the substrate extractor, so something earlier left the decision cached or the module stubbed |

| `test_executive_closure::test_closed_loop_coalesces_hierarchical_phi_refresh_tasks` | `pytest tests/ -k executive_closure` | `loop._hphi_task` is None where the test asks for a coalesced task, so an earlier test leaves the loop or its task tracker in a state where the refresh is not scheduled. Reproduced identically on a clean checkout at `36174cd12`, so it predates this session |
| `test_perception_keeps_up_while_she_acts` ×3 | `pytest tests/ -k "percept or broadcast"` | the perceptual compute budget reads 0.1 Hz where the test asks for 2.0, so an earlier test leaves the lane or generation state that `_compute_budget` consults |

Both affect files pass alone and pass as a pair, so that root is further up the
selection. `pytest tests/test_affect_behavioral.py tests/test_core_affect_models.py`
is green, which rules out the nearest suspect. The perception trio passes alone
and passes beside `tests/test_broadcast_consumers.py`, so the same is true of
it.

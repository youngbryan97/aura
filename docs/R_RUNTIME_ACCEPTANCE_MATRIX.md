# R Runtime Acceptance Matrix

This matrix groups the existing master obligations by causal contract. It
does not replace or close any obligation in AURA_1_0_MASTER_TODO.md.

## Execution order

1. Allocation and ownership: R03, R04, R05, R11. Trace one request's public
   and private capacities, prefill estimate, deadline, owner, progress and
   termination. Require the same accounting at admission and dispatch.
2. Persistence and health: R06, R07, R08. Exercise boot, active generation,
   concurrent writes, shutdown and restart. Separate state-lock contention,
   disk serialization, event-loop delay and actual worker failure.
3. Answer adjudication: R05, R10. Replay retained answers offline before
   spending another live generation. Pair valid paraphrases and code with
   incomplete answers, wrong calculations and unsupported effect claims.
   Lexical evidence alone cannot certify semantic correctness.
4. Delivery: R09, R11. Exercise disconnect/reconnect, idempotent replay,
   cancellation and follow-ups using the desktop transport. Compare the
   worker answer, durable journal and rendered final text.
5. Integrated replay: all remaining R items. Freeze the source identity for
   the run, then execute the matrix on one resident load. Correlate neural
   feed, worker logs, delivery records and visible effects.

## Closure evidence

Initial combined offline run: 88 tests passed across delivery journal,
delivery ceiling, shared disposition, semantic-program execution, latent
output quality and prose/code boundaries. Probe transport tests: 2 passed.
Smoke: 164 passed, 1 skipped. These results establish a regression baseline,
not live closure. The retained locking answer still demonstrates missing
semantic coverage in the lexical facet validator.

| Contract | Required positive cases | Required negative cases | Existing checks |
| --- | --- | --- | --- |
| Allocation | Short/long context; public/native thinking; measured/cold task bucket | Explicit insufficient budget; missing rates; stale rates | test_foreground_read_decode_allowance.py, test_latent_cortex_wiring.py, test_a_deadline_covers_reading_too.py |
| Ownership | Prefill/decode progress; one successor; retired handles | Dead worker; stale progress; cross-client replacement | test_mlx_client_resilience.py, test_mlx_retired_handle_liveness.py |
| Persistence | New pending write during old flush; settled producer before save | Failed write; cancellation; corrupt record | test_a_versioned_store.py, test_experiencer_shutdown_persistence.py, test_boot_profile.py |
| Adjudication | Equivalent explanations; code spans; exact executable results | Wrong result despite exit zero; missing requested result; fabricated action | test_latent_cortex_output_quality.py, test_prose_artifact_code_boundaries.py, test_semantic_program_execution.py |
| Delivery | One final; same final after reconnect; contextual follow-up | Duplicate submit; cancelled turn; interrupted client | test_chat_delivery_journal.py, test_chat_delivery_ceiling.py, live_desktop_conversation_probe.py |

These are entry points, not an assertion that the listed checks exhaust the
contract. Add missing executable cases before closing the corresponding R.

## Rules for this pass

- Fix one shared mechanism, then run all affected contract cases.
- Use offline retained-response replay to find validator defects first.
- Do not tune wording to obtain a live pass or expand keyword lists to fit
  a single answer. Reuse language and disposition contracts.
- Do not convert model confidence, HTTP 200, health readiness or process exit
  into semantic success.
- Every live attempt records its source identity and whether the intended
  mechanism actually ran. A fallback is not evidence for the requested lane.
- A new warning becomes a classified case in its owning contract, not an
  unrelated side project. Preserve unresolved cases in the master ledger.
- Mark each R only when its complete acceptance surface has receipts. A
  shared repair may close several items, but a unit test alone closes none
  of the live obligations.

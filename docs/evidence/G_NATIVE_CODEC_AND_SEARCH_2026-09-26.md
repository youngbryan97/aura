# Native codec and search checkpoint

The complete-epoch fold-2 control is finished. Its selected native candidate
scores 37/47 against incumbent 43/47, with zero gains and six regressions.
All 367 eligible training sources were visited. The independently checked
receipt is linked from the [native pilot record](G03_NATIVE_DECODER_SEMANTICS_2026-09-25.md).
Additional exposure did not produce a promotable candidate.

The role-relative register view now reaches source supervision, checkpoint
calibration, held-wording replay, fresh-schema scoring, constrained generation,
and unconstrained generation through one declared codec. Historical plans keep
their absolute wire. Unknown declarations refuse before model loading.
The shared transducer uses the same checked register conversion. The actual
resident tokenizer preserves the supervised register-identity spans.

Global native search reuses the typed grammar and retains alternatives to
local operation, reference, and termination decisions. Its best-first ordering
is proved only for a declared nonpositive local-decision objective. Receipts
distinguish unfinished search from top-k completion, preserve raw scores, and
record complete-graph reranking separately. Targets reach grading only after
generation and selection. This implementation does not establish model-backed
proposal reach or semantic accuracy.

Checks: 110 combined native/runtime tests passed; 32 shared-transducer tests
passed; the final trainer update passed 23 focused tests. Final smoke passed
164 tests with one skipped. The [design](../G_NATIVE_REGISTER_BINDING_DESIGN.md)
states representation reversibility and the remaining measurement obligations.

The next frozen source-only fit declares `role_relative_v1`, exact-length
prefix batches of four, 303 fit sources, 185 calibration cases, and 50 held
cases. Each observed batch size must pass full-logit and selected-projection
equivalence. Format, batching, and calibration coverage differ from the older
fold-0 fit; their combined result cannot isolate any one causal contribution.
The plan-only receipt loaded no model weights. Performance remains unmeasured.
G03-G05 remain open. Serving and fusion are unchanged.

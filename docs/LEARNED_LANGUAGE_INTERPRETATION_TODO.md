# Learned language interpretation — the todo

**Target.** Language interpretation is learned, calibrated and abstaining.

**Today.** Deterministic language rules plus one early learned semantic
substrate. Not the same thing, and the gap is measurable: `core/` holds
**2,210** compiled patterns, and the decisions that route a turn are made by
word lists.

This document is the work item. `docs/ONLINE_LEARNING_ROADMAP.md` §6 holds the
claim boundary and the measurements.

---

## 1. What the model can now do that it could not

`encode_hidden` (worker action, `core/brain/llm/mlx_worker.py`) added a role
the model did not have. Its roles were understand, reason, generate. This adds
a fourth that does not require it to speak:

    f(x) → h(x)

A reusable representation other systems consume directly. One causal forward,
no sampling, nothing to steer, no prompt. Measured on held-out paraphrases it
separates a decision better than a topical embedder does (AUROC 0.771 against
0.693) and cannot yet decide anything, because twelve examples do not make a
trustworthy boundary.

## 2. The debt, by site

| Site | What decides meaning today |
| :-- | :-- |
| `core/intent/declared_capability.py` | `_VERB_CLASSES` (run/execute/compute, search/find/browse, write/create/build), `_OBJECT_CLASSES` (code/script/python, file/document/path), `_PRO_VERBS`, `_DISCOURSE_LEAD`, `_ADDRESSES_THE_LISTENER`, handcrafted mood logic |
| `core/reasoning/positional_constraints.py` | English → formal constraints by regular expression: relation phrases, name detection, question detection, population counts |
| `core/runtime/desktop_objective_intent.py` | 17 patterns deciding actuation against observation against construction |
| `core/conversation/response_reliability.py` | 174 patterns, including the action-claim verb list this work started from |
| `core/phases/response_contract.py`, `core/phases/dialogue_policy.py` | 41 and 44 patterns shaping what a reply must contain |
| `core/conversation/chat_preflight.py`, `core/conversation/request_coverage.py` | 52 and 30 patterns deciding what was asked |

Every one has been wrong the same way at least once: a phrasing nobody
enumerated. Each repair widened a pattern, which is the move that will be
needed again.

## 3. The surfaces to learn

Each is one decision, independently calibrated, abstaining, trained on
receipts rather than on a list. In dependency order — the earlier ones supply
labels the later ones need.

| # | Surface | Ground truth available today |
| :-- | :-- | :-- |
| 1 | promise against completed action | tool receipts — **first consumer, shipped** |
| 2 | tool requirement | whether a capability was dispatched and succeeded |
| 3 | request against statement | whether the turn produced an effect |
| 4 | task state | whether an objective's acceptance contract passed |
| 5 | observation against inference | whether a grounding reading backed the claim |
| 6 | hypothetical against factual | whether the reply survived the reliability gate |
| 7 | correction against new instruction | whether the next turn repaired the last |
| 8 | uncertainty | calibration records in `core/evaluation/` |
| 9 | intent | which capability the turn eventually used |
| 10 | relevance | whether injected evidence reached the answer |
| 11 | contradiction | belief-revision records (`core/self/belief_history.py`) |
| 12 | appraisal categories | affect records, weakest ground truth — last |

## 4. The bar each must clear

The protocol exists: `core/language/substrate_measurement.py`, frozen set in
`config/`, receipt in `artifacts/language_substrate/`.

1. Fitted on declared examples only, by leave-one-out.
2. Scored on held-out wordings that are never examples and sit near the
   boundary.
3. AUROC, F1 and false-positive rate reported; abstentions reported as
   themselves, never as errors and never dropped.
4. A boundary is used only when its gap exceeds the spread of the examples it
   was drawn from.
5. It replaces a pattern only after it beats that pattern on the pattern's own
   declared examples. Until then it may only ADD, never remove.

## 5. What must not change while doing this

* **Abstention.** A surface that guesses in the middle is worse than the
  pattern it replaces.
* **Fail-open.** No worker, no boundary, no vectors — the caller keeps what it
  did before.
* **Off the critical path.** Deciding costs a forward pass; a turn answers
  from what is known and queues the rest.
* **No prompt engineering.** These read activations. Nothing is written to
  steer the model, and nothing asks it to classify in words.

## 6. Nearest blocker

Labels. Every surface above needs more than a dozen examples before its
boundary is usable, and the receipt-teaching path is what supplies them from
live traffic. Durable storage landed 2026-08-20, so they now accumulate across
restarts; the first surface's boundary should be re-measured once it has a few
hundred.

The other known blocker is per-sentence hidden states. A paraphrase is still a
first sighting because deciding one costs a forward pass the turn cannot
spend — while the model computed exactly those states to write the sentence,
and nothing keeps them.

# G03 paired methods and advisory review

The source of suggestions is Bryan's `Suggestions.pdf` and the subsequent
active-arbitration note. This is a development analysis, not a serving change
or a fresh transfer result. The immutable paired receipt is
`~/.aura/rlc-evidence/semantic-real-bank-mixed-fold0-20260922/method-overlap-v2.json`.
`tools/analyze_semantic_method_overlap.py` verifies each input receipt and
candidate-bank identity before comparing the four selected programs on the
same item. Outcomes are graded only after choices are frozen.

| Slice | N | Incumbent | Ranker | Direct | Operation probe | Any right | All wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source fold 0 | 24 | 10 | 15 | 23 | 19 | 24 | 0 |
| Exposed validation failures | 23 | 0 | 5 | 7 | 3 | 9 | 14 |

The operation probe is uniquely right on one source item and two exposed
failures. Direct is uniquely right on two exposed failures. These slices have
different selection histories and cannot be pooled into a production rate.
The exposed 23 are generated validation variants 3-5 selected because an
earlier portfolio failed, not independent live-user traffic or increased
program depth. Their correct programs are in the bank; the four methods all
choose wrongly on 14. The problem there is selection, not bank availability.

Agreement is not semantic evidence. A unique plurality among method programs
gets 22/22 on the source fold but only 4/17 on the exposed failures. Ranker
and direct agree on 12 exposed items, only five correct. Thus the proposed
"trust consensus" or "probe agrees with ranker" promotion would be unsafe.
Pairwise exact McNemar counts and Wilson intervals are in the receipt; small
samples do not establish a universal ordering.

## Disposition of the suggestions

- **Adopted:** item-level paired overlap, unique wins, shared failure,
  per-construction counts, exact paired comparison, uncertainty intervals,
  and immutable receipt checks. The operation probe remains a separate
  candidate selector with no serving authority.
- **Adopted in the existing architecture:** conflicting executable programs
  remain in `SemanticProgramPortfolio`. Its full bounded counterfactual probe
  set now reaches `plan_program_inquiries`, which executes every candidate
  under the same inputs and ranks questions by expected information gain.
  Only a separately observed result can reconcile a pending inquiry. The
  source request, program identities, predictions, and observation provenance
  are bound and checked. This implements active discrimination without
  treating a simulation as its own answer.
- **Already present but not a correctness oracle:** resident-model
  deliberation, episodic memory, local reference retrieval, and optional
  governed web access can propose interpretations or supply source material.
  Re-asking one model is correlated with its first proposal; search snippets
  and model summaries can descend from the same assumption. Neither a corpus
  hit nor repeated model agreement proves which synthetic program means the
  request. An unavailable web connection must not block local reasoning.
- **Deferred until independent evidence exists:** confidence calibration,
  learned routing, selective canary, and promotion. A new frozen
  policy-validation set and an untouched final test must be grouped by
  construction and include ordinary, hard, and boundary cases. The policy,
  costs, abstention behavior, worst-stratum criteria, and stopping rule must
  be fixed before the final test. The exposed 23 can diagnose failures but
  cannot tune the router or certify transfer.
- **Rejected:** a hard rule for the two probe-only wins, arbitrary 0.6/0.15
  routing thresholds, fixed review/rollback percentages, counting execution
  or type checking as proof of intended semantics, voting correlated methods
  into truth, and a 200-tree meta-learner fitted to 47 heterogeneous cases.
  The prototype selects an existing candidate; it is not a candidate
  generator. A wrong premise cannot become an independent measurement by
  being sent to the model, a search query, or a counterfactual simulator.

The next mechanistic gap is source-grounded discrimination of the 14 shared
selection failures. If no independently checkable consequence distinguishes
the retained hypotheses, the correct output is unresolved, not a fabricated
high-confidence pick. G03 and G04 remain open.

The v2 diagnostic compares the 14 shared-failure selections against frozen
target programs only after choices are made. Direct decoding keeps the right
operations but reverses a register role on nine; the other five keep the
roles but choose a wrong operation. The ranker has five binding-only, five
operation-only, and four combined misses; the operation prototype has three,
eight, and three respectively. These are descriptive labels from exposed
development targets, not routing features or newly acquired answer evidence.

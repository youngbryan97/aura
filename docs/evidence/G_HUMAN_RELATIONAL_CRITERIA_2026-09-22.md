# Human relational criteria for the G ledger

Bryan's 2026-09-21 and 2026-09-22 feedback is design input, not a source of
test labels. The frozen development and prospective cohorts must not be
rewritten to match these examples. His examples make the following distinctions
operationally important.

| Human distinction | Existing mechanism to reuse | Required test before claim |
| --- | --- | --- |
| Shared causal structure can transfer across unlike surfaces, but shared words or a component skill may not (scheduling/code versus boxing/MMA). | Typed programs, relational structure mapping, and source construction folds. | Hold out both surface and construction; test the changed toolset, goal, and constraints as negative analogies. |
| Plausibility is not proof; test claims against consequences and seek independent evidence (the salary/car example), while remaining revisable when direct evidence defeats a prior. | Source-grounded candidate banks, counterfactual witnesses, shared verification and knowledge provenance. | Score every plausible interpretation before the answer is revealed; measure revision after a decisive observation without treating unsupported suspicion as a contradiction. |
| Partial methods can contribute different useful pieces without any one method being sufficient. | Mixed candidate proposals, direct neural program decode, learned source ranker, and the universal executable floor. | Measure candidate union, selection, gains, regressions, and compute on the same tasks; ablate each method. A recovered failure-only subset is not net gain. |
| Claims and evidence carry reasoning; elegant commentary and analogies do not establish it. | Typed execution receipts, independent outcome checks, and relational evidence records. | Audit the intermediate claim, its evidence, and the inferred consequence separately. A metaphor or joke may be a transfer probe only if its implied mapping survives a counterexample. |
| A useful principle survives genuine attempts to falsify it in independent contexts; remembered phrasing does not. | `RelationalGeneralizer` and retained operator interventions. | Freeze a learned abstraction, change vocabulary and domain while preserving relations, then break a relevant relation. Expect transfer only in the first condition. |
| When representations fail, decompose the missing information and ask what would discriminate alternatives. | G03 candidate-stage diagnosis and source-only contrast generation. | Attribute errors to absent candidate, wrong selection, grounding, execution, or public answer before changing a component. |
| Conversation continuity is part of the observed quality of one agent. | Episodic/discourse memory and live turn receipts. | Evaluate multi-turn reference, repair, and identity continuity separately from typed-program accuracy. |

## Implemented interpretation of the feedback

`RelationalGeneralizer.scrutinize` checks each explicit consequence of a
plausible interpretation against an independently identified observation. It
does not turn an unobserved consequence into a contradiction.
`plan_discrimination` identifies which missing observations could distinguish
the interpretations and marks one-sided predictions as non-decisive. These
operations instantiate Bryan's salary/car and "ask why/how" examples without
hardcoding either domain. The caller must still obtain and validate the
observation; this module does not claim that fluent or confidently stated
premises are true.

`assess_revision` requires observations explicitly scoped to the old and new
times. It distinguishes evidence for an initially mistaken reading from
evidence for a changed fact, leaving missing or contradictory histories
unresolved. `PrincipleCandidate` now needs distinct evidence sources as well
as distinct contexts and a falsification attempt before consolidation.

`probe_transfer` reuses the existing structure mapper and scrambled null,
reports unmatched source and target relations, and requires a sourced role
correspondence when domains use different role types. The scheduling/code
analogy and a case with an extra target constraint are contrastive tests. A
matched structure is still not evidence of matching goals or task success.
Metaphor and humor can be evaluated through such relation-preserving and
relation-breaking probes, but these tests do not certify joke generation or
social pragmatics.

The G03 source experiment keeps source-bank alternatives and source-only
counterfactual contrasts as separate training views. The comparison tool
passes ordinary, ranker, and direct-decoder proposals to the pre-existing
`SemanticProgramPortfolio`: it executes each under the universal floor,
records disagreements, and plans independent inquiries. It separately counts
the oracle union after acquisition. Portfolio execution is not semantic
correctness, and an oracle union is not a label-free selection policy.

The relational evidence adapter is not yet a general live conversation
resolver. The G03 source experiment is pilot-only and has no serving authority.
Those limitations are retained on purpose; neither the human examples nor
the source-bank labels should silently become production rules.

For G03, the immediate experiment combines real retained alternatives with
source-training counterfactual contrasts. The source bank is fully acquired
before labels are used. Both views share the same input grounding, but the
contrast labels are source-training evidence only. This does not supply the
unknown answer to a new request. The existing language substrate is reused;
this note does not authorize a second one.

For G04 and G09, an analogy is a hypothesis about preserved relations, not an
equivalence certificate. New held-out tasks must separately test relation
preservation and a deceptively similar but invalid transfer. For G05-G08,
independent checks and public answer correctness remain distinct from internal
program selection. For G10-G12, live authority and frontier claims require
their own matched controls and prospective tasks. None of these criteria
alone establishes broad gain, developmental generality, or consciousness.

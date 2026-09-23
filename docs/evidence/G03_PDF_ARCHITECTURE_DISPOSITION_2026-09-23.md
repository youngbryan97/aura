# G03 external architecture proposals: disposition before broad evaluation

Reviewed in full: `Frozen_Transformer_Thesis.pdf` (11 pages), `Solutions.pdf`
(108 pages), `Solution2.pdf` (14 pages), and `Solutions3.pdf` (24 pages),
supplied on 2026-09-23.
They are advice, not training labels or evidence that a candidate works. This
record separates built mechanisms, existing mechanisms, later experiments,
and unsupported claims. No validation examples are used to design a selector.

| Proposal | Disposition and code owner |
| --- | --- |
| Preserve mention identity separately from current value and score complete role-to-mention bindings | Built as an opt-in extension of `ContextualProgramRanker`. It scores each operation/argument slot against the selected register and carries a source-span identity for inputs and intermediate registers. The legacy ranker remains checkpoint-compatible. This establishes representational capacity, not learned semantic correctness. |
| Optimize the meaning of a complete program, admitting multiple correct programs | Already present in `candidate_set_loss`, trained on the retained candidate bank with independently checked source-training labels. The candidate bank and floor preserve typed execution; neither is an oracle for the intended wording. |
| Train on hard competitors in the same operation-span distribution as runtime | Built `bank_factors_runtime`: source-witnessed role and operation contrasts are paired with answer-blind operation spans retained by the source bank. Each source, regardless of chart count, makes one average-gradient update. The runtime-chart local-head refit is a separate opt-in experiment. |
| Use counterfactual intervention to distinguish apparently equal programs | Already present in `semantic_graph_counterexamples`, source factor contrasts, and `semantic_program_inquiry`. Probe predictions choose a question; only independently acquired outcomes can resolve meaning. |
| CEGIS, rejected-class memory, and procedure/macro retention | Existing counterexample, source arbitration, procedure induction, and procedure registry paths provide parts of this. A unified cross-domain developmental loop and fresh-task causal reuse are not established by G03 and remain later G-ledger work. |
| Recurrent neural controller over an identity-preserving workspace | Research direction, not an immediate G03 repair. The current frozen/qualified recurrence evidence does not prove that a new controller learns useful state transitions. Require process supervision, depth scaling, matched compute, lesion/rescue, and fresh families before serving. |
| Program extraction plus exact execution | Already represented by `SemanticProgramIR`, the universal floor, and candidate-bank execution. Extraction may fail; exact execution proves the computed program's result, conditional on correct interpretation. |
| Damped frozen-middle-layer recurrence and latent workspace | Worth a bounded separate experiment with matched compute and causal lesions. Do not turn it on from an analogy or run it in place of the measured selection repair. |
| Spectral normalization, log-det rank loss, and a purported guaranteed contractive symplectic flow (`Solutions.pdf`, pp. 99-108) | Reject as stated. Bounding an adapter's weight matrices does not bound the entire frozen-transformer/attention/normalization/residual transition. The proposed `-log det(G + epsilon I)` is finite at rank collapse for positive epsilon; it is not an infinite barrier. A damped flow is not symplectic in the asserted sense. The listed throughput and complexity separations are not measured on Aura's 27B. |
| High cosine similarity proves contraction in 1-2 passes (`Frozen_Transformer_Thesis.pdf`) | Reject. Similarity of consecutive states does not bound distances between trajectories. Even a proven contraction factor of 0.9994 needs about 1,155 steps to halve error. The fixed-state argument is a space bound, not a ban on repeated computation. `Solution2.pdf` independently identifies these defects. |
| Universal floor plus learned macros guarantees exponential practical search speedups | Treat as a testable conditional claim. A shorter description can improve Levin-style ordering, but realized speed depends on the search distribution, interpreter cost, macro retrieval cost, and the tasks. No `2^-DeltaL` gain is credited without measured paired fresh-task reach. |
| Safe baseline means no regressions | Reject as a universal claim. Keeping an incumbent prevents replacement only where independent evidence can certify the alternative; natural-language interpretation is not always independently verifiable. Paired full-denominator non-regression remains required. |

## Solutions3: searchable compiler and source evidence

This PDF combines several independent reviews. Its proposed downloadable patch
is not attached here. The implementation below was made against the inspected
repository, not copied from the claimed patch.

| Proposal | Disposition and owner |
| --- | --- |
| Preserve the optimizer's selected argument mentions and definitions through complete-program selection | Built in `semantic_candidate_bank.py` as a receipt-bound v3 bank. The incumbent retains its IR argument spans and register-definition anchors, explicitly marked as anchors rather than optimizer-selected attachments. Alternative graphs retain selected mentions and definitions in topological execution order. `semantic_bank_replay.py` reads v2 and v3. A v2 bank cannot pretend to carry this evidence. |
| Learn a complete-program relation over the exact runtime evidence | Built as an opt-in argument-evidence channel in `ContextualProgramRanker`, with zero initial influence and a same-program, differing-mention lesion test. The ranker receives the operation, role, chosen mention, chosen definition when available, and register source identity. `evaluate_semantic_candidate_ranker.py` admits the channel only with real v3 bank training, not source-annotated synthetic rivals. Old checkpoints remain separate. This proves capacity and wiring, not held-out accuracy. |
| Preserve distinct evidence realizations for one program | Built in `_rankable(..., preserve_evidence=True)`. Legacy program-hash deduplication remains for old rankers. The source-only training labels still grade program meaning; they do not certify that one of two equivalent-program mention paths is the uniquely intended explanation. |
| Train on runtime-proposed charts and balance by source | The existing chart-view refit and new source-fold runtime factor mode cover this. Sources without any witnessed contrast are explicitly excluded from updates and counted. Real-bank evidence training and synthetic factor training remain different arms. Neither arm is a qualified result. |
| Search the candidate bank with an exact executor as if it were 2048 | Reject as a correctness claim. Execution determines the output of a proposed program; it does not provide a reward for fidelity to the sentence. MCTS over a fixed finite bank cannot create a missing proposal or repair an uninformative scorer. Existing graph search and inquiry remain useful for reachability and discriminating observations. |
| Train through executor reward with Gumbel-softmax | Reject as stated. The proposed indicator reward is discontinuous in a discrete program, and using the answer value as a semantic label admits accidental equality. The existing candidate-set objective uses independently verified source-training program labels and admits multiple correct programs. |
| Add a holistic text-program scorer | The existing `ContextualProgramRanker` already encodes the whole request and complete graph. The new evidence term adds the missing local role/mention relation. A second standalone scorer would duplicate that path without independent evidence. |
| Increase compute when uncertain, retain hard cases, ask a discriminating question | Existing `semantic_program_inquiry`, counterexample search, value-of-computation, and source-fold error receipts cover these mechanisms in part. Search depth, retention, and simulated counterfactuals are not correctness oracles. An inquiry resolves meaning only after an independent answer/observation. Calibration and live activation remain open. |
| Delexicalize all numbers, randomly jitter spans, and use a new Torch decoder | Reject those implementations. Masking numbers can erase task information; random jitter can cross a mention or operation boundary and create false supervision. Target-blind charts produced by the actual decoder give a measurable runtime-span distribution. The claimed O(1) batched ranker and sub-five-second epoch are not justified by the supplied code or host measurements. |
| Add mandatory proposal, selection, execution and public-answer attribution | Existing `semantic_failure_diagnosis.py` and bank receipts separate these stages. The v3 evidence now makes mention retention inspectable. Holdout reports must keep reachability, conditional selection, paired gains/regressions, and incomplete search separate. |

The PDF states `end-to-end accuracy <= r*a`, where `a` is conditional
selection accuracy given reachability. With those definitions the expression
is an equality, not an upper bound. The 84/84 annotated-span probe cannot
estimate runtime `a`. Its MCTS regret rate and claimed 10-30-point gains are
predictions without proof here; the 572/572-to-0/16 observation is not a
measurement of total-variation distance. We will not promote on these claims.

Before this evidence-preservation build, short source-fold pilots had already
falsified a simple ranker fix: an identity-binding bank-factor arm scored 6/24
against a 10/24 incumbent, and a normalized-pointer variant scored 4/24.
The new v3 bank needs fresh source-only acquisition. Neither negative result
is replaced or retroactively called a success. The next development check is
whether real runtime mention evidence changes the paired decision without
regressing other source constructions. Broad validation and G03 closure remain
open.

The four-source `source_factors_runtime` plumbing pilot at
`~/.aura/rlc-evidence/semantic-source-runtime-4row-pilot-20260923/fold-0.json`
completed one epoch with four balanced updates. One source gained a runtime
chart; three did not. On four held-out source-bank rows it selected 1/4 correct
against the incumbent's 3/4. This is a negative pilot, not a model comparison
with useful power. A separate v3-bank acquisition retained 16 candidates on
one source row with mention and definition evidence, but search was incomplete
and semantic verification unknown. Its receipt is under
`~/.aura/rlc-evidence/semantic-identity-evidence-bank-pilot-20260923/`.

A 42-source target-blind v3 bank was then acquired with four operation charts
and two argument graphs per chart. Correct programs were retained on 40/42
source rows; the incumbent selected one on 13/42. Two ordinary decodes yielded
no candidates. The evaluator now leaves those two in the denominator as
unavailable and never treats them as training positives. On source fold 0,
the grouped evidence ranker selected 3/12 correct versus 4/12 for a ranker
trained on the same evidence-path inventory with the argument-evidence term
absent, and 5/12 for the incumbent. Both learned arms are negative pilots.
The bank and fold receipts are under `~/.aura/rlc-evidence/` with names
`semantic-identity-evidence-bank-4x2-20260923`,
`semantic-identity-evidence-grouped-fold0-20260923`, and
`semantic-identity-lesion-grouped-fold0-20260923`. The first two training
attempts were not matched because one deduplicated same-program evidence paths;
the grouped rerun corrected that. A same-checkpoint evidence lesion is now
recorded by the evaluator on subsequent runs. None of these source subsets
establishes transfer or promotion.

The same-checkpoint lesion receipt at
`~/.aura/rlc-evidence/semantic-identity-evidence-lesion-fold0-20260923/fold-0.json`
found seven of twelve program choices changed when the learned evidence term
was zeroed, but treatment and lesion both scored 3/12. The term reaches the
decision. It has not earned an accuracy claim. Full-source training is the
next source-only test of whether the relation generalizes with adequate
construction coverage.

The immediate experiment remains source-only and opt-in. Focused tests check
that identical value representations at distinct source spans remain separable,
that the ranker score is differentiable through the binding terms, that source
bank labels do not select runtime chart views, and that old ranker checkpoints
remain loadable. Those checks do not measure generalization. Next, complete
the source-fold pilot, inspect errors and regressions by construction, and only
then run the frozen broad denominator. No G03, transfer, frontier, or serving
box is closed by this build.

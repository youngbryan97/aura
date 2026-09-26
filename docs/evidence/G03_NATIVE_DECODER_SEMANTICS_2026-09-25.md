# Native decoder semantic pilot

The source-atom fit did not improve the incumbent. This pilot instead uses the
resident model's pretrained decoder and vocabulary projection to score typed
programs. Its input is the unchanged source request under the installed chat
template. No strategy prompt, new system message, family router or held answer
is supplied. The installed template's existing reasoning configuration remains
unchanged.

## Shared computation

`FrozenDecoderPrefix` captures the original frozen embeddings and first 63
blocks. `NativeDecoderSuffix` references the original final block, normalization
and vocabulary projection. Four native projections receive rank-8 LoRA factors:
query, value, attention output and MLP output. All other parameters stay frozen.
The fit is an isolated research process; nothing is fused or activated in Aura.

For deterministic frozen prefix P and trainable native suffix S, evaluating
S(P(tokens)) gives the same computation as the split full decoder. Reusing P's
output preserves suffix gradients because P has no trainable parameters. This
equivalence does not establish that the resulting program means the right thing.
Real dense and hybrid MLX tests cover tied embeddings, quantized adapter updates
and restoration of the unfitted control. The first actual resident sequence
gave maximum absolute logit difference 0.0 over 111 tokens.

The resident tokenizer merges a newline across the prompt/continuation boundary.
Supervision now uses the existing offset-aware tokenizer surface and the exact
rendered template text. The crossing token is supervised once; no text is changed,
no private channel is exposed, and no sequence is truncated.

## Host defect found and repaired

The first launch failed before model loading. The default wired limit was 54.4
GiB while Metal permits 51.84 GiB on this host. The shared memory envelope now
caps default wiring at the measured device capacity, refuses an explicit oversized
request before mutation, and rolls back successful setters after setup failure.
Cleanup attempts all restorations even if synchronization or another setter fails.
Requested and effective wiring are both recorded. MLX's allocator limit is not
a hard process-RSS bound; the module documentation no longer claims otherwise.
The upstream [Metal allocator](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp)
checks the same device capacity.

## Complete result

The frozen fold supplied 303 source fit requests, eight source calibration
requests, and 18 held wording requests. Calibration continuation loss alone chose
between steps 32 and 64. It fell from 2.993303 to 0.297641; step 64 was selected.
No held labels chose weights or a checkpoint.

The paired bank comparison measured 143 unique program continuations:

| Mechanism | Correct / 18 |
| --- | --- |
| Existing source-fold proposer | 15 |
| Unfitted resident suffix | 7 |
| Fitted resident suffix | 14 |
| Observed equivalent candidate in the bank | 17 |

Against the unfitted suffix, training yielded eight gains and one regression.
Against the incumbent, it yielded one gain and two regressions. The latter
rejects promotion. The complete process took 527.949 seconds and released its
model owner. No desktop runtime was launched or changed.

The four native misses are reserved-alias reference binding, nested arithmetic
without observed bank reach, a natural-alias lookup, and a counterfactual count.
The last two regress incumbent successes. A target-blind native scorer can help,
but generic continuation likelihood is not yet a safe combined selector.

Evidence is under
`~/.aura/rlc-evidence/semantic-native-prefix-fold0-v2-20260925/`:

- Plan: `7cbd17e7be8763a9e85fa7cbc2d3d6883b1284e8ca96b0f793b0e8551a971d7f`.
- Selected weights: `7c3304a3eb69f3f7969d9e89b84a3fa7434eb1edb0d6d3f7b94fed75c8923d87`.
- Complete report: `e93549f8282dc91edb1da258656d65a4607eb91b10f242ba647639f6d262a44a`.

Sixty-one focused tests passed, including the real resident tokenizer. Smoke
passed 164 tests with one skip. Lint, compile, governance, layering and writing
passed. G03 remains open; no full development, fresh semantic-family transfer,
public-answer gain or frontier result is inferred from this pilot.

## Research disposition

[Pretrained semantic parsing](https://arxiv.org/abs/2109.15101) supplies a reason
to retain the existing decoder instead of learning another language encoder from
the small source cohort. Its experiments also show that transfer depends on
the pretraining domain; they do not guarantee universal transfer. The linked
[author discussion](https://www.reddit.com/r/MachineLearning/comments/hv4299/)
likewise distinguishes pretraining benefit from solved generalization. The
available page exposed the author's post, not a full comment discussion.

[Grounded compositional evaluation](https://aclanthology.org/2024.findings-acl.205/)
supports keeping unfamiliar combinations and lengths separate from ordinary
in-distribution accuracy. Neither paper establishes Aura's result. The next
diagnostic must separate likelihood attached to semantic graph decisions from
likelihood attached to formatting, then combine useful signals using source-only
calibration rather than selecting successful methods from held answer keys.

## Semantic-decision follow-up

The follow-up marks operation names and reference integers by their exact
character/token offsets. Training and ranking use only those decision tokens,
not JSON formatting or private-channel delimiters. Exact-length batches reuse
the frozen prefix without padding, truncation or changing the training order.
The resident batch again gives maximum absolute full/split logit difference 0.0.

The frozen run uses the same 303 eligible fit requests, 23 source calibration
requests selected by source identity, and the same 18 already-exposed held
wording requests. One complete fit epoch takes 303 updates. Source calibration
decision loss selects step 202 from checkpoints 101, 202 and 303; it falls from
2.533156 to 0.637823. The plan's inherited selection-name string says
continuation loss, but its explicit `loss_scope` and the executed objective are
`semantic_decisions`. No held labels choose weights or a checkpoint.

| Mechanism | Correct / 18 |
| --- | --- |
| Existing source-fold proposer | 15 |
| Unfitted resident suffix, same decision score | 8 |
| Fitted resident suffix | 17 |
| Observed equivalent candidate in the bank | 17 |

There are two gains and zero regressions against the incumbent, and nine gains
and zero regressions against the unfitted suffix. The remaining nested-arithmetic
request has no observed equivalent candidate in this incomplete bank. This is
not a proof that it is unreachable. The process finishes in 558.232 seconds and
releases its model owner. No serving, fusion or desktop state changes occur.

An independent replay verifies the plan, report, every row, checkpoint and weight
digest; recomputes the paired counts; checks score-selected identities and the
source-calibration checkpoint choice; and matches all bound implementation files.
Evidence: `~/.aura/rlc-evidence/semantic-native-decisions-fold0-20260925/`.

- Plan: `db24e6b1e668f906a8e7912c16b4fb0a4c089196c7226b28629e26296e9d7b87`.
- Selected weights: `f9553ce0c30bfdabcddc4aa8c69d1f9eb5adc4646b8949696659a3e86a1ee814`.
- Report: `9cab030c9f23b32c421cc60d5ddc950e8e04cd35273eddbe96f63d8b759e27de`.

Seventy focused tests pass, including the real resident tokenizer. Smoke passes
164 tests with one skip; lint, compile, governance, layering and writing pass.
The cohort is small and exposed. It establishes an exploratory paired gain on
held wording, not full-development success, fresh family transfer, freely decoded
public gain or frontier performance. G03 remains open. Whole-graph continuation
and completion decisions and competition against witnessed wrong programs remain
the next objective-level checks; the incumbent is not withdrawn.

## Complete-program objective

The next frozen implementation includes the comma between instructions and the
final steps-list closure in its semantic decision mask. Those tokens decide
whether the graph continues or ends. The rest of the canonical serialization
and the installed chat template stay unchanged.

For request x and candidate graph g, the score is the sum of native conditional
log probabilities at its operation, reference and termination decisions. The
source-only objective combines the gold graph's mean decision-token loss with
whole-graph competition:

```text
s(x, g) = sum(log P(token_j | request, graph_prefix_j)) over decision positions
L_token = -s(x, gold) / number_of_gold_decisions
L_choice = logsumexp(s(x, all_known_graphs)) - logsumexp(s(x, known_positive_graphs))
L = L_token + L_choice
```

For one positive, the derivative of L_choice with respect to graph scores is
softmax(scores) minus the positive's one-hot vector. Real MLX tests check this
identity and the multiple-positive form. This aligns training with complete
candidate discrimination. It neither proves that the native model can express
every required ranking nor guarantees transfer.

Contrasts come from the existing `source_program_contrasts` mechanism. It retains
a rival only after type checking and a differing output on independent input
probes. Source-fit peer programs supplement operation and reference variants.
Uncertain equivalence does not become a negative. No held target, construction
router, new language encoder or second execution floor is introduced.

The suffix computes every causal layer state before selecting the positions
sent to the original vocabulary projection. Dense/hybrid, tied/untied and
quantized-adapter tests compare selected logits and parameter gradients with
the full projection. The actual resident batch gives maximum full/split and
selected-projection differences of 0.0. The training schedule is frozen before
capture; only scheduled fit sources and source calibration are captured.
The unfitted checkpoint remains eligible for source-calibration selection.

[Semantic-aware contrastive parsing](https://aclanthology.org/2022.emnlp-main.269/)
motivates whole-representation discrimination alongside token likelihood. Its
two-dataset results are not evidence about Aura. This implementation uses
Aura's existing witnessed floor contrasts and native decoder rather than the
paper's representation encoder or sampling system. The reviewed
[public discussion](https://www.reddit.com/r/MachineLearning/comments/1adnq4u/d_whats_the_proper_way_of_doing_direct_preference/)
raised reference-policy and data-distribution questions; it supplies hypotheses,
not technical authority. This objective is supervised graph classification,
not DPO, and does not claim a preference-policy theorem.

The frozen 64-update plan is
`e419b8d0dd792ab6f386f01044d6c86512faa3216e5eae541079a77d87b26788`,
under `~/.aura/rlc-evidence/semantic-native-contrast-fold0-v2-20260925/`.
Its complete result must be appended after process exit and independent receipt
verification. No result or ledger closure is inferred from a running process.

## Checkpoint-bound replay

The 64-update fit writes all three declared checkpoint receipts, including the
unfitted checkpoint. Source calibration chooses step 64: loss 2.581903 versus
3.317516 unfitted and 6.639453 at step 32. The supervised prefix population is
348 continuations for 64 scheduled fit requests and 23 calibration requests.
Its process disappears during held evaluation with eight durable rows and no
final report. No process exit code or cause was recovered; no Python crash
report was found. This is an incomplete measurement, not a completed negative
or positive result.

`tools/evaluate_semantic_native_checkpoint.py` recovers evaluation from verified
checkpoint and weight digests. It requires the complete declared checkpoint
schedule and disjoint source fit/calibration/held identities, then selects the
minimum calibration loss without reading held rows. It validates the original
bank, source, model descriptor and pointer. Replay has zero fitting updates.
Durable rows are checked for score-selected identities and independently
graded bank outcomes before reuse; changed evidence refuses replay.

Historical adapters retain their original decision basis explicitly. The
operation/reference-only basis and the graph-termination basis serialize the
same program but use distinct token masks. An old adapter is not silently
rescored under the new mask and reported as the same candidate.

The two predeclared 50-wording replays retain the same bank and matched unfitted
controls. Their plans are frozen before either wider result is read:

- Step-64 contrast candidate: `4f1f8dbb2dcd73d2f963a7c85526350ae1ff428698c00bd9f138072f70d9a71b`.
- Step-202 decision candidate: `8b25c73de15d9f6d875e1b5c8361ed9b300f5ae87704ac90f67212c3590c256a`.

These candidates have different update counts and decision masks. Their paired
results can assess each frozen selector, but cannot isolate the effect of the
contrastive term. A union of their correct answers would be an oracle statistic,
not an available target-blind combined selector. Source-only arbitration remains
required before a mixed policy can be qualified.

Seventy-two focused tests pass, including the resident tokenizer, exact MLX
logit/gradient checks, source checkpoint selection and replay-row integrity.
Lint, compile, governance, layering and writing pass. Wider outcomes must be
appended after the corresponding complete report is independently checked.

### Complete contrast replay

The step-64 checkpoint completes all 50 frozen wording cases in 510.794391
seconds, with zero fitting updates. Incumbent: 41 correct; fitted native
selector: 38; matched unfitted native selector: 26; observed bank reach: 47.
There are three incumbent gains and six regressions. The candidate is rejected
for promotion. Improved native discrimination against its unfitted control does
not establish an improvement over Aura's incumbent.

The complete report receipt is
`7fac132a12b8d81f2d6605c3c9c52aef19418f242d2e4c9882e4ec8b90b4d9a4`,
under `~/.aura/rlc-evidence/semantic-native-contrast-replay50-20260925/`.
Row identities, score-selected programs, report totals and the source-selected
checkpoint are independently checked. The original interrupted fit remains
incomplete; this separately identified zero-fit replay supplies the wider
measurement without rewriting that history.

All three nominal-nested arithmetic requests lack an observed equivalent
program. Nine additional requests have an equivalent program that the fitted
selector does not choose. These are different obligations: proposal reach and
semantic discrimination. No domain-specific exception or held-label router is
introduced to conceal either. The earlier decision-only checkpoint's frozen
50-case replay is still pending.

### Complete decision-only replay

The earlier step-202 checkpoint completes the same 50 frozen wording requests
in 512.121912 seconds, again with zero fitting updates: incumbent 41, fitted
native 45, unfitted native 27, observed bank reach 47. It has five incumbent
gains and one regression. Its report receipt is
`e5ceb73fcc2c1f6b2b8e2cb477aac813dd4324d26fce1e6dc07cf8c962c3a9c5`,
under `~/.aura/rlc-evidence/semantic-native-decisions-replay50-20260925/`.
All 50 durable rows are independently checked against the original bank's
program identities and independently graded outcomes, not just report hashes.

The remaining reachable misses are a reserved-alias reference and a
natural-alias lookup request. The latter regresses an incumbent success.
Three nominal-nested requests still lack a reachable equivalent. These wider
results support useful source-learned discrimination across the sampled wording
constructions, but do not establish fresh-family transfer or universal
non-regression. Neither checkpoint receives serving authority or closes G03.

The next combined selector must learn from source-only candidate comparisons
and pass separate admission. It must not use these held-case labels to choose
between methods, nor treat the union of their successes as a callable policy.

## Source-only combined-selection acquisition

The existing crossfit proposer now has an explicit `source_calibration` bank
mode. It reuses the unchanged source-fit proposer, evaluates only the complete
separate calibration partition, and records distinct plan/report schemas.
Training, calibration and held identities must be disjoint. The source,
candidate and implementation identities are checked during acquisition.
Existing transfer-bank consumers reject this calibration schema rather than
counting calibration rows as new held evidence.

This provides actual target-blind proposal comparisons for the native learned
selector and the incumbent on the source calibration distribution. Labels are
added by the existing independent program diagnosis after proposal generation.
The existing calibrated candidate selector remains the arbitration owner;
there is no domain router or second arbitration architecture. Fitting and
separate admission are still required; acquisition alone grants no authority.
Partial acquisitions retain durable rows but do not publish a complete report.

The bounded acquisition check retained 12 of 185 calibration rows in
98.721498791 seconds and exited zero. Eleven rows contain an observed equivalent;
one has unresolved reach after incomplete search. All 12 have an ordinary
incumbent and known comparison outcomes. This partial population is not an
admissible calibration receipt. The native replay reader refuses it until all
185 rows and their report are present.

Native calibration scoring now preserves unknown comparisons as `None`, rather
than training them as failures. An unavailable ordinary answer cannot inherit
credit from the first alternative in a bank. Unrankable rows remain in the
population with no fabricated native scores or successes. The 23 instances used
to select the native checkpoint are recorded separately so independent arbiter
admission can exclude them. Implementation and source-bank identities remain
bound through completion.

For diagnosis only, an oracle union of the incumbent and both measured native
selectors reaches 46 of the 50 held requests, below the observed bank reach of
47. These labels are not available to a deployable selector. Even perfect
arbitration between these three outputs would leave one reachable selection
miss and three requests without observed equivalent proposals. Combined
selection and proposal reach therefore remain separate engineering obligations.

## Combined native evidence adapter

`tools/calibrate_semantic_native_choices.py` consumes the complete source bank
and one or more complete native calibration replays. Every durable score row is
checked against its report, source bank, original proposer and independent
grading. Native methods must score the same typed program inventory and use
the same resident identity. Duplicate method receipts are refused.

The adapter builds likelihood, relative likelihood, graph-depth and existing
joint-evidence views. No source words, construction names or correctness labels
are features. Checkpoint-selection instances are excluded. Whole construction
groups are separated into fit, tune and admission populations. The existing
binary evidence scorer and calibrated candidate selector own learning and
arbitration. Admission still requires gain and zero measured regressions.
`select_combined` applies that policy without reading comparison outcomes.
This is callable research infrastructure, not an admitted policy or a live
serving change. Source-native calibration is still needed.

`tools/replay_semantic_native_choices.py` applies the frozen admitted policy
to the original held bank. Checkpoint weights, training receipt, loss scope,
model identity and method order must match the calibration artifact. The replay
refuses calibration overlap and changed durable scores. Labels are read for
independent grading but never passed to `select_combined`. Eighteen focused
adapter/selector checks pass, including a synthetic admitted-policy replay.
That synthetic result is a contract test, not a measured Aura gain.

A shared native-prefix cache experiment was rejected before model replay.
Two of its 23 focused checks failed: full and cache-split hidden states differed
on unquantized dense and hybrid models at `atol=rtol=1e-4`. The cause of those
numerical differences is unmeasured. Original prefix computation was restored
exactly; no tolerance was relaxed and no cache speedup is claimed. The core edit
also stopped source acquisition at its implementation check after 105 durable
rows. Acquisition resumes against the original pinned implementation.

## Complete source acquisition and alternative coverage

Source acquisition completed all 185 cases and exited zero. Independent row
verification reproduces the complete origin/candidate binding and every nested
bank/diagnosis receipt. Observed equivalent proposals exist in 163 cases; 22
remain unresolved after incomplete search. The incumbent selects 122 correct
programs and the existing joint score selects 144. There are 298 equivalent and
971 witnessed-different candidate comparisons, with no unknown comparisons.
The plan is `8b86b0d7fe3d1c9f8855167481e950ed48b1a04b94e727574f4cb1e7fc9f1c02`;
the report is `e22320676e1ef6313487c605962fac4199fda7e16f8862e90e4a25e4ac2db300`.

The combined adapter now retains fitted-native, unfitted-native and joint-score
choices. They share the same independently validated typed inventory; labels
remain outside the choice construction. Including the unfitted alternatives
raises the exposed 50-case oracle union from 46 to 47, equal to observed bank
reach. Adding the joint choice does not increase this union. This is an exposed
architecture diagnostic, not a learned policy, fresh test or qualification.
Three cases still have no observed equivalent proposal. Sixty-nine focused
checks pass across the adapter, selector, replay, proposer and the newly merged
numeric-answer acceptance repair. No G03 checkbox closes.

## Full source-only native scoring and selector coverage

The decision checkpoint scored all 185 source-calibration banks with zero fit
updates. Its top choice is correct on 116 cases versus 122 for the incumbent:
29 gains and 35 regressions. The complete receipt is
`/Users/bryan/.aura/rlc-evidence/semantic-native-decisions-source-calibration-20260925/report.json`.
It is source-only measurement, not held replay or a serving decision.

The combined adapter now lets the existing calibrated scorer rank every
measured executable typed program, not only the fitted, pretrained and joint
argmaxes. It also exposes candidate-relative source operation/argument/
definition spans and path counts. Diagnosis labels and construction names
remain outside the feature view. Twenty-three focused selector tests pass.

Calibration on the complete source bank admitted the binary correctness scorer
(Brier 0.16699782 versus 0.2719135 for a constant predictor on its separate
calibration split). It did **not** admit a selection policy: the admission split
made zero switches, zero gains and zero regressions. Its source-only receipt is
`/Users/bryan/.aura/rlc-evidence/semantic-native-combined-source-coverage-20260925.json`.
No held replay or promotion follows from an unadmitted policy.

The receipt now reports construction coverage separately from accuracy. Eight
groups occur in the 185 scored rows, but excluding the 23 checkpoint-selection
instances removes every eligible example from two small groups
(`counterfactual`, 3; `natural_alias_source`, 2). The remaining six groups are
kept disjoint across fit/tune/admission, with 74/74/14 examples. Thus this
experiment tests transfer across the eligible groups only; it has no arbiter
admission evidence for the two excluded groups, and its small admission split
does not support a broad cross-family claim. Further source groups and
independent admission cases are required, not a relabeling of excluded cases.

## Incumbent-relative decision evidence

The same combined selector now exposes each candidate's score and source-path
evidence relative to the incumbent on that request. This is a comparison over
measured programs, not a construction classifier or a target label. Its
source-only calibration Brier improves from 0.16699782 to 0.15694262, but the
policy is still **not admitted**: no gain or switch reaches admission. The
receipt is `/Users/bryan/.aura/rlc-evidence/semantic-native-combined-relative-rank-20260925.json`.

The rank diagnostic locates the transfer break. In fit groups, raw ranking
gains 25 arithmetic cases without regression (51/61 versus incumbent 26/61).
In the disjoint tune groups it gains zero, and in admission it gains zero.
The group-disjoint result therefore rejects the tempting inference that better
per-candidate calibration or arithmetic performance establishes general
semantic selection. The receipt does not publish the rejected scorer as a
serving policy.

Only candidate program structure, native likelihoods, joint score, and the
retained source-span/path evidence reach this selector. The source bank does
not contain the live conversation, memory, web retrieval, inner deliberation,
or raw hidden-state relation vectors. Those cannot be credited to selection
without a separately captured, source-bound and replayable evidence path.

## Matched-relation support in the source cohort

The calibration tool now records independently verified typed-program relation
keys solely as a **dataset-coverage diagnostic**, outside the selector feature
view. After checkpoint-selection exclusions, 122 distinct verified structural
relations remain; only 5 recur across eligible construction groups. The
cross-group overlap is confined to cataphoric, role-binding and reserved-alias
groups. Arithmetic and fork-join provide no matched verified relation to a
different eligible group under this exact structural key. The source-only
receipt is
`/Users/bryan/.aura/rlc-evidence/semantic-native-combined-relation-coverage-20260925.json`.
This explains why a naive contrastive objective across all six groups would
mostly pair unlike relations. It does not prove that no broader semantic
analogy exists; the key recognizes exact connected program structure under
declared safe symmetries, not arbitrary real-world meaning.

The source-fit-only schema memory is now a frozen, label-free runtime input:
verified fit relation keys contribute a count of supporting construction
groups for each candidate program, with the row's own group excluded while
fitting. Replay reads only those counts and the candidate's executable graph;
held labels never enter the memory. Twenty-six focused checks pass. The
source-only result remains unadmitted: fit groups arithmetic and cataphoric
share no verified relation key, so schema support is constant in fit and
does not change Brier (0.15694262) or selection. Receipt:
`/Users/bryan/.aura/rlc-evidence/semantic-native-combined-schema-memory-20260925.json`.
Matching program structure is useful reusable evidence but is not, by itself,
proof that a new utterance expresses that program. Matched cross-construction
source examples and relation-flip controls are still required.

## Source representation audit

The new source-only diagnostic probes construction recoverability from the
incumbent's candidate feature vector. Leave-one-out nearest-centroid balanced
accuracy is 0.793695 across five groups with at least two eligible examples;
100 label permutations average 0.205987 (upper-tail p = 0.009901). A singleton
group is excluded from this diagnostic. The feature vector therefore still
contains strong construction information, even though construction names are
not explicitly supplied to the selector. Among independently verified correct
programs, 26 same-relation cross-group pairs have median standardized feature
distance 4.36848, versus 2.894964 for 5,976 different-relation within-group
pairs. Those pairwise observations are dependent and are diagnostics, not an
independent significance test or a proof of semantic impossibility. Receipt:
`/Users/bryan/.aura/rlc-evidence/semantic-native-combined-invariance-20260925.json`.
The selector remains unadmitted with no admission gain. The next representation
must compare source-bound role/dependency relations directly, and must show
held-group improvement against role-flip and surface-preserving controls.

## Role binding and partial structural precedent

Candidate evidence now compares each source-annotated argument definition with
the origin of the register that the executable program actually uses. It
separates input-role bindings from intermediate-result dependencies; absolute
token positions are not selector inputs. On this source cohort, adding that
feature changes the independent calibration Brier from 0.15694262 to
0.15692149, but held admission remains zero gains and zero switches. The
construction probe is still 0.75309 balanced accuracy versus a 0.202088
permutation mean. Receipt:
`/Users/bryan/.aura/rlc-evidence/semantic-native-combined-binding-20260925.json`.

The selector now also receives frozen, source-fit-only precedent for connected
typed subcomputations. Correct and incorrect fit programs are counted
separately by construction group, with a fit row's own group removed from its
features. This permits a partial A-to-B structural bridge without claiming
that shared subgraphs make entire programs equivalent. The frozen memory
contains 143 motifs. Source-only Brier is 0.1573255 and held admission is
still zero gains, zero switches. Receipt:
`/Users/bryan/.aura/rlc-evidence/semantic-native-combined-motif-20260925.json`.
Neither feature is promoted. These observations reject the present numeric
source-binding and subgraph-precedent representation as a sufficient answer to
unseen-family selection; adding more threshold gates would not repair it.

The broader suggestions map partly to existing runtime code rather than new
modules: `semantic_counterfactual_corpus.py` makes name/order-preserving and
witnessed role/operation-changing source examples;
`semantic_candidate_contrasts.py` supplies typed witnessed negatives;
`semantic_program_floor.py` executes/composes exact programs;
`core/cognition/structure_mapping.py` aligns typed relation graphs. Earlier
full-request contextual recognition is recorded in
`G03_CONTEXT_AND_PORTFOLIO_2026-09-21.md`. None establishes that the current
selector can infer an unseen utterance's intended relation. A next training
experiment must provide independent matched same-relation/different-surface
examples and relation-flip controls to the learned source representation,
then hold complete construction families out. The exposed validation fold
cannot be reused as fresh transfer evidence.

## Matched source relation controls

The existing counterfactual corpus now produces an explicit cross-construction
control inventory. A positive pair requires two independent source-training
examples in different construction families with the same connected typed
program relation. A negative is a type-correct rival on the same source whose
changed output is witnessed by the existing floor comparison. Equal lineage,
unproved rivals, and validation/test examples cannot enter. On the established
seed-24 program-first source corpus, 640 training examples yield 56 relations
and 560 cross-construction pairs. Receipt SHA-256:
`22e0a3b83a1a743f4940d426ca339bac5340144066f553c0dd16423644fdde2e`.
Of the 560 witnessed negatives, 440 preserve the operation sequence while
changing a role or dependency; 120 change an operation. The distinction is
retained so a role-binding result cannot be credited to operation recognition.
This inventory is callable source supervision and an acceptance contract for
the proposed relational learner. It has not trained a model, measured an
unseen construction, or changed serving. The preexisting native contrastive
objective only contrasts programs within a source; the cross-construction
positive relation needed a source-hidden-state training consumer and a
separately frozen held-family test.

The native semantic trainer now has an opt-in `relational` objective using
that same typed relation identity. It pairs fit examples across construction
families and unrelated contrast lineages, retains witnessed within-source
candidate contrasts, and optimizes the smooth maximum of the two source
losses. An unmatched scheduled source retains the ordinary contrastive loss;
the plan names every scheduled partner and counts paired updates. Only the
scheduled partner set is prefetched, not the entire fit pool. Calibration
examples are never used as partners. This arm is implemented and unit-tested,
but no resident training run or held-family result exists yet; it has no
serving authority. It is a testable relational-learning mechanism, not a
claim that source-pair optimization guarantees transfer.

The first read-only plan against the frozen fold-0 source cohort completed
without loading model weights: 303 fit sources, 8 calibration requests, 18
declared held requests, 11 paired updates in a 16-update pilot, and 23 unique
fit sources to cache. Five updates are explicitly unmatched. Plan receipt:
`06a27aa1b396d2a97091a9c6acc176144cbde214646ffa9270abe265a35b912e`
at `~/.aura/rlc-evidence/semantic-native-relational-plan-fold0-20260926/plan.json`.

The first model-active pilot stopped before training at the prefix-equivalence
gate. The full model and frozen-prefix-plus-suffix logits were bit-identical,
but selecting only the eight supervised output positions before the BF16
vocabulary projection changed an unrelated logit by 0.0625. The supervised
target log-probability differed by at most 0.001065, with identical argmaxes
on the checked positions. The gate now retains the strict full-path check and
compares the selected path at the actual supervised token probabilities, with
a tolerance derived from output dtype precision. A deliberately wrong target
probability still fails the focused test. The failed directory is retained at
`~/.aura/rlc-evidence/semantic-native-relational-pilot-fold0-20260926/`.

The fresh 16-update pilot completed at
`~/.aura/rlc-evidence/semantic-native-relational-pilot2-fold0-20260926/`.
It cached 128 supervised sequences and retained the unfitted step-0 checkpoint:
calibration loss rose from 2.800060 at step 0 to 3.181365 at step 16. Its
held readout was 8/18 native-correct versus 15/18 incumbent-correct, with
zero learning gains and seven regressions relative to the incumbent. Receipt:
`ca33f818862cd00fffb9c7b27d1c7e207fccd0eaaaa60b0a5ee1300321bd92de`.
This is a negative result for this paired-loss pilot, not evidence of broad
transfer or a serving candidate. Pairing source forms does not by itself
align their representations or solve candidate selection.

The floor's existing finite counterfactual comparison now also accepts an
explicit transition feedback slot and bounded horizon. It can distinguish
programs that agree on the initial input but diverge after their output is
fed into a declared state input; a direct helper generates distinct, seeded
typed initial contexts through the existing counterfactual input generator.
Identical repeated contexts are deduplicated, execution failures and jointly
undefined traces are counted, and finite agreement remains `unknown`. This
is available to a caller with a genuine state-transition interpretation; it
is not inferred automatically from arbitrary prose or used to select a
candidate without external observations. In particular, more internal
simulation alone cannot tell which of two distinct programs matches the
user's intended relation.

The existing `structure_mapping` and `transfer_search` engines now accept
connected typed programs through a diagnostic graph adapter. The adapter
retrieves source-fit precedents by structure, carries the shuffled null, and
checks proposed matches with the universal floor. It cannot establish that
an utterance intended the retrieved program. On the untouched source-only
admission constructions, the bounded audit found 95/95 candidate programs
passing the structural analogy threshold, including all 75 graded incorrect
ones; none was floor-proven equivalent to a fit precedent. The 14 admission
sources and 95 candidates therefore provide a negative result for analogy
as a discriminator, not a transfer gain. Fit precedent used only verified fit
labels; admission labels graded the result after retrieval. Receipt:
`~/.aura/rlc-evidence/semantic-program-analogy-held-audit-20260926.json`
(`18d18bac8dcfd50001291044da4075e9375be66d935b4aa99cdd2b77919b6871`).
No serving authority or G03 closeout follows. Multi-hop similarity would
compound these false positives unless every edge had independent source and
meaning evidence; a path alone is not a proof of semantic transfer.

The next opt-in native objective takes one step beyond paired source losses.
It embeds a request's last pre-answer hidden state through the trainable native
suffix, pulls a same-relation source from another construction closer, and
pushes a different-relation source from the same construction away. All three
sources belong to fit; the positive has a different contrast lineage. The
negative must have a changed-meaning witness from the existing floor
comparison. Structural difference alone is not accepted. This auxiliary
metric loss runs beside the source-level contrastive loss. It does not use the
answer continuation to form its embedding, and calibration retains its
ordinary source loss. A changed embedding geometry is an optimization target,
not proof that a new utterance will choose the correct program.

The read-only fold-0 plan, before loading model weights, has 303 fit sources,
8 calibration requests, and 18 held requests. Ten of 16 scheduled updates
have witnessed triplets, requiring 31 unique fit sources in the prefix cache.
The immutable plan at `~/.aura/rlc-evidence/semantic-native-metric-plan3-fold0-20260926/plan.json`
has SHA-256 `e97ce207d02a663904942e0d1414e488814f40a4555038bd8c56ed2b36eae0f7`.
No model-active metric result or G03 closeout follows from this plan.

The model-active pilot completed in 317.05 seconds. Its selected step-16
checkpoint lowered calibration loss from 2.800060 to 2.716242. On 18 held
utterances, native selection rose from 8 correct before fitting to 11 after
fitting: three learning gains and no learning regressions. The existing
incumbent selected 15 correctly. Relative to that incumbent, the trained
native selector had no gains and four regressions: two natural-source cases,
one arithmetic case, and one cataphoric case. The held labels were unavailable
to fitting and checkpoint selection. The immutable report is at
`~/.aura/rlc-evidence/semantic-native-metric-pilot-fold0-20260926/report.json`
with receipt SHA-256
`3d3df20e31ab2ad69da0ffb0f3174e24f4ee005e6d3742c7aac8c125b62d9bf6`.
This establishes a small within-native learning gain, not incumbent
non-regression, broad transfer, or serving authority. G03 remains open.

A full-fit run then visited all 303 fit sources in 320 updates, with 230
witnessed metric triplets. It selected step 256 from ten scheduled checkpoint
intervals and the eligible step-0 baseline. Calibration loss fell from
2.800060 to 0.555287, although intermediate checkpoints varied sharply.
On the 18 exposed held utterances, it selected 17 correctly, versus 8 for
the unfitted native suffix and 15 for the incumbent. There were two paired
incumbent gains and no paired regressions. The sole miss was an arithmetic
case that the incumbent also missed; the observed bank reached 17/18.
Training finished in 1,351.02 seconds. The run receipt is
`~/.aura/rlc-evidence/semantic-native-metric-full-fold0-20260926/report.json`
(`9ff0f5631c5a428fa4bc463a4e9b53ce4ab68c22b6d872c9fbd9c0a49f90dec3`).

A fresh-process replay independently selected the source-calibration-best
checkpoint, rescored the same frozen bank without fitting, and reproduced
17/18, the two gains, zero regressions and the 17/18 reach ceiling. Its
receipt is `~/.aura/rlc-evidence/semantic-native-metric-full-replay-fold0-20260926/report.json`
(`ebff1ff357a92960e60c46c2f86b8c52d87bfd8bde9c34d074ea69ed137d0dbf`).
These 18 utterances are an exposed development subset. This result neither
establishes fresh-family transfer nor qualifies a serving configuration.

The same frozen step-256 checkpoint was replayed, without further updates,
against all 50 held utterances in this fold. It selected 46/50 correctly,
versus 41/50 for the incumbent and 26/50 for the unfitted native suffix:
five paired gains and zero regressions. The candidate bank reached 47/50.
Three failures share one arithmetic construction and lack a correct bank
candidate; the remaining reserved-alias failure has a reachable correct
candidate. The independent replay receipt is
`~/.aura/rlc-evidence/semantic-native-metric-full-replay50-fold0-20260926/report.json`
(`c890680e5c442686db24493642b0afc2cbd5af7c28ef82d867a03d99f946d5a5`).
This completes this fold's exposed held readout. It is still one known
construction partition, with no fresh-family or broad reasoning claim.

The same 320-update objective on utterance fold 1 did not replicate the
incumbent-relative gain. The frozen source-only plan had 303 fit sources and
50 held utterances. Its candidate bank reached 47/50; the incumbent selected
41/50. The fitted native suffix selected 41/50 at source-calibration step 288,
with four paired gains and four paired regressions. The regressions included
arithmetic, cataphoric, and two reserved-alias cases. Calibration loss fell
from 5.154188 to 0.585496, but that optimization did not improve the held
total. The fit used 153 witnessed metric updates, compared with 230 in fold 0;
this difference is observed, not established as the cause of the outcome.
The immutable report is at
`~/.aura/rlc-evidence/semantic-native-metric-fold1-full-20260926/report.json`
(`e218718334c3640cc35b1e2ca284551ec54948881264101aea7f48317665168e`).
This negative replication prevents promotion or a general-transfer claim.

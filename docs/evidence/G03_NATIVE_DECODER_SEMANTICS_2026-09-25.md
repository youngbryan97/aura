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

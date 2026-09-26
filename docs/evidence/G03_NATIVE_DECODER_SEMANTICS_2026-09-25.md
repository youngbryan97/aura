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

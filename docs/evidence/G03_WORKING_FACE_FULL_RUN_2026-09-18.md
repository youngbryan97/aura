# G03 working-face full paired run, 2026-09-18

This record closes the pending full paired measurement for the joint semantic
graph working face. It is a development validation result, not a G03 closure,
promotion, serving authorization, fresh-transfer result, or broad-reasoning
claim.

## Run identity

The detached run completed all three paired arms without restart or timeout:

- incumbent: the pre-run transducer;
- refit: the three-round joint graph refit;
- fit-start: the same starting transducer fit under the run's retained
  background, pointer, argument, and graph objectives.

The validation used 500 rows, with zero test examples supplied to the decoder
(`test_examples_used=0`). The run used source-anchored exact-program scoring;
the validation artifact reports `expected_answers_available=false` and
`gold_program_available_to_decode=false`. It therefore measures the declared
program-equivalence contract rather than a public-answer score.

The detached receipt reports `passed`, `restart_count=0`, and 7,967.572 seconds
(132.79 minutes). The run and its output artifacts are outside the repository:

```
run:        ~/.aura/rlc-evidence/g03-working-face-full-run-20260918/
candidate:  ~/.aura/rlc-evidence/semantic-working-face-full-20260918.json
validation: ~/.aura/rlc-evidence/semantic-working-face-full-20260918.validation.json
```

The candidate transducer uses model-basis SHA-256
`c2ccefe0c102df230a114172b40e8bd562f6c644b62a1f5b37b7ded4ac190cc8`.

## Paired result

| Arm | Exact | Equivalent | Paired gains | Paired regressions |
| --- | ---: | ---: | ---: | ---: |
| Incumbent | 488/500 | 488/500 | 0 | 0 |
| Fit-start | 474/500 | 474/500 | 7 | 21 |
| Refit | 386/500 | 390/500 | 0 | 102 |

The selector retained the incumbent. The fit-start arm's seven gains were
outweighed by 21 regressions; the refit lost 102 cases and produced no paired
gain. The background-supervision integration therefore did not establish a
general improvement, and the joint refit remains rejected.

This is a useful failure boundary: the run completed the learned constraint
fit and all paired validation work, but satisfying the internal graph
constraints did not preserve the incumbent's source-anchored program
selection. The next repair must address the source-to-runtime selection gap,
not relax the regression criterion or promote the higher-loss candidate.

## Receipts and checksums

```
semantic-working-face-full-20260918.json
  2a58a1649822ad284b157036e4c463d92e694b379010e42e8413654d6f0fc125
semantic-working-face-full-20260918.validation.json
  f56b8d6dd7a8216cdca19907a1ecfe1fcb763cba51600566d85b3c0765fec4fc
detached_status.json
  c87fc685b42d3bae89b432b0c68d6deb3ebbbd842e69c85bac18075840e3b75d
detached_receipt.json
  7d5291f5c9ed6d5015131d3cb6466bacba29ef7c6e4c8558eaae1c949c14eca1
```

G03 remains open. No coefficient is promoted, no serving package changes, and
no later G04-G12 claim follows from this negative result.

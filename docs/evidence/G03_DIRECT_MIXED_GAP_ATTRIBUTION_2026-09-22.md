# Direct and mixed method attribution on the exposed G03 gap

This is a development diagnostic on 23 previously identified validation
failures. It is not a fresh transfer test, a serving promotion, or G03 closure.
Validation labels were used only to grade frozen proposals after selection.

The source-trained direct decoder and mixed candidate ranker were compared on
the same frozen gap bank. The bank contains a correct program for all 23 cases;
the incumbent is correct on none. The ranker chooses 5 correct programs and
the direct decoder chooses 7. Their union is still only 7. The existing
answer-blind portfolio chooses 1 correct program and retains 19 inquiries.
Execution establishes that a candidate runs, not that it expresses the
requested meaning. The portfolio therefore preserves its incumbent when
executable methods disagree without an independent observation. Calling the
7/23 oracle union a selector result would leak the validation answers.

| Construction | Cases | Direct | Ranker | Portfolio |
| --- | ---: | ---: | ---: | ---: |
| cataphoric variant 3 | 2 | 0 | 0 | 0 |
| cataphoric variant 4 | 1 | 1 | 0 | 0 |
| cataphoric variant 5 | 5 | 0 | 0 | 0 |
| reserved alias variant 4 | 4 | 0 | 0 | 0 |
| reserved alias variant 5 | 8 | 4 | 4 | 1 |
| role binding variant 4 | 2 | 1 | 0 | 0 |
| role binding variant 5 | 1 | 1 | 1 | 0 |

This separates two problems. The learned methods do not select a correct
program on 16 of these cases even though the bank contains one. On six more,
one learned method selects a correct program but the portfolio has no
independent evidence to replace the incumbent. A source-only expansion of
construction coverage and a calibrated, independently tested selection
signal are needed; neither can be inferred from these validation labels.

The numbered constructions are wording/template variants, not program depth.
These validation programs have two instructions; training has other wording
variants of the same operation families. A larger bank sampled only from the
existing source variants cannot by itself establish lexical transfer.

Receipt:
`~/.aura/rlc-evidence/semantic-real-bank-mixed-fold0-20260922/direct-mixed-gap-attribution.json`.
The receipt binds the source fold, checkpoint, candidate bank, gap report, and
per-construction counts. The prior full-development result remains 477/500
selected, with 486/500 candidate availability.

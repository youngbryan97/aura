# G05 pilot: her free answers already carry the reading

G05 asks whether an internal gain reaches the answer a person reads. The
internal gain is G03's frozen program reader (v12). Before planning a
confirmatory run, a pilot on consumed requests measured how her own answers
behave with and without what the reader found. Consumed requests can be
looked at; the confirmatory population is fresh and planned from this.

## Arms

`tools/run_g05_public_answers.py`. Each answer is a free, greedy decode by
her persona 27B through her own chat template, native thinking on, at the
medium reasoning effort she serves with:

* ordinary: the request alone;
* assisted: the request, then the reader's program and its executed values
  as runtime evidence, the way her runtime gives any turn evidence. It holds
  steps, values and a result, and no word about how to answer;
* sham: the same form, holding another request's reading.

The answer is the last exact integer in her public reply; her reasoning is
never read.

## Two defects found before any number counted

* The final-answer rule read her correct "**2,104,802,751,450**." and
  "**476 583**" as 4 and 9 in every arm. Bold kept the final number from
  looking like one, and a space-grouped number split in two. Fixed in
  `3a364f2be` (`answer_tokens`), with positive and negative controls; the
  verifier used by the latent-cortex experiments shared the defect and shares
  the fix.
* The sham took its reading from the next request in the same stratum, and a
  bundle without strata made every request its own stratum: the sham showed
  each request its own reading. Fixed in the harness; the sham rows written
  before the fix were deleted, not regraded.

## Result

48 requests from composition bundle v1 (five steps, products up to thirteen
digits); stopped after nine with every arm and ten shams, the regime being
plain:

| Arm | Exact | Median tokens | Median seconds | Stopped at the budget |
| --- | ---: | ---: | ---: | ---: |
| ordinary | 9 of 9 | 794 | 52.4 | 0 |
| assisted | 9 of 9 | 691 | 46.5 | 0 |
| sham | 10 of 10 | 1,128 | 72.0 | 0 |

The reader read all 48 correctly. Her ordinary answers were already right:
she works thirteen-digit products out in her reasoning. With the right
reading she reached the same answer in 13% fewer tokens. With a wrong
reading she spent 42% more and still answered from the request, so she
checks evidence against what was asked instead of copying it.

## What follows for G05 and G06

On requests her free decoding already gets, a reader has nothing to add to
the answer, and a paired comparison there has no disagreements to count. The
confirmatory population has to be one where free decoding is known to slip
and an exact reader does not: counting and indexing in long lists
(`core/learning/semantic_g05_long_sequence_corpus.py`, 40 to 64 entries, the
most the value algebra holds). A pilot on that family, on its own seed, sizes
the plan before any confirmatory request is decoded.

## What this does not show

* Anything confirmatory: nine requests, consumed, stopped early by choice.
* That the evidence never misleads her: ten shams is a small sample.

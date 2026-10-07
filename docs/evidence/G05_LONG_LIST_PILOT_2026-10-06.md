# G05 long-list pilot: long lists do not make her slip either

The [first pilot](G05_PUBLIC_ANSWER_PILOT_2026-10-06.md) found her free
answers right on five-step compositions, so a comparison there would have no
disagreements to count. It proposed counting and indexing in long lists as
the population where free decoding slips and an exact reader does not. This
pilot tested that proposal before any confirmatory request was drawn.

## Requests and reader

`g05_long_sequence_v1`, pilot seed 20261007, 24 requests: a `count_of` or an
`at` over a list of 40 to 64 entries, then one scalar step, in consumed
phrasing. Features from her persona 27B, manifest
`4402a9a23d459a5d722dbe023a48894168ca6b5de4c3c6382afe628a838af7ab`,
`~/.aura/rlc-evidence/g05-long-sequence-pilot-20261007-features`. These 24
requests are consumed.

The G03 reader (v12) first read 21 of the 24. The literal span bound was
fitted on training mentions (29 tokens for a sequence) and was also applied
to literals, so a long list written out was never offered as an argument.
With a literal's extent left to its grammar (`9cc8cd592`), the reader read
24 of 24, every program the reference.

## Answers

`tools/run_g05_public_answers.py` from `9cc8cd592`, the three arms of the
first pilot, greedy, native thinking on, medium effort. Rows:
`~/.aura/rlc-evidence/g05-long-sequence-pilot-20261007-answers/rows`.
Stopped after seven requests, about five minutes each with all three arms,
because the answer to the question was already plain:

| Arm | Exact | Median tokens | Median seconds | Stopped at the budget |
| --- | ---: | ---: | ---: | ---: |
| ordinary | 7 of 7 | 836 | 55.2 | 0 |
| assisted | 7 of 7 | 616 | 41.8 | 0 |
| sham | 7 of 7 | 873 | 58.5 | 0 |

Five lookups and two counts. In her ordinary answers she walks the list by
index in her reasoning and lands on the right entry, at index 52 of 64 as
easily as at 1.

## What follows

* Long lists are not where her free decoding fails. A paired comparison here
  would again count no disagreements, so this family is dropped as G05's
  confirmatory population.
* The reader's evidence cut her tokens by about a quarter (616 against 836),
  as the first pilot found (13%), and a wrong reading did not mislead her
  once in seventeen shams across the two pilots. Both are compute and safety
  observations, not the correctness gain G05 asks for.
* G05 needs a population where her ordinary answer is measurably wrong. Two
  pilots and 16 requests have not found one in this request family; the next
  step is to find where she fails before choosing what to compare there.

## What this does not show

* Anything confirmatory: seven requests, consumed, stopped by choice.
* That she never slips on long lists: seven is a small sample.

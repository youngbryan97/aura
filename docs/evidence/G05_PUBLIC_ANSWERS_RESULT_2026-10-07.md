# G05 public answers: the reading reaches the answer a person reads

Plan `9bbc11dc58b2` ([preregistration](G05_PUBLIC_ANSWERS_PREREGISTRATION_2026-10-07.md)),
run as frozen from `23082a4a2` with G04's candidate v17 as the reader. All
552 decodes completed: 48 compositions in four arms and 120 lists in three.
None is missing. Rows, the runner's report and the independent check are in
`~/.aura/rlc-evidence/g05-public-answers-v1-run/`.

## Result

G05's closure rule holds. The primary comparison rejects and every right
reading was translated into an exact public answer.

| Stratum | Arm | Exact | Median tokens | Median seconds |
| --- | --- | ---: | ---: | ---: |
| compositions | ordinary, closed | 29 of 48 | 723 | 43.0 |
| compositions | assisted, closed | 48 of 48 | 7 | 1.6 |
| compositions | sham, closed | 33 of 48 | 696 | 40.8 |
| compositions | ordinary, open | 42 of 48 | 876 | 50.6 |
| lists | ordinary, closed | 108 of 120 | 446 | 26.5 |
| lists | assisted, closed | 120 of 120 | 2 | 1.7 |
| lists | sham, closed | 57 of 120 | 3 | 2.0 |

* Primary: assisted exact where ordinary was not 19 times, the reverse
  never; exact one-sided p = 1.9e-6, alpha 0.05. Rejects.
* Translation: the reader was right on all 168 requests, 48 and 120, and her
  assisted answer was exact on all 168. Holds.
* Secondary, assisted against her open channel on compositions: 6 to 0,
  two-sided p = 0.031 at alpha 0.025. Does not reject. With her channel open
  she spends 31 times the assisted arm's seconds and misses 6 requests the
  reading gets; six discordant pairs cannot carry a test at this alpha.
* Secondary, ordinary against sham on lists: 55 to 4, one-sided
  p = 8.5e-13. Rejects. A wrong reading shown as evidence cost her 55 lists
  she reads right alone. Of her 63 wrong answers, 20 are the shown reading's
  own number. The gain needs the reading to be right; in these 168 requests
  it was.
* Descriptive: on lists the reading adds 12 exact answers and loses none.
  On compositions she was exact 33 times with a wrong reading and 29 with
  none. One
  ordinary and one sham composition stopped at the 4,096-token budget.

## Four rows the runner misgraded

The plan's rule is "the last exact integer in the public reply; a budget
stop counts as not exact". The independent verifier
(`tools/verify_g05_independently.py`) reads it as written. The runner
differed on four rows, which the verifier lists without resolving. Read
from her replies:

* Ordinary `855b2f2a`, expected 871: stopped at 4,096 tokens, still
  deciding whether a position counts from 0 or 1 after writing "Answer: 871."
  The runner graded it exact. By the rule it is not.
* Open channel `c1761b67`, `24faa937`, `1f82e11d`, expected 888211, -6975
  and 56562: each ends `$$\boxed{…}$$` with that value. The runner's parser
  read 20, 18 and 8, numbers from the line above the box. By the rule all
  three are exact.

The table and tests above use the rule. With the runner's grading the
primary is 18 to 0 (p = 3.8e-6) and the open-channel comparison 9 to 0
(p = 0.0039, rejecting). Three of those nine were the runner's misreadings,
so the open-channel comparison is reported as not rejecting.

Both defects are fixed for later runs; these rows are not regraded in place.
The runner applies the budget-stop rule (`answer_exact` in
`tools/run_g05_public_answers.py`). The shared final-answer reader
(`answer_tokens` in `core/brain/llm/latent_cortex/experiment_tasks.py`)
opens `\boxed{}` and `\fbox{}` and strips math delimiters and LaTeX thousands
spacing. That is the same class as the bold answers misread on 6 October.

## Independent verification

`independent_verification.json`: every committed task regenerates from its
seed with the committed source hash; no task text was consumed; the plan
matches the spec; each expected answer agrees with the verifier's own
interpreter. Primary 19 to 0 (rejects), translation 168 of 168 (holds),
closure rule holds, checks pass.

## What this does not show

* Gains outside bounded integer programs: these are G04's request families
  at fresh seeds.
* Live use: the evidence was injected in her runtime's evidence role in a
  standalone decode, not delivered through a desktop turn (G11).
* Safety against a wrong reading: the sham stratum shows one misleads her.
  The reader's own refusals and its accuracy decide whether that matters.

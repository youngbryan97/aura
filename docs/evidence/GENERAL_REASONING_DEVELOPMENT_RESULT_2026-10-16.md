# General reasoning: the development round (16 October 2026)

The plan is GENERAL_REASONING_PROGRAM_2026-10-14.md. This round ran three of its mechanisms on development
halves, on her 27B, with the runners at c62cf0c01 (worktree `gr-tools`): mechanism A (her greedy and sampled
derivations; where they part, the crux settled by a program, her corpus, her constraint engine, or derivations
of the crux alone) on MMLU-Pro and MuSR, and mechanism B (her corpus for knowledge questions) on SimpleQA.
Every arm of a benchmark is graded by the same function (`run_general_reasoning.correct`,
`run_knowledge_with_corpus.grade`), and every comparison is paired, request by request, with an exact two-sided
McNemar test. These are development halves: they show where a mechanism has room, and claim nothing. BBEH's
non-mechanical tasks were in the plan and were not run this round.

## Results

| Benchmark (requests) | Her ordinary answer | Mechanism | Self-consistency, about equal tokens | Mechanism against ordinary |
| --- | ---: | ---: | ---: | --- |
| MMLU-Pro (60) | 54 | A: 54 | 54 | 0 gained, 0 lost |
| MuSR (40) | 34 | A: 35 | 34 | 2 gained, 1 lost, p 1.0 |
| SimpleQA (200) | 9 | B, passages: 58 | not run | 50 gained, 1 lost, p 4.6e-14 |
| SimpleQA (200) | 9 | B, fused passages: 68 | not run | against passages alone: 11 gained, 1 lost, p 0.006 |

Tokens per request, ordinary / mechanism / self-consistency: MMLU-Pro 1,286 / 3,192 / 2,777; MuSR
2,213 / 4,861 / 4,398; SimpleQA 740 ordinary, 543 with passages, 501 with fused passages.

**MMLU-Pro.** Her two derivations agreed on 57 of 60; on the other 3 her constraint engine settled the crux,
each time on the answer she would have given. Her six wrong answers were all agreed: both derivations made the
same mistake, so mechanism A, which acts only where they part, had nothing to work on. Four of the six are
knowledge questions (history, psychology, health), the kind mechanism B is for.

**MuSR.** All three changed answers came from cruxes her constraint engine settled. It found the right answer
for an object placement and a team allocation she had wrong, and the wrong suspect in a murder mystery she had
right.

**SimpleQA.** Allowed to decline when her answer with passages and a second, sampled answer with the same
passages disagree (1,081 tokens a request for the two), she answered 50 of 200: 44 right, 6 wrong, 150
declined. Of what she answered, 88% was right; her ordinary
answers were right on 9 of 197 answered. Passages were found for 192 of the 200 questions. 126 questions were
wrong in every arm.

**Passage recall** (200 SimpleQA questions, seed 7, a different sample from the run above): the answer is in
the top 1, 3, 5 and 10 paragraphs for 42, 54, 59 and 68 questions by words, and 40, 58, 65 and 69 by her
embedder's order.

## What the round shows

Mechanism B has room, and A, as built, has little. On SimpleQA her own knowledge is close to nothing (9 of
200) and her corpus raises it to a third, with fusion adding more; on MMLU-Pro and MuSR her derivations
mostly agree with each other, right or wrong, so a mechanism that acts on disagreement rarely acts.
Self-consistency at about equal tokens never beat her ordinary answer either, so on these two benchmarks more
sampling is not where a gain is.

## What it does not show

No claim: these are development halves, and the plan's claims are decided on test halves under a plan
committed before they are drafted. Nothing here measures open-ended reasoning or writing.

## Next

A test-half plan for mechanism B on SimpleQA, committed before drafting; and mechanism B on MMLU-Pro's
knowledge questions, on development first. Files: `~/.aura/rlc-evidence/gr-dev-<benchmark>-20261014/`
(`summary.json`, `rows/`), `gr-passage-recall-20261014/recall.json`, and the smoke runs beside them.

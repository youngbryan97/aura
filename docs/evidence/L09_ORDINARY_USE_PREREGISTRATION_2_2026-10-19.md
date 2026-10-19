# L09, second half, attempt 2: her kept procedures on against off over requests people make. Preregistration

Fixed on 19 October 2026 by the commit that adds this file, before any request below has been drawn, routed,
drafted or graded.

The first attempt ([result](L09_ORDINARY_USE_RESULT_2026-10-18.md)) was underpowered: her own chat history
reaches her kept procedures on 9 of 195 requests, too few for the five discordant pairs its plan required. Its
result names the way on: a plan that samples realistic requests beyond what has been typed to her so far. This is
that plan. Everything not changed here is as in the [first plan](L09_ORDINARY_USE_PREREGISTRATION_2026-10-18.md).

## The requests

The opening requests people made of a chat assistant in WildChat-1M (Zhao et al. 2024, ODC-BY), in
`~/.aura/datasets/wildchat-1m`, from its shards 1 to 13. Shard 0 is left out because L01's rehearsal draws from
it (`tools/what_people_ask_who_it_is.requests_people_make`), so nothing here is a request her model may be
trained to answer. The filter is that function's: English, not toxic, not redacted, not flagged by its own
moderation, the length of a chat line, not a short line about the assistant itself; one request per distinct
spelled-out text. The requests are put in an order by `random.Random(20261020).shuffle` over their sorted texts,
and routed in that order.

## The book and the route

As in the first attempt: her live book as installed at 13:24 on 18 October (sha256 beginning `4cfbfa11`, the
same book), and her chat's own route (`kept_answer_for`): by the words, by the meaning among the kinds the words
leave near, and by her model's restatement.

## How many

A request counts once the route gives it a kept answer: by the words, or by her model's restatement when the
restatement passes her chat's own check. Counting stops at **33**. That number comes from the first attempt: 3
of its 9 routed requests were discordant, and 33 is the least count at which, at that rate, at least eight
discordant pairs (enough for a verdict that survives one loss) come with probability 0.9 or more (0.905). Kinds
people ask about may be discordant less often than the number theory her history held; if fewer than five
discordant pairs come of the 33, the result is again reported as underpowered.

## The answers

Before any draft, each counted request's answer is computed by a program written for it (sympy, Python's
datetime, sorted(), integer arithmetic), and the program is kept beside the answer in `truths.json`. A counted
request no program can settle, because it is not a question with one answer, is a **misroute**: graded as in the
first plan, and still counted.

## The arms and the grader

The arms are drafted on her persona 27B with her app down, exactly as in the first attempt.

The grader is `states` with one change, made before any draft and to the rule only: for an answer that is a
number or a list of numbers, the paragraph read is the last one that states a number the request did not give,
and not simply the last paragraph. The first attempt graded "**331 × 571 × 34,231**" wrong because the reply
closed on a sentence naming only the number asked about. Yes or no answers and word orders are read as before.

## The criteria

As in the first plan: gained against lost on the discordant pairs, one-sided exact binomial, p < 0.05; no harm
(lost not more than gained at p < 0.05); every discordant row read by hand before the verdict, reported as
registered and as read. L09 closes when the primary criterion holds as read and no harm is found.

## What it can and cannot show

Whether, on what people ask chat assistants, her kept procedures change her answers for the better where they
reach. WildChat's people asked another assistant, not her; the requests are realistic, and they are not hers.
How often the route reaches such requests is reported beside the result.

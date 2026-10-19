# L09, second half: her kept procedures on against off over her ordinary use. Result, 18-19 October 2026

**Underpowered, as the plan says such a result is reported, not a pass or a fail.** Of the 195 distinct requests
typed into her chat (18 September to 18 October), her kept procedures reach 9. Over those 9 the registered grader
finds 3 gained and 1 lost (p 0.31); read by hand, the one loss is the grader's and the count is 3 gained, 0 lost
(p 0.125). Fewer than five discordant pairs: the plan
([preregistration](L09_ORDINARY_USE_PREREGISTRATION_2026-10-18.md)) does not let either count stand as a verdict.
Every gain is a number-theory request on which her ordinary reply ended inside her private reasoning with no
answer given.

## What ran

Drafting on her persona 27B with her app down, from 16:56 to 19:03 on 18 October (worktree l09-run, 50d8b9428),
graded at 19:03:59 by `states` in `tools/l09_ordinary_use.py`, the one grader fixed before any draft. Records:
`~/.aura/rlc-evidence/l09-ordinary-use-20261018/` (`rows/`, `truths.json`, `graded.json`).

| Request | Kind | Off (her reply) | On (with her kept procedures) |
|---|---|---|---|
| r038 "Quick one: what's 17 times 23?" | multiplication | right | right |
| r181 sort six words | word sorting | right | right |
| r182 gcd of 4862 and 7854 | gcd | right | right |
| r183 1203321 from base 4 to base 10 | base conversion | right | right |
| r189 "Is 829348951 a prime number?" | primality | **no answer** | **right** |
| r190 prime factors of 6560061883 | prime factors | right | right |
| r191 "Is 828044761 a prime number?" | primality | **no answer** | **right** |
| r192 prime factors of 6469693231 | prime factors | right | right (grader: wrong) |
| r193 prime factors of 21000063 | prime factors | **no answer** | **right** |

Misroutes: none. Every routed request was a question of the kind it was routed to.

## Each discordant row, read

- **r189.** Off: her model ended the decode inside its private reasoning after 29,005 tokens (1,768 s); nothing
  was said to the person. On: restated "Is 829348951 prime?", the kept answer (not prime; 7919 × 104729) shown to
  her as evidence, she replied "No, 829348951 is **not** a prime number." Right. A gain.
- **r191.** Off: the private reasoning ran the full 32,768 tokens (1,990 s) and never closed. On: "No, 828044761
  is **not** a prime number." Right. A gain.
- **r192.** Graded a loss; it is not one. The on reply reads "The prime factors of 6,469,693,231 are:
  **331 × 571 × 34,231**", then closes with "All three are prime numbers, and their product equals 6,469,693,231."
  `states` reads only the closing paragraph, which names the product and none of the factors. Both arms are right.
- **r193.** Off: the private reasoning ended without closing after 20,831 tokens (1,228 s); no answer. On: by the
  words, the kept procedure's "3, 7, 1000003", served. Right. A gain.

## What this shows and does not

Where her kept procedures reach her ordinary use, they help on one thing: number theory hard enough that her own
reasoning does not finish (three of four such requests, at 17 tokens a second: 20, 29 and 33 minutes). They harmed
nothing read by hand. But her ordinary use reaches them rarely: 9 of 195 requests, most typed to test her. Her
history cannot supply the five discordant pairs the plan asks for; the second half closes only when her use
grows, or on a plan that samples realistic requests beyond what has been typed so far.

The registered grader has one defect found here: a reply that states its answer and then closes on a sentence
about the answer is graded wrong. It is reported, not changed after the fact.

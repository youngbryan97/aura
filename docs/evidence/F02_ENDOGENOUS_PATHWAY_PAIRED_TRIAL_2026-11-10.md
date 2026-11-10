# F02: her endogenous pathway, at the strength it earned (10 November 2026)

Item F02 asked for the 18 October refit of her endogenous head to be installed after a voice check, and for a paired
trial showing the pathway changes her words.

## It was installed without one

The refit was already in the directory her app reads (`artifacts/endogenous_language`, fitted 18 October 15:24,
verdict `content_bearing`). Her log holds 6,787 lines of "Endogenous vocabulary bias ACTIVE (alpha=0.600, 3084
tokens)" from 20 October to 10 November. Alpha 0.6 was the flag's default. The head's own report says it was
evaluated at 0.05.

## The trial

`tools/measure_endogenous_pathway.py`, on her persona 27B with the installed head, through her worker's own processor
(the bias touches only tokens at least a tenth as likely as the model's first choice). Eight ordinary chat prompts are
decoded greedily for 48 tokens, once with no head and once under each of eight states she recorded. Her voice is
scored as F01's was: 32 of her recorded replies, the first 64 tokens of each after the turn "Go on.", log-likelihood
per token with the head under the state each was said in, minus with no head. Twenty-six had enough tokens to score.
The bars are F01's: the words change on more than half the prompts, two of her states decode differently on more than
half, and her replies are not less likely. The pathway runs at the largest alpha of a sweep at which all three hold;
with none, it is shut.

The voice bar was first written as the head's average bias on her words, without the model. A bias added to every
logit alike changes no probability, so the bar was replaced before any run on her 27B; the tool's docstring keeps
both versions.

| Run | Alpha | Words change | Differ by state | Her replies, own state | Borrowed state | Verdict |
|---|---:|---:|---:|---:|---:|---|
| 11:44, seed 20261110 | 0.6 | 8 of 8 | 8 of 8 | -0.0572 | -0.0548 | fails |
| 11:54 sweep, same seed | 0.3 | not decoded | | -0.0124 | -0.0119 | fails |
| | 0.2 | not decoded | | -0.0046 | -0.0045 | fails |
| | 0.1 | not decoded | | -0.0005 | -0.0005 | fails |
| | 0.05 | 7 of 8 | 6 of 8 | +0.0002 | +0.0002 | holds |
| 12:01, fresh seed 20261113 | 0.05 | 6 of 8 | 6 of 8 | +0.0002 | +0.0004 | **holds** |

Nats a token. An alpha whose voice cannot hold is not decoded. The confirmation was planned on disk before it ran
(`~/.aura/rlc-evidence/f02-paired-20261110/confirm/PLAN.md`); every report is in that directory.

## Installed

`earned_alpha.json` beside the head says 0.05 and the weights it was earned on. Her worker now runs the head at the
alpha it earned unless one is set explicitly, and a head with no certificate, or a refitted head whose weights differ
from the certificate's, is shut. Her app takes the change at its next start.

## What it does not show

At 0.05 her words change with her state and do not drift from her. Her own state did no better for her own words than
a state borrowed from another turn (+0.0002 against +0.0004 on the fresh seed). The head moves her words with her state;
this does not show that the state it reads makes her sound more like herself. Every reply scored was itself decoded
under the head at 0.6, so the voice reading is not independent of the head.

# G05 public answers: preregistered before any request is answered

Plan `9bbc11dc58b2`, written by `tools/g05_public_answer_protocol.py` at
`23082a4a2` and published in the commit that adds this page, before any of
its requests was generated into features or answered. The plan, spec and
summary are in `preregistrations/`.

## What is measured

Whether G04's frozen reader (candidate v17, the second G04 plan's candidate)
reaches the answer a person reads. Her runtime renders a settled answer with
the private channel closed, so that is where the comparison is made; the
[closed-channel pilot](G05_CLOSED_CHANNEL_PILOT_2026-10-07.md) on consumed
requests found her ordinary answers exact 61 times in 96 there and the
assisted ones 96 times.

Arms, one greedy decode each through her own template:

* ordinary: the request alone, channel closed;
* assisted: the request, then the reader's program and values as runtime
  evidence, channel closed;
* sham: another request's reading in the same form, channel closed;
* ordinary_open: the request alone, channel open at her serving effort
  (compositions only).

The answer is the last exact integer in her public reply; the private
channel is never read; a budget stop is not exact.

## Requests

* compositions: 48 fresh five-step requests with large numbers,
  `natural_weave_replication_6x5` at seed 20261012 (two per schema and
  domain), planned from the pilot's conservative bounds (discordance 0.269,
  assisted share 0.918 over 96 pairs): power 0.928 at alpha 0.05;
* lists: 120 fresh counts or lookups over 40 to 64 entries,
  `g05_long_sequence_v1` at seed 20261013, planned for the sham comparison
  from its pilot (12 harmed, none helped, over 24).

No request repeats any of the 2,732 consumed requests.

## Claims, fixed before the run

* Primary: assisted is exact more often than ordinary on compositions,
  exact one-sided McNemar at 0.05.
* Translation: wherever the reader's answer is right, in either stratum,
  the assisted answer is exact.
* G05 closes only if both hold.
* Secondary, for G06, alpha 0.025 each: assisted against ordinary_open on
  compositions (two-sided: what the cheap path costs against the expensive
  one); ordinary against sham on lists (one-sided: what wrong evidence
  costs). Tokens, seconds and terminations per arm are reported.

## What this does not show

* Anything yet: this page is the plan.

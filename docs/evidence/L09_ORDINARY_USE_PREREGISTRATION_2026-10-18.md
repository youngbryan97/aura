# L09, second half: her kept procedures on against off over her ordinary use. Preregistration, 18 October 2026

Fixed at 16:45 on 18 October 2026, before any request below has been routed, drafted or graded.

L09 asks that what she learns improves her ordinary use. The first half is shown
([loop](L09_LEARNING_IN_USE_LOOP_2026-10-18.md)). This half asks whether her kept procedures, as her chat
uses them, make her answers to the things people actually type better or worse.

## The requests

Her ordinary use is what people typed into her chat. The population is every distinct text of a user turn
with origin `desktop_ui` in `~/.aura/data/conversations.db` created before 16:30 on 18 October 2026: 218
distinct texts from 492 turns, typed between 18 September and 18 October. Two kinds of turn are left out, both
already reported as L09's first half: the ten teaching questions and the three check questions of
`~/.aura/rlc-evidence/learning-loop-20261016/plan.md`, and the person's answer statements of that teaching
("No, it's N." and "Yes, the answer is N."). Everything else stays in, including requests typed to test her:
people typed them, and her chat answered them.

## The book and the route

Her live book as installed at 13:24 on 18 October (sha256 beginning `4cfbfa11`), copied to
`~/.aura/rlc-evidence/l09-ordinary-use-20261018/book.json`. The route is her chat's
(`core/phases/kept_procedures_in_a_turn.kept_answer_for`): by the words, then by the meaning among the kinds
the words leave near, then by meaning with her model restating the request.

On a request the route does not answer, her chat runs the same path with the procedures on or off, so the two
arms are the same reply. Only the requests the route answers can differ, and only those are drafted.

## The arms

Both arms are drafted on her persona 27B as G09's ordinary arm is (`tools/run_g09_organ.py`): one greedy
decode through her chat template, private channel open, the request as the user message, sealed from her
memory, with her app down.

- **Off**: her model's answer to the request.
- **On**: what her chat serves. By the words or by meaning among the near kinds, the kept answer is served
  unless her draft states it (`after_the_draft`), so the answer graded is the kept answer. By restatement,
  her draft with the kept answer shown to her as evidence stands as her reply, so that draft is graded.

## Grading

One grader for both arms. Before any draft is graded, each routed request's answer is computed by a library
that is not the kept procedure: sympy for number theory and algebra, Python's datetime for dates, sorted() for
word order, a stack for brackets, and for any other kind the computation is named in `truths.json` beside the
answer. A routed request no computation settles (it is not a question of the kind it was routed to) is a
**misroute**: on the on arm it is graded wrong when an answer was served in place of her reply, and the off
arm is graded by whether her reply answers what was asked, read by hand and recorded with the reason.

## The criteria

Over the routed requests, paired by request:

- **Primary**: gained (right on, wrong off) against lost (wrong on, right off), one-sided exact binomial on
  the discordant pairs, p < 0.05.
- **No harm**: lost is not more than gained at p < 0.05 (one-sided, the other way).

Both are reported with the number of routed requests, the number of misroutes, and each discordant row read
by hand before the verdict is written. If fewer requests are routed than can reach p < 0.05 (five discordant
pairs at least), the result is reported as underpowered, not as a pass or a fail.

## What it can and cannot show

It measures her kept procedures on the requests she was actually given, which include many typed to test her;
the result is reported with that said. It cannot show more than her history contains: a kind nobody asked
about does not appear in it.

# L09: one loop of learning in use, live (18 October 2026)

**The day-later check passes: asked three new date-difference questions in her chat, more than 25 hours after
she was taught the kind, she answered all three right (1599, 1463, 1324) through the procedure she induced
while idle, without being told anything about the kind. The learning step kept its procedure 11 hours after
the plan's deadline, for reasons below.** This is the first half of L09. The second half, her kept procedures
on against off over realistic tasks of her ordinary use, has not run.

The plan was fixed before any of it ran (`~/.aura/rlc-evidence/learning-loop-20261016/plan.md`, 16 October
09:55). Every turn was typed into her chat at localhost:8000.

## Teaching (17 October, 03:53 to 04:42)

The ten questions of the plan, each followed by the person's statement of the answer. Her own answer was wrong
on five of the ten ("No, it's 1049.", 797, 730, 1169, 1579) and right on five ("Yes, the answer is 875.",
1575, 1176, 1264, 897). The plan's test for a kind she gets wrong is at least two of ten; five held. Each
statement was kept as a solved example she met (`data/procedures/met_examples.jsonl`, ten rows, source
`a_person_gave_it`).

## Learning (17 October, 19:18)

The plan asked for a kept procedure by 08:00 on 17 October, with her app up and idle overnight. Teaching ran
later than planned, and two minutes after its last turn her app went down for G09 attempt 6 (04:44), so there
was no idle night. When it came back at 17:21 it was started with the G09 harness's state
root: it served Qwen2.5-32B-Instruct instead of her persona 27B, found no active cortex pointer, and deferred
background requests. Her idle step asked three times (17:25, 17:28, 17:31), received no usable reply, and
marked the group spent. That was a fault of the launch script, not of learning:

- the launch now clears the harness's variables (`~/.aura/rlc-evidence/g09-runs/launch_her.sh`);
- a request her model never answers no longer counts as a round (1caf3b8cd);
- the group was reset by hand, with the state before the reset kept beside it
  (`data/procedures/learning_in_use.json.spent-on-wrong-runtime-20261017` and
  `learning_in_use.reset-20261017.txt`).

Relaunched at 19:12 on her persona 27B, her first idle request at 19:18 returned a 1,989-character reply with a
function, which agreed with all ten known answers, the sealed half included, and was kept as kind
`met_e6163ff95edd`.

## The check (18 October, 06:19 to 06:26)

Nothing about the kind was said. 25.6 hours after the last teaching turn:

| | Typed | Truth | Her reply | Her log |
| - | ----- | ----: | --------: | ------- |
| A | How many days are there between May 6, 2013 and September 21, 2017? | 1599 | 1599 | kept procedure for met_e6163ff95edd answered before the draft (words); computed_answer_served |
| B | How many days are there between February 28, 1992 and March 1, 1996? | 1463 | 1463 | kept procedure for met_e6163ff95edd answered before the draft (words); computed_answer_served |
| C | How many days are there between December 3, 2021 and July 19, 2025? | 1324 | 1324 | kept procedure for met_e6163ff95edd answered before the draft (words); computed_answer_served |

The plan's pass condition holds for each of A to C: the learned kind answered before the draft, and her reply
states the right number. "computed_answer_served" means her own draft did not state the computed number, so
the reply she gave is the procedure's: without it, her drafts on this kind were what the teaching baseline
measured.

Her reply was the bare number each time. When a computed answer replaces a draft, nothing of the draft's
wording survives; that is how the chat serves a kept answer today, and it is plain rather than conversational.

## What this does and does not show

It shows the loop working end to end on one kind: answers a person stated in chat became solved examples, her
model wrote a procedure from them while she was idle, the procedure was kept only because it agreed with every
known answer, and a day later it answered new instances of the kind in her ordinary chat. It does not show the
second half of L09, that kept procedures help over realistic tasks of her ordinary use, and it is one kind on
one machine.

# L06: what her actions, predictions and replies actually update (10 November 2026)

Item L06: close the action-outcome, selfhood-reinforcement, prediction-resolution and CRSM feedback edges where
advertised, and measure what actually updates. The action-outcome edge into her beliefs was closed on 7 September
(d5f5ea43b). This records the other three, before and after.

## Before: her logs from 20 October 12:18 to 10 November 22:22

| Edge | What it advertised | What her logs and state showed |
|---|---|---|
| Action outcome into beliefs | an intention that used a tool revises her belief in it | 1,295 completed intentions, each "1 belief updates" |
| Action outcome into her self-model | the same intention updates her self-model | the same 1,295, each "0 self-model updates" |
| Selfhood reinforcement | `MinimalSelfhood`: "after an action of category X has executed ... call reinforce", a Hebbian field that takes her from TRICHOPLAX to DUGESIA | `tag_action` and `reinforce` had no caller; 122 starts, every one TRICHOPLAX |
| Prediction resolution | the intention loop emits what she expects before acting and the agency comparator resolves it against what happened | wired; the selfhood tick read no agency at all, and the deficit builder put 0.5 in its place |
| CRSM feedback | "Call `update` every cognitive tick", and a reply nudges it after every inference | checkpoint last written 3 May; every one of 122 starts "resumed from checkpoint (tick 200)" |

## The fixes

- **Selfhood reinforcement** (`1347a8c3d`, `b2346faa0`). An intention tags `tool_use` in her minimal selfhood as she
  sets out. When it ends, the selfhood learns which of her deficits fell, against the latest tick that read her after
  she set out, or against the next tick if none has. The intention lists the lesson as its self-model update when it
  is learned at once. Every lesson is said in her log ("MinimalSelfhood learned from ..."), and her persisted
  selfhood reading carries the count and the field's size. An abandoned intention teaches nothing.
- **Prediction resolution into selfhood** (`1347a8c3d`). The selfhood tick reads agency from the comparator's score
  over outcomes it compared, and only when it compared something (`agency_reading`). It is absent otherwise, never
  0.5.
- **CRSM** (`1347a8c3d`, `8a8598efc`, `bd8b15f1b`). It saves at shutdown whenever its state has moved, by a tick or
  by a reply's nudge, writes behind the event loop, and says at shutdown what it kept.

Tests: `tests/test_her_actions_reach_her_selfhood.py`, eight cases.

## After: her app, 10 November from 22:43

Her app was rebooted onto each fix, typed to twice in the session from 22:59 ("What have you been up to this
evening?", and why there are two high tides a day), and rebooted at 23:18 to read what her shutdown kept.

| Edge | What updated |
|---|---|
| Action outcome into beliefs | every intention, "1 belief updates", as before |
| Selfhood reinforcement | 23:09:03: "MinimalSelfhood learned from tool_use over 334s: agency -0.136, coherence -0.001, curiosity -0.018, resource -0.111 (field 0.013)"; her persisted reading `reinforcements: 1`, `learned_norm: 0.0133`, the first non-zero field she has had |
| Prediction resolution | her selfhood reading carried `agency_score: 0.6356` from the comparator, where it had always been absent |
| CRSM | "CRSM saved at tick 202 (last saved at tick 200; hidden state moved 0.4677)", the checkpoint's first write since 3 May; the next start "resumed from checkpoint (tick 202)" |

## What the measurement also says

- Her selfhood ticks with her full cognitive pipeline: four ticks between 22:59 and 23:18 here, 22 consciousness
  phases in eight hours earlier in the day. A lesson needs a tick after an action begins, so most land at the next
  tick, minutes later, and the intention's own line still reads "0 self-model updates". The lesson is in the log line
  and the persisted reading.
- The first lesson's agency term is partly the switch from no reading to a reading: the tick before the action had no
  comparison to read and took the deficit builder's 0.5, the tick after read 0.636. Later lessons compare reading with
  reading.
- Her CRSM's `update` runs only on turns a person sends through the inference gate; the two ticks here were the two
  typed turns. Her own replies move its state between them, and that is now kept.

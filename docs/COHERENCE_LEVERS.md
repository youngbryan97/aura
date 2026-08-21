# Coherence levers

Every number governing how much of a conversation Aura can see is a named
environment variable, read **at call time** — a live runtime picks up a change
without a reboot, though the prompt cache will re-warm.

Set them in the launcher environment. For a soak instance, export them in
the shell that starts the run — they are read at call time, so a value set
after boot still takes effect.

## Why these exist

The reported failure — "great for a few turns, loses the plot, great again,
worse every time" — was not cognition. `build_system_prompt` sized its whole
trimming regime against a docstring claiming *"the 32B model has ~8K tokens"*.
The registry says `Qwen2.5-32B-Instruct` = **32,768**, and the live desktop
system prompt measured **2,189 chars ≈ 550 tokens**. She was discarding
continuity to defend a budget she was using about 2% of.

The policy was also inverted: continuity *shrank* as the conversation grew
(1800 → 600 → 400 chars past depth 20 and 30). At the depth in the reported
transcript, 46 turns, the entire prior conversation was allowed **400
characters**.

## Continuity budget

How much prompt is reserved for the thread. **Grows with depth** — continuity
gets more load-bearing as the raw transcript scrolls out of reach.

| Lever | Default | Effect |
| --- | ---: | --- |
| `AURA_CONTINUITY_FLOOR_CHARS` | `1800` | Budget at turn 0. |
| `AURA_CONTINUITY_CEILING_CHARS` | `4800` | Budget once the ramp completes. |
| `AURA_CONTINUITY_RAMP_TURNS` | `40` | Turns taken to reach the ceiling. |

```bash
AURA_CONTINUITY_CEILING_CHARS=9000 AURA_CONTINUITY_RAMP_TURNS=25
```

Raising the ceiling is cheap: 9,000 chars is roughly 2,250 tokens against a
32,768-token window. Restoring the old behaviour (do not — this is the bug) is
`AURA_CONTINUITY_FLOOR_CHARS=400 AURA_CONTINUITY_CEILING_CHARS=400`.

## Continuity ledger

The durable record. Entries are whole propositions with the turn they came
from; over budget, whole entries are **evicted by salience and counted**, never
shortened. A retained entry is byte-identical to when it was written.

| Lever | Default | Effect |
| --- | ---: | --- |
| `AURA_CONTINUITY_LEDGER_CHARS` | `3200` | Hard cap on the rendered block. |
| `AURA_CONTINUITY_LEDGER_CAPACITY` | `240` | Entries held before eviction. |
| `AURA_CONTINUITY_HALF_LIFE_TURNS` | `60` | Turns for recency weight to halve. |
| `AURA_CONTINUITY_SUBJECT_TRAIL` | `6` | Subject transitions retained. |

Salience weights decide what survives an eviction. A thing the person said
about themselves outranks a topic marker, because losing it is what makes her
sound like she has never met them.

| Lever | Default |
| --- | ---: |
| `AURA_CONTINUITY_W_DISCLOSURE` | `3.0` |
| `AURA_CONTINUITY_W_COMMITMENT` | `2.6` |
| `AURA_CONTINUITY_W_POSITION` | `2.0` |
| `AURA_CONTINUITY_W_QUESTION` | `1.6` |
| `AURA_CONTINUITY_W_SUBJECT` | `1.2` |

**Longer memory for facts about the person:**

```bash
AURA_CONTINUITY_HALF_LIFE_TURNS=120 AURA_CONTINUITY_W_DISCLOSURE=5.0
```

**Cheaper prompt on a loaded host** (continuity degrades gracefully rather
than vanishing at a depth cliff):

```bash
AURA_CONTINUITY_LEDGER_CHARS=1600 AURA_CONTINUITY_CEILING_CHARS=2400
```

## Reading whether it is working

`quality_metrics` now carries the numbers that matter:

```
📊 quality_metrics | … | ledger_entries=41 | state_coherence=0.814 |
   thread_abandoned=False | overlap_turn=0.31 | overlap_thread=0.22 | assessment=ok
```

* `ledger_entries` — how much durable continuity exists. Flat at 0 deep into a
  conversation means compaction is not folding into the ledger.
* `thread_abandoned` — the reply did not engage the user's turn *or* the live
  thread. This is the octopus case.
* `state_coherence` — kept for continuity with older logs. It is
  `state.cognition.coherence_score`, an **internal** score that never looks at
  the conversation; it read `0.814 ok` on the turn before a total non-sequitur.
  Do not read it as a measure of the thread.

A reply that abandons the thread also logs on its own:

```
🧵 Reply abandoned the thread (overlap_turn=0.0 overlap_thread=0.0) — user=… reply=…
```

## What is deliberately *not* a lever

`MAX_WORKING_MEMORY` (150) is a **count** bound, not a token bound, and stays
fixed. Raising it does not buy coherence — the ledger is what carries the
conversation past the compaction horizon, and it is bounded in characters
precisely so a 500-turn conversation costs no more prompt than a 50-turn one.

# Her training buffer, only what people said: 19 October 2026

## What was found

`~/.aura/data/learning/experience_buffer.jsonl` is what her live learner, the weight-compounding loop and the genuine
learning pipeline read as training data for her model. On 19 October it held 9,638 rows. 9,138 of them had no source
and opened with her own runtime's requests, with her replies to them:

| Rows | The user turn opens |
|---:|---|
| 1,428 | "Think deeply about autonomous agent swarm protocol" |
| 1,071 | "[SYSTEM ROLE: EPISTEMIC ARCHITECT] You have just ..." |
| 890 | "You are fixing a bug in your own code. FILE: core/..." |
| 884 | "You are the Master Synthesizer. Review the original ..." |
| 813 | "[SWARM PROTOCOL: You are 'The Critic'. Analyze the ..." |
| 806 | "Summarize the following sequence of internal system ..." |
| 581 | "Check in with your state and tell me right here: ..." (tools/conversation_endurance_probe.py) |
| 481 | "[SWARM PROTOCOL: You are 'The Architect'. Design ..." |
| 187 | "Ok. Once more. You with me?" (written by core/self_model.py and core/goals/standing_objective.py) |

9,138 of the 9,638 rows opened with a "You are Aura." system text. Trained on, they would teach her the scripts. The
cause: the learning phase recorded every tick's objective with her reply, and a tick's objective is whatever the kernel
pursues, a person's message or one of its own.

The other 500 rows are her conversations with Bryan from 21 March to 25 August 2026 ("Hey, Aura. My name is Bryan.
Bryan Young."), recorded as conversation turns before `conversations.db` began on 20 September.

## What changed

- `core/learning/live_learner.py` (2ff84fb6b): a tick is kept as training data only when a person started it, a
  user-facing origin the runtime has not named a probe or a benchmark of its own.
- `tools/quarantine_what_no_person_said.py`, applied at 12:45 with her app down: the 9,138 rows no person said moved,
  whole, to `experience_buffer.no-person-said-20261019.jsonl`; the buffer copied first to
  `experience_buffer.jsonl.before-quarantine-20261019`. 500 rows remain. Nothing was deleted.

## What is left

The same role texts are in other stores: her decision preferences (5), subjective choices (398), task commitment
state (3), continuity record (2), dream fragments and the stream of being. The task engine's model decomposition is
still a written instruction ("Decompose this goal into a sequence of concrete steps ... Rules: ..."), and the ReAct
loop opens "You are Aura, an autonomous cognitive agent."

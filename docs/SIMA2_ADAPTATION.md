# SIMA 2 in Aura

Bryan, 25 Sep 2026: adapt everything SIMA 2 can do into Aura, general and
clean-room, learning how DeepMind did it before building. This file is the
working plan: what SIMA 2 does, what Aura already has for each faculty, and
the order the rest is built in. The research behind it is in
`~/.aura/research/reports/SIMA 2 adaptation for Aura.md` (primary sources:
the SIMA 2 technical report, arXiv 2512.04797, and the SIMA 1 report, arXiv
2404.10179).

## What SIMA 2 is, in one paragraph

A Gemini Flash-Lite model fine-tuned to read 720p frames and write one text
stream that mixes private reasoning, dialogue and actions: 96 keyboard keys,
mouse clicks and discretised relative mouse moves, parsed deterministically
into input events. It is trained on human play, a small set of successful
spans Gemini Pro annotated with reasoning ("bridge data"), and online RL on
tasks with programmatic success checks. It then improves itself: one Gemini
model proposes tasks it judges achievable from the current frame, a second
scores each attempt 0-100 against a rubric (50 passes), and the scored
experience trains the next generation. On its training games it scores 65%
(SIMA 1: 33%, time-limited humans: 76%); on unseen games about 14% (ASKA) and
13% (MineDojo). Gemini Pro prompted to act with no action training scores 7%.

## Six rules the build follows

1. **The 27B sits where Gemini Pro sits in SIMA 2's hierarchy**: planner,
   narrator, task setter and judge, called on events, never on every frame.
2. **Per-frame acting is a separate executor** that learns from action data,
   and Aura's own logs are action data. Look-ahead search over her world model
   handles short precise sequences (SIMA 2's weakest category is combat, 25%
   against 64% for humans); a small policy distilled from search and verified
   successes handles routine ones.
3. **One action grammar** for games and the desktop, parsed deterministically.
4. **A grade is checked before it is trusted.** Programmatic verifiers first;
   the language judge sees only the instruction and the final frames with
   their OCR, uses a prompt different from the setter's, and never scores the
   held-out suite.
5. **The 27B stays frozen.** Action training cost Gemini a quarter of its
   AIME score.
6. **Practice happens where a mistake costs nothing**: save files, virtual
   machines, her own world model. Every new faculty ships behind a switch so
   its worth can be measured by turning it off.

## What Aura has, faculty by faculty

Status: **have** (built and wired to screen work), **part** (built, not
wired to screen work, or covering part of it), **none**.

### Perception and grounding

| Faculty | Aura today | Status |
|---|---|---|
| Pixels as the only input | `core/perception/eyes_of_their_own.py`, `what_the_pixels_show.py` (Vision OCR in a child process) | have |
| Reading on-screen text to act | OCR feeds the screen loop's reading and goal checks | have |
| Image and sketch prompts | `core/brain/llm/mlx_vision_client.py` (Qwen3-VL-4B) answers "read my screen"; not used to set a goal | part |
| Objects where OCR finds nothing | none on the screen loop | none |

### Action and real-time control

| Faculty | Aura today | Status |
|---|---|---|
| Keyboard | `screen_pursuit_surface.press_key`: taps only, a bounded key set | part |
| Held keys with durations | `window_server` can post key-down and key-up; nothing holds a key | none |
| Absolute clicks | `host_automation.click_at`, `screen_pursuit_surface.click_normalized` | have |
| Relative mouse (camera look) | none | none |
| Scroll, drag | `host_automation.scroll`, `computer_interface.drag` | have |
| Action chunks with empty slots | none: one act per look | none |
| Latency hiding | eyes and search off the loop; no measured perception-to-action delay per world | part |
| Staying still once done | the loop stops on success; no Done token | part |
| Precise control by look-ahead | `looking_ahead`, `a_world_compiled` on boards | part |

### Language, instructions and dialogue

| Faculty | Aura today | Status |
|---|---|---|
| Following an instruction | `core/runtime/watched_goal.py` reads a request into a goal, keys and a finish | have |
| Multi-step chains | one goal per run | none |
| Multilingual, emoji | the 27B reads them; nothing normalises them into a screen goal | part |
| Voice | speech-to-text exists in the voice stack | part |
| Completion reports, narration | `saying_what_a_move_does`, narration every move | have |
| Clarifying questions | conversation-level only, not when a screen offers two candidates | part |
| Answering questions by exploring | none | none |
| Explaining intentions | the standing strategy line, spoken | have |

### Reasoning, memory and horizon

| Faculty | Aura today | Status |
|---|---|---|
| Reasoning that conditions action | `core/agency/standing_strategy.py` (a line her search is scored by) | have |
| Choosing when to think | `core/agency/worth_thinking_about.py` | have |
| Orchestrator above the executor | language every few moves in the screen loop | part |
| Memory past the context window | per-world records in `what_she_learned`; no rolling summary per run | part |
| Standing rules ("do the opposite") | none | none |
| Recovery after failure | restart handling; no written failure note read on the retry | part |
| Goal verification | `goal_reached`, layout check | have |
| Active exploration | `HowItMoves.worth_trying` (acts that split hypotheses) | part |

### Transfer and retention

| Faculty | Aura today | Status |
|---|---|---|
| Transfer from breadth | rules composed per world; kinds of world carried (`CARRIES_TO_A_WORLD_LIKE_IT`) | part |
| Zero-shot unseen worlds | `tools/measure_getting_there.py` (9 grid worlds) | part |
| Concept transfer across games | none | none |
| Keeping general ability | the 27B is not trained on actions | have |

### Learning and self-improvement

| Faculty | Aura today | Status |
|---|---|---|
| Imitation of her own play | action logs exist; nothing trains on them | none |
| Causal and hindsight labels | `core/environment/experience_replay.py` (hindsight replay, environment kernel only) | part |
| Bridge data (reasoning written onto successes) | none | none |
| RL on verifiable tasks | rehearsal in her model (`rehearsing_in_her_model`, `working_out_what_matters`) | part |
| Task setter | `core/environment/curriculum.py` (environment kernel only) | part |
| Reward model / judge | none for screen tasks | none |
| Experience bank and retraining | `core/learning/verified_replay_sft.py` (language, not actions) | part |
| Learning in a generated world | her compiled world model | part |

### Evaluation

| Faculty | Aura today | Status |
|---|---|---|
| Three success functions (ground truth, programmatic, human) | programmatic only, per run | part |
| Suite rules (persisting success, post-completion cap, chains, held-out states) | none | none |
| Human baselines | none | none |
| Skill taxonomy | none for screen tasks | none |

### From agents SIMA 2 does not have

| Mechanism | Aura today | Status |
|---|---|---|
| Verified skill library | `core/agency/skill_library.py`, `macro_skill.py` (not admitted by a verifier) | part |
| Plan memory keyed on screen | per-world records | part |
| Best-of-N attempts | none | none |
| Inverse dynamics labelling | none | none |
| Training inside the world model | rehearsal tunes judging, not a policy | part |
| Unified action space across OS and games | none | none |
| Confirmation gate for consequential acts | governance and `task_commitment_verifier` | have |
| Learned success detector | none | none |
| Explore first, name the task after | none | none |

## Build order

Each batch lands with its tests, one commit, and a measurement.

1. **Action grammar and executor.** One text grammar: key taps, holds
   with durations, relative mouse moves, normalised absolute clicks,
   scroll, drag, an empty slot, `Done`, `Think`. A deterministic parser,
   an executor that plays a chunk through Quartz events, per-world key
   and mouse calibration learned from what each act does (the way
   `HowItMoves.way_of` learns a key's direction).
2. **The orchestrator seat.** The 27B issues a subgoal on events, keeps a
   rolling run summary, holds standing rules and multi-step chains as a
   queue, asks when the screen offers more than one match, and reports at
   subgoal boundaries.
3. **Setter, judge and experience bank for screen tasks.** Tasks proposed
   from the current screen and ranked by learning progress; programmatic
   verifiers first, a separate language judge on the final frames and OCR;
   episodes tagged causal or hindsight and kept with their grades.
4. **The screen task suite.** SIMA's eight skill categories plus desktop
   categories, the suite rules above, trial counts and confidence intervals,
   a held-out set scored by nothing that trained her.
5. **The fast executor.** Filtered imitation of passing episodes and of
   search's own choices, into a small policy; an inverse dynamics model on
   her logs; latency measured per world and trained on.
6. **Transfer.** Skills indexed by verb and object across worlds, verified
   before admission; plan memory keyed on screen.

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
| Keyboard | `core/agency/what_hands_do.py` grammar, `core/capabilities/hands.py` player; letters and digits pressable, Command never | have |
| Held keys with durations | a key named across slots is one press held (`hands.play`) | have |
| Absolute clicks | `host_automation.click_at`, `screen_pursuit_surface.click_normalized` | have |
| Relative mouse (camera look) | `QuartzHands.move_by` sets the event's delta, which games read; gain learned per world (`how_the_view_moves`) | have |
| Scroll, drag | `host_automation.scroll`, `computer_interface.drag` | have |
| Action chunks with empty slots | `Chunk` of slots, `.` for an empty one, `done` and `think` | have |
| Latency hiding | eyes and search off the loop; no measured perception-to-action delay per world | part |
| Staying still once done | `done` ends a chunk; a made layout is not moved out of | have |
| Precise control by look-ahead | boards: `looking_ahead`; moving things: walking while steering, 21 of 30 chases caught; a lead by drift is built and switched off (11 of 30) | part |

### Language, instructions and dialogue

| Faculty | Aura today | Status |
|---|---|---|
| Following an instruction | `core/runtime/watched_goal.py` reads a request into a goal, keys and a finish | have |
| Multi-step chains | one goal per run | none |
| Multilingual, emoji | the 27B reads them; nothing normalises them into a screen goal | part |
| Voice | speech-to-text exists in the voice stack | part |
| Completion reports, narration | `saying_what_a_move_does`, narration every move | have |
| Clarifying questions | a trip where two things answer to the name hands the choice back with both named (`going_to_what_she_sees`) | have |
| Answering questions by exploring | looking around for a named thing, half a view a step (`in_a_world_through_a_camera`) | part |
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
| Active exploration | `HowItMoves.worth_trying`; the camera sweep | part |

### Transfer and retention

| Faculty | Aura today | Status |
|---|---|---|
| Transfer from breadth | rules composed per world; kinds of world carried (`CARRIES_TO_A_WORLD_LIKE_IT`) | part |
| Zero-shot unseen worlds | 9 grid worlds (`measure_getting_there`); generated camera worlds with remapped keys and mouse (`measure_in_camera_worlds`) | have |
| Concept transfer across games | none | none |
| Keeping general ability | the 27B is not trained on actions | have |

### Learning and self-improvement

| Faculty | Aura today | Status |
|---|---|---|
| Imitation of her own play | (view, chunk) pairs kept per episode; nothing trains on them yet | part |
| Causal and hindsight labels | `core/environment/experience_replay.py` (hindsight replay, environment kernel only) | part |
| Bridge data (reasoning written onto successes) | none | none |
| RL on verifiable tasks | rehearsal in her model (`rehearsing_in_her_model`, `working_out_what_matters`) | part |
| Task setter | `core/agency/setting_herself_a_task.py`: untried first, then nearest even odds; mastered sends her looking elsewhere | have |
| Reward model / judge | the world's own answer: words that came or a prompt that went with its thing still in front (`in_a_world_through_a_camera`) | have |
| Experience bank and retraining | `core/agency/what_she_tried.py`: every episode per world, who set it, chunks and the view before each; nothing trains on it yet | part |
| Learning in a generated world | her compiled world model | part |

### Evaluation

| Faculty | Aura today | Status |
|---|---|---|
| Three success functions (ground truth, programmatic, human) | programmatic only, per run | part |
| Suite rules (persisting success, post-completion cap, chains, held-out states) | `tools/measure_in_camera_worlds.py`: generated worlds, per-category Wilson intervals, a random-act null | part |
| Human baselines | none | none |
| Skill taxonomy | navigation, using, finding, asking, chasing | part |

### From agents SIMA 2 does not have

| Mechanism | Aura today | Status |
|---|---|---|
| Verified skill library | `core/agency/skill_library.py`, `macro_skill.py` (not admitted by a verifier) | part |
| Plan memory keyed on screen | per-world records | part |
| Best-of-N attempts | none | none |
| Inverse dynamics labelling | none | none |
| Training inside the world model | rehearsal tunes judging, not a policy | part |
| Unified action space across OS and games | one grammar for keys, holds, relative mouse, clicks in window shares, scroll | have |
| Confirmation gate for consequential acts | governance and `task_commitment_verifier` | have |
| Learned success detector | none | none |
| Explore first, name the task after | none | none |

## Proven live (25-26 Sep)

In `tools/a_room_to_walk_in.py`, a first-person room in a real window read
by her own eyes and driven by Quartz events: she learned which key walks and
how the mouse turns the camera by trying them, opened the Door (6.1 s), then
looked around for the Chest behind her, walked to it and opened it (6.8 s).
"Walk over to the chest in Python and open it", passed to `pursue_on_screen`
as a person would say it, did the same in fourteen chunks.

## Measured (26 Sep)

Generated camera worlds, thirty per category, each with its own walking key,
mouse direction and gain, layout and prompt keys; she is told none of it.

| Category | Her | Random acts |
|---|---|---|
| navigation | 30/30 | 0/30 |
| using | 30/30 | 0/30 |
| finding | 30/30 | 0/30 |
| asking | 30/30 | 0/30 |
| chasing | 21/30 | not run |

Live, in the test room, three fresh rooms: door and chest opened six times
of six, 8 to 12 seconds a trip. Self-set practice: she set herself both,
opened both, and did not repeat what she had managed.

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

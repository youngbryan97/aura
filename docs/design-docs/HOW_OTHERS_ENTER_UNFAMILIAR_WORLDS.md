# How others enter unfamiliar worlds

Status: Research map · What the systems that play unfamiliar games best do, what people do, and where each idea lives in Aura or is still to be built

Gathered 10 October 2026. Each line says what a system or a study found,
then where Aura does the same thing, or that she does not yet.

## People

**What a person brings before the first frame.** Dubey et al. (ICML 2018,
"Investigating Human Priors for Playing Video Games") masked one kind of prior
at a time in a small platformer. With everything visible people finished in
about 1.8 minutes; with all object priors masked it took about 20. In order of
how much each mattered: that the world is made of objects (about 4x slower
without it); that things which look alike behave alike (about 4x); what things
are (coins, fire, keys, ladders: about 2.4x, worse when reversed); what they
afford (about 2.6x); physics and controls (about 1.7x). Objects act as
subgoals: people walk to the key before the door, and once they learn what one
texture does they expect the same of its look-alikes. A curiosity-driven RL
agent was barely slowed by any of this, which says the slowdown is lost
knowledge, not visual clutter.

- Objects: `core/perception/what_moves_in_the_picture.py` (things over a
  backdrop), `core/perception/shapes_that_look_pressable.py` (still shapes).
- Look-alikes behave alike: kinds in `what_moves_in_the_picture.py`; a kind
  not yet met is judged as one like it (`core/agency/what_meeting_things_does.py`).
- What things are: `core/cognition/what_things_are.py`,
  `core/agency/mechanics_she_knows.py`.
- What they afford: partly, through `what_things_are.py` ("a platform is stood on").

**How a person learns one.** Gee (2003) names the cycle good games ask for:
probe the world, form a hypothesis, probe again with it in mind, and keep or
rethink it. Tsividis et al. (2021, EMPA) built that into an agent that learns
Atari-style games in minutes, as people do: a theory of the game (object
classes, what each pair does on contact, what wins and loses), explored by
making contact with each pair of kinds never yet seen to touch, because one
contact says nearly everything about the pair; dangers avoided once found;
the count that can go to zero taken as the likely end. Taking exploration away
was what hurt most: the agent failed games whose ends needed a chain of steps.

- Contact as a question: `what_meeting_things_does.py` learns what meeting
  each kind did, and a kind she has not met is one to meet unless what it looks
  like says it is trouble (`what_things_are.py`); her fast loop goes to a known
  good thing before a still one she has not met.
- Hypotheses held and overturned by play: the evidence tiers of
  `core/cognition/what_this_place_is.py` (a name, then what is seen, said,
  and played).

## Systems

**The Pokémon runs (Gemini 2.5 and 3 Pro, Claude, GPT-5).** What carried them
was the harness: a mental map kept only from what the screen showed, with
"reachable unseen tiles" listed; map markers and a notepad for plans and
hypotheses; three goals (the next step on, what enables it, what is handy)
put back after every summary; a critique pass that read the history for bad
goals and invented facts; a pathfinder and puzzle solvers called when
needed; later, code the agent wrote for itself (`press_sequence`). The fixes
that mattered were real values read in place of the model's beliefs (0 PP
that was not), summaries that broke loops, and "ground what you know in what
you see, not in other games".

- Screens as a map, untried acts first, the nearest screen with something
  untried next: `core/agency/where_things_lead.py`.
- Goals and a plan from the start to the end: `core/cognition/a_plan_to_an_end.py`.
- Notes and counsel: `core/cognition/a_guide_to_a_place.py`,
  `core/cognition/taking_stock.py`.
- Real values over belief: counters, bars and readings measured on the
  screen (`what_meeting_things_does.py`, `core/agency/what_she_has_left.py`).
- **Not yet:** a map of a world bigger than one screen. She sees the view go
  by (`core/perception/how_the_scenery_goes_by.py`) and keeps no account of
  where that has taken her or how far she has come.

**Voyager (Minecraft, 2023).** A curriculum it sets itself from where it is and
what it has done; a library of skills written as code, kept when verified and
reused in later tasks; programs revised from what running them showed.

- Setting herself the next thing to practise: `core/agency/setting_herself_a_task.py`.
- Composed acts: `core/cognition/an_action_she_composed.py`.
- What worked kept and borrowed by the most alike world:
  `core/agency/the_world_it_is_most_like.py`, `core/agency/what_worked_before.py`.

**SIMA 2 (DeepMind, 2025).** Reads only the screen and sends keys and the
mouse. Learns a new game by its own play: a task and an estimate of reward
proposed, the attempt kept as experience, failed tasks tried again.

- Screen and hands only: her whole play path. Tasks she sets and keeps at:
  `setting_herself_a_task.py`, `core/agency/what_she_tried.py`.

**Cradle (general computer control, ICML 2025).** Six parts: gather what the
screen shows, reflect on the last act, infer the next task, curate skills,
plan the act, remember. Same as hers: `core/skills/screen_pursuit*.py`.

**ARC-AGI-3 explorers (2025).** The third-placed agent used no learning: it cut
each screen into parts, ranked acts by how much each part stands out, and went
to the nearest state with an act not yet tried. Built in
`where_things_lead.py` after this research the first time.

**Worlds as programs (WorldCoder 2024, PoE-World 2025, AutoManual 2024).** A
model of the world written as code or as rules, fitted to what play showed,
planned in; or a manual of rules kept by one part of the agent and read by
another. Hers: her rule of a world (`core/agency/a_world_compiled.py`), the
rules she reads (`core/cognition/reading_the_rules.py`) and the guide.

**What the benchmarks say fails.** BALROG (2024): models know things they do
not act on (that rotten food is dangerous, and eat it), the knowing-doing gap;
and they do worse given pictures than text. VideoGameBench (2025): the best
model finished under 1% of its games, mostly because the game moved on while
the model thought. Her fast loop plays by measurement at the speed the
pictures come (`core/agency/playing_as_it_happens.py`), and what she was told
becomes the stances it plays by.

## To build

1. A world bigger than the view: where the view going by has taken her,
   how far she has come along the way it goes, and where she has not been.

# How people and systems act toward an end

Status: Research map · How people, game designers, game-playing agents, computer-use agents and robots get from where they
stand to what they are there to do, by chains of acts that use one thing for another; and where each idea lives in Aura

Gathered 10 October 2026, as the second half of
[HOW_OTHERS_ENTER_UNFAMILIAR_WORLDS.md](HOW_OTHERS_ENTER_UNFAMILIAR_WORLDS.md), which covers how a place is first
explored and learned. This one covers what comes after: an end held in mind, the means the place offers, and the chain of
acts between them. Each entry says what a study or a system found, then where Aura does the same, or that she does not
yet.

The occasion: on 10 October she read a trap-building game's lesson and goals, made a sound plan (open the device library,
pick a type that fits the gap, put each device at the end of the last one's arrow, turn it toward the cage), and then
clicked the shapes on the page and tested an empty trap. The plan and the acts never met.

## People

**Means-ends analysis.** Newell and Simon's General Problem Solver (1972) is still the plainest account of goal-directed
action: compare where you are with where you want to be, find an operator that reduces the difference, and where its
precondition is not met, make meeting it a subgoal. People work this way on unfamiliar puzzles; the Tower of Hanoi
protocols show the subgoals being stacked and unwound.

- The step before a step: `core/skills/screen_pursuit_decision.py` (`_reaching_what_the_next_step_needs`) takes, where
  nothing on the screen does the next step, the act found to bring up what that step uses.
- Working backward from what is out of reach: `core/cognition/what_would_have_to_be_true.py`.

**People plan on a simplified picture, chosen for the plan.** Ho et al. ([Nature, 2022](https://cocosci.princeton.edu/papers/ho2022people.pdf))
found that people leave out of their mental picture whatever would not change the best route ("value-guided
construal"). Correa et al. ([PLOS Computational Biology, 2023](https://cocosci.princeton.edu/papers/correa2023decompose.pdf))
found that people choose subgoals that cut the cost of planning while keeping the result, and that the subgoals they
pick are the bottleneck states a route has to pass through.

- Subgoals as bottlenecks: the screens a way on passes through, in `core/agency/where_things_lead.py`.
- **Not yet:** a construal step that drops from her picture what cannot change the route before she plans.

**Tool use is imagined, tried and corrected.** Allen, Smith and Tenenbaum ([PNAS, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7703630))
had people solve physical puzzles by choosing an object and placing it (to launch, block or support a ball). People
solved most within a handful of tries. Their model of it, Sample, Simulate, Update: object-based priors propose a few
tools and placements, a rough mental simulation runs each, and the outcome of each real try updates which tools and
places are promising.

- Proposing what to use from what things are: `core/cognition/what_things_are.py`, the plan's means in
  `core/cognition/a_plan_to_an_end.py`.
- Simulating before acting: `core/agency/looking_ahead.py` over her compiled rule of a world
  (`core/agency/a_world_compiled.py`).
- **Not yet:** a simulation of placed parts for a chain-building game; her plan's steps are tried in the world.

**Affordances.** Gibson's word for what an environment offers an animal; Norman's for what a designed thing signals it is
for. A handle affords pulling; a lit square signals "put it here". Dubey et al. (2018, in the companion doc) measured how
much slower people play without knowing what things afford.

- What she has found each thing on a screen does: `WhereThingsLead.did` and `what_things_do()` in
  `core/agency/where_things_lead.py`, given to her plan.

**Objects persist and correlate.** Infants expect a hidden object to still be there and to reappear where its path
leads (Baillargeon's violation-of-expectation studies; [ADEPT](https://www.mit.edu/~k2smith/pdf/Smith_et_al-2019-Modeling_Expectation_Violation.pdf),
Smith et al. 2019, tracks objects through occlusion with a particle filter). They find themselves in the world by
contingency: Bahrick and Watson ([1985](https://infantlab.fiu.edu/publications/publications-by-date/publications-1975-1989/1985_bahrickwatson_dp_detection-of-intermodal-proprioceptive-visual-contingency.pdf))
showed five-month-olds tell a live video of their own legs from a recording, by whether it moves when they do.

- The same thing when hidden: `core/cognition/entity_track.py` for windows and pages, and in her fast loop
  `core/perception/what_moves_in_the_picture.py`: a moving thing gone out of sight is kept a few seconds with where
  its going was taking it, and a thing of its kind coming back into sight near there is that thing again, its number,
  path and way kept (before, she forgot a thing a quarter-second after it went behind something).
- Which thing is hers, by what answers her presses: `core/agency/which_one_answers_to_her.py`.

**Looming.** What grows fast in the eye is about to arrive, and animals flinch from it before they know what it is. Lee
(1976) showed the rate of growth gives the time to contact without knowing distance or speed.

- `core/agency/warning_signs.py`: a thing of a kind she does not know yet, closing on her faster than she can answer, is
  kept clear of until she knows what it is.

## Games as teachers

**Games teach by their layout.** World 1-1 of Super Mario Bros. places a ? block high enough that a player jumps at it,
and bricks over a Goomba so the jump lands on it; Miyamoto's aim was that the level holds everything a player needs to
work it out. Nintendo's designers describe a level as introducing one mechanic safely, developing it, twisting it, and
ending (kishōtenketsu, [Game Maker's Toolkit on Super Mario 3D World](https://www.mcvuk.com/development/video-nintendos-level-design-secrets-in-four-steps)).
A first meeting with a mechanic is meant to be survivable.

- A lesson read and followed: `core/cognition/reading_the_rules.py`, and watching a lesson play before acting.
- **Not yet:** reading the layout as the lesson (a thing placed high is to be jumped at; a lone enemy on open ground is
  the first one to learn from).

**A game is patterns to learn.** Koster's *A Theory of Fun* (2004) treats play as learning a pattern until it is
mastered; Hunicke, LeBlanc and Zubek's [MDA](https://users.cs.northwestern.edu/%7Ehunicke/MDA.pdf) separates a game's
mechanics (its rules), its dynamics (what the rules do in play) and its aesthetics (what a player feels). An agent can
read the mechanics in code and the dynamics only by playing.

- Mechanics from code: `core/cognition/reading_the_code.py`. Dynamics from play: `core/agency/what_meeting_things_does.py`,
  `core/perception/what_the_world_does.py`.

**Progress is gated.** Dormans' missions and spaces split what a player must do (the mission, a graph of locks and keys)
from where they do it (the space); his dungeons are built from cycles such as lock-and-key and hidden shortcut
([cyclic generation in Unexplored](https://boristhebrave.com/page/7)). Metroidvanias gate areas behind abilities.

- The step before a step, and the screens a way on runs through: `where_things_lead.py`.
- Locks remembered: `core/cognition/locks_she_met.py` keeps what a screen said would not open, with what it wants in its
  own words; a gain that names it ("You found the brass key!") makes it a place to go back to by the screens she knows.
- **Not yet:** a lock seen rather than said (a gap too wide, a ledge too high).

**Action games telegraph their attacks.** In Sekiro every enemy move is shown before it lands, by a wind-up, a flash or a
sound, and each has its answer ([a design analysis](https://www.superjumpmagazine.com/the-art-and-science-of-sekiros-combat/)).
Pixel-only reinforcement learning on all twenty-two Dark Souls bosses learned almost nothing
([DSLE, 2026](https://arxiv.org/pdf/2608.09902)); the state-based environments that do learn read the animation timers
directly ([SoulsGym](https://pypi.org/project/soulsgym)).

- `core/agency/warning_signs.py`: what each kind of thing does just before she loses (it stops, grows, rushes, turns
  toward her) is held against how often a loss follows any moment, and a change that warns more often than not is a
  tell; a thing showing its kind's tell is kept clear of until the moment passes, and the tell is said once.

## Systems that play

**Planners in shipped games.** Orkin's goal-oriented action planning for F.E.A.R.
([GDC 2006](https://pages.cs.wisc.edu/~dyer/cs540/handouts/gdc2006_orkin_jeff_fear.pdf)) gives each act preconditions and
effects and searches with A* from the world as it is to a goal state; a character replans when the world changes under
it. Baumgarten's A* agent won the [2009 Mario AI Competition](https://gpbib.cs.ucl.ac.uk/gp-html/Togelius_2010_cec.html)
by running the game's own physics forward over every move it could make.

- Acts with what they brought about, searched toward what a step needs: `WhereThingsLead.toward`.
- Running a world forward: `looking_ahead.py`, `a_world_compiled.py`, `core/agency/what_a_press_does.py` (a jump's
  curve, learned by watching, run forward to choose).

**Plans that check themselves.** DEPS ([NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/file/6b8dfb8c0c12e6fafc6c256cb08a5ca7-Paper-Conference.pdf))
has a model describe what happened when a subgoal failed, explain why, and replan, and orders parallel subgoals by how
near each is. ADaPT ([2023](https://arxiv.org/pdf/2311.05772)) breaks a task down only when doing it fails, and
recursively. AERA on ARC-AGI-3 ([2026](https://arxiv.org/html/2605.25931v1)) explores until its open questions are few,
tries a few acts meant to prove its hypothesis wrong, then plans, comparing each step's result with what the hypothesis
predicted and going back to exploring on a mismatch.

- `core/cognition/a_plan_to_an_end.py`: each step says what will show once it is done; the screen after it is held to
  that; a step that did not show it is not done, and a second miss in a row has the plan made again with what each was
  to show and what showed instead. A step tried until it is passed over (it did nothing as written) has the plan made
  again around it, the way ADaPT breaks down only what failed.

**Models of a world written as code.** On ARC-AGI-3 the strongest open systems write the game's rules as a program,
check it against every transition seen, simplify it, and plan in it before spending moves
([executable world models, 2026](https://arxiv.org/pdf/2605.05138): 15 of 25 public games solved); raw frontier models
scored under one percent. Coding agents given only an observation and action interface build controllers that beat
StarCraft II's built-in AI and win Civilization ([Compiled Agency, 2026](https://arxiv.org/pdf/2609.18996)), and access
to the environment while building added 10 to 78 points over building from the description alone.

- Her rule of a world, compiled and searched: `a_world_compiled.py`. The program's own code read into a manual:
  `reading_the_code.py`.

**Machines built from parts.** In [BesiegeField](https://arxiv.org/html/2510.14980) models write machines as a tree of
parts, each attached to a named face of its parent; that form gave valid machines far more often than coordinates did.
Models fail at placing parts precisely, at planning the mechanism, and at turning what the simulation showed into the
right edit: they patch locally where people redesign. State feedback (how far it went, which part broke) helped more
than a score.

- A chain built from where the last part went: `core/agency/putting_things_in_place.py` (`in_chain_order`), carried to
  the place her eyes find for "the end of the arrow" (`core/perception/where_the_words_point.py`).
- Editing over starting again: `KEEPS_THE_WORK` in `reading_the_rules.py`.
- A part put down turned until its output points on toward where the chain must go, by her eyes on the output's end
  and the goal, each named in the place's own words: `core/agency/aiming_what_was_placed.py`.
- **Not yet:** a chain held as parts and joints (which part joins which), so that a failed test edits the joint that
  broke.

**Open worlds and long play.** [Lumine](https://arxiv.org/html/2511.08892v1) (2025) finished the five-hour opening of
Genshin Impact at human pace and the openings of two other games untrained, seeing at 5 Hz, acting at 30 Hz, and
reasoning only when needed. [Game-TARS](https://arxiv.org/html/2510.23691v1) (2025) trains one model on keyboard and
mouse across games. VARP ([2024](https://arxiv.org/html/2409.12889v2)) won most easy and medium fights in Black Myth:
Wukong from screenshots with libraries of situations and actions, and lost the hard ones because a model looking every
few seconds cannot answer a wind-up.

- A fast loop that plays by measurement and thinks only beside it: `core/agency/playing_as_it_happens.py`.

## Systems that use computers

**Learning what each control does.** AppAgent ([2023](https://arxiv.org/pdf/2312.13771)) explores an app before using it:
a screenshot before and after each act, the change written as that control's entry, and the entries read when it works
there later. OmniParser ([2024](https://arxiv.org/abs/2408.00203)) gives each element on a screen a caption saying what
it is for, and the models acting on its list of elements chose the right one far more often than models acting on the
picture.

- `WhereThingsLead.did`: after every act that answered, what came up and what went, kept between sittings and given to
  her plan as "what she has found things here do".
- A screen as a list of things to act on: `core/perception/element_inventory.py`.

## Robotics

**Useful and possible.** SayCan ([2022](https://arxiv.org/pdf/2204.01691)) scores each skill twice, by a language model
for how much it helps the instruction and by a learned value for whether it can succeed from here, and takes the
product; grounding this way nearly doubled success.

- What a model proposes is held to the place: a plan step naming nothing the place has shown is dropped
  (`a_plan_to_an_end.py`); what the place says outranks what her model supposed (`what_this_place_is.py`).

**Closing the loop.** Inner Monologue ([2022](https://arxiv.org/pdf/2207.05608)) feeds a planner each step's success,
the scene and any correction while it acts; a wrong success detector was its weak point. REFLECT
([2023](https://arxiv.org/pdf/2306.15724)) summarises a failed run at three grains (what the senses showed, key events,
the end of each subgoal), finds the subgoal that failed, and plans the correction from there.

- Each step held to what it was to show, and the misses described to the next plan: `a_plan_to_an_end.py`.

**Affordances as constraints.** VoxPoser ([2023](https://arxiv.org/pdf/2307.05973)) has a model write code that builds a
3D map of where to go and what to avoid; ReKep ([2024](https://arxiv.org/pdf/2409.01652)) writes each stage of a task as
small functions over keypoints (the spout over the cup; the pot kept upright), solved stage by stage.

- A carry's end found by her eyes from the words that name it: `where_the_words_point.py`.
- **Not yet:** a step's end written as a relation between things on the screen (this part's arrow end on that square;
  the chain's end on the cage) and checked by measurement.

**Learning affordances by touching.** Where2Act ([2021](https://arxiv.org/pdf/2101.02692)) learns, from its own pushes
and pulls in simulation, where on an object an action does something and how likely it is to work; interactive
perception builds the same map on a real robot ([2025](https://arxiv.org/pdf/2501.06047)).

- Contact as a question: `what_meeting_things_does.py`; what each control does: `WhereThingsLead.did`.

**Task and motion planning.** TAMP joins a symbolic plan (on, in hand) with the geometry that must make it possible, and
fails where the symbols promise what the geometry cannot do ([Garrett et al., 2020](https://courses.cs.washington.edu/courses/cse571/22sp/slides/15-tamp.pdf)).
LLMs now write behaviour trees that a failure interpreter edits when a step fails ([2025](https://www.ijcai.org/proceedings/2025/980)).

- The plan's steps run by the moves that do them, a step that keeps doing nothing passed over (`reading_the_rules.py`).

## Built from this, 10 October

1. What each act does: `WhereThingsLead.did`, `what_things_do()` (AppAgent, Where2Act).
2. Reaching what a step needs: `WhereThingsLead.toward`, `_reaching_what_the_next_step_needs` (means-ends analysis,
   goal-oriented action planning).
3. Each plan step held to what it was to show, and the plan made again from described misses (Inner Monologue, DEPS,
   REFLECT, AERA).
4. Warning signs and looming: `core/agency/warning_signs.py` (telegraphs, Lee's time to contact), counted once as each
   change begins, and only where a thing of hers is in play.
5. Locks and gains: `core/cognition/locks_she_met.py` (lock and key, Metroidvania gates).
6. Things kept through occlusion in her fast loop: `what_moves_in_the_picture.py` (`out_of_sight`).
7. A placed part turned toward where the chain must go: `core/agency/aiming_what_was_placed.py`.

## Still to build

1. A chain held as parts and joints, where it ends now, and a failed test that edits the joint that broke.
2. A step's end written as a relation between things on the screen and checked by measurement.
3. Locks seen rather than said: a gap too wide, a ledge too high.
4. Reading a level's layout as its lesson.
5. A construal step: what cannot change the route left out before planning.
6. A thing out of sight still counted as a danger near where it was going.

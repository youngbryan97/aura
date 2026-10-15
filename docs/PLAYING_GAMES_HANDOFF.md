# Playing games: what is proven, what is not, and what to run

Status: Handoff · Written 2026-10-13 · Cloud container, no models, no Flash games

Aura is to play three games drawn at random from the 56 Flash games of the
Cartoon Network list, on the owner's Mac. This session could not reach those
games: archive.org is blocked here and the offline bundle lives only on the
Mac. Everything below was built and measured against the test worlds in
`tests/fixtures/worlds_that_move/`, played from their own code, and through
the whole of her play as a live game is handed to it.

The owner's direction for this work: code and architecture, no fine-tuning.
Nothing here trains a model.

## What she has now

**She reads a game's own code before she plays it.**
`core/perception/reading_a_world_s_code.py` reads a page's scripts for her
body (its size, start, paces per key, the rectangle it is kept in, jumps,
pushes, the colour it is drawn in), the things in the world (what touching
each does, how each moves, how fast, what it looks like), the ground to be
gone over (its grid, the point of her that marks it, the share that wins,
the colour of its marks), the clock and the lives. For Flash programs,
`core/perception/reading_a_program.py` turns the decompiled ActionScript into
the same shape (`from_how_it_plays`). The page's reading is cached per page
as "a world read" in `~/.aura/state/worlds/`.

**What the code says reaches her play.** Through the guide to the place:

- which thing on the screen is her body, by the colour the code draws it in;
- which kinds cost her and which are worth getting, before she meets them;
- how fast her keys move her, where her trials have not measured it, and
  that a key the code paces her by is no jump;
- that nothing inside the play area stops her, where the code says so;
- how far each kind could swerve, for the berth she keeps;
- the ground's own grid and which of its cells are marked, read off the
  screen in the colour the code marks them in;
- that a strip on that ground is ground and not a bar of what she has left.

**Her decisions are tested against the world running its own code.**
`tools/oracles/a_world_run_from_its_code.py` runs a world's JavaScript in
Node with a driven clock, checked equal to Chromium.
`tools/oracles/her_choices_on_a_world.py` feeds her real choosing the exact
state at a chosen pace, with her keys arriving late by whole decisions and,
with `jitter`, uneven waits between decisions, as live pictures come.

## Proven

Each line has a test that fails when it stops holding.

**Her reading of a world's code is what the code does.** 83 checks against
the running code, 14 deliberate misreadings caught
(`tests/test_what_a_world_s_code_says_is_what_it_does.py`). The colour read
for each thing is the colour on the canvas, in 15 worlds
(`tests/test_what_a_world_s_code_draws_is_what_is_on_its_canvas.py`).

**She wins the paint world at the pace she decides.** In its own code, on
seeds 1 to 6, each key arriving a decision late, the waits between decisions
uneven by up to 30%, the floor's marks seen
(`tests/test_she_plays_at_the_pace_she_decides.py`):

| Decisions a second | Won | Lives lost of 18 | Slowest win |
|---|---|---|---|
| 2 | 2 | 16 | 89 s |
| 3 | 6 | 1 | 68 s |
| 4 | 6 | 0 | 56 s |
| 15, keys on time | 6 | 0 | 50 s |

With plans that change keys before her next decision, she lost 7 lives at three
a second and 1 at four.

**She plays game 25 from its own card.** `tests/fixtures/worlds_that_move/paintcan.html`
is Camp Lazlo: Paintcan Panic as its hand-read card says it plays, held to the
card by `tests/test_a_world_made_from_a_game_s_card_plays_as_its_card_says.py`.
Given only what her reading of a Flash program gives, on seeds 1 to 12: level 7
at three decisions a second, 6 won; level 20 at eight a second, 12 won. Her
chasers are imagined turning as fast as they have been seen to where she
answers slowly; imagined going straight on, level 7 at three a second gave 1
(`tests/test_a_thing_that_comes_for_her_turns_as_fast_as_it_was_seen_to.py`).

**The keys she takes from the 56 games' own words are the keys their code
listens for**, game by game, where both were read
(`tests/test_the_keys_a_game_s_words_name_are_the_keys_its_code_listens_for.py`).

**At 15 decisions a second the code's ground decides it.** On seeds 1 to 6:
with the code's ground and plans worth the cells they cover, six wins, the
slowest in 50 s, no life lost. Plans that do not count cells won five, the
quickest in 73 s. Her own cells as ground, or no reading of the code: no win
(`tests/test_the_ground_her_code_names_is_the_ground_she_goes_over.py`).

**A painted floor is not a bar.** Rows of floor painted in a browser are
read as bars that fall without the code, and none is taken for what she has
left with it
(`tests/test_a_strip_of_floor_she_paints_is_not_what_she_has_left.py`).

## Found in the whole of her play, and fixed

`tools/measure_whole_play_on_worlds.py paint` hands the paint world to the
same entry a live game is handed to (`core/skills/sovereign_browser_drawing.py`),
from its title screen. Here it played at 3.1 to 5.1 pictures a second over the
stretches measured. Each fault below was found there and is fixed in code,
with a test.

1. **Plans changed keys faster than she decides.** A plan turned a sixth of a
   second ahead when her next decision was a third of a second ahead, so she
   went right and left on the spot while a bug reached her. Plans now change
   no sooner than her next decision (`core/agency/imagining_ahead.py`).
2. **Cells crossed between pictures were not counted.** She went back over
   them for seconds at a time. Cells along a straight row or column between
   two pictures now count, but not along a cell's edge, where the world's
   choice of side is a guess (`core/agency/ground_her_code_names.py`).
3. **A cell her steps kept carrying her past held her for 15 s.** A cell she
   goes for near at hand for 2.5 s without marking is left for 12 s.
4. **The floor was read as bars.** One strip was named for the "PAINTED 2%"
   written above it. Each shortening counted as a loss, and she came to think
   holding down cost her (`core/agency/what_its_code_says_she_sees.py`,
   `a_strip_of_ground`).
5. **"Nothing here answers to me" ended a stretch she was winning.** Her going
   is measured only between pictures 0.35 s apart or less; at three a second
   it went unmeasured for 8 s. Her body seen to go somewhere while a key is
   held now counts (`core/agency/which_one_answers_to_her.py`, `went_at`).
6. **Code paces were applied before she knew she follows the pointer.**
   Fixed by order (`core/agency/choosing_where_to_be.py`).
7. **On Bounce she watched the ball fall three times, game after game.** The
   ball set going by her click on PLAY was counted as the click's answer, not
   as the world moving, so the start of play looked like a scene to wait out
   (`core/perception/watching_it_happen.py`, `in_motion`).
8. **Four games' words were misread for their keys**: a key named in quotes,
   "Enter" as a verb, "pick it up Press", and number keys
   (`core/agency/the_controls_a_game_names.py`).
9. **Her taps went unseen by any game that reads its held keys each frame.**
   A tap was a key sent down and up together. Flash's Key.isDown works this
   way, and so do the shooter, doors, timing and food fight worlds. In
   Chromium 0 taps of 10 reached the food fight; held 20 ms or more, 10 of
   10. A tap is now held 90 ms, let go of while she plays on
   (`core/skills/keys_tapped_as_a_hand_taps.py`).
10. **In the food fight, a duck dodged nothing**, because the world let food
    fall through her ducked shape. With the world fixed, she ducks what is
    on its way to where the things that cost her were last seen
    (`core/agency/thrown_at_what_shows.py`, `WhereHarmLands`). Over seeds 1
    to 3 she won 6 games of 26 and was hit 0.13 to 0.18 times a second.
    With the held taps alone it was 1 of 17, hit 0.40 to 0.46 times a second.
11. **She learned how long her throw takes from the other side's food.** The
    food fight draws no splat; what came into sight where she aimed was his
    food leaving his hand, and her throws were led by 0.25 to 0.67 s where
    they fly 0.83 s. A landing now has to stay where she aimed, and her throw
    seen in flight says how long it has to go. Seeds 1 to 3 each ended with
    her win, by her own record, in 0.7 to 0.9 minutes; before, none did.

After the fixes, from the title screen with nothing remembered: the first
game was lost at 67% painted in 59 s, the second won in 77 s with no life
lost. Before them, three runs won their later game in 112, 114 and 124 s, with lives
lost. In both, she said "Its code says the bugs cost me." and "Its code says
which one is me" within half a second of play starting.

**Why she still loses lives here.** Recorded beside the world's own record of
each hit: she decides on a picture taken 0.25 to 0.59 s before, and her keys
take about as long again to land. A bug that bounces off a wall or turns in
that time is where she did not plan for it; one hit came as she ran into a
corner a bug was bouncing towards. Her eyes take 190 ms of each picture
here. On the Mac they are measured at 20 pictures a second alone. Cutting
the age of the picture she decides on is the work that most changes this.

## Not proven

- **The 56 Flash games.** None was run here. The reader's Flash path was
  checked against one hand-written card (game 25, Camp Lazlo: Paintcan Panic,
  `tests/fixtures/flash_games_as_their_code_says/25.json`), with the program
  itself absent.
- **Her pace on the Mac.** `core/perception/seeing_play_elsewhere.py`
  records 7 to 9 pictures a second live with her eyes in her own process,
  and 20 with them alone. This session measured only this container.
- **Two decisions a second.** With a decision's delay of half a second she
  won the paint world on 2 seeds of 6 in its own code.
- **Of nineteen tests that failed here on the code before this session, six
  still fail**: the brawl punch, the beam charge, the needle press and the
  paddle's returns (each played in a browser in real time, and slower here
  than where they were written), and two of the code executor's, which need a
  sandbox this container lacks. Fixed: the key-trial tests (their fakes lacked
  a field), the eyes in a process of their own (shared memory left open), the
  async write ratchet (two writes on the loop). The word-mending tests needed
  the system word list, which macOS has; it was installed here.

## Run it on the Mac

This plays three games drawn at random from the offline bundle, the way
the demo draws them, with no model loaded and nothing on port 8000.

1. Make sure that the bundle exists:
   `ls ~/.aura/flash-campaign/offline-bundle/programs | wc -l` must show 56.
2. Run:

   ```bash
   cd ~/.aura/live-source && .venv/bin/python tools/measure_on_archived_games.py \
     --bundle ~/.aura/flash-campaign/offline-bundle --random 3 --minutes 6 \
     --trace /tmp/aura-three --out /tmp/aura-three/report.json
   ```

3. Read the first line: it gives the games drawn and the seed. To draw the
   same three again, add `--seed` with that number.
4. Read one line per game: `runs`, `won`, `error`, and `last_seen`. The
   folder `/tmp/aura-three` holds each game's log and its last screen.

To let the next cloud session play the same games, push the bundle to a
private repository and add it to that session:

```bash
cd ~/.aura/flash-campaign/offline-bundle && git init -b main && git add -A \
  && git commit -qm bundle && gh repo create youngbryan97/aura-flash-museum --private --source . --push
```

## Next, by what it is worth to the demo

1. Get the 56 games where a session can play them, and put each program's
   reading to its running, as was done here for the test worlds.
2. Find out whether the punch, charge, needle and paddle tests are this
   container or faults: run them on the Mac. On brawl the whole of her play
   gave up after five runs as "not getting better at it".
3. Her eyes cost 190 ms of a 270 ms picture in the whole of her play here
   (`WhatMoves.see`). Exact speed-ups not yet tried: gradients computed once
   a picture in `_refine`, `_with_the_view` (8 ms), `_blobs` (16 ms). A
   rewrite of the sampling in numpy gave the same numbers and was slower, and
   was taken out.
4. Survival at two decisions a second.
5. **Shooter, through the whole of her play.** Given the world's state, her
   choosing now plays the shooter's minute out on seeds 1 to 6 at seven
   decisions a second, a decision late: it shoots what costs her wherever it
   lands instead of dodging it, and takes her ship, not its gun, for her.
   Through the whole of her play it is not yet better: seeds 1 to 3 still
   lost each game in about 10 to 15 s, best 0 to 40 points. Find what the
   live path does that the harness does not: her eyes, the fire pace
   (`_trigger` in `core/agency/playing_as_it_happens.py`), or whether the
   code's reading reaches her choosing there.

## Mac checkpoint, 15 October 2026

Main now includes the browser observation deadline repair (5e82e7082),
unfinished proof checkpoints (7fa671d25), and retained material before a
joined outline's larger point consensus (fd06569ed). The last repair
reproduces a moving body's measured motion becoming zero when stationary
texture enters its outline. Generated-pixel tests now preserve that body
through the join and split with both trackers. The focused perception set
has 249 passes; smoke has 165 passes and one skip. These are offline
component results. No original game was won or native prompt proved here.

The new record is
`docs/evidence/steering/2026-10-15-retained-marks-before-the-joined-outline.md`.
It also maps Bryan's five-repair attachment against the current code.
The control-identification thresholds remain unchanged.

The machine reboot removed `/private/tmp/codex-steer`, including its raw
captures and diagnostic scripts. Earlier dated logs retain their original
claims, but those missing captures cannot be replayed. Keep new run evidence
under `~/.aura/flash-campaign/evidence/`. `tools/run_proof_step.py` now saves
prepared and running records before waiting, with `passed=false`, and a
command's exit code leaves its task outcome for independent review.

Claude's G09 run is active. Preserve its model and process ownership; no
desktop runtime or browser play run was started alongside it for this
checkpoint. Once that lane is available, the next proof is a fresh assigned
original-case-9 run with durable actual-frame and input receipts, followed
by ordinary native-app prompt validation. Keep holdouts 31, 32, 33, 34, 35,
36, 39, 42, 47 and 49 unopened. Body control, task-goal grounding and the
56-game campaign remain open. Presentation readiness is unproved.

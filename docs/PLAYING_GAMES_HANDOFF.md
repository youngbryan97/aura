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

## CPU continuation while G09 owns the model lane, 15 October 2026

The archived runner now has durable per-case checkpoints, exclusions before
asset reads, a supervisor lease and settled-attempt resume. A 56-case plan
retained 45 pending cases and excluded case 9 plus the ten holdouts. It did
not open any program. A process exit and Aura's own win claim still supply
no independent presentation qualification.

Independent local point fits reduced median observation time by 20.7 percent
on the generated scene used in the paired measurement. The tracking set
has 255 passes. This is not a measured original-game speed-up.

For probes beside G09, use `--cpu-only`. The earlier combination of software
graphics and CPU OCR was insufficient: scene recognition launched a 4B GPU
vision worker during the first case-18 probe. That diagnostic group was
stopped; G09 was not signalled. The explicit model-worker prohibition now
reaches the generic subprocess gateway, vision client before lane admission,
and worker entry before MLX import. The complete CPU retry started no vision
worker. Separate real tests proved SwiftShader WebGL2 readback, keyboard
delivery into screenshots and native OCR with CPU acknowledgement.

The case-18 retry still lost: zero landings, one crash, no selected body or
moving key. Internal gain counts were inconsistent with useful progress.
The latest feedback repair preserves unowned strips and unnamed numeric
readings without awarding reward, with 119 focused passes. It has not yet
been measured on the original game. Keep the distinction between these
CPU component/browser results and ordinary native model-backed play.

New evidence is under `artifacts/flash-campaign/` in the steering worktree.
The dated evidence logs describe each checkpoint and the aborted probe.
These paths survive temporary-directory cleanup. Do not resume its
unfinished record blindly or reuse an old report path for a new attempt.
Native prompt proof, reliable body control and the 56-case campaign remain
unfinished. Preserve the holdout boundary and G09 ownership.

The later CPU check at 97ac86d8f finished in 79.13 seconds. Its actual
fast-loop gain and loss totals were both zero after the feedback repair.
It still reported zero landings and one crash, with no selected body or
measured moving key. The final picture is after a restart; the terminal
wording is in OCR logs, so this remains unqualified for independent
terminal-image proof. Next capture the actual fast-loop frames and paired
input receipts durably; the current twenty-second pictures cannot explain
each identity failure. See the appended measurement in
`docs/evidence/steering/2026-10-15-unowned-readings-do-not-award-reward.md`.

## Pixels, owned calibration and actual contact, 15 October 2026

Continue in `.claude/worktrees/codex-steering-1012`. G09's induction process
4559 remains owned by another lane; original case 9 and the ten holdouts
remain excluded. Do not launch a model-backed runtime beside that job.
The following general repairs are committed and pushed to main:

- `4f7040fb4`: record the actual browser frames and input calls, with byte
  hashes, capture clocks, bounded asynchronous writes and explicit gaps.
- `f54cef691`, `4ab5de94b`, `b00b242a9`, `2c85070e2`: surviving material marks,
  pieces of one visible body, qualified affine image deformation, and a
  shared fresh-mark admission budget. Recorded-pixel replay improved
  identity retention. These changes do not create control authority.
- `39b41cd50`: current menu classifications reach navigation; the notice
  expression preserves JavaScript word boundaries and requires an exposed
  hit target.
- `1851fbbcc`: body selection copies sample lists rather than sharing them;
  captured velocities cannot count twice. A changing strip needs a known
  role before supplying reward or scarcity, even after body selection.
- `3b4b3e124`: provisional pulse/rest paths rank finite calibration trials
  by visible clearance. Trial ownership is separate from statistical
  independence; feedback-chosen calibration does not teach body identity.
- `d0decf11c`: the observation reservation reaches each newly delivered
  pulse/rest phase. Existing input deadlines and the overall end stay fixed.
- `5830b1de5`: an identified inertial body's calibration can wait for four
  actual settled samples, then release and observe its rest. The existing
  1.8-second hard maximum and two-attempt allowance remain bounded.
- `543ef27db`: the optional recording keeps the actual selected target and
  predicted plan at the choice boundary without rerunning target selection.

Fresh CPU-only original case-18 play now sometimes confirms the body in
four seconds and measures all three accelerations. It has still made no
landing. The latest inspected run at `5830b1de5` crashed against the platform:
the ship remained tracked immediately before impact, then exploded. Inspect
`artifacts/flash-campaign/2026-10-15-case18-observed-acquisition/` and its
dated evidence document. Its last five recording events were dropped at
the raw-byte limit; the crash precedes that gap.

The next run, `2026-10-15-case18-made-choice`, is intended to preserve the
made destination and arrival plan. Inspect its supervisor/process before
launching anything else. Do not diagnose that collision as a tracker loss
or invent a surface policy before reading which target was actually chosen.

CPU results remain distinct from ordinary native Aura prompt integration.
No native runtime restart, full 56-case pass or independently witnessed
presentation qualification has been established. Focused tests and smoke
passed at each retained checkpoint; the full suite was not run beside G09.

## Explicit goals and supplied visual references, 15 October 2026

Later checkpoints are `f51927d86` (resolved inertial axes), `65598e717`
(requested spatial destinations and damaged repeat labels) and `6d9f707cd`
(normal pursuit options retained with their reader snapshot). All were
pushed to main. The original case-18 browser runs remain CPU-only and lost.
`2026-10-15-case18-requested-role` exercised the end-screen handoff but made
no normal target choice. `2026-10-15-case18-offered-controls` has a complete
recording: 229 motor frames, three normal screen snapshots, 78 input events
and five normal choices. All five kept the requested landing relation and
reported an unbound destination. They did not choose scenery instead.

The source's exported artwork now supplies an offline reference experiment
without another model. Unmodified FFDec 26.3.0 and a Temurin 17 JRE are
isolated under the worktree's untracked reference-tools artifacts; published
archive hashes were checked. Headless Java2D used one CPU and a 512-MB heap.
Only original case 18's source was read. `-select all:1` failed upstream;
the later `-select 1` succeeded but exported every sprite frame. Do not infer
first-frame-only custody from its directory name.

The new general correspondence component is
`core/perception/a_visual_reference_in_the_picture.py`, with an offline
probe in `tools/measure_supplied_visual_references.py`. It combines bounded
masked scale search with the existing original forward/reverse patch fits
and redundant geometry. On a captured gameplay frame it retained 27 points
of the platform reference and seven of the ship reference. The ship also
matched its instruction-screen preview; the platform did not. These are
static appearance correspondences and grant no control identity or task
role. They have not been wired into the live observer or target binder.

Read the dated evidence documents before the next integration. A reference
name from source is not automatically the instruction's semantic role;
preserve ambiguity, capture ownership and the existing causal thresholds.
The next work must connect qualified material to current observed entities
without assigning a goal merely because it is unnamed or nearby. G09's
induction PID 4559 remained active during this work. Leave its model lane,
original case 9 and the ten holdouts untouched.

The existing two one-shot continuation heartbeats were rearmed for
12:10:40 PM and 5:11:40 PM Pacific on 15 October, as Bryan requested. Their
prompts prohibit duplicate workers while this chat is already active, retain
the G09/runtime boundary and retire each follow-up after it runs.

## Source custody for visual references, 15 October, after noon

`d1c361e72` keeps newly read directions current in the running actor, ordered
by capture time, even when the wording repeats a prior request. `80853022e`
keeps visual exports with source namespaces, dictionary IDs and definition
and export addresses in `ProgramRead` and the current guide. Invalid asset
metadata cannot discard read controls or assign an observed role. Main's
concurrent changes were merged in `0a831a553` and the combined checkpoint
was pushed.

The new explicit external export utility is
`tools/export_program_visual_reference.py`. It stages a read-only source,
uses the existing bounded supervisor, and records renderer and output hashes.
`-select ID:1` selects a sprite's first frame; bare `-select 1` selected the
main timeline and had exported all sprite frames. The new case-18 export
and its 27-point match to recorded gameplay are retained in
`artifacts/flash-campaign/2026-10-15-custodied-reference/`.

Only static source 18 was read. Its current procedural code reader has
empty `hers` and `ends`; source names alone still do not bind the instruction's
landing platform. See the new dated source-custody evidence before wiring
these references into control. There is still no landing, full campaign or
independent native-app demonstration. Do not weaken causal thresholds to
promote the reference match into body identity.

The first continuation fired while this chat was active and was deleted
through the app tool. The later 5:11:40 PM Pacific continuation remains.
G09 PID 4559 and the excluded original 9/holdouts remain untouched.

The source-effect reader now retains reaching conditions, assumptions and an
immutable invocation on each call/write. An unspecified free callback keeps
symbolic `this`; it no longer becomes its own receiver. See
`docs/evidence/steering/2026-10-15-source-effects-keep-their-own-guards.md`.
Its 72 focused checks, 165 smoke checks (one skipped), compile, Ruff,
layering, dependency and writing gates passed. Case 18 still exhausts 64
paths and stops at `Delete`, so neither complete source semantics nor an
exact source controller is qualified. These new records are not wired into
task-role binding or used to replace pixels.

`4a68839a1` repairs animated navigation candidates and two outer completion
bypasses. Case 3 (zero-based Runaway Robot) exposed both: it was left on
instructions and falsely described as made. The repeat now recorded a
dynamic `CONTINUE` offered and selected, and ended honestly as time up;
it still did not clear the instructions or win. Both raw runs and the dated
navigation/completion evidence are retained. The component and normal
handover suites passed 84 tests; smoke passed 165 with one skipped, and
Ruff/layering/dependency/writing gates passed.

Next inspect cursor hit ownership. Small footer arrows lie outside the
coarse grid's last centre sample, and their hover captions can be displaced
from the physical hit point. The current cursor-to-label association keeps
only a boolean, so a later click uses the caption centre. Preserve the
measured hit, revalidate it before delivery, and record the normal physical
input without treating delivery as task completion. The non-placement
maker's internal changed-screen completion heuristic also remains weaker
than the placement proof; the last change closes outer bypasses only.

`06e2b604d` carries cursor candidates through normal click geometry, checks
for a current hand before pointer-derived delivery, extends the boundary
sweep and records normal page port requests/completions. The 69 focused
checks and normal gates passed. The repeat in
`2026-10-15-case3-owned-cursor` reached an actual room, as its final picture
shows, although its textual report retained old instructions. It did not
win. Its `Continue` captions had no pointer anchor, so their successful
transport is not proof of the new caption-anchor mechanism. Read the dated
cursor evidence before drawing a stronger conclusion.

The next actionable gap is cursor-cache custody: words alone do not
identify a scene, and samples from a different crop cannot be reused in
another coordinate frame. Preserve capture geometry and current pixel
identity. Normal input records are now available to test that connection.

Deployment also remains open. The default live-source checkout was on
`main` at `ca975c498`, 131 commits behind the pushed branch. Do not assume
the native app contains these changes. Do not update that shared checkout
under an exclusive job or start a model-backed runtime beside G09. The
later continuation remains scheduled, with the first already retired.

## 15 October, ordinary cursor navigation follow-up

The worktree is `codex-steering-1012`. `a73fc355f` added current-corner
cursor proposals, `54eb608f6` bounded the full sweep while retaining finished
physical samples, and `2df71d077` distinguished view-navigation routes from
following-body control. Each was pushed to main. `fa7da1c27` preserved two
concurrent upstream changes by merging them unchanged after a rejected push.

The three-minute original case 3 repeat at `2df71d077` finished in 213.55
seconds with 14 moves, no win, and no fast body-control stream. Its final
image shows a kitchen. Later clicks were repeatedly rejected because a
remembered 2 by 2 lattice differed from the scene's inferred arrangement.
The current repair gives a unique current captured pointer target its own
admission evidence, while keeping the grid check for keyboard actions.
See the dated evidence for its negative tests and the next original repeat.

Next: measure that admission repair, check whether hover samples wait for
the browser's response, and fix the stale `last_seen` report. The report
retained old instructions after visual room entry, so inspect the actual
recording and final image. Do not read it as current scene text.

Accelerator ownership continues under G09. PID 91159 ended; successor PID
668 runs the same organ entry point with about 15 GB resident. Only resource
ownership was checked. Stay CPU-only, leave the shared default checkout
alone, and preserve original 9 and the recorded holdouts. No native app was
started and no runtime restart occurred. The 5:11 PM continuation remains
scheduled; the fired first continuation was retired. None of this is an
independently witnessed native-app result or a cleared 56-game campaign.


## 15 October, cursor material and pointing feedback

The `b76c34b02` case 3 repeat completed with 17 moves, zero grid-admission
refusals and no win. Its final picture shows another room and no inventory;
its current/historical text fields now stay separate. The next fault is input
to view coupling: cursor inspection pans the picture, and a later nearby-hand
search can select another affordance.

The captured page path now samples material under physical hand cursors,
refreshes the reading after inspection, and uses unique pixel correspondence
plus measured pointer feedback to reach that material before down/up. It makes
no extra approach move after final validation. Component and Chromium normal
port tests pass; original-game performance is the next measurement. See
`docs/evidence/steering/2026-10-15-cursor-material-and-pointing-feedback.md`.
The governance baseline gate has existing drift; the exact effect inventory
of the production files changed here matches HEAD. No baseline was raised.
This code has not been deployed to the default checkout or native runtime.


The original repeat at `911ee48d0` was interrupted after 148.17 seconds when
scene motion was misread as a well. It reported 39 pieces and 198 clears before
returning that none of its keys moved a piece; the outer actor already
rejected that ending. No game win or corrected cursor click was proved.

Candidate wells now need current distinct horizontal control responses, spend
the same admission budget across changing/resting branches, and cannot report
or save new play outcomes before admission. Failed attempts preserve prior
hypotheses without mutating verified rule/appearance objects through aliases.
Offscreen or invalid well crops return no piece instead of crashing. The well
pixel worlds and admission regressions pass. Read the dated candidate-mode
evidence. Next couple material capture to the original physical cursor hit;
resampling it after the sweep loses the pointer/view relationship.


Material acquisition now runs inside the original physical cursor sample,
including physical validation of a synthetic sweep, before the next move.
It has no extra approach and spends the existing sweep deadline. The fresh
reading retains only uniquely reconciled samples. Opted-in recordings include
reference patches and correction/comparison pictures in declared RGB with
reference sequence links, so the exact correspondence can be replayed.
The Chromium normal-port recording replays to the same target coordinates.
The focused suites and normal gates pass; the next step is an original case 3
repeat with these changes and the candidate-well admission repair. See
`docs/evidence/steering/2026-10-15-cursor-captures-stay-with-their-physical-sample.md`.


At `f29daccdb`, the next original case 3 repeat completed in 219.58 seconds
with six moves and no win. It retained four current material correspondences
on several reads and recorded three corrected cursor presses, each after a
zero-error comparison. The final image still shows the house; collection or
accepted task progress is unproved. The complete 235-picture recording has
no drops and remains presentation-unqualified.

The source audit and trace identify edge-driven rate scrolling. Holding the
pointer keeps moving the view, so the former static inverse learned the wrong
response. The controller now measures two frames at a held pointer before
learning that inverse; continuing drift uses bounded damped correction instead.
Physical survey ordering also spreads its early capture budget across the
view. Pixel rate-control worlds and a strengthened Chromium fixture pass. The
fixture now shifts the layout after discovery and requires multiple correction
cycles; its earlier version could succeed in a single arrival. See the dated
continuing-motion evidence, then run the original repeat before claiming gains.

## Startup receiver checkpoint, 15 October 2026

The `e72751c94` original case 3 repeat finished in 220.61 seconds: 12 moves,
no win. Eight corrected cursor requests have a zero-error held-pointer
comparison immediately before delivery. The final kitchen picture does not
prove collection or completion. Repeated material under changing point labels
still needs persistent action memory.

The 25.25-second startup delay was traced to a warning that appears on hover
and receives the first click. The intended play control receives only the
second. Startup now retains the trusted down/click paths and retries promptly
when a different element received the press. The normal startup function on
the original file completed in 2.365 seconds, exposing the title screen.
See `docs/evidence/steering/2026-10-15-startup-presses-keep-their-actual-receiver.md`.
Focused checks: 43 passed. Smoke: 165 passed, one skipped. No native runtime
or G09 changes. Gameplay and native presentation readiness remain unproved.

## Delivered material exploration checkpoint, 15 October 2026

Each page surface retains a bounded history of completed pointer-material
presses. A unique local correspondence can remove an anonymous point's
never-tried bonus after its coordinate label changes. This is an exploration
cost, never a scene or body identity, input authority, accepted action or win.
Current capture custody is required; named controls keep their behavior.
An optional-history failure cannot discard a completed physical click.

The original eight-click trace retrospectively yields counts
`0, 0, 0, 0, 1, 2, 1, 0`. The Chromium port check presses a fresh point after
the first point moves and changes label. Broad navigation checks passed 172;
final integration passed 85; final smoke passed 165 with one skipped.
See `docs/evidence/steering/2026-10-15-delivered-presses-follow-their-local-material.md`.
An original repeat is next. G09 and native runtime ownership remain unchanged.

## Current program evidence checkpoint, 15 October 2026

The original case 3 repeat at `c5ce472b3` completed 15 normal input requests,
including nine corrected material clicks. Four normal decisions received the
new material-press costs. The 202.39-second run still did not win or establish
collection. Its complete recording keeps `presentation_qualified: false`.

Revisits no longer substitute an address-keyed summary for current program
bytes. The normal reader reacquires and parses each program; inline world
reads also inspect current scripts. Failed acquisition cannot restore old
source authority. The owned CPU browser's two original case 18 reads both
retained 707 code lines, 18 complete action streams and 33 visual declarations
in 0.103 and 0.070 seconds. This was a source check without game input.

Focused tests: 94 passed. Smoke: 165 passed, one skipped. See
`docs/evidence/steering/2026-10-15-current-source-survives-a-revisit.md`.
Source artwork still needs a fresh observed registration before it can name
a live destination. G09, holdouts and native runtime ownership are unchanged.

## Source-coordinate reference checkpoint, 15 October 2026

`tools/export_program_visual_reference.py --with-registration` now exports
a sprite/button coordinate frame and requires an independent CPU raster to
align with its PNG. Source origins can then be projected through a fresh
appearance fit with exact source/reference/picture byte custody. Changed
bytes and missing or unsupported registration refuse. This does not name
an object instance or certify contact or input.

Original case 18's source origin projects 83.084443 pixels from its artwork
box centre in the recorded frame. The second-raster calibration error is
0.001177 pixels. The registration packet and RGBA evidence are retained in
`artifacts/flash-campaign/2026-10-15-registered-reference-raster/`.
Source comparisons use object origins, so the visual centre is insufficient.

The material replay also shows several platform pieces sharing kind 0 with
the controlled ship. Instance naming must avoid assigning that asset's name
to its entire visual kind. The explicit destination lookup currently uses
`choosing.others()`, whose heuristic removes still same-kind objects before
the goal can resolve them. Inspect that gate and preserve name ambiguity.

Focused checks: 75 passed; smoke: 165 passed, one skipped. See
`docs/evidence/steering/2026-10-15-reference-images-retain-source-coordinates.md`.
Runtime reference acquisition, instance binding and source contact geometry
remain open. Native-app demo readiness is unproved; G09 remains untouched.

## Explicit referent gate checkpoint, 15 October 2026

Explicit spatial goals now resolve against all published observed instances,
before ordinary interaction heuristics remove still same-kind objects,
writing or attached things. A filtered rival still competes for its name;
ambiguity and current-geometry checks remain. The ordinary interaction list
and hazard checks are unchanged. Tests: 84 passed; smoke: 165, one skipped.
See `docs/evidence/steering/2026-10-15-explicit-referents-precede-interaction-filters.md`.

The next naming gap is in `LookingAtThings` and `seen_in_play`: one marked
instance is classified, then its answer is stored for the entire visual kind,
and another instance of that kind is never asked. Track and bind the named
instance instead. Preserve the semantic classification as a guess and the
existing causal-control thresholds. No native runtime or G09 changes.

## Instance naming checkpoint, 15 October 2026

`LookingAtThings` now queries individual tracks. Its private capture receipt
retains observer epoch, number, birth, kind, capture time, box and pixel hash
through delayed semantic replies. `seen_in_play` stores the result on that
instance, and the explicit-goal resolver consumes the scoped names. Shared
kind guesses require agreement among current peers; an unknown or conflicting
peer removes the unmeasured class supposition. Measured contacts and told
rules remain. The provider prompt and causal-body thresholds are unchanged.

Tests: 116 focused passed; smoke 165 passed, one skipped. Lint, compile,
layering, DEPS consistency and writing passed. See
`docs/evidence/steering/2026-10-15-visual-names-retain-the-boxed-instance.md`.
The producer-to-guide-to-chooser test supplies semantic replies and an existing
body identity. Recognition accuracy and original-game wins remain unproved.
Next: connect source-reference geometry to current observed material without
turning a reference appearance into control identity or guessed contact rules.
No native deployment, G09 or original-game-9 changes.

## Oriented reference coordinates checkpoint, 15 October 2026

`reference_appearances` accepts a bounded orientation bank and composes
resampling geometry back to original reference samples. The offline probe
accepts repeated `--angle-degrees`; scale-count times angle-count must not
exceed 32. Failed orientation proposals no longer exclude a valid orientation
at the same place. Default zero-angle callers retain their search policy.

Tests: 88 focused passed; smoke 165 passed, one skipped. Lint, compile,
layering, DEPS consistency and writing passed. A fresh static case-18 ship
export still matches recorded frame 304 with seven points; this is offline
appearance evidence. See
`docs/evidence/steering/2026-10-15-reference-orientation-keeps-source-samples.md`.
Next remains current material attachment and source-contact geometry. The
reference matcher is not yet an ordinary runtime source-asset provider.

## Source point and consumed-pixel checkpoint, 16 October 2026

`SourceImageCoordinates.on_track` attaches a calibrated point through unique
current material ownership and a current named-point basis. It preserves
the existing motion origin and returns no control or task authority. The
ordinary observer now retains the consumed source RGB hash/shape/time across
worker publication. Attachment requires that exact capture. `inspect`
reports stale, missing or replaced tracks/bases and precision refusals.

A recorded case-18 replay attached the exported source origin through seven
owned correspondences. It retained 16 of the next 20 frames; the four losses
remain explicit. Nine independent fresh appearances agreed within one
working pixel (largest difference 0.216533). This is retrospective pixel
and material evidence, with no new action or gameplay pass. See
`docs/evidence/steering/2026-10-16-source-points-follow-current-owned-material.md`.
Focused: 241 passed; smoke: 165, one skipped; normal code gates passed.

Next: preserve the original observation owner and its captured material
basis while the matcher works off the event loop. A result computed on an
old frame must be carried through current corresponding marks, or refused;
it cannot be applied to a new observer merely by overlapping boxes. Runtime
asset acquisition, destination occurrence binding and source-contact control
remain open. Native deployment and independently witnessed play are still
unproved. The October-16 single continuation is scheduled in this chat for
12:24 PM Pacific; retire it after its run. G09 remains exclusive.

## Captured material owner checkpoint, 16 October 2026

`keep_material_frame` keeps source RGB pixels, masks, point bases and original
observer instance receipts for detached reference work. A copy does not
become a different observer. `SourceImageCoordinates.on_track` can attach on
that saved frame and transport the result to the original current material
basis; a different observer or new basis cannot inherit it. Copy budgets
refuse the whole snapshot instead of dropping inconvenient competitors.

The replay tool supports `--deliver-after-frames`. A frame-304 result reached
its current track after three newer captures (0.468009 seconds of source
time). Thirteen subsequent frames retained the point; seven independent
fresh matches agreed within one working pixel. Four losses remained explicit.
This is delayed recorded-data evidence, not live worker performance. See
`docs/evidence/steering/2026-10-16-detached-reference-work-keeps-its-owner.md`.
Focused checks: 247 passed.

Next: an ordinary runtime source-reference provider with bounded off-loop
work and explicit source/capture lifetimes. This copy/transport contract is
ready for that provider, but no normal actor acquires source artwork yet.
Destination occurrences and source-contact requirements still need their
own source-bound connection. Do not substitute a box centre for an asset's
source origin or promote its appearance into controlled identity.

Captured-owner checkpoint gates: 247 focused passed; smoke 165 passed,
one skipped; lint, compile, layering, DEPS consistency and writing passed.
The G09-owned general-reasoning process was still active at closeout.

## Ordinary reference worker checkpoint, 16 October 2026

Normal `PlayObservation` now acquires registered source-reference exports
and matches them on one bounded optional CPU operation. It keeps the
captured observer owner, transports delayed origins through current material
and refuses changed sources, tracks, bases and excessive precision error.
Source declarations are replaced on each current program acquisition; a
failed read clears their scope. The catalog is read-only during play.
`tools/publish_program_visual_reference.py` registers an existing export;
`tools/export_program_visual_reference.py --with-registration --publish`
can register a new export under the selected runtime state root.

Measured scale and orientation guide reacquisition without replacing broad
search or qualification. No-match searches cannot erase continuing material.
Agreeing pose hypotheses produce one current estimate with an enclosing
error allowance. Different instances remain separate. None of this sets
controlled identity, task roles, source collision rules or the motion origin.

In 100 source-paced recorded case-18 frames, the initial worker published
points on 27 frames. Measured-view search and pose coalescing published them
on 57 frames. Fresh pixel matches checked 31 final points, all within one
working pixel (largest difference 0.630021); 26 checks remain unmeasured.
The final replay's largest loop pulse gap was 0.017494 seconds. This is the
ordinary asynchronous observation path over recorded data, with no game
input, native-app request or gameplay pass. See
`docs/evidence/steering/2026-10-16-ordinary-reference-work-follows-current-material.md`.
Focused: 219 passed, followed by 57 post-publication-boundary checks.

Next: fresh source-asset acquisition for uncached programs, and a verified
destination/source-contact connection to control. A composite static
reference can match pixels while lacking a single existing material owner;
do not turn its box centre or a partial component into the source origin.
The current runtime publishes source geometry but the task controller does
not yet consume it as a source-defined contact condition. Native deployment,
original wins, the full 56-game campaign and independent demo readiness are
still unproved. No native runtime or G09 change was made. The one-time
October-16 12:24 PM continuation has been retired after this run.

Final checkpoint gates: smoke 165 passed, one skipped; lint, compile,
layering, generated DEPS consistency and writing passed. Governance retains
131 existing regressions; these new runtime modules add none. The module
size gate retains its existing unrelated regressions. Baselines were not
changed. The exclusive corpus/model job remained active (PID 53965).

## Visible source frames and sampling checkpoint, 16 October 2026

Independent raster-marker tests found a sample-centre bias in reduced
reference matches. The matcher now composes that offset. The native point
stream also carries its actual per-axis sampling grid, rather than treating
scaled point indices as original pixel centres. Earlier fresh-match
agreement shared the matcher bias and did not prove absolute accuracy.

Normal observation now publishes frozen named material before object
grouping. A separate source-frame channel transports source geometry through
those original visible marks when no single object owner qualifies. It
creates no controlled identity or role. Initial appearance still needs six
correspondences; rigid continuation needs three non-collinear named marks,
affine continuation six, with 70 percent retained and the existing precision
ceiling. The object/control qualification gates were not relaxed.

A measured failure showed that capacity allocation discarded two younger
marks while all four reference anchors were still visually valid. Active
consumers can request bounded, expiring retention of current marks and
propose source regions for new sampling. The request passes through the
ordinary asynchronous and process-worker observation paths. It cannot
restore missing material or replace original identities.

The 100-frame ordinary replay published the composite's source frame on
eight frames. Seven fresh pixel checks agreed within 0.375641 working
pixels; one remained unmeasured. Focal sampling did not improve that count.
An original supporting mark was still lost despite an admitted retention
request. This remains far below demo quality. See
`docs/evidence/steering/2026-10-16-visible-source-frames-and-sampling.md`.

Next: source-defined contacts and current instance binding. The controller
still cannot turn source-frame geometry into a verified landing predicate.
The source reader reports unsupported deletion and path-budget exhaustion
in the case's movement method; raising the budget would not prove semantics.
Fresh uncached asset export and ordinary native play are also open. No model
or resident runtime was started, stopped or restarted. G09 and holdouts
remain untouched; the exclusive corpus job is still active.

Checkpoint gates: final focused set passed; smoke 165 passed, one skipped;
lint, compile, layering, DEPS consistency and writing passed. Governance
retains 131 existing regressions and module size the same unrelated
regressions, including total excess of 197 lines. Baselines were unchanged.

## Shared matcher transforms, 16 October 2026

The reference search now shares current-scene transforms across its bounded
template bank. Direct pixel sums check the colour objective. It retains
the same candidate, reverse-patch, material and precision checks, with one
floating-point colour kernel active at a time. Focused tests: 131 passed.

Five alternating timing pairs on each of two recorded captures measured
median reductions from 0.261416 to 0.179305 seconds and from 0.279757 to
0.199373 seconds. Pose components and supporting correspondences were
exactly equal. The ordinary replay still produced the composite source
frame on eight of 100 frames; lower matching cost did not close that loss.
Object-bound points appeared on 53 frames. The maximum loop pulse gap was
0.021283 seconds. These remain recorded-pixel measurements with no game
input. Source contacts, instance binding and independent native play remain
open. No G09, holdout, model-job or resident-runtime change was made.

Shared-transform gates: smoke 165 passed, one skipped; lint, compile,
layering, generated DEPS consistency and writing passed. Native deployment
has not been performed.

## Deletion in the source sidecar, 16 October 2026

The bound-method reader now keeps property deletion and its unknown stack
result, invalidates cached property/descendant values and supplied constants,
and preserves reaching guards and invocation. Deletion remains an
uncomposed mutation. Scope deletion and unresolved dynamic property names
remain unsupported. Known string locals still work as member names.

The recorded case's movement method now passes its former unsupported
deletion, retaining 15 deletion receipts across 64 paths. Twelve path-budget
issues remain; complete/composed verdicts are both false. The bound-method
sidecar is not yet the normal controller's source-contact model. This is
static-reader progress, not a changed gameplay result. Focused: 139 passed.
See the appended source section in
`docs/evidence/steering/2026-10-16-visible-source-frames-and-sampling.md`.

The former corpus PID ended. Another exclusive model job is active as PID
72875. Preserve it and continue CPU-only work. No G09 run, holdout, model
job or resident runtime was changed. Source contacts, occurrence binding,
fresh asset export and independently witnessed native gameplay remain open.

Deletion checkpoint gates: smoke 165 passed, one skipped; lint, compile,
layering, generated DEPS consistency and writing passed.

## Boolean source typing, 16 October 2026

Boolean byte operands now remain boolean values. The old decoder retained
integers, which made strict boolean/number comparisons wrong. Numeric source
operations deliberately convert known booleans to zero or one; strict
equality preserves type distinctions. Unknown input stays symbolic.
Focused: 157 passed. Smoke: 165 passed, one skipped. Lint, compile, layering,
DEPS consistency and writing passed.

The recorded movement method still reaches 64 paths and eight path-budget
issues under each of five supplied input contexts. Its incomplete and
uncomposed verdicts remain false. This correction does not establish source
contact semantics or a live win. The next visual question is whether a
previous measured appearance can propose fresh current-pixel correspondence
when original global marks disappear, without asserting continuing object
identity or relaxing material/control qualification.

## Fresh current-pixel source geometry, 16 October 2026

Normal observation now uses measured source poses to propose fresh reference
fits on the worker that consumes each picture. Coarse local colour matching
precedes the unchanged bidirectional patch and six-correspondence gates.
Source geometry can qualify without a canonical object or global named
marks. The new channel creates no continuing object identity, role or input
authority. Capture, source, sampling and lever-arm precision remain checked.

One reference is measured per frame. Previously published geometry supplies
priority hints for missing work, with fair audits every fourth slot. A
capacity latch now retires old search hints explicitly instead of blocking
all later fresh work. Agreeing origin hypotheses carry an enclosing error
allowance and do not multiply support or count physical objects.

On the same 100-capture recording, the final ordinary run supplied the
composite origin on 42 captures through fresh or continuing geometry,
compared with eight in the earlier material-only path. All 38 fresh points
passed capture verification and fresh global matching, with largest
difference 0.160980 working pixels. Object-bound source points appeared on
51 frames, compared with 53 earlier; this is not controlled-body proof.
The final maximum loop gap was 0.040515 seconds and frame call 0.163468
seconds. This replay starts no model and sends no game input.

See `docs/evidence/steering/2026-10-16-current-pixels-refresh-source-geometry.md`.
Source-instance/role binding, source-defined contacts, uncached asset
acquisition and native gameplay proof remain open. The earlier reader
wording means `complete=False` and `effects_composed=False`: it remains
incomplete. No native deployment, G09 run or holdout was changed.

Fresh-geometry gates: 298 focused passed, one busy contention test
deselected; smoke 165 passed, one skipped. Lint, compile, layering, DEPS
consistency and writing passed. Governance retains 131 existing regressions
and module size the same unrelated regressions, including 197 excess lines.
Neither baseline changed. The exclusive model process remained PID 72875.

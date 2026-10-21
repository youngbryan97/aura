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

## Guarded source effects, 16 October 2026

The new on-demand `ProgramRead.method_effects` reads bounded acyclic source
flow as a DAG with guarded stack, register and memory joins. It retains
opaque calls, getters, coercions and deletions in memory order. It refuses
loops, inconsistent stacks, unbound registers and unproven constant pools.
Graph coverage never grants executable controller or identity authority.

The recorded case-18 movement method now has all 450 instructions and
19 branches represented in 657 nodes. The older path reader still stops at
64 paths. The outcome-display helper remains incomplete at the NewObject action (0x40);
the two-method inspection returned 1 and preserved that result. Focused
tests: 86 passed, including comparison with a separate concrete fixture VM
over 100 input assignments across 16 unknown branches.

See `docs/evidence/steering/2026-10-16-guarded-source-effects.md`. This is
source representation progress. It is not source contact qualification,
a changed gameplay result or native deployment. The current exclusive
model process is PID 75699; preserve it, G09 and the recorded holdouts.

Final review moved register loads and stores into the ordered VM-state
graph, so opaque effects cannot leave a falsely cached register value.
The final source measurement is 771 nodes for the same 450 instructions
and 19 branches. Focused source tests: 188 passed. Smoke: 165 passed, one
skipped. Lint, compile, layering, DEPS consistency and writing passed.
Governance and size retain their existing regressions; baselines unchanged.

Object creation and callbacks are now represented as opaque ordered source
effects. The outcome helper lifts all 237 instructions plus its virtual
exit in 298 nodes; callback creation never runs the nested body. Cyclic
methods retain their acyclic prefixes while remaining incomplete. The
reset method now retains 199 of 215 reachable addresses, including visual
attachments and the movement callback assignment; it formerly lost the
whole read to its later loop. This opens the source-instance binding
investigation without claiming that binding is already qualified.

The eleven-method inspection lifts six completely and preserves explicit
loop-summary refusals for five. All remain uncomposed. Focused: 196 passed.
See the appended creation section in the guarded-source evidence record.

Creation checkpoint gates: smoke 165 passed, one skipped; lint, compile,
layering, DEPS consistency and writing passed. Native proof remains open.

## In-process imports during CPU probes, 16 October 2026

CPU-only campaigns now scope a known model-backend import boundary around
lease, preparation and play. Already-loaded backends refuse admission;
ordinary backend imports fail before package execution. The case retains
the actual active-boundary report. This complements the existing worker
prohibition, CPU vision request and software-driver acknowledgement. Its
scope is this Python interpreter, not arbitrary native code or children.

Focused: 81 passed, two explicit native tests skipped. See
`docs/evidence/steering/2026-10-16-classical-probe-import-boundary.md`.
The next measurement is a recorded CPU browser attempt on development
case 18. Do not run G09 or the ten holdouts, or start/restart native Aura.

Import-boundary gates: smoke 165 passed, one skipped; lint, compile,
layering, DEPS consistency and writing passed. Governance remains at its
131 existing regressions, with no baseline update.

Actual control records now retain copied source-reference geometry, its
original capture receipts and current matching statuses. This will show
which channels reached the actor in the next CPU run. Missing, empty,
stale and truncated channels remain distinguishable. Source geometry
cannot set the selected body. The focused checks passed 78 tests.

Recording checkpoint gates: smoke 165 passed, one skipped; lint, compile,
layering, DEPS consistency and writing passed.

## Current CPU run stopped before gameplay, 16 October 2026

The recorded case-18 run at `e9ed28187` ended after 69.92 seconds on the
title screen, won=false. All five screen-option records read `BTART` on
the visible Start button, but its animation flag removed it before click
ranking. Only title text survived as a click; the actor tried arrow keys.
There were no fast-loop control or source-reference measurements.

The CPU boundary remained active with no loaded known backend and refused
five attempted MLX imports. The measured browser renderer was SwiftShader.
The recording closed without dropped events. G09 and holdouts were excluded.

Measured OCR now has an explicit bounded, unambiguous one-character path
through animated-control filtering and captured-menu handoff. Exact semantic
labels keep their strict grammar. Readouts, stale pixels and reset variants
do not gain this new path. Focused: 125 passed. The next original UI attempt
must show the press reaching gameplay. See the dated animated-navigation
evidence record. This remains CPU browser work, not native demo proof.

Navigation gates: smoke 165 passed, one skipped; lint, compile, layering,
DEPS consistency and writing passed. No baseline was changed.

## Native clock values in the CPU browser actor, 16 October 2026

The fresh `abd07b990` attempt clicked both damaged Start and Play labels
and reached gameplay. Its 77 control observations selected body 4 on 41
captures, but supplied no play choices or source references; every source
status was `capture_not_measured`. The game still lost. Supervisor: 64.01s.

The optional reference paths were receiving the rich float-subclass capture
clock while their strict pixel-geometry readers expected the plain scalar
already used by the eyes. The observation boundary now preserves the rich
receipt for the eyes and input timing, and supplies its scalar capture
value to source matching. Both real page and canvas clock regressions
failed before the fix; 113 focused checks passed afterward. The next
original CPU run must prove actual catalog and source publication.

Clock-boundary gates: smoke 165 passed, one skipped; lint, compile,
layering, DEPS consistency and writing passed. No native restart or G09
intervention was performed.

## Later continuous control intervals, 16 October 2026

The fresh `68f5b8f65` CPU browser attempt proves current source publication
in the actor: 33 of 67 observations had fresh geometry, 23 had object-bound
source geometry, and 31 selected the independently identified body. It still
lost with zero landings and zero recorded goal-directed choices. The final
active image was independently viewed. This is browser evidence, not native
demo proof. The classical boundary refused four MLX imports and reported no
loaded model backend. The recording closed without drops.

The code had another measurement gate: a missing picture sealed the first
two-sample prefix and discarded all later samples from that held input.
Acceleration needs four continuous samples. Later intervals can now satisfy
that requirement separately, while the press remains one independent input.
The original prefix, input deadline and control budget stay intact. Input
completion and contradictions retain their distinct meanings. The final
focused group passed 176 checks, including normal control identification and
the actor's recorded interval selection. See
`docs/evidence/steering/2026-10-16-control-intervals-after-missing-pictures.md`.

The next browser run must measure the gameplay effect. Also inspect why the
known successful title button was omitted from exploration after the loss:
novelty scoring can supply only untried actions to the later navigation
weights. Source contact/role binding and independently witnessed native play
remain open. G09, holdouts and the resident lifecycle remain untouched.

## Known navigation after discovery, 16 October 2026

The control-interval checkpoint is pushed at `792ac1118`. The next diagnosed
gate was in navigation: novelty scoring omitted already tried actions while
untried keys remained. Navigation weights therefore never received a button
whose recorded forward use was already known. Six ordinary-pipeline
regressions failed before the fix; 111 focused checks passed afterward.

Current offered navigation can now retain its recorded forward use without
being counted as new information. The path checks its current screen and
transition evidence, excludes resets and still passes task checks. See
`docs/evidence/steering/2026-10-16-known-navigation-after-discovery.md`.
Fresh browser measurement remains required. Native play and a win remain
unproven, and no protected process or holdout was used.

Fresh measurement at `ed2d1b41d`: navigation reuse carried into ordinary CPU
browser play, with repeated re-entry after losses. Gameplay still failed:
118 observations, no confirmed body, no goal-directed choice, no win. Ten
observations had fresh source geometry and 12 had object-bound geometry;
no owned later interval was published. The recorder closed without drops.
Keep this failure beside the passing component checks. The next input
investigation concerns current keyboard focus and discovery time. The
native checkout remains at `24c04f575`; it was not updated or restarted.

## Current keyboard targets, 16 October 2026

Both page input paths cached their first successful focus query. Three
actual-consumer regressions reproduced delivery after another element took
focus; a real Chromium fixture showed the second key missing the drawing.
The helper now checks the drawing's target for each press, down or tap.
Release does not refocus. Document-level input without a drawing retains
its prior behavior. Opt-in recording keeps current focus separately from
input completion and game acceptance.

Focused: 58 passed, one existing browser test deliberately deselected.
The actual repaired Python ports then delivered two of two keys for each
of normal, held and tapped input in a real software-rendered Chromium
fixture. It also checked drawing-free document input. See
`docs/evidence/steering/2026-10-16-current-keyboard-targets.md`.
This does not establish a cause for the preceding original-game losses,
which had no focus receipts. Gameplay still requires fresh measurement.

The pointer path can set the old focus flag without a keyboard query.
Three additional regressions and a repeated actual-port Chromium fixture
cover the first key from that state. All six drawing keydowns and the
document-level key were received, with no recorder drops. The native
checkout and protected process remain untouched.

The fresh `bb884ece0` original case 18 CPU browser run still lost: zero
landings and three crashes in the independently viewed final image.
All 27 focus receipts were positive, while all 241 control observations
left body identity unknown. Twenty-five observations had fresh source
geometry and 16 had object-bound geometry. The recorder closed without
drops (403 images, 98 input-port events). Keep this failure beside the
passing input fixtures. The remaining visual-to-control boundary still
needs repair; focus alone did not close it.

## Reference catalog readiness, 17 October 2026

Completed reference metadata used to spend the one-second interval meant
for expensive pixel searches. A regression reproduced the first search
waiting despite the catalog being ready. Catalog completion now releases
that first search; subsequent searches keep their full cadence and the
global single-worker permit. The first focused group passed 54 checks.
See `docs/evidence/steering/2026-10-17-reference-catalog-readiness.md`.
This scheduling repair supplies no controlled identity or game win.

Final: 55 focused checks passed, smoke passed 165 with one skip, and lint,
compilation, layering, dependency consistency and writing passed. The
remaining work is current visual continuity into independently proven
control, source contact/role binding and ordinary native-app proof.

The readiness checkpoint was merged with the concurrent main commits and
pushed at `a40ac6c7d`. The combined revision passed 55 focused checks and
smoke (165 passed, one skipped). The protected source checkout was not
updated. A point-capacity prototype did not improve recorded tracking and
was rejected. The next change permits bounded wider reference reacquisition
when the local window has no qualified fit. Two large-displacement cases
failed before it; 50 focused checks passed afterward. A paired 59-picture
replay found one reference on 16 captures instead of eight. Another
reference was never seeded, so its fallback remains unmeasured. See
`docs/evidence/steering/2026-10-17-reference-reacquisition-outside-a-local-window.md`.
No gameplay improvement or controlled identity is established by that replay.

The wider fit is pushed at `2197fbc45`. Ordinary asynchronous publication
then observed all 59 source-paced recorded frames. Eleven fresh source
geometries passed independent current-pixel checks (maximum difference
0.129063 working pixels); zero object-bound points and zero continuing
source frames were published. This is the remaining upstream connection:
current correspondence exists without a continuing named material path.
It supplies neither causal control nor a game win.

## Current words and old control coordinates, 17 October 2026

Fresh actor run at `b1404b177`: still no win, 38 control observations with no
body, five positive focus receipts, three fresh source geometries and no
continuing or object-bound source geometry. The independently viewed final
image was the instructions screen. The recorded end screen revealed old
Start/Play labels replacing a percentage and a new retry label in layout.
CPU OCR reread the percentage correctly from the saved bytes.

The cause was the separate 30-second coordinate-only navigation-label
cache, applied after the pixel-backed reader. Eight regressions failed
before its retirement. The first focused group passed 81 checks afterward;
the existing pixel-backed recovery and reset handling remain. See
`docs/evidence/steering/2026-10-17-current-words-cannot-inherit-old-coordinates.md`.
A source-guided point-sampling prototype gave one more attachment and two
fewer point continuations on 59 recorded frames; it was not promoted.

The coordinate-label change also passed smoke (165 passed, one skipped),
lint, compilation, layering, dependency consistency and writing. Its
ordinary gameplay effect remains to be measured. The skill module remains
below its 2,000-line cap. No protected process or native checkout was changed.

Pushed at `0d71007e1`. The fresh CPU actor run preserved the current
percentage in layout, so the label repair carried over. Gameplay still
failed: 75 observations with no selected body, eight positive focus
receipts, nine fresh geometries, three object-bound and six continuing
geometries, zero goal-directed choices. Final image: zero landings, one
crash. The record closed without drops (218 images, 50 input-port events).
Actual candidate press counts and statistics are already in each control
event's `evidence` field; use them before adding more instrumentation.
None cleared both thresholds. Do not interpret a zero ratio as no motion:
it also represents insufficient repeated groups. The pinned emulator uses
delegated shadow focus; no focus change was made merely from a suspicion
about the outer player. Continue tracing visual sample continuity.

## Resting material continuity, 17 October 2026

The corrected-label failure is preserved at `9e323a76c`. A separate red
regression then showed the ordinary unmatched-object path discarding a
measured point origin despite unchanged visible pixels and current marks.
It can now continue through the ordinary estimator when current named motion
agrees with matching pixels and boundaries. Missing or contradictory
support remains unknown. The focused group passed 72 checks.
See `docs/evidence/steering/2026-10-17-resting-material-keeps-its-origin.md`.
The 75-frame last-game pixel replay was unchanged (40 point continuations,
22 moving material-path samples). Do not attribute a landing improvement
to this stationary-path repair; the moving-body failure remains open.

Smoke and the main code gates also passed under the bounded supervisor
(90.63 seconds). Continue from the moving visual path and physical-control
timeline; this checkpoint does not qualify the native demo.

## Read before waiting on a moving clock, 17 October 2026

The corrected-label recording's first fast picture arrived 2.620 seconds
after its Play click completed; the first key completed 3.278 seconds after
the click. The log includes a 1.3-second scene wait. `OnAPage` was deciding
whether to wait from the preceding menu. A known running clock now gets a
current reading before scene settling. Fresh menus retain navigation;
unknown and waiting environments retain the wait. Six ordering regressions
failed before the change, 41 focused checks passed afterward, then 59
navigation and pointer checks passed. See
`docs/evidence/steering/2026-10-17-read-the-current-screen-before-waiting.md`.
Measure ordinary gameplay before claiming any improvement from the saved
time. No protected source checkout or runtime lifecycle was changed.

Smoke (165 passed, one skipped), lint, compilation, layering, dependency
consistency and writing passed. The combined group took 98.56 seconds.

Original follow-up at `16371505e` still failed (140.16 seconds; `lost`,
`unread`, `unread`; final image instructions). Of 149 control observations,
none selected a body. Fresh geometry: 22; object-bound and continuing: zero.
The first image/key request worsened to 4.280/5.712 seconds after Play.
The log measured a 2,151.2-millisecond cold observation startup after
contention selected the separate process. Do not claim a timing or gameplay
gain from the wait repair. Test preparation on a current menu next. The
record closed without drops, with software graphics and no model loaded
in the guarded interpreter.

## Prepare classical perception on a menu, 17 October 2026

The ordinary handoff can now prepare an empty classical observer on a
captured menu in a known running world, before input starts play. Its
bounded startup and retirement are reused; no picture, body state or input
is transferred. Normal backend choice remains. Learned selections skip
this preparation. The focused group passed 49 checks, including pending,
failed and cancelled preparation. A real cold-process probe measured
0.722226 seconds on the menu and 0.000707 seconds for the next observer
using the same child, with zero pictures and bodies. See
`docs/evidence/steering/2026-10-17-classical-observation-prepared-on-a-menu.md`.
Original-game carryover still needs measurement. This is not native proof.

Explicit local-only observation also skips subprocess preparation. The
policy and deadline group passed 29 checks; smoke (165 passed, one skipped)
and the main code gates passed in 111.96 seconds. The skill is 1,980 lines.

Merged 22 intervening main commits into the isolated checkout at
`b7041896b`. The combined focused group passed 51 checks, then smoke and the
main code gates passed in 123.02 seconds. No native source checkout or
runtime was changed. Measure the actual menu-to-play route next.

At `82822b7dd`, menu preparation reached the original lander before Play
(1.110 seconds). The first fast image arrived 1.119547 seconds after Play;
the first key completed at 1.799338 seconds. The lander still had zero
landings and no confirmed body across 309 observations. Source geometries:
53 fresh, five object-bound, 70 continuing; these do not establish the play
phase or control. The food-fight case also had no demonstrated win; its
final image was active play. Both complete records closed without drops.
The two-case campaign took 264.89 seconds. A 40-frame alternate original
tracker replay kept the same 25 point continuations and 25 moving material
samples, cost more time and identified no body; it was not promoted.
The next red regression shows discovery repeating a twice-delivered counter
probe when it starts its ordinary key order. Retain delivered counts for
ordering, without creating any causal evidence or changing thresholds.

## Discovery retains delivered probes, 17 October 2026

The actual hold/deadline route reproduced a third first-key delivery after
the counter programme's two pulses. Unknown-body discovery now retains a
bounded ledger of delivered independent probes across those strategies and
prefers a less-exposed current key. Counts supply no response or identity;
neutral phases, finite allowances and physical deadlines remain. Duplicate
holds cannot renew across epochs. The broad timing group passed 155 checks.
See `docs/evidence/steering/2026-10-17-discovery-keeps-its-delivered-probes.md`.
Measure ordinary original-game carryover before claiming any outcome gain.

The cold-start order also permits an ordinary pulse before the counter
programme. That second ordering failed with the first repair. Both strategies
now respect the same delivered count: a new counter pulse cannot repeat an
allowance already delivered by ordinary discovery. The final group passed
156 checks in 19.07 seconds; smoke (165 passed, one skipped) and the main code
gates passed in 88.05 seconds. Governance still reports 130 regressions, the
same recorded count. No original outcome gain has been measured yet.

The original follow-up at `53fc0d28e` completed in 133.29 seconds with three
losses and zero landings. Its first controls now went up, up, left, proving
the order reached play. No body qualified across 97 observations. The first
0.464213-second pulse retained one settled sample: two initial capture
intervals were available, but reservation required three. The next pulse's
three-sample prefix also fell short. Continue with cold-start observation
reservation and preserve the four-sample and identity requirements.

Three cold-start reservation regressions failed, including the actual trial
executor retaining a 0.45-second deadline despite two measured capture
intervals. Reservation now uses the slowest valid recent interval as soon
as one exists, inside the existing maximum and without changing any delivered
deadline. The focused timing group passed 152 checks. See
`docs/evidence/steering/2026-10-17-first-trial-reserves-observed-frames.md`.
Original carryover still needs measurement.

Smoke (165 passed, one skipped) and the main code gates passed after this
repair in 92.65 seconds. No native lifecycle or protected case was touched.

The original follow-up at `a702a104e` retained five and four settled samples
from its first two thrust intervals. That carryover supplied two acceleration
measurements, but neutral comparison intervals still broke: a 0.424726-second
capture gap discarded 28 anchors, then a contradiction and another long gap
prevented neutral evidence. No body or win qualified across 160 observations;
the final image was instructions. A serial 40-frame asynchronous local/separate/
local comparison had no operation over 0.35 seconds, so switching processes
alone is not supported as the explanation for the live gaps. Measure the
ordinary capture/observation/action stages before changing their scheduling.

Ordinary timing wrappers then measured optional fresh geometry at up to
0.477594 seconds, with a 0.362145-second p95. Bar reading reached 0.245547
seconds. Both diagnostic attempts still failed. Three new regressions showed
four source poses being refitted in one frame. Optional geometry now rotates
one pose per reference visit and retains the unsearched hints; all published
matches still need current pixels and precision. The focused group passed
38 checks. See `docs/evidence/steering/2026-10-17-optional-geometry-has-a-pose-budget.md`.
Measure carryover and then address the measured bar-discovery spikes.

The nearby contracts passed 55 checks; smoke (165 passed, one skipped) and
the main code gates passed in 96.57 seconds after the pose-budget repair.

At pushed `7f5e22242`, the ordinary CPU follow-up finally selected body 1
through independent repeated up/neutral trials: statistic 27.038360, effect
26.948735; both thresholds remain 20. The independently viewed capture places
that box on the vehicle. Nineteen of 124 observations retained the selection.
Fresh geometry's observed p95/max fell to 0.106731/0.142820 seconds. The run
still did not win and ended in active play. A later left calibration carried
the body off the left edge and its track was lost. Bar reading still reached
0.168235 seconds. Continue the bar scan and bounded calibration observability;
this milestone does not qualify the native demo.

Two bar micro-optimizations preserved outputs but did not justify promotion.
The retained architecture instead spreads discovery over pictures, at most
eight sampled rows per read. Delayed candidates must pass the current colour,
run and thickness checks; established meters remain sampled every picture.
The focused group passed 56 checks. Complete scan outputs matched the prior
implementation on 17 pictures; incremental ordinary reads had a 0.012662-second
maximum in that fixed-pixel probe. See
`docs/evidence/steering/2026-10-17-bar-discovery-keeps-time-for-pictures.md`.
Measure original carryover after the gates, then address calibration that
drives an identified body beyond observable bounds.

Two additional current-thickness refusals passed in the 12-test focused file.
Smoke (165 passed, one skipped), lint and compilation passed. Layering first
refused the newly registered canary import; a narrow manual organism DEPS
entry fixed that. Layering, dependency consistency and writing then passed
in 37.59 seconds. Governance retains 130 regressions; its baseline is unchanged.

At `1e277a2c7`, the ordinary CPU follow-up measured bar-reading p95/max at
0.016604/0.022147 seconds. Complete observation p95/max were 0.245581/0.260533
seconds. Sixty-four of 145 observations selected a body across the complete
record (the earlier interim count of 41 covered its first stretch). It still
ended with zero landings and one crash. Only up remained known: left was
interrupted early, and right's first settled frame was contradicted before
further gaps. Trace the current rotating material and motor-trial gate next.

On October 18, a 90-picture paired replay reproduced the lateral failure
with both classical CPU trackers; neither learned a lateral key. The first
lost origin followed a 0.350355-second gap. Its ordinary timing trace shows
mandatory motion and copying finished before another 0.105878 seconds of
optional geometry delayed control. The observation worker now budgets these
optional refits against measured essential work. One operation may overdraw
its allowance, but later refits must repay the cost; deferred frames supply
no fresh geometry. No identity threshold or input lease changed. The focused
group passed 89 checks, and smoke (165 passed, one skipped) plus the main code
gates passed in 143.79 seconds. See
`docs/evidence/steering/2026-10-18-optional-geometry-shares-observation-time.md`.
Measure ordinary carryover next; lateral control, landings and the native
demo remain unproven.

The ordinary follow-up at pushed `4f00cac1a` still reported two losses and
no selected body. A viewed loss image shows zero landings and one crash;
the final recorded image is active play after another start. Budgeting
deferred 26 refits and seven fresh geometry records were published, but
one admitted refit still took 0.193575 seconds. The first thrust retained
only three settled samples before a gap. Its default eyes ran in a separate
CPU process this time, so the timing distribution is not paired with the
earlier local run. Split the remaining multi-position fine-refit operation
before further original testing. No native demo or campaign pass is shown.

Ordinary optional refits now fit one candidate position of one pose per
admitted visit. Failures rotate through all three local and three wider
peaks for every pose; only current precision-qualified material returns
that pose to local tracking. Untried positions remain explicit, and the
complete offline search API remains available. The focused group passed
105 checks; smoke (165 passed, one skipped) and the main code gates passed
in 195.05 seconds. See
`docs/evidence/steering/2026-10-18-one-position-per-optional-refit.md`.
Measure original carryover next. No original win or native readiness is
established by these component results.

The two-minute original follow-up at `db593d175` completed in 142.52 seconds
and still selected no body across 168 observations. Its independently viewed
final screen shows zero landings and one crash. Refit p95/max were
0.101025/0.134989 seconds, while complete observation still reached 0.460861
seconds. The vehicle's first thrust retained two samples and its second
retained five; neutral intervals retained eight and seven. Neither the
statistic nor the effect jointly qualified. The next diagnostic is point
correspondence versus capacity eviction at the first material-path break.
Do not infer demo readiness from the completed scheduling repairs.

The first material break is not a capacity eviction: four owned names fall
to two valid correspondences at a 0.207845-second interval. A dense source
fit works at a later break but fails at the first; native LK and a larger
context also fail to select a body. No such fallback was promoted. See
`docs/evidence/steering/2026-10-18-information-in-observed-intervals.md`.
A post-hoc precision-aware reconstruction dates velocity at its displacement
interval midpoint, separates local displacement error from shared reference
error, and includes capture-time uncertainty. The recorded two thrust and
two rest holds then give a candidate statistic/effect of 38.402551/20.793608.
This estimator is not integrated, and these are estimated errors, not
guaranteed bounds. Build and falsify that general measurement path before
another gameplay claim; retain independent holds, contradiction retirement
and the existing thresholds.

The interval measurement is now connected to ordinary perception and control
attribution. It carries current named-pixel displacement precision through
the detached observer, uses capture-bounded input ownership, and dates each
velocity at its interval midpoint. Both velocity and midpoint-time uncertainty
reach the acceleration estimate. Physical holds remain the independent
trials; existing exclusions, resolved motion and 20/20 thresholds remain.
The focused groups passed 230 tests before the time-axis refinement and 97
after it. Three serial saved-pixel replays selected the vehicle on 15/44
observations with the new path and on 0/44 both without the estimator and
without capture clocks. The corrected first statistic/effect is
27.434242/20.325563. See
`docs/evidence/steering/2026-10-18-responses-from-measured-material-intervals.md`.
Complete checkpoint gates and fresh ordinary carryover next. No landing,
original-game win or native presentation readiness is established.

The interval checkpoint is pushed in `3830ca8e4`, which includes the latest
other sessions' work through `a96be13d2`. Post-merge focused checks passed
170 tests, and smoke (164 passed, one skipped), lint, compile, layering,
DEPS and writing passed in 78.48 seconds. The ordinary host-GPU browser
follow-up still won nothing: 171 of 583 retained observations selected the
vehicle, but all six planning choices had an unbound requested destination.
Its partial recording dropped 250 events and cannot support a full replay.
See `docs/evidence/steering/2026-10-20-a-capture-clock-can-reach-instance-naming.md`.
The next repaired connection is visual naming's timestamp gate: it rejected
the ordinary capture's float subclass before making any query. Finite real
capture clocks now pass into the existing instance-owned request path; 101
focused tests passed. Complete its gates and a bounded real-vision ordinary
follow-up. The resident runtime, G09 and holdouts remain untouched.

Visual naming's capture-clock fix is pushed at `75ac772ab`. Its checkpoint
gates passed. The first small-vision admission attempt stopped before loading:
sharing the resident ledger was refused by the state guard, and the default
vision weights were not found by the registry. The guard stays intact. The
registry now resolves an already-cached snapshot after missing its local
inventory, without fetching or granting a semantic serving role. Its focused
registry group and checkpoint gates passed. A new diagnostic must use the
supported private ledger plus current host process/resource checks and retain
its limited coordination scope. Continue ordinary play with the existing
small vision client; no native runtime restart or demo qualification occurred.

The owned small-vision follow-up at `e1fc9acd2` ran in 138.91 seconds but
failed: zero wins, zero observed landings, 188 selected bodies across 657
control observations and no planning choice. Fourteen instance names now
reach ordinary play. The complete recording dropped no events. The owned
worker was shut down and its PID confirmed gone; no native runtime or cortex
was loaded. The run exposed a phase error: initial terminal screens could
exhaust menu handoffs and resume body discovery, including assigning a wreck
as the player. Current asserted end-word pixels now return that stretch
without a novelty or menu-budget requirement and retire pending body guesses.
The actor and integration group passed 121 tests. See
`docs/evidence/steering/2026-10-20-current-end-pixels-stop-body-experiments.md`.
Finish gates, push this checkpoint and measure ordinary carryover next.
G09, holdouts and the native runtime remain untouched. Demo readiness is open.

The phase checkpoint is pushed as `56502de79`; gates passed in 78.12 seconds.
Its ordinary small-vision follow-up finished in 134.88 seconds with no win.
The complete recording retained 308 selected bodies across 582 observations,
48 choices, and zero visual-fallback body assignments. All 48 choices were
`wait` with no bound requested destination. The motion classification also
changed from accelerated to direct while useful acceleration estimates
remained stored. See the whole-chain mechanical audit:
`docs/evidence/steering/2026-10-20-why-the-whole-play-chain-still-fails.md`.
Expose those exact refusal/model-selection branches, then falsify and repair
them before another original test. PID 90144 was owned, cleaned up and
confirmed gone. No native demo or campaign pass is established.

The whole-chain audit now has executable branch records on the ordinary
path. Goal binding records its actual refusal and bounded candidate names;
motion-family selection records the actual predicates, body number and
evaluation time. No thresholds or choices changed. The focused groups passed
138 and 62 checks. Finish gates and measure those branches before choosing
the next behavioral repair. The complete mechanical explanation is in the
whole-chain evidence file above.

The diagnostic checkpoint is pushed as `74da4b72e`; gates passed in 78.65
seconds. Its fresh ordinary run retained 570 selected bodies in 659
observations, but all 413 choices waited with no destination. The exact
refusals were 410 absent named referents and three unavailable body/request
states. Three motion-family refusals came from the effect threshold; no
pace or release predicate was visited in those refusals. Prioritize task
grounding over an unproven classifier change. The final viewed frame shows
zero landings and one crash; the owned worker was cleaned up.

Missing task referents now reserve one current visual query slot, while
the normal player/scene queries keep the remaining slots. Geometry ranks
observations and grants no target authority; the unique current binder and
provider prompts are unchanged. The integration group passed 119 checks.
Finish gates, push and measure this attention connection. Whole-object role
grounding and perception-budget exhaustion remain open.

October 20 CPU-only continuation: Bryan has reserved the GPU for demo runs.
No model worker, G09, holdout or native runtime was used. Checkpoints
`a08ca0c7b`, `9cd56e4de`, `1107b0fb5`, `3e2820a0a` and `5d54bf2f3` add owned
wordless-surface custody, comparable forward motion evidence, discovery/input
arbitration, refusal-specific referent requests/contact unknowns, and joint
contrast qualification. Main's concurrent decoding work was preserved at
merge `17a5f74e56`. The exhaustive proposal ledger and bounded measurements
are in `docs/evidence/steering/2026-10-20-play-chain-proposal-disposition.md`.

The additional Argon PDF was rendered and read in full. Its named code and
verification suite are unavailable; reported original-game scores remain
unverified. Its composite binder manufactures width and merges unrelated
fragments; its asserted angle/lateral invariant fails after source channel
saturation. Those patches must not be adopted as written.

`ProgramRead.resolved_method` now exposes bounded arithmetic from current
uniquely bound source and supplied primitive values. Actual bytecode fixtures
match an independent interpreter over 100 states. Missing state, unknown
guards, calls and deletion remain named evidence requests without partial
writes. The post-merge group passed 196 tests; smoke 164 passed, one skipped;
lint, compile, layering and writing passed in 70.21 seconds. Live hidden-state
estimation, source call composition, whole-parent/role/contact producers and
the ordinary prepared-controller bridge remain open. Do not claim demo
readiness from this checkpoint; continue the next measured CPU repair.

The source arithmetic checkpoint is pushed as `b94a32218`. Whole registered
artwork now has an independent, current semantic observation path through the
ordinary worker, naming and actual chooser, with current pixel/edge support
and separate birth/material identity. It stays outside body tracking. The
matcher no longer suppresses that material when an origin attaches to a
fragment. The final group passed 200 tests; smoke 164 passed, one skipped;
lint/compile/layering/writing passed in 72.55 seconds. Physical contact height,
arrival limits and observed task roles remain unknown.

CPU replay of twelve retained original-18 pictures produced no whole
referent. The destination reference was located, but point sampling had not
kept enough source-supported marks. Fix the sampling handoff before requiring
global material identity; use current matched pixels to request sampling,
without changing identity thresholds. The new tool is
`tools/measure_whole_referents_in_recorded_pixels.py`. Preserve the zero result.

The additional supplied 2,050-line Python lab was read and independently
reproduced: nine self-checks, six toy wins and 50/50 toy Monte Carlo wins.
It also accepts future foreign custody, manufactures fragment width,
miscomputes containment and verifies an accelerated law for a stationary body.
Its repaired body selection is assigned True and its world supplies exact
state. The new PY checklist in the proposal ledger records every mechanism,
rejection and remaining calibration contract. Do not install its functions
as purported drop-in Aura repairs or count its scores as original-game proof.

The source appearance now reaches current point sampling before an object
owns global marks. The fast refitter preserves affine shape and visible
clipping; one fit supplies both the sampling hint and the current material
attachment. The ordinary reference owner accepts those bindings and retains
their measured points. Optional accounting covers both sides of observation.
The regression group passed 200 tests, the final owner group 17; smoke 164
passed, one skipped; lint/compile/layering/writing passed in 89.48 seconds.

The final twelve-frame CPU replay has seven consumed hints and eight frames
with current source material, against no destination material at the original
checkpoint. It still has zero whole referents. The next measured refusal is
`whole_reference_out_of_view`: the visible building's ladder leaves the bottom
of the capture. Treat image clipping separately from in-frame occlusion,
preserving unknown roles and contact geometry. Results and intermediate
failures are in `artifacts/flash-campaign/2026-10-20-current-reference-sampling/`.
No GPU, physical input, native runtime, G09 or holdout was used.

Clipping now remains explicit on a partial source assembly. All material
inside the captured image keeps the colour/edge gates, and a missing edge
cannot supply a relation point or region. The final group passed 156 tests;
smoke 164 passed, one skipped; lint/compile/layering/writing passed in 76.85
seconds. The retained original still supplies zero qualified referents: the
next refusal is a current pixel mismatch, mean 0.07436 and edge error 0.50458,
despite 99.4248% of opaque source centres being in view.

Photometric fitting and affine-first experiments improved residuals but did
not qualify the object; both remain private experiments. The SVG's native
bitmap exactly matches the rendered PNG's opaque RGB samples, so extracting
that bitmap alone is not the repair. Inspect forward raster formation and
the actual sampling/filter model next. Current artifacts remain under
`artifacts/flash-campaign/2026-10-20-clipped-source-assemblies/`.

Source visual variants now retain complete declared frame boundaries,
selected-frame export/registration and distinct coordinate identities even
when bitmap bytes match. A general preparation tool fills missing frames
for registered asset families; the read-only catalog exposes actual coverage
and visits each family before its next variant. Labels grant no task role
and a matched static reference grants no active execution-frame identity.

The private original-18 corpus now has all five top-level frames of asset 52
and both frames of asset 57. It covers no nested state combinations or other
games. The expanded group passed 176 tests; final canaries 17. Smoke: 164
passed, one skipped. Lint/compile passed; layering/writing passed after adding
the explicit source-frame canary dependency. Seven-variant replay still has
zero qualified whole referents, despite current source material in eight
of twelve pictures. GPU and resident runtime remain untouched.

Next connect current source material groups to semantic attention while
keeping their geometry unqualified. A full-copy refusal must not erase a
valid place to look; a partial group must not authorize movement, invent
width or acquire its source label as a task role. Preserve the zero results.
The current variant artifacts are under
`artifacts/flash-campaign/2026-10-20-reference-frame-variants/`.

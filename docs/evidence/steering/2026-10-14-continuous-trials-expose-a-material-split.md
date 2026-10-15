# Continuous trials expose a material split — 14 October 2026

The original-game run from `ec43fab8d05a64fb2abb6086468f4b29332beb41`
completed in 165.36358199990354 seconds, exit 0, without a wrapper timeout.
The game was not won and no body was assigned. The repaired observation
connection now records ordinary material observations during physical
pixel trials. A moving candidate loses its entity number before repeated
independent controls can establish its identity.

Artifacts:
`/private/tmp/codex-steer/continuous-trials-original9-1014`.
`choices.jsonl` has 558 decisions, zero assigned-body decisions and median
picture age 0.22279473871458322 seconds. There are 750 records in
`candidate-measurements.jsonl` and 560 in `pixel-search-admission.jsonl`.

## Physical input and observation

Line 170 of `pixel-search-admission.jsonl` records the first intervention:
24 admitted presses, 117 ordinary observations, release confirmed, no
qualified region and no confirmed body. The selected settling interval is
0.6991215467453002 seconds. The median actual hold is
0.7263560830033384 seconds.

The first pair starts at 527098.3538630009. The last after-capture ends at
527124.7939527035 and the last release is acknowledged at 527124.810884125.
All 117 ordinary observations occur in identification epoch three. The
preceding run had zero ordinary observations during its first 24 presses;
that acquisition disconnect is repaired in this run.

Line 542 records another intervention: 14 admitted presses and 71 ordinary
observations before an observed phase change stops the experiment. Release
is confirmed. Its incomplete cohort supplies no region or identity.
The selected interval is 0.7484355688095092 seconds.

## A moving candidate cannot retain its trial history

The first pair's image shows a small moving sprite near frozen RGB
coordinates `(93, 75)`. The nearby tracker candidate is number 84 at
working coordinates `(70.5, 56.5)`. Image extent explains the coordinate
difference. This is a strong body candidate, not a proven player identity.
Source information or color alone does not authorize control attribution.

During press zero, `down`, candidate 84 moves from `(74, 56)` to
`(103.5, 51)` across four observations and earns one independent down
trial. Subsequent up presses move it across the screen in the other
direction. At candidate line 185, its F statistic is 9.12 and effect is
38.83 working pixels per second, rounded here from the recorded values.
At line 186, those values are 6.62 and 42.49. Physical travel and measured
effect are present; the statistical criterion and repeated independent
groups still have to qualify on one continuous entity.

By line 191, candidate 84 has three up trials and only one down trial. Its
second down trial starts while it overlaps a painted parking-line region.
The following records show the discontinuity:

| Candidate log line | Number | Box `[x,y,w,h]` | Motion | Material marks |
| --- | --- | --- | --- | --- |
| 191 | 84 | `[52,60,12,8]` | measured points | 7 |
| 192 | 84 | `[56.5,74,17,36]` | measured points | 5 |
| 193 | 84 | `[57,73.5,16,37]` | unavailable | 0 |
| 194 | 84 | `[57.5,73.5,15,37]` | unavailable | 7 |
| 195 | 131 | `[58.5,59,11,8]` | unavailable | 0 |
| 196 | 132 | `[64,58,12,8]` | unavailable | 0 |

The box grows from a small sprite to a tall connected shape before it
splits. The marks at line 191 are
`34766,34758,34763,34755,34759,34752,34754`.
At line 192, five remain:
`34758,34754,34763,34766,34752`.
At line 194, the pose includes retained names
`34764,34766,34754,34752`, together with new names
`35713,35712,35715`. The added points and enlarged box require a material
membership check; shared connected pixels do not establish one rigid body.

The new numbers 131 and 132 both retain birth time 527093.2454249859 from
candidate 84, but their independent input receipts are empty. Candidate
132 resumes measured motion at line 197. Its first up trial appears at
line 200. This new entity cannot inherit candidate 84's causal evidence
merely because it retained the earlier birth time.

Later moving patches receive numbers 185, 189, 227 and 273. Their physical
relationship still requires a verified material bridge. Reassigning all of
them the old entity would conceal the original tracking failure.

## Source boundary

The independently read source states that `MPlayer` is appended after the
ordinary cars and then passed to `Player.mCreate`; `Game.mIsYourCar`
compares that object identity. Source references are
`/private/tmp/codex-steer/verified-source-9-1013.txt`, lines 1053 through
1054, 1467 through 1469 and 1617 through 1637. The source-audit agent
independently verified these bindings. This artifact audit does not read
the live game's private object arrays or grant authority from them.

Candidate 84's geometry follows the visible small moving sprite. This
audit has not independently matched its entire observed track to the
original `MPlayer` clip. Other cars can respond indirectly through contact.
The entity split remains an observable perception failure regardless of
which car this candidate represents.

## Next necessary connection and evidence limits

The tracker must preserve a measured moving material subset across contact
with static painted structure, or report its continuity as unresolved. A
current point bridge must bind the separated child to the previous object
before the identity observer can keep the previous trials. Connected
component size, old birth time and similar appearance cannot supply that
authority alone.

This audit read completed logs, paired images and existing source. It ran
only bounded metadata summaries; it did not launch a model, browser or
real-time experiment. It did not change source, identity thresholds or
game state. The source audit and observation repair are separate evidence
layers. Native-app prompt execution, correct player attribution, a game win
and presentation readiness remain unproved.

## Code trace after the completed run

The first enlarged box occurs after a capture gap of 0.268130064 seconds,
between candidate lines 191 and 192. At this revision three separation
paths each reject a preceding observation older than 0.25 seconds:

- `texture_that_continues_when_outlines_join.py`, lines 60 and 77, rejects
  both eligibility and transformed-texture proposals.
- `parts_when_things_touch.py`, line 43, excludes the recent color model.
- `a_small_thing_in_a_joined_patch.py`, line 62, excludes the recent small
  object's seed.

Therefore none of these paths can use candidate 84's preceding observation
to separate that first joined patch. Current material association allows a
0.35-second bridge in `continuing_material_before_outlines.py`, line 47.
The enlarged blob is assigned to 84, and `_moved` then overwrites its
box, patch and support in `what_moves_in_the_picture.py`. The existing
trace does not identify which matching branch made that assignment.

At the later split, the preceding box area is 555 pixels and the separated
sprite box area is 88 pixels. Their ratio is 6.306818181818182, above the
ordinary `_cost` limit of four at `what_moves_in_the_picture.py`, line 791.
Its material fallback still needs current, unique, spread-out geometry;
the available metadata does not expose the exact fallback refusal.

The broader support ledger also records the transfer. At line 194,
number 84 carries
`34752,34754,34758,34763,34764,34766,35712,35713,35715`.
At line 195, number 131 carries
`34752,34754,34758,34763,34764,34765,34766,36096,36097`.
The production `WhatThePointsSay` can report a uniquely matching lost
entity from this point overlap. Its caller at
`what_moves_in_the_picture.py`, line 1100, discards the returned lost
entity number and restores only the earlier birth time. That explains how
new numbers retain birth without receiving the old entity's trial ledger.
This weak point-overlap result cannot alone justify transferring causal
authority; it lacks the stronger current geometry required by the material
matcher.

A minimal repair target is to preserve the measured small material subset
before a connected outline replaces its support. Current owned motion and
its uncertainty can bound whether that subset remains measurable across
the actual capture gap. The corresponding regression must include a small
moving texture crossing a static painted line, a 0.268-second gap, and
separation afterward. It must also reject a competing body or incompatible
point geometry. A larger fixed time limit or copying identity by birth
alone would leave the membership problem unresolved.

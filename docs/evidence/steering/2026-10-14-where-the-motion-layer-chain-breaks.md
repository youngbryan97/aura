# Where the motion-layer chain breaks — 14 October 2026

## Correction to the preceding run record

The original-game run at `3f4621c8ab1c7e2277d33d2f559ca76d2ceca8b5`
qualified and applied a responsive region in its first experiment. The
second experiment returned no region. The preceding motion-layer evidence
record said both experiments returned no region; that statement was wrong.
This audit preserves the earlier record and corrects it here.

The run remains unwon, with zero assigned-body decisions. A successful
search hint did not become confirmed control.

Evidence root:
`/private/tmp/codex-steer/motion-layer-original9-1014`.
`completion.json` records exit 0, no wrapper timeout and 166.593384 seconds.
`pixel-search-admission.jsonl` has 688 records;
`candidate-measurements.jsonl` has 689 records. Each experiment directory
has its own schedule and 24 press JSON files with separate before/after
PNG captures. These are the corrected, separate artifacts. The mixed
files from the preceding settled-exposure run were excluded.

## Discovery and sampling reached their consumers

At line 276 of `pixel-search-admission.jsonl`, the first intervention
records the following result:

| Field | Recorded value |
| --- | --- |
| Status | complete |
| Admitted physical presses | 24 |
| Physical release confirmed | true |
| Tested region | `[144, 24, 168, 48]` |
| Motion reference | distributed translation |
| Sampling hint applied | true |
| Retained priority points | 64 |
| Fresh priority points | 64 |
| Selected settling interval | 0.35598709106445314 seconds |

A numeric replay used the stored signatures, original action labels and
cohort labels with the production `region_from_press_features` function.
It returned the same region, discovery probability 0.010, confirmation
probability 0.004, effect 5.079448941142653 and sampling bounds
`[120, 0, 192, 72]`. No optical-flow recomputation, new images or game
state were supplied to this replay.

The discovery cohort has 12 presses, six per key. In its body cell
`(row=1, column=6)`, translation support is available on 10 presses and
rigid-angle support on six. The production availability gate requires
the channel to be measurable on every discovery press, so these channels
are excluded. The local patch-angle channel is available on all 12 and
supplies the accepted contrast. There are 84 usable cells overall and one
eligible cell. Confirmation retains its fixed pattern and has all required
measurements. The returned region establishes that the discovery, camera,
support, area, ambiguity and confirmation gates passed for this experiment.

At line 633, the second intervention records 24 admitted presses and
confirmed release, with no qualified region or applied hint. Its body cell
has translation and patch-angle channels available throughout discovery,
but its maximum-statistic discovery probability is 0.042, above the 0.010
limit. Its body-cell contrast magnitude is 1.0902219729243436. No cell
qualifies, so this experiment stops before region confirmation.

Both experiments used the same two controls, `up` and `down`. Their
schedules were recorded before outcomes. The median measured holds were
0.389345479 and 0.386882063 seconds. This audit did not change thresholds,
substitute trials or select a favorable subset for qualification.

## The experiment's observations did not reach identity

The first experiment begins at monotonic time 522607.229255583. Its last
physical release is acknowledged at 522620.202365541. During that entire
interval there are zero `WhichIsHers.saw` records.

The records immediately surrounding it are lines 277 and 278 of
`candidate-measurements.jsonl`, at 522607.00984477997 and
522620.28444767. The green sprite candidate, number 74, has these same
independent trial counts on both sides:

| Control | Settled independent presses |
| --- | --- |
| up | 1 |
| neutral | 10 |
| left | 4 |
| down | 2 |
| right | 2 |

Its F statistic stays 1.6489461482973147 and its effect stays
11.152394710041898. The accepted search adds five current material marks
in the following record. It does not add any of the 24 missing trials to
the identity observer.

The code at the measured revision collects paired captures and extracts
motion, then calls `moves.see` once to apply the finished hint. It does not
call the normal tracker or `hers.saw` while collecting those captures.
`WhichIsHers.saw` can accumulate a trial only when a current measured
entity is observed under its delivered input. A region cannot supply those
missing observations after the input ends.

The coordinates have different working extents: the frozen experiment
image is 320 by 190, while this tracker uses a smaller working image. The
region surrounds the sprite near frozen coordinates `(151, 28)`;
candidate 74 is near working coordinates `(113.5, 21)`. Comparing these
raw positions without their extents would incorrectly report no candidate
inside the responsive region.

## The following physical response also fails existing identity gates

After the accepted hint and before the next screen reset, candidate 74
appears in 232 snapshots. Motion is measured in 219 of them, with at most
eight visible material marks. Its box center stays within working x
coordinates 113.0 and 113.5, and y coordinates 21.0 and 23.0.

The largest recorded effect is 11.665944277932635, at line 355, below the
required 20 working pixels per second. Its best F statistic is
60.88895838547907, at line 486, with effect 5.476809617802186. A large F
statistic alone does not qualify it.

At line 411, time 522643.0107097626, the identification epoch changes
from two to three and the candidate's control range starts again. By line
486, its recorded range is
`[112.81090206113386, 21.142118290758297]` to
`[113.78947007050643, 23.443101105209617]`.
The larger axis spans only 2.30098281445132 pixels, below the required four.

At this revision the complete identity path requires all of the following:

1. A current visible entity, resolved measured motion, and no scenery,
   gauge or exclusion classification.
2. Displacement greater than four working pixels on at least one axis,
   through `motion_is_resolved` in `controls_measured_on_one_thing.py`.
3. Two independent presses in each of at least two distinct control
   groups, with at least two measured settled samples per press.
4. Current identification epoch, finite F statistic greater than 20 and
   effect greater than 20, through `CausalIdentification.choose`.
5. A distinguishable winner, followed by a current motion check before
   assigning the body.

Candidate 74's post-recheck displacement and effect fail independently.
Earlier observations show it crossing substantial screen distance before
the search; the small later excursion therefore requires investigation of
the current physical circumstances. This audit supplies no reason to
weaken either movement guard.

## Independently read source constraints

This audit directly read the 2,341-line parsed source at
`/private/tmp/codex-steer/verified-source-9-1013.txt`. The source-audit agent
separately verified that a fresh `read_flash` of the original SWF produces
the same text. This audit independently checked the SWF's size and hash:
932113 bytes, SHA256
`ef2498224df3a9bfdea35a1441e5b42ad06538998624e5b4155313595095b93d`,
at `/Users/bryan/.aura/flash-campaign/offline-bundle/programs/9.swf`.

The parsed text retains jump offsets rather than reconstructing every
branch as structured source. The following observations use explicit
assignments, guards and calls, with that limitation:

- Lines 1116 through 1124 guard keyboard handling with
  `Player.pKeybEnabled`, poll four controls independently with `Key.isDown`,
  call acceleration for up/down and turning for left/right. The independent
  polls permit acceleration and turning within the same frame.
- Lines 1801 through 1823 compute acceleration from the current angle,
  reverse it for the down action, add it to velocity and add a stopped
  vehicle to the moving collection. These controls are relative to heading.
- Lines 1826 through 1835 guard turning with `pMoving`, alter the angle and
  reconstruct velocity from that angle and current speed. Turning a stopped
  body alone cannot establish this conditional response.
- Lines 1683 through 1706 decay velocity, test for stopping and update
  proposed position; lines 1726 through 1732 copy the position and angle
  into the visible clip. Pixels follow this physical update chain.
- Lines 1214 through 1229 show a top-boundary branch that reverses and
  reduces velocity, disables keyboard input and adjusts position. Lines
  1689 through 1695 include re-enabling input as movement stops. Other
  boundary branches and `Player.mBounce` also disable keyboard input.
- Lines 1253 through 1269 test the target car's bounds against its matching
  parking bounds, update the parking count and request transition when all
  targets qualify. Moving the controlled body is an intermediate task.

The captured sprite is near the top border and immediately above another
car. Boundary or collision responses are plausible causes of the small
excursion. The captures and existing logs do not prove which hidden branch
ran, whether input was temporarily disabled, or which collision occurred.
Those remain hypotheses rather than a reconstructed private game state.

The source supports general requirements for the next experiment: observe
every delivered input, preserve the same entity, account for unavailable
controls, and test conditional responses in an informative physical state.
Single-key tests cannot measure every composition of movement and steering.
The game name, character labels and extracted coefficients need not enter
the general controller.

## Audit scope and validation

This was a read-only code and artifact audit plus bounded numeric replay of
stored signatures. No real-time run, resident model call, browser action,
new control trial or test suite was launched for this audit. No holdout
game was opened. No production source or threshold changed.

The earlier checkpoint's 83 related passing tests and repository gates
remain recorded in the preceding motion-layer document. They were not
rerun here and do not validate the root agent's later uncommitted changes.
The root agent is separately implementing and testing continuous ordinary
observation during pixel experiments. This audit establishes the original
disconnect and the later measured refusals; it does not prove that repair,
an original-game win, native-app prompt execution or demo readiness.

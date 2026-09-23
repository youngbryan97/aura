# ISC-v5, preregistered

Drafted on 21 September 2026 and committed on 22 September with the code that
computes it, before any v5 number has been read from any run of Aura. ISC-v1 to
ISC-v4 are kept; every line of each is still computed and reported, and a
failure under any of them stays a failure.

ISC-v5 is ISC-v3 with the two irreducibility lines read by intervention instead
of by regression. Nothing else moves.

## Why a new version

The v3 line asks whether knowing the rest of the system improves the held-out
prediction of one side's next turn. It is blind to one kind of integration.
Three input-driven pipelines, ten domains of four columns, one fresh input per
turn processed by every other domain in a fixed order, differ only in how the
other domains enter: additively, as a gain on the input, or not at all. The
first two are integrated by construction. On one row per turn, as the battery
reads Aura, `phi_do` scored all three between 0.003 and 0.016 at every coupling
tried, and scored the one with no coupling no lower. The fresh input fills the
loss the cut is divided by, so a cut that removes real cross-domain dependence
costs almost nothing against it. Aura is that shape: each turn begins with what
arrives and is carried through the phases in order.

So the question is asked the other way round. From one snapshot, hold one side
of a cut at its values and let the other side run; then the reverse; compose the
two free halves; compare with the untouched run. If the halves need each other,
the composed run differs from the untouched one by more than two untouched runs
differ from each other.

## What v5 asks

`partition_irreducibility` (v5): every one of the 511 bipartitions of the ten
domains is decided.

A cut is decided when the lower confidence bound of its excess rate clears the
sham floor. The excess rate and its bound are `core.subject.v25_cut.decide_cut`,
unchanged: the Fisher-Rao divergence between the intact and cut futures, per
unit horizon, less the divergence between two untouched forks, with a paired
bootstrap over matched anchors.

- **Horizon.** One turn: 33 frames at the experiment clock, one turn per arm.
  The battery reads every other transition at the turn grain, and a cut matters
  if holding one side for one turn changes where the other ends it. Two turns,
  66 frames, are scored on the same anchors and reported beside it; that
  horizon decides nothing and draws no anchors of its own.
- **Anchors.** Sequential, at looks of 8, 16, 32, 64, 96 and 128 anchors. A cut
  is scored on its first `n` anchors at each look in turn and stops at the
  first look where its bound clears. Each look is read at alpha / 6 = 0.00833,
  so the chance that a cut which costs nothing is decided at any look stays
  under 0.05. A cut undecided at the last look is undecided, and the line
  fails.
- **Bootstrap.** 1000 draws, because the bound is the 0.00833 quantile and
  needs draws below it to be a quantile: 8.3 of the 1000 fall there.
- **Intersection-union.** The claim is the conjunction over all 511 cuts. Each
  cut is tested at its own level; no correction across cuts is needed or made,
  and one undecided cut refuses the claim.
- **Not a screen, not a shard.** A sweep of a sample of the cuts never decides
  the line, and a shard decides nothing until the shards are merged and their
  anchors pass the exchangeability check the merge already runs.

`partition_beats_nulls` (v5): the line needs `partition_irreducibility` first;
then the playback control is decided at no cut, and no null architecture that
passes every v3 line except irreducibility is decided at all 511 under the same
design.

The v3 comparison set compares magnitudes, because a regression score has an
estimator floor that surrogates reach. An interventional decision has its floor
built in, the two untouched forks, so the surrogate's role is taken by playback:
for each cut, at the look it stopped at, the cut arm is replaced by the intact
trajectory, which cannot differ from it, and read by the same rule at the same
level and draws. For the architectures the question stays the one v3 asks:
whether anything the rest of the battery cannot tell from her is judged
irreducible too. Which architectures that is comes from the campaign's own null
table (`core.subject.isc_v5.nulls_passing_the_rest`); each is swept by
`tools/validate_interventional_cut.py` at this design. On camp7's table the list
is empty.

## Where the design lives

`core/subject/isc_v5.py` holds the numbers above and the two lines. The runner
takes them from there: `tools/run_subject_core_v25.py --v5` sets the looks,
draws, level, horizons and 128 anchors. It learns the grain as well, which v5
does not read and the carrier run needs for its authority, in the coordinator
while the workers sweep. (Until 22 September 08:30 the preset skipped the grain,
which would have left J*'s carrier term unresolved; no run under it was scored.) The design enters the run's fingerprint, so a run to another design is
another campaign. `tools/score_isc_v5.py` joins a campaign report and its v5
sweep into the verdict, and refuses a sweep not run to this design.

## Changes to the runtime sweep that come with this

- Every look is read at alpha divided by the number of looks, for every sweep,
  not only v5. Before, `sweep_cuts` read each of its rounds at the full alpha.
- A sweep can name which horizons decide. Before, a cut kept drawing while any
  horizon on the ladder was undecided, and one frame decides nothing in a
  system that updates once a step, so every cut drew to the last look.
- The v25 playback null is decided by the cut rule on the same anchors. Before,
  it passed when its rate was under a quarter of the honest rate, which is a
  comparison with 1e-6 when the honest rate is zero.

## What was measured, and read, before this was written

All on synthetic systems unless said otherwise. With the tool's decision read
from the right slot (13d1b7019; before it, the tool decided on the point
estimate and called the common driver irreducible at all 511 cuts):

| system | lag 4, 96 anchors, one look |
|---|---|
| reference (recurrent) | 511 / 511 |
| independent | 0 |
| common driver | 0 |
| star | 511 |
| hub | 509 |
| one-way chain | 508 |
| ring | 494 |
| pipeline, additive | 468 |
| pipeline, modulated | 466 |
| pipeline, independent | 0 |

At lag 1 nothing is decided, the star included: in a system that updates once
per step, a side held at its anchor for one step holds the values the untouched
run used, so the cut changes nothing until the second step. A turn of Aura is
many steps of her phases reading each other, so one turn is not that case.

The sequential design above is being run on the reference, the independent
system, the common driver and the star at seeds 3, 7 and 11, at lag 4 with 128
anchors, one numeric thread each (tools/validate_interventional_cut.py, from
30cdfaf3e). A decision at 1000 draws took 9.2 seconds at 8 anchors and 19.7 at
128 on this host while it was loaded, so the two systems that should decide
nothing, which run every cut to the last look, take many hours. The decisive run was started at the commit that carries this
document, before that study finished. Its table is appended below as an
addendum, dated, before any v5 number from Aura is read; if it shows the design
cannot decide the reference, the design is changed by an amendment before any
v5 number is read, and the run started under this one is not scored.

One reading of Aura was seen before this was committed, and it is disclosed
here. On 22 September, sizing the run's cost, the log of the v25 screen of 21
September (942ee4c5a, seed 7, 16 anchors, two looks at the full alpha, 24 of
the 511 cuts) was read: no cut was decided at any horizon, and its playback
null was reported as not collapsed. At twelve anchors no control's cut was
decided, the star included; sixteen was not measured on its own. Nothing in the
design above was changed after the log was read: the looks, the level, the
draws and the horizon are the ones the power study was started with on 21
September, before it. The playback finding is the comparison-with-1e-6 defect
named above.

## What this version does not license

- It does not replace v3. Both are computed and both are reported.
- It does not apply to any run from which a v5 number has been read, and it
  does not apply to the v25 screen of 21 September, which is not a v5 run.
- The organism a v5 run measures is named in its report. The runner builds the
  offline organism with the stub language organ under the testing flag, as the
  battery's campaigns do; a verdict on it is a verdict on that organism. Her
  whole self, with the live language organ decoding greedily, is a separate
  run. With the stub a clamped arm costs 5.7 seconds (the v25 screen ran 768 of
  them in 75.7 minutes); the live cortex costs more a turn, so 511 cuts at up
  to 128 anchors is not affordable on this host with it, and no whole-self v5
  verdict is claimed until one is run.

## Addendum, 22 September 2026, 20:30: seed 19 is void and seed 23 replaces it

No v5 number from Aura had been read when this was written.

The decisive run started beside this document used seed 19. The host
restarted twice that morning and the run was relaunched each time. Its last
launch (8447bc297, 14:17) died at 18:34. The campaign, the content run and
the sweep's coordinator were each refused a write, `subject_core.action_probe
called outside governed context`, after a sixty-second disk stall. The probe
held one governed scope over a read and a write, and a governance token lives
thirty seconds, so the stall expired it. Each probe write now opens its own
scope (0f6c80784). The five sweep workers were stopped by hand. No campaign or
sweep number from seed 19 was read.

Two numbers from seed 19's content run were read by accident at 18:10, while
checking whether it had reached its displacement stage: agreement rho 0.414
(p 0.001) and design recovery rho 0.620 (p 0.001). They belong to the content
run J*'s structure term reads, not to v5, and that run is void with the rest.
The content run under the new displacement (44a64d4ba) reads a seed no number
has come from.

The decisive run is now seed 23 at 44a64d4ba, launched at 19:40: the campaign,
a v5 sweep with five workers and a coordinator, and the content run. No run of
seed 23 existed before it. The code that computes v5 did not change between
8447bc297 and 44a64d4ba: `core/subject/isc_v5.py`, the sweep in
`core/subject/v25_cut.py`, `tools/validate_interventional_cut.py`,
`tools/score_isc_v5.py` and the battery are the same files. The one change to
`tools/run_subject_core_v25.py` (e49d2beb3) is how its report words the
bridge. The organism did change, and a verdict from seed 23 is a verdict on it:

- a reminder lifts an intention for as long as it is in mind and is then taken
  back, so finished goals stop climbing to urgency 1.0 (e33940236);
- a goal takes the pressure its need has now, not the pressure it had when it
  was chosen (d41fea974);
- a campaign's conversation turns meet a person, and the partner has a field
  (5a7b0b25b);
- the content run's displacement pushes her valence before the turn as well as
  the reference her feelings are judged against, because recall runs before
  affect in a turn (43c574800, 44a64d4ba).

The power study is still the gate. It runs at 8447bc297 from its own tree; the
validator has not changed since. The three recurrent-reference jobs and the
three star jobs are running, and the independent and common-driver jobs are
queued behind them. Its table goes here, dated, before any seed-23 v5 number
is read.

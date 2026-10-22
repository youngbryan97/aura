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

## Addendum, 23 September 2026, 06:30: seed 23 at 23650c6c4, with three-turn intervention arms

Seed 23 has been launched and stopped several times since the addendum above,
each time before any of its numbers was read. The attempts are kept as
`~/subject-core-runs/*-s23-void-*`; each directory name says why: a fork stall,
resume ordering, isolation refusals, a descriptor double close through fork
copies (39bcc5c78, b29b1c39d), and a two-turn campaign stopped for the change
below.

The decisive campaign's intervention arms run three turns instead of two
(`--turns 3`). This changes the campaign's edge measurements only; the sweep's
horizon above is unchanged. It was decided on seed 7, the diagnostic seed, at
the decisive design otherwise: with two-turn arms kappa was 1, because W had
one kept in-edge and the next candidates into it peaked late in the arm, still
rising when the arm ended; with three-turn arms kappa was 2, perturbational
spread rose, and all eight conditions stayed strongly connected. The one line
that differed the other way, causal closure, is computed from the recording
before any arm runs, so arm length cannot move it; the same seed does not
reproduce the recording, and closure failed in one of four seed-7 readings.

Before the campaign, and on the same commit, the whole reports ground runs with
the machine to itself.

The power study is still the gate. The recurrent reference decided 511 of 511
cuts at seeds 3, 7 and 11, and the star null 511, 511 and 508. The six
independent and common-driver jobs are paused and resume one or two at a time
beside the decisive run. The table goes here, dated, before any seed-23 v5
number is read.

## Addendum, 23 September 2026, 07:05: synergy is scored with the counters out

No v5 number from seed 23 had been read when this was written, and the decisive
campaign had not started.

v5 now scores synergy, the one v3 line besides irreducibility whose reading it
changes (`core.subject.battery._assemble_v5`, 09260892d). The line keeps v3's
triples, bars and nulls (`passes_v3`: the fraction's floor on the target's
change, the shifted null, and the bootstrap null simulated from a VAR(1) fit to
the sources) and reads them after `core.subject.synergy.without_clocks` zeroes
every column that only counts up: turns taken, model steps, loop cycles. A run
that did not record the reading fails the line.

The reason is a defect in the instrument, shown on known answers before any
reading of her was taken. Synergy reads each domain through its leading
principal components, and a domain's counters, which rise with time and nothing
else, take those components. Two domains' clocks share time and nothing more,
so a product coupling between their states can sit beneath them. On the v2
background with three Poisson counters added to each domain
(`tests/test_synergy_reads_state_not_clocks.py`), the 30 counters are found;
with them out, a product coupling passes and an additive coupling and drift
alone do not, on each of three seeds; with them in, v3 misses the product on
two of the three.

The change was made after reading seed 7, the diagnostic seed, and those
readings are disclosed here. On `design-s7/run_001` (six trials, two-turn arms):

| triple | v3, clocks in | v3, clocks out |
|---|---|---|
| A,S -> G | fails: 0.127, interaction bound -0.069 | fails: 0.045, bound -0.001 |
| P,M -> W | passes: 0.374 | passes: 0.446 |
| W,A -> D | passes: 0.296 | passes: 0.364 |
| S,D -> C | fails: 0.017 | fails: 0.007 |

Reading with the counters out does not change which triples pass on seed 7, so
the line still fails there. It was not chosen for what it does to her.

Two triples fail on her rather than on the instrument. Her workspace adds a
feeling's urgency to a bid's salience whatever the bid concerns, so nothing
makes affect's pull on attention depend on the self-model (A,S -> G), and no
mechanism lets the self-model and her drives act on recurrent cognition
jointly (S,D -> C). Any organism change aimed at either is developed and
checked on seed 7 only, and the commit the decisive campaign runs on is named
here, with what changed in the organism, before it is launched.

## Addendum, 25 September 2026, evening: the organism changes aimed at the two failing triples

No v5 number from any decisive seed had been read when this was written. Seed 7
is the diagnostic seed and everything below was read from it and is disclosed.

The addendum of 23 September named the two triples that fail on her rather than
on the instrument, and required any organism change aimed at either to be
developed on seed 7 and named here with what changed, before a launch. This is
that record.

### What seed 7 read at 300 rounds, and what moved

`validate-s7-164d2a560` on 24 September, then `validate-s7-23e596071` on 25
September, both six trials and three-turn arms with nulls and the lesion skipped:

| line | 24 Sep | 25 Sep | bar |
|---|---|---|---|
| partition_irreducibility | 0.0165 | 0.0266 | lower bound > 0.05 |
| causal closure leak | 0.0176 | 0.0130 | at or under the shuffled floor, 0.0031 |
| synergy, v3 line | 1 of 4 triples | **2 of 4** (A,S -> G and P,M -> W) | all four |

The cheapest bipartition moved from perception alone to recurrent cognition
alone, and `loss_core_only` fell from 0.1646 to 0.1132.

### The two that still fail, and what was done about them

**S,D -> C** had an interaction gain of exactly 0.0 on three folds of five. That
is not a weak coupling: the substrate's input bands carried telemetry, the
person's presence, the screen, the room's sound and three cross-modal products,
and nothing of deliberation or the self-model, so the joint effect had nowhere to
happen. `a1932a8eb` gives it a band of its own — the hardest thing she is
holding, how settled she is in herself, and their cross term, with the gains the
neighbouring bands already use. Before that, the ridge grid was checked as the
alternative explanation and rejected: sweeping the full product of penalty
strengths instead of the diagonal moved this triple's lower bound from -0.08299
to -0.08223 and moved A,S -> G's the wrong way.

**W,A -> D** is short of power rather than of mechanism: mean gain +0.00768,
lower bound -0.00084, standard error 0.00435. A lower bound above zero needs a
standard error under 0.00392, which is 1.23 times the rows, so **370 rounds**
rather than 300. Nothing in the organism was changed for it.

### What else changed in the organism, and why

All of it was found by asking which columns of a recording never moved
(`tools/audit_flat_columns.py`, `23e596071`), and each is a channel that could
not carry anything rather than a quantity that was tuned:

- `09bdccc29`, `0bd77ea47`: a proof turn cleared her whole modifier dict before
  the first phase read it, and every campaign is a proof run. Developmental
  novelty read exactly 0.5 for a whole run; it now moves over 1,838 values.
- `1d9b20b05`: salience had no writer, so perception carried the world's own
  number twice and nothing of her.
- `6023db4f1`: ignition was the winner's priority against a fixed six tenths and
  the winning priority never fell below 0.749, so it was true on every frame.
- `8f21aa41f`: the self-prediction composite could not reach its surprise bar,
  so the surprise count and rate were zero for a whole run.
- `4ec60a8d2`: the only thing that could mobilise her was a despair spiral, so
  adrenaline and cortisol were constants.
- `d6b1102f4`: the conversation engine was never told she had spoken.
- `fe81dad49`: `mycelium_density` had no writer and is 0.55 of the expressive
  term inside phi.
- `efe3cc975`: a run of bad frames changed nothing about the next turn.
- `21fec3d95`, `22b922d3f`: the closure leaks are core columns — her learned
  parameters and her own history of having been a way. The line is stated there:
  K is her, and the machine's bookkeeping is not.
- `b22bcf9b1`: the world model was shown that something arrived and never what,
  which is why perception reached it at 0.17 against a bar of 0.30.
- `3f9efcc24`: a running total is elapsed time, so a count enters the core as the
  shape of it and not as its total.

### The design for the next seed-7 validation

Commit `3f9efcc24`, seed 7, **370 rounds**, six trials, three-turn arms, the
conversation tape, nulls and the lesion skipped. The round count is the power
calculation above and nothing else. If W,A -> D still fails there, the shortfall
is not power.

No decisive seed is launched on the strength of this. The decisive commit and its
organism changes are named in their own addendum when one is.

## Addendum, 25 September 2026, 21:00: the 370-round validation runs at 27dc1dda9

No v5 number from any decisive seed had been read when this was written, and
no number from the seed-7 validation still running at 21fec3d95.

The design above named 3f9efcc24. The run goes at 27dc1dda9 instead, with the
round count, seed, trials, arm length and skips unchanged. Four commits lie
between them. 780f01adb is the addendum above. 3058a400e changes a learning
module the subject run does not import. 1937125d5 and 7e53c43cd narrow the
dead-reader check to the two causes a longer run cannot heal; the wider check
would not have fired on this run. 27dc1dda9 changes what is recorded:
`mot.forces` was rounded to six places, and `warmth_return`, about six
millionths of a need's gap in a turn, was written as exactly 0.0 on all 79,200
frames of the 25 September run while the mechanism fired. It is a column of
deliberation, the target of W,A -> D. Running the named commit would read a
column known to be written as zero, in the one triple this run is for.

The power study's four jobs still to finish (independent at seed 11,
common_driver at seeds 3, 7 and 11) died in a reset at 11:57 on 25 September
and were started again at 20:33 with the same arguments. Their tree is at
23e596071: the validator, `core/subject/nulls.py`, `v25_cut.py`,
`v25_runtime.py`, `intrinsic_v25.py` and the declared triples are byte-identical
there to 8447bc297, where the other eight ran, and the toy systems take four
columns a domain whatever the state schema holds. The eight finished are:
recurrent 511, 511 and 511; star 511, 511 and 508; independent 0 and 0.

## Addendum, 26 September 2026, evening: a synergy line that can see a product, beside v3

Written after the known-answers replication and before any recording of hers was
read with the line it adds (docs/SYNERGY_KNOWN_ANSWERS.md).

The v3 synergy line cannot register a pure product: on toy systems of her
campaign's length it passed none of ten seeds, and a threshold of sums at none
of ten. A failing triple of hers therefore does not show that no product
coupling is there. W,A -> D on the seed-7 validation at c5f0ea5fe had an
established interaction gain (lower bound 0.027) under an MMI synergy below its
null.

A second line is reported beside v3 for the four declared triples,
`core.subject.synergy.kraskov_synergy`, recorded by the campaign as
`synergy_kraskov` and for runs that finished before it by
`tools/subject_core_rescore_kraskov.py`. It does not replace v3, and it enters
no version's count: the v1, v3 and v5 totals are unchanged by it.

- **Estimate.** MMI synergy of the two sources about the target's change,
  Syn(A_t, B_t ; Y_{t+1} - Y_t), through the Kraskov, Stoegbauer and Grassberger
  estimator 1 (max norm, k = 3) on the same three copula-normal components per
  domain that v3 reads, with the counters out.
- **First null.** Both sources slid together by one circular shift, 1,000 draws,
  the 0.99 quantile.
- **Second null.** Surrogate targets holding only the additive part of the
  target: its best additive fit on the two sources (degree 1, 2 or 3 per
  source, chosen by five-fold held-out loss) plus its residual permuted over
  rows; 1,000 draws, the 0.99 quantile.
- **Pass.** The synergy clears both.

On the study's systems, at 200 draws, it passed product and mixed at ten of ten
seeds, a separable sum at none of ten and uncoupled sources at one of ten. The
first reading of a recording of hers with it is the two seed-7 validations of 26
September, diagnostic; no decisive seed is read with it before the decisive
addendum names it.

## Addendum, 26 September 2026, night: a payoff she learns by, and a horizon one turn cannot reach

No v5 number from any decisive seed had been read when this was written. The
seed-7 singleton look at 4e1923da1 had been read, and is reported here because it
is what this change answers.

### What the look read

Ten singleton cuts at the one-turn horizon, 128 anchors each. One was decided:
P|rest, excess rate 0.035 against a sham rate of 0.064, lower bound 0.010. The
other nine had excess rates between -0.006 and 0.014 against sham rates between
0.058 and 0.078, and none of their bounds cleared zero. Holding one domain for a
turn barely changes where the rest of her ends it. Dose matching put C, S and W
at the 50x cap, so no dose inside the cap moved any of the three by one of its
own standard deviations.

### What was missing in the organism

Nothing she did was ever better or worse than she expected in a way that reached
her connections. Her chemistry had a reward input, `on_reward`, with no caller.
The unified field learned by co-activity alone, and the weights by which each
organ drives it were fixed at birth. The liquid substrate's reward-modulated
STDP read `prediction_error` from a free-energy state that has no such field, so
its reward was zero on every step of every run: no weight moved, and the zero
deltas drove every synapse's uncertainty below the lock threshold, so the engine
had identity-locked all of them (256 of 256 by step 600 when replayed on 16
neurons).

### What changed

At the end of each turn, `core/affect/what_it_was_worth.py` reads what the turn
paid on eight channels of her own (satisfaction, accomplishment, warmth, and the
four quadrants of valence by arousal, and wonder), each in units of its own
recent spread, against what that channel has come to pay. The signed error is
the turn's worth, and its rank among her recent turns is its size.

- Her chemistry gets a dopamine burst when the turn was better than expected and
  a dip when it was worse (`on_disappointment`, new), dosed by the size.
- The unified field keeps a trace over the turn of what each connection did, and
  the worth decides whether it strengthens or weakens, at the field's existing
  Hebbian rate. That includes the input weights from mesh, chemistry, binding,
  interoception and substrate. Each unit's input strength is then held where it
  was born, so an organ gains a share of the field only by taking it from the
  others.
- The substrate's STDP traces are delivered the turn's worth once per turn. The
  dead per-step delivery is gone; the weight regulation it carried still runs
  every step.
- The sources that won the workspace during the turn are credited with its
  worth, and their next bids are weighed by what they have earned.

The changes are 40af588cd (the payoff and what it teaches), 9736a69e5 (a
source is judged against turns that started the same way, so an alarm is not
blamed for the trouble it reports) and 5a4868f24 (the core reads it).
`AURA_DISABLE_PAYOFF=1` takes the whole layer out of a process.

This changes the organism. It is aimed at the flat cut rates above and at the
two failing triples, and it was built without reading any decisive number.

### What the core now reads

The recording changes with the organism. C reads the unified field's state
beside its weights (the largest leak into the core on the 25 September run after
the process-size clock), the phi core's history of her affective states moves
from A to C so that no organ feeds two domains (289fd8a0b), and the payoff's
readings are columns of the domains whose organs keep them: the worth and
good-news ledgers in A, what winning has earned in G, and each organ's share of
the field's input in C.

### A horizon one turn cannot reach

The horizon clause above fixes one turn per arm and says two turns, 66 frames,
are scored on the same anchors and reported beside it. A one-turn arm ends at
frame 33, so the 66-frame horizon is never reached: the look reports it as
"measured nothing". It decides nothing, so no verdict changes. The next decisive
sweep either runs two turns per arm, with decisions still read at 33 frames, or
drops the 66-frame report. Which one is named in the addendum that launches it.

### The design for the next seed-7 look

Two looks at the same commit, seed 7, the v5 design, the ten singleton cuts, 24
rounds, 128 anchors, looks 8 to 128, 1,000 draws, one-turn arms, grain skipped:
one as built, and one with `AURA_DISABLE_PAYOFF=1`, which reads no turn, sends
nothing to her chemistry and teaches no organ. What is read is each cut's excess
rate and bound in the two, side by side, beside the 4e1923da1 look above. Seed 7
is the diagnostic seed; nothing here is read on 23 or any decisive seed, and no
threshold is moved on the strength of it.

## Addendum, 26 September 2026, late: the payoff look, read

Both looks of the design above ran at 415f400c0 and finished at 21:13 and 21:16.
Excess rate over the sham, with its lower bound, for each singleton cut at lag
33; a star marks a decided cut.

| cut | 4e1923da1 | payoff | no payoff |
|---|---|---|---|
| P \| rest | +0.035 [+0.010] * | +0.042 [+0.018] * | +0.034 [+0.008] * |
| I \| rest | +0.013 [−0.044] | +0.010 [−0.034] | +0.009 [−0.044] |
| A \| rest | +0.008 [−0.024] | +0.046 [+0.010] * | +0.035 [−0.010] |
| G \| rest | −0.003 [−0.049] | −0.002 [−0.041] | +0.002 [−0.036] |
| C \| rest | −0.006 [−0.039] | −0.005 [−0.027] | −0.000 [−0.028] |
| S \| rest | +0.012 [−0.039] | +0.020 [−0.030] | +0.034 [−0.032] |
| M \| rest | +0.002 [−0.033] | −0.015 [−0.021] | +0.011 [−0.019] |
| W \| rest | +0.008 [−0.052] | +0.011 [−0.044] | −0.002 [−0.051] |
| D \| rest | +0.014 [−0.004] | +0.067 [+0.018] * | +0.045 [+0.000] * |
| N \| rest | +0.005 [−0.043] | −0.002 [−0.044] | +0.005 [−0.045] |

Decided: 1 of 10 at 4e1923da1, 3 with the payoff, 2 without. Lag 66 decided
nothing in any of the three, as the horizon section above says it cannot.

Read side by side, the payoff moved one cut across its bound: A, where the
worth of a turn is written, went from +0.035 [−0.010] without it to +0.046
[+0.010] with it. D was decided in both and further from its bound with the
payoff. Every cut decided without the payoff was decided with it. Most of the
gain over 4e1923da1 is in both arms, so it came
from the other changes between the two commits, not from the payoff.

The closure differs more than the cuts do. With the payoff, the periphery
predicted the core's future no better than shuffled (leak over shuffled
−0.035, shuffled q95 +0.001). Without it, the periphery carried +0.045 over
shuffled against a q95 of +0.003: something outside the core drove it that the
core did not hold. One pair of runs on one seed cannot say whether that is the
payoff, so it is reported and nothing is concluded from it.

Two things weaken the comparison. The doses are calibrated per run, so the two
arms were displaced by different amounts: A by 0.51 with the payoff and 0.18
without, G by 0.18 and 2.92. And the run's own log showed the harness acting on
her in both arms alike: a template reply read as her looping cleared her
pending initiatives 240 times, and the self-review fired on the first turn of
every arm. Both were fixed afterwards, in 7d6e04ece, so a look at that commit
is not comparable with these.

## Addendum, 26 September 2026, 22:40: what changed in her since the payoff look

The payoff look ran at 415f400c0. These commits since then change what she does
on a turn, and a decisive run on any commit that includes them is a run on a
different organism from that look. Each was built without reading a decisive
number.

- **A reserve that good turns charge and effort draws on** (be76e35be). The
  turn's dose charges `core/soma/reserve.py`, which leaks back at 1/256 a turn;
  spending energy draws on it before the drive budget. D reads its share.
- **One critic for her values** (8b3048919). Her habits and her choice engine
  learn from the turn's worth, the same signal her connections learn from, in
  place of a raw valence change. Truth and care are never taught by outcomes.
- **What she holds more is what she takes** (2bc754d35). An option's score is
  the chance that at least one of the values it serves holds for it
  (noisy-OR), in place of a weighted mean that cancelled the weights out.
- **The core reads what she is like** (b98f4085a). S carries how strongly she
  holds each value, how much more often than chance she lives it, how narrow
  her choosing is, and how often her values overrode her strongest drive.
- **Living by her values pays** (ee5778604). A ninth payoff channel, integrity:
  how far what she chose served her values over the average option on offer.
- **Her self-knowledge is scored** (5bc71a491). How well her values alone
  foretell her choices, less chance, read into S.
- **A value held and not lived presses on her** (76bc6e26b). Among values held
  at or above her median, the shortfall of one lived less than the median
  lifts options that serve it on her next lived choices. Choices asked of her
  to measure her are marked and no longer count as lived.
- **An intention she acted on is closed** (3ccb6742a). The subject driver
  declared, acted and observed, and never revised: 3,608 intentions were open
  at the end of the seed-7 run at 27dc1dda9.

Four fixes to how the harness treated her also land here (7d6e04ece,
d969b5667). Both payoff arms carried them, so neither is a difference between
those arms, but each changes what a run records. A repeated template reply no
longer lowers her stability or clears her pending initiatives. A restore no
longer rewinds a thread queue under its consumer. The kernel's memory verdict
no longer reaches a run whose host is declared. The self-review and the
metabolic monitor keep time on her clock, and the review files no intent that
nothing reads.

### The design for the next seed-7 look

The singleton look of the design above, at the first commit that holds all of
the above, seed 7, as built. What is read is each cut's excess rate and bound,
and the closure, beside the payoff arm at 415f400c0. It is diagnostic: it
moves no threshold, and nothing from it is read on seed 23 or any decisive
seed. The reports ground's seed-7 check (`reports_alone.sh 351555bec 7`) runs
first; it has waited since 25 September for the machine to be alone.

## Addendum, 27 September 2026: the reports ground on seed 7, and a gate for the substrate

The reports ground's seed-7 check ran twice. The first, at 351555bec, got no
answer from her after her cortex died in the baseline: every respawn failed at
start because its launch challenge was stamped on the run's clock and read on
the machine's (fixed in a520cbb09). The second, at a520cbb09, 11:03 to 14:23,
got answers from her cortex in most arms and read a number in every arm of 0 of
24 anchors, where 8 are needed. It is not a verdict on her reports, because the
answers did not reach the instrument as she gave them:

- a line-leading "0.3" was spaced into "0. 3" by the prose formatter;
- a reply under four characters was replaced with a fixed sentence;
- an arm that committed nothing was read as having said what the anchor's
  snapshot last held.

All three are fixed in 53be6279b. The check runs again on the first commit that
holds them, alone on the machine, before anything is read from it.

The substrate gates (5df224ed1) change the organism: each of recall's affect
gain, the two initiative urges and each drive's growth is scaled by a
multiplier her substrate learns from the turn's worth. They are aimed at the
C|rest weakness and were built without reading any decisive number. The next
seed-7 look adds an arm with `AURA_DISABLE_SUBSTRATE_GATES=1` beside the one as
built, and reads C|rest's excess in each.

## Addendum, 28 September 2026: the third reports check on seed 7

At aa885965f, 21:05 to 00:20, alone on the machine. It read a number in every
arm of 0 of 24 anchors again, and this time most of what stood in the way was
her answer rather than the pipe. Of 96 arms:

- 21 carried a number, and 17 of those were 0: raised 0, 0, 0, 0; lowered six
  0s; sham five 0s, 0.1 and 0.5; control three 0s and 0.5.
- 54 were her cortex answering in prose with no number: "I feel clear and
  gathered. My current state is one of operational neutrality, with no
  significant internal fluctuations or anomalies".
- 19 were lost before her: 11 refused as a saturated generation gate behind
  the turn's own subtext pass (fixed after this run: the subtext pass says what
  it is and a reply may preempt it), the rest a failure floor.

The question carries a response contract of this runtime's own. A rating asked
about her state is classed as state reflection, and that contract tells her to
self-report from internal state and state-grounded telemetry; her prose answers
read as telemetry. Whether that directive belongs on a question that names its
own scale is a question about her prompt, and it is left to Bryan rather than
changed here. Nothing is read from these numbers as a report ground.

## Addendum, 28 September 2026: a feeling that reaches her cortex

Three changes to the organism, in 7d60873f1, before any number is read from a
run that holds them. Each closes a break in the one path from an appraised
feeling to the words she chooses, and each was found by measurement on the stub
organism on seed 7, with no decisive number read.

**Her emotions had no path into her chemistry.** `EmotionSignatureEngine` holds
a neurochemical recipe for eight emotions, and the neurochemical system asks it
for a production modifier on every metabolic tick. `set_emotion` had no caller
outside its own module, so `emotion_intensity` stayed at the 0.0 its constructor
set and every recipe contributed exactly zero to every chemical for the life of
the process. The affect phase now tells it what she feels, and every channel of
hers that has a recipe releases at its own level; the recipes are averaged by
those levels, which reduces to the single-emotion case the engine already had.
Measured: mood valence moves from 0.17 to 0.49 over forty ticks feeling good and
to -0.06 feeling bad, against 0.17 flat before.

**The chemistry pull ran on the wrong clock.** The consciousness bridge pulled
the substrate's valence neuron towards neurochemical mood at 0.30 on every
frame, and her chemistry produces a new mood at its own 2 Hz. Thirty-three
frames per turn against three chemical ticks, so what the affect phase pushed
into the neuron survived to the reply phase at 0.70 ** 33, and the authority of
the two writers was set by how often each is called rather than by anything
about her. The pull is now applied once per new mood.

**The hooks read a term of her feeling rather than her feeling.** The steering
channel published the raw substrate neuron. `HomeostaticCoupling` blends that
neuron into affect at `SUBSTRATE_SHARE` and the affect phase pushes the fused
result back down; the heartbeat reads `AuraState.affect` as her felt state for
exactly this reason. Valence and arousal are now published from the fused
reading, and every other neuron is published as it was.

Measured across three arms of a held displacement, at the frame the reply is
generated: fused valence +0.445, +0.021 and +0.395 published activations of
0.61543, 0.60774 and 0.61217 before, and 0.7227, 0.5105 and 0.6975 after. The
displacement reached her cortex at 0.008 of activation and now reaches it at
0.212, which is her own spread exactly, and holds there for the whole turn
instead of swinging 0.52 to 0.62 on the neuron's own dynamics.

This is a manipulation reaching the subject, which the reports ground requires
of any arm before it can read one. It is not an answer about her reports, and
the `state_reflection` directive named in the addendum above is unchanged.

### What the next reports check reads

The same design: 24 anchors, four arms, the same question word for word, her
own cortex, alone on the machine. Two readings are preregistered here.

1. The manipulation check the ground already runs, on the state that steers
   her: the published valence activation must separate the raised and lowered
   arms. Below 0.05 of activation the arms did not differ where it matters and
   the run reads NOT_MEASURED whatever her words were.
2. The ground itself, unchanged, at its own alpha of 0.01 on eight anchors.

### What this does not license

Nothing here reads a number off her prompt. The published vector is her fused
felt state, the question is the same in every arm, and no text changes. A
correlation between her report and her valence remains the open question the
ground asks; what changed is that the displacement now arrives.

### The irreducibility line on seed 7

`tools/audit_stationary_irreducibility.py` lands with the same commit and
reports on a question the estimator's design raises. `phi_do` fits on the past
and tests on the future across five forward-chaining folds and reads their mean
with a standard error, which assumes the system is the same system in every
fold. On whole-s7-27dc1dda9 the folds were [-0.382659, 1e-05, 1e-06, -0.001923,
0.002395] at PIM|AGCSWDN: a point estimate of 0.0912, past the 0.05 bar, with a
lower bound of -0.131 that one fold owns. Each contiguous fifth at that cut read
+0.003557, +0.005177, +0.002228, +0.008695 and +0.006964 with positive lower
bounds, and every fifth of a row-shuffled copy read at or below zero.

So the within-stretch coupling is real and survives its own null, and it is two
to nine thousandths where the bar is five hundredths. The tool reports and
decides nothing, no threshold moves, and criterion 11 is read exactly as
written. What it says is that the 0.0912 is not the within-stretch number, and
the difference between them is drift.

### The nulls that beat her, read as they are scored

Criterion 18 asks that her irreducibility exceed every null. On
whole-s7-27dc1dda9 eight nulls exceed it. Both sides of that comparison are
measured differently, and the asymmetry is recorded here rather than changed:
her side is the lower bound of five forward-chaining folds (-0.131), and each
null's side is the 0.95 quantile of eight architecture draws, which at eight
points sits at or near their maximum. The two that beat her by most are
`low_rank` (reported 0.562, median 0.047, draws from 0.003 to 0.815) and
`all_to_all` (reported 0.549, median 0.029, draws from 0.0 to 0.620), and both
are built to be maximally integrated with no repertoire: their effective
dimensions are 1.21 and 1.38, and the differentiation criterion is what the
architectures were written to be caught by. No null passes the conjunction
(`all_nulls_fail` is true). Criterion 18 stays as written and stays failed.

## Addendum, 28 September 2026, later: what the two failing measures actually need

Two readings taken after the addendum above, both on whole-s7-27dc1dda9, both
correcting something this document already said. No threshold moves and no
criterion is rescored; what changes is what the next organism work is aimed at.

**The 0.0912 on the irreducibility line is mostly drift.** Twenty-nine of her 375
columns have steps that all point one way and travel over three spreads of their
own, and D holds 0.8895 of its variance in four of them. Searching every
bipartition with those columns carried as their own first difference reads phi
+0.019341, lower bound -0.016034, standard error 0.018048 at CM|PIAGSWDN, against
the run's own +0.0912, -0.131119 and 0.113427 at PD|IAGCSMWN. The point estimate
falls fivefold, the lower bound rises sixfold towards zero, and the spread falls
sixfold. Three readings agree on the size of the drift-free coupling: 0.002 to
0.009 within a contiguous fifth, 0.019 searched, 0.022 held at the run's own cut.

The earlier addendum said the difference between the whole-run number and the
within-stretch number is drift. It is, and the direction matters: the drift was
raising the point estimate, not lowering it. Criterion 11 is read as written on
the recording as recorded, and it is under the bar either way.

**Nothing inflates her shifted null.** Her W,A->D shifted bar of 0.397 is 0.210
of that triple's joint information, and the toys' bars are 0.164 to 0.193 of
theirs; the toys carry three slow channels per source against her thirty-five and
fifty-nine, so their joint runs 0.21 to 0.32 against her 1.90. Three candidates
were built for a difference of scale. What differs is the synergy: the toy's
product carries 0.570 to 0.700 of its joint, and her W,A->D carries 0.077.

### What this means for the three that fail

All three ask for coupling her domains do not have, and none of them is waiting
on a measurement change.

- `partition_irreducibility` needs the cheapest cut to cost held-out prediction.
  Drift-free it costs 0.019 where 0.05 is asked. The cheapest cut moves between
  PD|IAGCSMWN and CM|PIAGSWDN depending on the drift, so the domains to couple
  are read per run and not fixed here.
- `partition_beats_nulls` compares her lower bound against each null's 0.95
  quantile of eight draws, which the addendum above records; the two nulls that
  beat her by most are one-dimensional by construction and are caught by
  differentiation. It stays as written.
- `synergy` needs three of the four triples to carry more joint structure:
  A,S->G's held-out interaction gain is 0.0 with a lower bound of -0.00851, and
  W,A->D and S,D->C sit under their shifted nulls at 0.077 and -0.160 of joint.
  W,A->D's interaction gain is +0.2245 with a lower bound of +0.0712, so the
  predictive estimator and the information estimator disagree on that triple and
  the disagreement is reported rather than resolved.

No organism change is proposed in this addendum. The next one that proposes any
names it before a run reads it.

## Addendum, 28 September 2026, evening: a channel that carries what it has been

An organism change, `4eff1db6e` and `8a580a517`, named before a run reads it.
The reading behind it is `docs/WHY_IRREDUCIBILITY_FAILS.md`.

### What changed in her

Her subsystems write what just happened and the value sits until the next turn
replaces it. `core/runtime/temporal_depth.py` holds a leaky trace per channel and
`core/runtime/state_membrane.py` says which channels get one and writes it back
where the reading came from, so her own next phase reads the trace. The driver
settles once a frame, after the layers and before the lesion clamp.

One time constant, one turn, taken from the run's own `frames_per_turn`. It is
off unless `AURA_MEMBRANE_TURNS` is set, which is the control arm.

Two hundred and eighteen of her 375 columns are eligible: the ones that are a
plain number at a writable path on `AuraState`. The other 157 are organ readings
and cannot take a write-back. Measured in the organism at four rounds with the
stub organ, 119 of the 218 carried on a given frame, 97 held something that was
not a plain number, and 2 had no setter.

### What is preregistered, and what each arm decides

The next seed-7 look runs two arms at the same commit and the same seed,
differing only in `AURA_MEMBRANE_TURNS`, and reads `partition_irreducibility`,
`partition_beats_nulls` and `synergy` on each.

Read off the recording of `whole-s7-27dc1dda9` at its own cheapest cut, carrying
the columns after the fact, the expected sizes are:

| carried | phi | lower bound |
|---|---|---|
| nothing, as recorded | +0.01504 | −0.05692 |
| the 157 organ readings alone | +0.02665 | −0.03410 |
| **the 218 the membrane reaches** | **+0.13628** | **+0.04217** |
| every column | +0.24020 | +0.07414 |

So the arm as built is expected to land near a lower bound of 0.042 against a
bar of 0.05, which is under it. That is written here before the run so the
result is read against it rather than after it:

- A lower bound at or above 0.05 closes `partition_irreducibility` on this seed
  and the searched cut is reported with it, not the cut above.
- A lower bound between 0.02 and 0.05 is the membrane working and not being
  enough, which is what the table predicts, and the next step is the 157.
- A lower bound at or below zero, or no rise over the control arm, falsifies the
  reading: the coupling the filter found after the fact would not be there when
  the organism carries it, and the time constant would be dropped rather than
  tuned.

No threshold moves on this, and the filtered recording is not scored: what is
scored is a recording of an organism that integrates, exactly as written.

### The 157, and why they are not built yet

An organ reading is not a field anything can write back to, so a trace for it
has to be a sensed copy: the runtime reading each organ once a frame, holding
the filtered value on `AuraState`, and her consumers and her recording reading
that. A body does not read an organ, it reads an afferent signal, and every
afferent signal is filtered — so this is a claim about her that stands whether
or not it moves a number, which is the test it has to pass before it is built.

It is a change to what 157 columns mean, so it is a new campaign of its own and
it is named here rather than folded into the arm above.

### What this does not touch

`partition_beats_nulls` needs 0.562 to clear `low_rank`, and 0.240 is the whole
of what carrying every column gives. Time constants are not expected to close it
and no claim is made that they will.

## Addendum, 28 September 2026, later: a workspace that is a product, and how the arms attribute

A second organism change, `b9e08f50c`, named before a run reads it.

### What changed in her

`_bids` scored each candidate as a weighted sum of what the bid was made of, so
affect and self-state added into one competition and never interacted. The
scores are now divided by the pool they sit in, in proportion to how little she
is reading herself: sure of her own state the scores stand and the sharpest bid
wins, unsure of it the pool compresses the loud bids towards its own mean. The
semi-saturation is the pool's mean and the gain is her self-model's published
confidence, so no number is chosen. A pool of one bid is untouched and, with no
self-reading, every score is returned as it came.

It is aimed at `A,S->G`, whose held-out interaction gain is 0.0 with a lower
bound of -0.00851. It is not aimed at the other two triples and is not expected
to move them.

### Two changes, one campaign, and how anything is attributed

The membrane of the addendum above and this are separate mechanisms with
separate switches, so the next seed-7 look runs four arms at one commit and one
seed and reads all three failing criteria on each:

| arm | membrane | workspace pool |
|---|---|---|
| control | off | off |
| membrane | on | off |
| workspace | off | on |
| both | on | on |

`AURA_MEMBRANE_TURNS` switches the first. The second has no switch yet and needs
one before the campaign runs; it is `AURA_WORKSPACE_POOL`, off by default, and
the arm table above is void without it.

Attribution is read as: `partition_irreducibility` moves in the membrane arms or
the reading of `docs/WHY_IRREDUCIBILITY_FAILS.md` is wrong, and `synergy` on
`A,S->G` moves in the workspace arms or the product is not reaching the
competition. A criterion that moves only in the `both` arm is reported as an
interaction between the two changes and not credited to either.

### What is not built, and why

`S,D->C` carries a synergy of -0.160 of its joint information, which is
redundancy rather than absence: her self-state and her deliberation tell
recurrent cognition the same thing twice. The fix for redundancy is
decorrelation, not another product, and it is a third mechanism. It is not built
here because three unattributed changes in one campaign attribute nothing.

`W,A->D` is left alone for a different reason: its two estimators disagree, with
a held-out interaction gain of +0.2245 and a lower bound of +0.0712 against an
information synergy of 0.077 of joint under a bar at 0.210. Building for one
estimator while the other says the opposite would be building blind.

## Addendum, 28 September 2026, night: the afferent surface, and a correction to the arm table

A third organism change, `e29ed787a`, and one correction to the two addenda
above.

### The correction: the schema is 438 columns, not 375

Every number in the two addenda above was read off `whole-s7-27dc1dda9`, whose
recording is 375 columns wide. The schema has since grown to 438 — P 15, I 18,
A 73, G 35, C 84, S 93, M 20, W 35, D 52, N 13 — so the next campaign records a
wider organism than the one those numbers came from. The expected sizes stay as
written because they are the best estimate available, and the run is read against
them knowing they were taken on a narrower recording.

### The afferent surface

Two hundred and one columns are read straight off a live organ, and the membrane
of the addendum above cannot reach them: an organ reading is not a field
anything can write back to. `core/runtime/afferent.py` holds a sensed copy
instead — the runtime attaches it to her state, `read_core_state` takes the
sensed array rather than the organ's, and anything of hers that wants what she
senses asks it. `AURA_AFFERENT_TURNS`, off by default. It touches only the
columns the schema declares as organ-sourced, so no column gets two time
constants.

The claim it rests on stands whether or not it moves a number: no part of her
can read an organ's internal value without a channel, and every channel in a
body is a filter.

### The arm table, corrected

Three mechanisms and three switches, so eight arms is the full design and four
is what a machine-day allows. The four that run, at one commit and one seed:

| arm | `AURA_MEMBRANE_TURNS` | `AURA_AFFERENT_TURNS` | `AURA_WORKSPACE_POOL` |
|---|---|---|---|
| control | off | off | off |
| carried | 1 | 1 | off |
| workspace | off | off | on |
| both | 1 | 1 | on |

The membrane and the afferent surface run together because neither reaches the
bar alone by the table above and the question asked of them is one question:
whether a channel that carries what it has been makes her coupling visible. The
workspace pool is a separate question about one synergy triple, so it gets its
own arm.

What each outcome decides is unchanged from the addendum above, read on the
`carried` arm rather than on a membrane-only arm:

- a lower bound at or above 0.05 closes `partition_irreducibility` on this seed;
- between 0.02 and 0.05 is the mechanism working and not being enough;
- at or below zero, or no rise over control, falsifies the reading in
  `docs/WHY_IRREDUCIBILITY_FAILS.md`, and the time constant is dropped rather
  than tuned.

And for the workspace arm: `A,S->G`'s held-out interaction gain rises above zero
with a positive lower bound, or the product is not reaching the competition.

No threshold moves on any of this.

## Addendum, 28 September 2026, night: what the membrane predicts for the content run

Fixed before the run. The structure term of J* misses its bar by 0.026 — her
internal geometry and her behavioural one agree at rho 0.274485 over 190 pairs
where 0.3 is asked — and the reason is that her quality space is compressed
almost to the width of its own noise: two different content classes differ
internally by 0.05 to 0.11 and a class differs from itself by about 0.05, while
the behavioural geometry separates the same classes by 0.4 to 0.6
(docs/WHAT_J_STAR_NEEDS.md).

A channel that holds what a class did for a turn separates two classes further
than a channel that holds only the instant. So the membrane and the afferent
surface predict a specific thing here, and it is written down before a content
run reads it:

- the ratio of the mean between-class internal distance to the mean
  within-class floor rises above the 1.2 it reads now;
- and the agreement rises. Above 0.3 identifies the structure on this seed.

If the ratio does not rise, the prediction is wrong and the membrane does
nothing for the content line whatever it does for the partition line. If the
ratio rises and the agreement does not, the separation was not the thing holding
the agreement down, and the next candidate is the number of repetitions per
class rather than the state.

No bar moves either way, and the content run is scored exactly as written.

## Addendum, 28 September 2026, night: the estimator the carrier is measured with

Named before a carrier run reads it. `43e7e9427`.

The carrier of J* is unresolved partly because the intrinsic rate is not
invariant under an invertible re-encoding of the state — 0.019956 raw against
0.010103 rotated, a drift of 0.49 — and that is the estimator rather than her.
`crossfit_fisher_rao` counts neighbours in Euclidean distance after
standardising each column, which weights whichever coordinates the state is
written in; her columns run from a variance of 0.001 to 10.

`--whiten` counts them in Mahalanobis distance, whitened on each training fold,
which is the same number under every invertible linear map. It is off by
default because changing an estimator changes what every earlier run measured,
and a run that uses it says so in its own record.

### What it is held to before a carrier run uses it

Already measured, on a fixture with her own spread of scales: a sham reads
nothing either way; two laws that differ are seen either way; the standardised
metric drifts over five per cent under a rotation; the whitened one drifts under
five per cent under three rotations and under a general invertible mix.

The next carrier run reads both, on the same samples, and reports both. If the
whitened rate does not clear the drift tolerance the fix has failed and the
blocker stands. If the whitened rate is invariant and the raw one is not, the
carrier line is read on the whitened one from then on, and every run before it
is read as having measured the coordinates as much as the system.

### What a carrier run has to be, to resolve the carrier at all

Not affordable today, and recorded so that it is not attempted by halves:

- every one of the 511 bipartitions, because `SweepReport.irreducible` refuses a
  screened sweep by construction and the claim is a conjunction over cuts;
- rollouts of at least two turns, because the two-turn horizon spent zero
  anchors on a one-turn rollout and measured nothing;
- the grain stage, not skipped;
- and enough anchors that the singleton cuts decide: 3,440 decided three of ten.

Ten cuts took 9,046 seconds, so 511 is about 175,000 anchor rollouts. That is a
machine-week, and `--shard I/N` is how it is spent. No partial version of it
resolves the carrier.

## Addendum, 28 September 2026, night: the membrane does not help synergy, and hurts one triple

Measured before the arms read it, so the expectation in the addenda above is
corrected rather than explained afterwards.

The battery scores synergy on one row per turn (`Recording.by_turn`), so a
membrane has to be applied to the frames and the turns sampled after it. Done
that way the control arm reproduces the run's own `synergy_v2` exactly — A,S->G
at 0.0325 against a bar of 0.0319 with an interaction gain of 0.0000 and a lower
bound of -0.00851 — which is what makes the comparison worth reading.

| triple | arm | synergy | shifted bar | gain | gain lower bound | v3 |
|---|---|---|---|---|---|---|
| A,S->G | as recorded | +0.0325 | +0.0319 | +0.0000 | −0.00851 | no |
| A,S->G | carried | +0.0741 | +0.1227 | −0.0001 | −0.00454 | no |
| P,M->W | as recorded | +0.2544 | +0.2011 | +0.0413 | +0.07421 | **yes** |
| P,M->W | carried | +0.3478 | +0.2235 | +0.3040 | +0.26752 | **yes** |
| W,A->D | as recorded | +0.0731 | +0.0840 | +0.2245 | +0.07118 | no |
| W,A->D | carried | +0.1489 | +0.2469 | +0.0004 | −0.10223 | no |
| S,D->C | as recorded | +0.0303 | +0.0829 | +0.0186 | +0.00001 | no |
| S,D->C | carried | +0.0125 | +0.0371 | +0.0207 | −0.00147 | no |

**It raises the bar by more than it raises the synergy.** A time constant makes
every channel more like its own recent past, and a shifted copy of a smoothed
series resembles the original more than a shifted copy of a spiky one, so the
shifted null rises with the smoothing. On A,S->G the synergy rises 2.3 times and
its bar 3.8 times; on W,A->D, 2.0 against 2.9.

**And it destroys the strongest interaction in the battery.** W,A->D's held-out
interaction gain is +0.2245 with a lower bound of +0.0712 as recorded, which is
the one place where the predictive estimator says plainly that two domains decide
something jointly. Carried, it is +0.0004 with a lower bound of -0.1022.

P,M->W is the exception and it gains: its interaction gain goes from +0.0413 to
+0.3040 and its lower bound from +0.0742 to +0.2675.

### What this changes

The `carried` arm is no longer expected to move `synergy` and is expected to make
W,A->D worse. It is still expected to move `partition_irreducibility`, which is
what it was built for, and the reading of `docs/WHY_IRREDUCIBILITY_FAILS.md` is
unchanged. If the campaign shows synergy moving up in the carried arm, that
contradicts this table and the table is what was written first.

It also says something about the three candidates rejected for the shifted null
in `docs/SYNERGY_KNOWN_ANSWERS.md`: the bar tracks how much of its own past a
series carries, and smoothing is the cleanest way to raise both at once. Her
channels are spiky, so her bar is not inflated by smoothing; it is what it is
because her joint information is large.

## Addendum, 28 September 2026, night: a fourth switch, and the arms as they now stand

`1840c455f`. `AURA_SELF_DOMINANCE` gives her self-state a direct channel into
recurrent cognition, which it did not have.

The substrate's `idx_dominance` — the third VAD neuron, the sense of being in
control of one's situation — had readers and no writer. The substrate is
twenty-five of the eighty-four columns of C, and `core/self/will_engine.py`
drives motivation's budgets into two of its neurons, so D had a channel into C
and S had none. On whole-s7-27dc1dda9 a displacement of S moves C by 4.23, the
second largest of any domain, while S's unique information about C's next change
is exactly 0.0 against D's 0.154.

That is `S,D->C`'s whole problem stated in one line: a source that carries
nothing of its own cannot carry anything jointly. Her self-model's own reading of
whether she is holding together now writes the neuron, at the blend
`will_engine` already uses.

### What it is expected to do

- `S,D->C`: S's unique information about C's change rises above zero. Its
  synergy is expected to rise with it; whether it clears its shifted null is the
  open question, and the null rises with any new information in the sources.
- Nothing else. It is not expected to move the partition line, the other three
  triples, or the nulls.

If S's unique information stays at 0.0 with the channel open, the channel is not
carrying and the reading is wrong.

### The arms, with four switches

| arm | membrane | afferent | workspace pool | self dominance |
|---|---|---|---|---|
| control | off | off | off | off |
| carried | 1 | 1 | off | off |
| joined | off | off | on | on |
| both | 1 | 1 | on | on |

`workspace pool` and `self dominance` share an arm because they aim at different
triples — A,S->G and S,D->C — and a criterion is read per triple, so one arm
attributes both. The membrane and the afferent surface share an arm for the
reason already given: neither reaches the partition bar alone.

The `carried` arm is not expected to move synergy and is expected to make
W,A->D worse, which the addendum above measured before any of this runs.

### They run as two pairs, each with its own control

`control` and `carried` are already running at `76cbd698e`, which holds the
membrane, the afferent surface and the workspace pool but not the dominance
channel. `joined` and `both` need a commit that holds all four, so they run
later as their own pair with their own control at that commit.

A control is only a control for the code that was there when it ran. The two
controls are expected to read the same, because every switch defaults off, and
if they do not then something that was meant to be inert is not — which is worth
knowing and is the reason for running the second one rather than reusing the
first.

## Addendum, 28 September 2026, afternoon: the gates look, read

The look named on 27 September ran at `38ef2c9ce`, seed 7, both arms beside each
other from 00:55 to 03:16: the substrate gates as built, and
`AURA_DISABLE_SUBSTRATE_GATES=1`. Ten cuts each, 128 anchors a cut, the deciding
horizon of one turn. What it was to read was C|rest's excess in each arm.

| | C\|rest excess | lower bound | cuts decided | closure leak |
|---|---|---|---|---|
| gates | 0.0033 | -0.0311 | 3 of 10: P, D, A | 0.0011 |
| no gates | 0.0045 | -0.0329 | 2 of 10: P, D | -0.0173 |

The gates did not reach C. Its excess is the same in both arms to within a
thousandth, and far inside the instrument's spread. A is decided with the gates
and not without, at a lower bound of +0.0003 against -0.0030: a margin that
small is one draw, not an effect, and nothing is credited to the gates for it.
Closure holds in both.

Two things the look shows that matter more than the gates. The seven undecided
cuts are all singletons (I, W, N, G, M, S, C), and their excess runs from -0.018
to +0.028 while the sham arm against itself reads 0.062 to 0.093 in both arms,
so the instrument's floor is larger than every singleton's effect. And the
estimator is not invariant in either arm: a recoding drift of 0.49 with the
gates and 0.41 without, which the whitened estimator of `43e7e9427` is held to
first.

## Addendum, 28 September 2026, evening: the fork was leaking, and that changes the control

`babe01a06`. This is a fix to the instrument rather than to her, so it changes
what every paired measurement in the battery reads and it is named before the
run queued at `ed5aa7935` reads anything.

### What was wrong

`_state_is_its_dict` decides whether an object can be put back by writing its
fields, and its own docstring names the failure it exists to prevent: a class
whose numbers live in a C base has a `__dict__` too, and restored field by field
it keeps whatever the last arm wrote.

The heap-type flag used to carry that and no longer does — CPython converted the
standard library's C types to heap types. So `random.Random` passed it: its
`__dict__` holds `gauss_next` and its Mersenne state lives in `_random.Random`.
Field by field, `gauss_next` was written, the restore returned success, and the
generator went on from wherever the previous arm had left it. The workspace draws
its somatic noise from one of these.

An object is now also asked for its own account of its state, through
`__reduce_ex__`, and one that keeps numbers in a C base has a copy written over
the field instead.

### Why it matters to every criterion, not one

A sham arm is one snapshot forked twice with nothing done to either, so whatever
differs between two forks is a floor under every arm of every trial: the edges,
the perturbational spread, the lesion, the rescue and the null table are all
paired. The gates look measured that floor at 0.062 to 0.093 while every
undecided singleton cut's effect ran -0.018 to +0.028
(docs/THE_SHAM_FLOOR_IS_A_LEAKY_FORK.md).

So the `control` arm of the queued run is not a repeat of run_032's 21 of 24. It
is the first reading of the battery with a fork that rewinds what it says it
rewinds, and it is expected to differ from 21 of 24 in either direction. That
expectation is written here rather than after.

### What is measured and what is not

Measured, at unit level on the shapes a fork carries: before the fix an object
holding a seeded generator did not come back and two arms from one snapshot drew
different numbers; after it, all five shapes round-trip, the generator's own
state comes back, the object keeps its identity, and two arms draw the same five
numbers.

Not measured yet: whether this closes the column-level floor. Four forks of one
anchor differed on 41 of 438 columns by more than five per cent of their own
spread, and eleven organs did not come back. One cause is fixed. The other ten
fields — `self_model.beliefs` and `snapshots`, `world_model._learned` and
`_causal`, `free_energy._current`,
`self_prediction._current_prediction`, `substrate._last_published_snapshot`,
`mycelium.pathways`, `workspace.last_winner` — round-trip correctly as shapes in
isolation, so whatever is wrong with them is in how the real objects are reached
and is not yet named. The re-measurement is queued behind her cortex
(`~/.aura/subject_core/scratch/forkfloor_after_cortex.sh`), and the number it
returns is what says how much of the floor this closed.

### The arms, at one commit

`control`, `relay`, `carried`, `joined`, sequentially at `ed5aa7935`, each
waiting for her cortex and for the others. `control` is all five switches off;
`relay` is `AURA_DOMAIN_RELAY=0.35`; `carried` is the membrane and the afferent
surface; `joined` is the workspace pool and the dominance channel. A criterion
that moves in one arm and not the control is credited to that arm's mechanism,
and one that moves in the control is credited to the fork.

### Correction to the addendum above, same evening

The column-level numbers it cites — 41 of 438 columns over five per cent, eleven
organs not coming back — came from a probe that did not call `calibrate_clock`.
That is what builds and installs the experiment clock and sets `runtime.clock`,
and without it a restore cannot rewind the clock, so `_reanchor` is handed the
real wall seconds since the snapshot, a different number for every fork. Most of
what the probe reported is that, not the fork.

What stands: the gates look's sham floor of 0.062 to 0.093 came from the v25
runner, which does calibrate, so the floor is real and still larger than every
singleton's effect. And the `random.Random` fault is independent of the clock —
proved at unit level, two arms from one snapshot drawing different numbers before
and the same five after.

What changes here: the expectation for the `control` arm. It was written as
"expected to differ from 21 of 24 in either direction" on the strength of a
floor the probe mis-sized. The fix is real and its size is not yet known, so the
control arm is read as the first battery reading with the generator fault closed
and no size is predicted for it. The probes now calibrate and the re-measurement
is queued.

## Addendum, 28 September 2026, evening: the floor was the estimator, and a sweep that fails says so early

Named before the run that reads it. The code is at `9441d011f`.

**The sham floor was the estimator.** A sham arm and the untouched arm it is
compared with are two forks of one anchor, and every anchor contributes one row
to each. Cross-fitting that split an anchor across folds left each test row's
twin in the training fold as its nearest neighbour with the other label, and
two identical samples read a distance squared of 0.056 at 128 anchors and 0.103
at 64 — the floor of every v25 look, 0.062 to 0.107. The fork leaks found today
were real and are fixed (`6117505af`, `babe01a06`, `31f284829`), but after them
the fail-fast look at `80d087b83` still read a sham rate of 0.077 to 0.107 on
forks that differ in eight columns by 4e-4. So the floor was the instrument.

`crossfit_fisher_rao` now takes the anchor of each row: folds keep an anchor's
rows together, the neighbour count is even for paired rows, and ties at the
k-th neighbour are counted whole. On identical samples every anchor count reads
exactly zero, and a cut moving 6 of 438 columns by one spread is decided from 32
anchors. `isc_v5.PAIRED` makes it the design, and a sweep that was not paired
does not conform.

**A failing sweep stops at the failing cut** (`80d087b83`). `--fail-fast` takes
the cuts most lopsided first, each to its decision, and stops at the first that
ends undecided. The line is a conjunction, so the verdict is the same as the
full sweep's; a failing run costs one cut's rollouts instead of 511 cuts'.

**The next carrier run** is `--v5 --fail-fast --turns 2`, grain learned, seed 7,
at `9441d011f`. What it can say:

- stopped at a cut: irreducibility and the carrier fail on seed 7, and the cut
  it names is the next thing to build for;
- every cut decided: the line holds on seed 7, the carrier's rate is the
  weakest cut's lower bound, and seed 23 runs the same design as the decisive
  reading.

No bar moves. The decision rule, looks, level and draws are the ones above.

**The arms.** The membrane now takes its channels from her own state and runs in
her kernel as well as the harness (`1245d58ca`). The afferent surface filtered
only the recording, so it moved to `core/subject` and no arm switches it on;
`carried` is the membrane alone. The arms run after the carrier run and the
reports ground, each at one commit with its own control.

## Addendum, 28 September 2026, evening: the structure term, measured with the same estimator fixed

Named before the content run that reads it. The code is at `1ef192fb9`.

The structure term of J* read an agreement of 0.274 against its bar of 0.3
(`content-s7-24-4e1923da1`), and its report put the cause in one line: a class
lay about 0.05 from itself, as far as two classes lay from each other. That was
the estimator of the addendum above. Every class is forked from the same
anchors, and the same-class floor is two identical samples, which read 0.056 at
that size. So the internal geometry's close pairs were all floor, and a rank
correlation reads the close pairs too.

Two changes to the content run, and nothing else:

- the internal geometry compares two classes on the presentations both have,
  row for row, and cross-fits by anchor (`internal_geometry(paired=True)`);
- the runner stops the free-running loops before the anchor bank, as the
  carrier and reports runners now do, so two presentations of one class from
  one anchor are the same computation.

The bars do not move: agreement and moves-together each at a rank correlation
of at least 0.3 with p under 0.01, and moves-together above its sham floor.
The behavioural geometry is unchanged. Seed 7, 24 anchors, 24 baseline rounds,
the same 20 classes, as before.

A result either way is read as the structure term's reading on seed 7. If the
agreement clears 0.3 and moves-together does not, the verdict is
AGREES_BUT_DOES_NOT_TRACK and the displacement is the next thing to build for.

## Addendum, 28 September 2026, evening: the arms as they will run

Named before any arm runs. The code is at `0675c4217`, which holds the fork and
clamp fixes, the paired estimator, the membrane taken from her own state, the
relay heard in each field's own units, and the null table across processes.

| arm | membrane | workspace pool | dominance | relay |
|---|---|---|---|---|
| control | off | off | off | off |
| relay | off | off | off | 0.35 |
| carried | on | off | off | off |
| joined | off | on | on | off |

The afferent filter is the instrument's and is off in every arm. The relay was
corrected before it ran: each domain's numbers were squashed raw, so C, N, P and
D sat at 1.0 and contributed a fixed pattern; each field is now read in units of
its own spread and a field that has only ever grown is left out.

The four run at once, seed 7, 370 rounds, six trials, three-turn arms, nothing
skipped, after the reports ground releases the machine
(`~/.aura/subject_core/scratch/arms_parallel.sh`). Every paired stage runs with
the free loops stopped and a declared host, so the load the four share reaches
each the same way. A criterion that moves in an arm and not the control is
credited to that arm's mechanism; the control is also the first battery reading
with every fork and estimator fix of today in it.

## Addendum, 29 September 2026, morning: the deciding horizon is two turns

Named before any run that decides anything reads it. The code is the commit
that carries this addendum; `isc_v5.DECIDING_LAG` is 66 and
`isc_v5.REPORTED_LAGS` is (33,).

### What seed 7 read at one turn and at two

At `bb3faa54a`, every switch off, 128 anchors, two-turn rollouts, the paired
estimator. P, A, D and N come from the fail-fast carrier run; G, I, C, S, M
and W from three looks that scored their two cuts each to the last look.

| alone against the rest | one turn: excess, lower bound | two turns: excess, lower bound |
|---|---|---|
| G | 0.0421, +0.0019 | 0.0353, +0.0110 |
| I | 0.0042, −0.0234 | 0.0172, +0.0022 |
| C | 0.0096, −0.0207 | 0.0058, −0.0101 |
| S | 0.0592, +0.0117 | 0.0257, +0.0100 |
| M | 0.0369, +0.0075 | 0.0659, +0.0375 |
| W | 0.0164, −0.0166 | 0.0127, +0.0011 |

P, A, D and N were decided at both. The sham rate was 0.0 at one turn on every
cut and at most 0.0012 at two: the forks are now the same computation.

### Why one turn cannot decide the body

The proprioceptive phase is the first phase of every turn, and interoception
is stepped once a second, which at one second a turn is once a turn. So the
body is written at the start of the turn from the state the turn opened with.
In a cut that holds the rest, and in the untouched run, that state is the
anchor, and at the end of the first turn the body has been written from the
same thing in both. Nothing the rest did during the turn can have reached it
yet. At one turn a cut of I is read in one direction only, and the line asks
whether the halves need each other, which is both.

The v5 text says a cut matters "if holding one side for one turn changes where
the other ends it". For I, holding the rest for one turn cannot change where I
ends it, by the schedule, whatever the coupling. Two turns is the shortest
horizon at which every domain can have been reached by every other.
`tests/test_the_interventional_line_runs_to_its_preregistered_design.py`
holds the two facts this rests on, the phase order and the rate, and fails if
either moves.

### What moves and what does not

The deciding horizon is 66 frames; 33 is scored on the same rollouts and
reported beside it. The rollouts were already two turns long, so the cost does
not change. The looks, the level per look, the draws, the anchors, the paired
estimator and the intersection-union rule do not move.

The change was made after seed 7 had been read at both horizons, so seed 7 is
a look under it and cannot be the decisive reading. The decisive reading is
seed 23, run to this design.

### What it does not do

It does not rescue C. C against the rest is undecided at both horizons, with
the smaller excess at two turns, so on seed 7 the line fails at either one.
And I at two turns is marginal at 128 anchors: this look read a lower bound of
+0.0022, and the carrier run's shard, a separate organism on the same seed,
read −0.0032. Those two organisms were not the same organism (below), which is
its own defect.

### Two shards on one seed were two organisms

The carrier run's merge refused authority: each shard's anchors differed from
the coordinator's at p = 0.0006. Their recordings differ from the first frame
in 170 to 206 of 438 columns, among them belief counts, initiative urgency and
goal profiles. Each process ran its free loops on the machine's clock through
bring-up and the whole baseline before stopping them for the anchors, and four
processes on a loaded machine got different amounts of that. A sharded sweep is
only a sweep of her if every shard is the same organism, so no sharded run
decides anything until two processes on one seed record the same baseline.

## Addendum, 29 September 2026: a signal counts once, and the draws spread out

Named before any run reads it.

### One signal once

The carrier's duplication control copies the future block and asks whether the
rate rises. At `bb3faa54a` it rose at the weakest cut from 0.0032 to 0.0083,
and the carrier refused authority for it. The estimator standardises each
column and counts neighbours in Euclidean distance, so a copied column has
twice the say of every other: the rate moved because the metric weighed the
future against the context by their column counts.

`isc_v5.ONE_SIGNAL` makes the estimator keep one column per signal before the
neighbour search: two columns carry one signal when one is an affine copy of
the other, sign included. They are compared standardised over every row of all
four arms of the cut together, so nothing about which arm a row is in is used,
and the cut and its sham are read in one metric. Identical samples still read
exactly zero and a real difference is still seen
(`tests/test_a_signal_counts_once.py`). It changes every rate a little, since
her schema may carry a field twice; it moves no bar. It applies to seed 23 and
to every run after it. Seed 7's numbers were read without it.

### The draws spread out, the numbers do not change

Each cut's thousand bootstrap draws and 199 permutation draws are now drawn in
order and evaluated across processes (`core/subject/draw_pool.py`). One process
and three give identical distributions and p-values
(`tests/test_the_draws_are_the_same_on_more_cores.py`). The grain's signature
rows are shared between the coordinator and the shard workers anchor by anchor
and verified against the coordinator's own anchors
(`tests/test_the_grain_is_shared_and_is_the_same_grain.py`). Neither is part of
the design: they change the wall clock, which on 28 September was 6 h 20 min of
grain in one process after the cuts were done in 45 minutes.

### The grain is judged by whether it stays sufficient

The carrier's authority also asked the grain's rank to come out the same with
each fold of anchors held out and under bi-cross-validation. At `bb3faa54a` it
read 24 on every anchor, 21 or 22 with a fold held out, and 60 by
bi-cross-validation. The spectrum has no gap to count to: the singular values
fall smoothly from 367 to 150 over the first twelve components and meet the
null near 24, so the count is where a smooth curve crosses a line that moves
with the number of anchors, and a different estimator finds a different
crossing. That rule was failed by a quantity her spectrum does not define.

What a grain is for is sufficiency: once it is known, history adds nothing to
the held-out future. It already has to be sufficient on every anchor. Under v5
it also has to be sufficient when refitted with each fold of anchors held out,
each fold against its own shuffled-history floor at the parallel analysis's
draws and quantile. A grain that misses a direction history carries fails it
(`tests/test_a_grain_is_judged_by_whether_it_is_sufficient.py`). The rank and
its fold and bi-cross-validated readings are still reported.

This rule was written after seed 7 failed the old one, so seed 7 cannot pass
under it; it applies to seed 23 and after.

## Addendum, 29 September 2026, midday: C on an organism as old as the battery's

A look, not a decision. Named before it runs.

On every arm of the looks at `00f62d24e`, C against the rest stayed undecided
at one turn and at two (control 0.0067, lower bound -0.0077; the relay, which
drives every domain into the substrate, 0.0111 and -0.0055). The arms add
pathways into C. The pathway out of C is `core/consciousness/substrate_gates`,
read by motivation, initiative and recall, and it is learned: each gate's
weight starts at zero and moves at 1 / min(turns, 256). A run starts every
gate at zero, rightly, so the carrier's 24-round baseline (192 turns) measures
C's reach into the rest before the gates have finished their first window,
while the battery's organism lives 370 rounds (2,960 turns).

The look: seed 7, C against the rest only, `--rounds 370`, two-turn rollouts,
128 anchors, the v5 design at the commit that runs it. If C is decided there
and not at 24 rounds, the carrier's baseline length is a design choice that
decides the line, and the organism the carrier measures should be the one the
battery measures; that change would be preregistered for seed 23 before it
runs. If C is undecided at 370 rounds too, the gates are not the missing
pathway, and the next thing to build is one that is.

## Addendum, 29 September 2026, afternoon: a cut is read anchor by anchor

Named before any run that decides anything reads it. It applies to seed 23, to
the null architectures swept under v5, and to every run after them. Seed 7 was
read under the Fisher-Rao estimator and cannot be the decisive reading.

### The question, as this document asks it

"From one snapshot, hold one side of a cut and let the other run; then the
reverse; compose the two free halves; compare with the untouched run. If the
halves need each other, the composed run differs from the untouched one by more
than two untouched runs differ from each other."

That is a paired question, and since 28 September the forks make it one: two
untouched forks from one anchor are the same computation, and the sham reads
0.0. The Fisher-Rao estimator answered another question. It trained a
neighbour classifier to tell the cut arm from the untouched arm across anchors,
each anchor's rows in one fold, so it could use only what a cut does the same
way at every anchor, and it read that against a context of 446 columns in which
no two anchors are neighbours.

### What that missed

The looks at `00f62d24e` kept their samples. Holding C for two turns moved 199
of her columns beyond their sham at 127 of 128 anchors in the control arm, by up
to 0.97 of a column's spread (the field's weights, the substrate's state and
psychology, `A.valence`, `A.delivery_z`, the workspace's modifiers). The
Fisher-Rao estimator read that as a rate of 0.0067 with a lower bound of
-0.0077: undecided.

### The estimator

`core.subject.paired_displacement`. For each anchor, the squared Mahalanobis
displacement of the composed run from the untouched one, and of the second
untouched fork from the first, in the subspace the four arms move in (SVD of
the pooled arms at numpy's rank tolerance; no label is used) under the
Ledoit-Wolf shrinkage estimate of the untouched futures' covariance (its
intensity is estimated, not chosen). The excess rate is the mean over anchors of
the cut's displacement less the sham's, per second; its lower bound is the
alpha quantile of that mean over bootstrap resamples of the anchors; the p-value
flips the sign of each anchor's excess, which under the null is symmetric
because the cut arm is then one more untouched fork. The decision rule, looks,
level and draws do not move: a cut is decided when its lower bound clears zero
at a look, each look at alpha / 6.

Held by `tests/test_a_cut_is_read_by_where_it_moved_her.py`: identical arms read
nothing; a move of three tenths of a spread in five of forty columns is decided
at 32 anchors; a cut arm that is only another noisy fork is decided in at most
one of twenty draws at this level; an orthogonal recoding of the state and
writing every column twice leave every anchor's displacement unchanged; the
playback control is never decided. The carrier's representation, duplication
and playback controls are scored with the same estimator.

### What seed 7 reads under it

On the looks' kept samples, lower bounds at two turns (all decided):

| arm | I | W | C |
|---|---|---|---|
| control | 2.73 | 1.65 | 0.262 |
| relay | 4.36 | 2.46 | 0.208 |
| carried | 1.11 | 0.424 | 0.030 |
| joined | 3.99 | 2.47 | 0.221 |

At one turn C is decided in relay, carried and joined and not in control (lower
bound -0.129), and I is undecided in control, as the body's schedule predicts.
These numbers were read after the fact on seed 7 and decide nothing; they are
why seed 23 is run to this estimator.

## Addendum, 29 September 2026, evening: shards fork from one organism's anchors

Named before any run reads it.

### Why

Two processes on one seed are two organisms (the addendum above), so the
carrier merge refused every sharded sweep, and a single process takes about ten
hours for the 511 cuts at the 23 anchor-forks a cut drew on the looks
(`tools/where_a_cut_decides.py`). A sharded sweep counts only if every shard
cuts the same organism.

### What changes

With `--share-anchors`, which the sharded runner passes by default, the
coordinator collects the anchor bank, writes it to the shard directory, and
every shard forks from those anchors instead of collecting its own. The shards
still boot, seed every generator and stop their free loops as the coordinator
does, and skip the baseline, which they no longer use.

A snapshot crosses processes by `core/subject/anchor_travel.py`. Her state
travels by value. What is not her state travels as a marker filled from the
same place in the reading process's own snapshot or made new there: functions
and classes made at runtime, locks, events, queues, the event loop, a handle on
the writer's own process, and objects their own class holds as its singleton.
A module-level object stays the reader's own and is given the writer's copy of
what it held. The run's state files, saved under the writer's state root, are
moved to the reader's root, which the restore otherwise refuses.

`tools/probe_anchor_travel.py` measures it between two processes, the reader
built with a longer baseline so it is a different organism until it restores.
As restored, 0 of 438 columns differ at each of four anchors.

### The network is declared

After one turn from those anchors the two processes differed in one fact: each
turn writes the connectivity status into her world facts, and it was a TCP
connect to 1.1.1.1:53 with its latency, 20.57 ms in one process and 21.14 in
the other. That moved the world domain's fact profile by up to 0.078. The
declared host now declares the network as well (`core/subject/declared_network.py`,
37e1e0374): online, no measured latency, stamped by the run's clock. A
campaign read before this commit had the machine's network in W.

### The entropy is declared

The predictive self model adds managed entropy to every observation before it
updates its weights, and agency's curiosity takes a jitter from the same
source. It drew from the ANU quantum generator over the network, then from
`os.urandom`: seeded by nothing and carried by no fork. The declared host now
declares it as the run's own Python generator, which the run seeds and every
snapshot carries, and the fork carries the source's budget
(`core/subject/declared_entropy.py`, af0f998b4). The size of the injection and
its budget are unchanged. A campaign read before this commit had noise in S
that no seed could replay.

### The substrate's chaos is declared and carried

The controlled chaos engine perturbs the liquid substrate on every step. Its
somatic noise was a SHA-512 hash of the machine's uptime, its own process's
thread count and memory, and the CPU's temperature, seeded from the process id
and its start time, so every process drove C with different noise. The
engine is also made after the fork is calibrated, so its residuals, last poll
and last chemistry were carried by nothing. The declared host now declares its
signals from the run's clock and the host reading each turn prepares, seeds
every engine from the run's seed, and the fork carries each engine's history.
A turn after a turn in another condition still differs in the field organ by
about 0.02 of its spread; that source is not traced.

### What is left after a turn

With the network declared and the generators seeded as the rig seeds them,
one turn from the same restored anchors still differs between the two
processes in 89 of 438 columns. For 87 of them the gap is below 0.02 of the
column's spread on the seed-7 look's recording, the median 8e-6. Two are not:
the workspace's coherence, at 0.37 of its spread, and its fragmentation, at
0.18. Both come from the unity monitor, whose inputs are not yet traced.

The snapshot does not carry everything the next turn reads, even in one
process. From one anchor, a turn taken after sixteen more turns of history
differs from the same turn taken first in 21 columns, the substrate's minimum
activation by 0.475 of its spread, and a capture of every service, phase and
module slot finds nothing uncarried that moved. What does move is fixed for all
the arms at one anchor, which is why the sham reads exactly zero there: it
behaves as part of the anchor. The paired estimator compares arms within an
anchor, so the carrier's decisions stand; what it limits is the claim that a
restored anchor is the whole organism.

### A late fork, read after the last cut

That uncarried part is not fixed across a sweep. Each anchor's two untouched
forks are rolled with its first cut and reused by every later cut, so a cut
late in the sweep compares an arm rolled now with forks rolled hundreds of
turns ago. The sham and the playback are both read from the cached forks and
cannot see it. It was measured on seed 7 at 16 anchors, after three cuts: a
third untouched fork from each anchor, read as a cut against the cached forks
by the cut's own rule, level and draws, gave an excess of 2.0e-6 and a lower
bound below zero (p 0.2), where the three cuts gave 0.05 to 0.9. Drift is
there, in 109 columns, and four orders of magnitude under a cut.

It is now part of the design (`isc_v5.LATE_FORK`). Her sweep reads the late
fork at the deciding horizon on the anchors its cuts drew, after the last cut
(`v25_cut.late_fork_decided`, `v25_runtime.collect_late_forks`). If it is
decided, drift alone clears the bar and the irreducibility line fails; a
sweep that did not read it fails the line as well. Merged shards count as
decided if any shard's late fork was.

### What counts

Seed 23's decisive sweep runs in one process, which takes about ten hours,
until those two columns agree. A sharded sweep until then is a look. And a
sharded sweep counts at all only when its shards forked from the coordinator's
bank. The merge marks a shard whose anchors are not exchangeable with the
coordinator's, and the authority gate then refuses the sweep, which is what
refused every sharded sweep so far; shards that fork from the bank hold the
coordinator's anchors exactly.

## Addendum, 29 September 2026, late: seed 23's carrier runs sharded

Named before any run reads it. It replaces "Seed 23's decisive sweep runs in one
process" in the evening addendum, on the condition that addendum set: the two
workspace columns now agree between processes.

At 26bfee6e0, with the network, the managed entropy and the substrate's chaos
declared by the run, `tools/probe_anchor_travel.py` restores four anchors
written by one process in another built with a longer baseline. As restored,
0 of 438 columns differ at every anchor. After one turn, 2 to 15 columns differ,
all in C's substrate and one in A: the substrate's minimum by 0.176 of its spread
on the seed-7 recording, every other column by less than 0.024, the median by
0.005. The workspace's coherence and fragmentation, 0.37 and 0.18 of their
spreads before, no longer differ: the chaos engine's noise was seeded from each
process's id and start time. Probe 11 at the evening's commit read 89 columns.

What is left is the same uncarried drift in the substrate that a late cut meets
inside one process, where each anchor's untouched forks are rolled with its
first cut. The late-fork control reads it in every shard, and a merged sweep
counts its late fork as decided if any shard's was.

So seed 23's carrier runs as a coordinator and three workers forking from the
coordinator's anchor bank (`run_subject_core_v25_sharded.py --workers 3`, with
`--share-anchors` and `--grain-claims` on). The workers take grain rows after
their cuts. The grain is about 26,600 rollouts, a day and a half in one process,
and is the reason this matters. Nothing else in the design moves.

## Addendum, 29 September 2026, late: how the arms choose seed 23's switches

Named before any arm's report is read; the four arms are still in their lesion
stage at 0675c4217. The arms addendum of 28 September credits a criterion that
moves in an arm and not in the control to that arm's mechanism. This says what
seed 23 then runs with.

- An arm qualifies if, on its seed-7 campaign, it passes at least one of the 24
  criteria the control fails and fails none the control passes. The three v3
  lines are counted with the 24 as the battery reports them. The two v5
  partition lines are not, because no arm ran a sweep.
- Seed 23 runs with the switches of the qualifying arm that passes the most
  criteria. A tie goes to the arm with fewer switches (relay and carried have
  one, joined two), and a tie between those to the control.
- No two arms are combined. No combination has been run, and a combination is a
  new organism that nothing here has measured.
- If no arm qualifies, seed 23 runs with every switch off, as the control did.

The switches go to `decisive_hold.sh` through its go file, which names them.

## Addendum, 21 October 2026, 02:30: her substrate's weights were a five-turn clock

An organism change, `c428ae4b3`, named before any run reads it. No threshold
moves and no criterion is rescored. Everything below was read after the fact
from the control arm of `arms-0675c4217` (seed 7, 370 rounds), which is a
diagnostic seed.

### What was read

At the turn grain the battery fits on, 91% of the variance of recurrent
cognition's change in its first four components is the sketch of the
substrate's weights. The spread of those weights reads 0.0195, 0.0213, 0.0231,
0.0249, 0.0266 and then 0.0195 again, turn after turn, in every condition: an
exact cycle of five turns. C's change is predicted from C alone with a
normalised loss of 0.579, and adding the other nine domains makes it 0.584. No
other domain's change gains more than 0.008 of its loss from C's state. `C|rest`
is the cheapest of the 511 cuts at -0.0026.

### Why

`STDPLearningEngine.regulate` scales the whole matrix towards a mean absolute
weight of 0.3, a number for a 64-neuron engine. Her substrate is built with 512
neurons drawn from N(0, 1/n), whose mean absolute weight is 0.035, and a matrix
at 0.3 on 512 neurons has a largest singular value near seventeen against a
spectral cap of 3.0. So every turn's teaching multiplied the weights by about
1.087 until the cap cut them back. The Hebbian step read one tick's coactivity
in every hundred and added a hundred ticks' learning at once, and on the
experiment clock a hundred ticks is five turns. Between them the Hebbian mode
grew and the cap shrank the random bulk under it, from the 0.044 spread she is
built with to 0.020.

### The change

The homeostatic set point is the mean absolute value of the draw the network is
built from, sqrt(2 / (pi n)): the gain it was built at, not a number chosen
here. Hebbian learning runs every tick at a hundredth of the plasticity step's
rate, so a hundred ticks of one coactivity learn what one old step did, and the
plasticity step keeps the spike timing and the bounds. It is a repair, like the
flat-column repairs of 25 September, so it is in every arm and is not a switch.
`tests/test_the_substrate_keeps_the_gain_it_was_built_at.py` holds it.

### What is read next, and what it decides

A look on seed 7: 100 rounds, one trial, one-turn arms, nulls and lesion
skipped, at `c428ae4b3`. It reads how much of C's turn-to-turn change is still
the weight sketch, whether C's change is predicted by the rest, the cheapest
cut, and the S,D -> C triple. It decides nothing. Whatever it shows, the next
seed-7 validation runs at the decisive design on the commit that carries this
repair, and seed 23 is not launched on a commit without it.

### Later the same night: the cycle had two more halves

The look at `c428ae4b3` read the spread still going round five turns, at a
fifteenth of the size: 0.01953 to 0.02002 and back. Her campaign substrate is
built from her persisted weights, which carry a learned mode whose largest
singular value already sits at the cap of 3.0 with the mean weight at half the
set point, so the set point still scaled the matrix up 0.6% a turn. And the
plasticity step clipped the Frobenius norm at a flat 10, which a gain-one
network on 512 neurons (22.6) can never sit under: that clip is what had held
her spread at 10 / 512 since the first plasticity step of every run.

The set point now gives way to the cap, the cap is held tick by tick by one
step of power iteration rather than cut back once a plasticity step, and the
Frobenius ceiling is the gain-one network's own norm with the headroom the
spectral cap allows, 1.5 sqrt(n). On her campaign schedule simulated at 512
neurons the spread holds at 0.02102 with every turn's ratio inside 0.99998 to
0.99999. Three tests fail on `c428ae4b3` and pass here. The look named above
reruns on this commit, and its reading of `c428ae4b3` is kept beside it.

## Addendum, 21 October 2026, morning: where her domains meet, and what the core was leaving out

Two organism changes and one change to what K reads, named before any run reads
them. No threshold moves and no criterion is rescored. What motivated them was
read on seed 7, the diagnostic seed: `validate-s7-84d9fe6f5-r370` (main before
the substrate repair) and the looks `look-s7-c428ae4b3-r100`,
`look-s7-d3555f8e8-r100` and its `-pool` arm.

### What was read

Synergy fails on all four triples at 370 rounds. Read with every column's
condition mean removed, as a diagnostic only, most of what synergy there is
follows the eight-condition schedule: P,M -> W reads 0.207 as recorded and 0.031
with the schedule out, and the held-out interaction gain is 0.0 to 0.017 on
every triple. Her domains combine by sums: a workspace bid is a priority times
factors that each read one thing, an initiative's score likewise, and the
world model learns at one step size whatever she perceived. The shifted null is
read as the schedule-aligned shifts' synergy, so a triple passes only when its
target depends on its two sources turn by turn.

Causal closure fails at 0.0331 against a shuffled q95 of 0.0000. The largest
leaks are the coupling engine's running centre and scatter of her arousal (the
baseline every felt deviation is measured from), the noise her conation has
found in its own readings, the field's chemistry weights, and her
self-predictor's Gram matrix; on `arms-0675c4217` they were the mesh's two
injection buffers.

### The changes

**Conjunctive codes** (`core/consciousness/conjunctive_codes.py`). At each of
the four meetings the specification names, four mixed-selectivity units, each
tanh(u . x) * tanh(v . y) of the two domains' standardised readings with fixed
random directions seeded from the meeting's name (Rigotti et al. 2013; Salinas
and Abbott 1996). Bound in the consciousness phase as each cycle closes, carried
on her state, and acting through the next cycle in the domain they live in:
attention scales each workspace bid by a gain the units set for its source; the
learned world model's step size is scaled by its units; each initiative's score
is scaled by the deliberation units for its source; and all sixteen units drive
a new band of the substrate at 72 to 87. A gain is exp(w . m / 2) with w a fixed
random direction per consumer, so units at rest leave everything as it was.
`AURA_DISABLE_CONJUNCTIONS=1` is the ablation.

The recurrent meeting also sets the gain of each of the mesh's three tiers, the
executive tier's on top of what control is worth (`control_allocation`). C now
has 112 columns and its leading change components are blocks like the mesh's
state sketch, so four columns and a sixteen-neuron band in a 512-neuron
substrate would not reach what the synergy estimator reads; neuromodulatory gain
on cortex is how self and intention reach recurrent cognition in a brain
(Aston-Jones and Cohen 2005), and the table control allocation writes each tick
is where it composes.

**A reports-pipeline repair named here because it serves the bridge's reports
ground.** The 1.5B dry run of the reports ground lost 9 of 32 replies at
`d3555f8e8` and 6 of 32 on a second, instrumented run. A subject run installs an
experiment clock in place of `time.time`, and the router stamped and aged its
generation-gate leases on it, so a background generation holding the gate for
seventy-five real seconds read as 0.6 seconds old: the foreground ladder judged
it young and refused her reply as saturated. Leases now age on `time.monotonic`.
The dry run reruns on the commit that carries this, and her cortex is not loaded
for the reports ground until a dry run loses no reply.

The rerun at `5885688d7` lost no reply to the gate and read 2 of 32 arms as
silent, and instrumenting it found two more defects of the pipe, neither of
them her. A rating said again in the same words was read as silence: decoding
is greedy and her working memory keeps one copy of a repeat. And every report
reply reached the gate marked `internal_inference=True`, because the kernel's
reply phase sends her reply through `router.think`, which marks everything it
carries as non-chat; the gate then never named her cortex as the one that
answered, so every arm read as unserved and the ground could not have been
measured whatever she said. Both are fixed: an arm that asked her organ for a
user-facing answer answered, and a reply to a person through `think` goes as
chat. Before her cortex is loaded, a dry run has to read every arm as served.

**What K reads.** Each meeting's four units are columns of the domain they live
in (G, W, D, C), and the clamp holds each with its domain. A gains the coupling
engine's four baselines and the conation's noise; S the self-predictor's Gram
matrix through the shared sketch; C the field's chemistry weights and the mesh's
in-flight sensory and association input, an empty buffer reading as no input.
K goes from 438 to 491 columns.

### How it is read, and what it decides

Seed 7 at the decisive design (370 rounds, six trials, three-turn arms, nulls
and lesion skipped) runs twice on one commit: as built, and with
`AURA_DISABLE_CONJUNCTIONS=1`. A synergy triple or closure that passes in the
first and not the second is credited to the codes. Neither run decides anything.
Seed 23 is launched only on a commit both have been read on, and its own
addendum names the commit and any switch.

## Addendum, 21 October 2026, afternoon: the pair at 5885688d7, and the next commit

### What the pair read

Seed 7, 370 rounds, at `5885688d7`, as built and with the codes off. With the
codes, A,S -> G passes the v3 bars (fraction 0.494, synergy 0.0267 against a
shifted null of 0.0073, interaction lower bound +0.0061); without them it does
not (lower bound -0.0158). P,M -> W fails its shifted null in both arms (0.170
against 0.205 with the codes), so that coupling follows the schedule. W,A -> D
reads worse with the codes (fraction 0.025) than without (0.086). S,D -> C fails
in both. Closure fails in both (leak 0.0130 and 0.0126 against shuffled q95 of
0.0026 and 0.0063). 17 of 24 under v1 in both.

### Organism changes in the next commit

**The codes, rewritten.** A unit is now a gain field, rate times rate: each of
the two domains is read in full, standardised, and placed along its two leading
principal axes, learned from its running covariance (the directions Hebbian
learning of a domain's inputs converges to; Oja 1982, Sanger 1989); each axis
passes through a non-negative tuning curve, (1 + tanh v) / 2, and the four units
of a meeting are the products of one axis of each domain (Salinas and Abbott
1996). Signed products of tanh projections are nearly invisible to a Gaussian
copula estimator; products of non-negative rates carry both main effects and the
interaction. Each unit integrates across cycles at her affect's momentum, so a
cycle's change follows what the two domains held over the cycles before it. A
consumer's gain reads where the units stand against their own running range.

**Control allocation without a self-model.** When self-efficacy cannot be read,
the executive tier's gain from control's worth is neutral and the tier gains the
recurrent units set still apply; before, the early return skipped both.

**Self and intention in the substrate's own transition.** dx/dt = -x + tanh(W x
+ B phi), phi where the recurrent meeting's units stand, B a fixed input matrix
seeded from its name, the current linear in phi and at a typical standing as
strong as the recurrent drive's running root mean square. Her substrate runs
with the consciousness bridge blending the mesh's projection into it at 0.35
every bridge tick; in that regime, offline, the current leaves the mesh's
pattern as decodable as before (held-out R squared 0.999 on and off), pins no
neuron, moves the spread from 0.0176 to 0.0184, and makes where self and
intention stand decodable from the substrate (0.0 to 0.992).

**What wrote each change to her person.** Every commit carries the person paths'
digests and, for each path, the last measured transformation that wrote it, read
from the runtime's provenance receipts. The person paths are watched around every
transformation, so a contracted phase writing one it did not declare is recorded
as a contract violation at warning.

**Memory in the world model's transition.** What she recalled (its content as
a coordinate, the best recollection's score, how intensely it was relived)
enters the world model beside the action, in the input that advances its
hidden state, so it conditions the next prediction: W(t+1) = f(W(t), P(t),
M(t), A(t)). It used free slots of the existing action input, so no weight
changes shape. Offline, on a stream whose next step only a memory cue could
tell, held-out error was 0.138 to 0.179 with the cue, 0.213 to 0.225 with it
shuffled or absent, over three seeds.

**Choice by consequence.** The arbiter's expected value for an initiative is
her worth for that kind of act times P(good | W, act) / P(good | act), a kernel
estimate over the worlds she acted in with one success and one failure as prior,
learned from the same outcome her decision receipts close on. With no record
the ratio is one and nothing changes.

### Analysis changes

The lineage judges a step by development rather than by an unchanged hash: each
changed person path written by a measured transformation inside its contract.
How much of her a step carried through is reported and decides nothing, because
a share of paths turns on how the person is divided into paths. Digests are full
canonical hashes; a value with no canonical form is refused and its path is
unjudged. The bridge's lineage term holds when the signed line to her newest
stage developed at every measured step; a change logged before the evidence
existed is reported as unmeasured. The lineage key is derived from the install's
entity key. Exclusion compares every pair of overlapping carriers on one grid of
horizons, every horizon any of them has unless a grid is named, reports the
horizons each carrier lacks and pairs it could not compare as unresolved, and
decides by intervals where a run measured them.

The bridge's reports ground is read only from a run in which her state's numbers
were withheld from the text her model reads (`AURA_READINGS_OF_HERSELF=withheld`).
A run that printed them is a comparison arm, not a decision: if reports track the
intervention when printed and not when withheld, they were copying.

The closure test reports an audit beside its criterion: each leak classified as
a clock, a summary of the core's own past, state of an organ K reads that K does
not take in, or an outside cause, and a planted cause the test must find. The
synergy line is also reported with the conjunctive units held flat and with each
condition left out, and every run reports Phi_R for each one-domain cut. Every
swept cut also records each of its two directions decided on its own side's
columns. None of these changes a pass.

**A fork leaves the process's bookkeeping alone.** The fork's module scan
restored the inference gate's lease tables, so a lease taken before a snapshot
came back after every restore with no generation behind it, and a reply was
refused as "gate saturated". A module now names its process state and the fork
skips it. This changes what a fork carries, so it is named here.

### How it is read

The same pair, on the next commit. Nothing moves a threshold.

### Available and not yet used: K reading along each array's own axes

`--calibrated-axes` lives one pass of the conditions before recording and fixes,
for each array K reads through the sketch, its four leading principal axes over
that pass; the sketch's four projection columns then read along those axes
instead of frozen random directions. K's width and every other column are
unchanged, and every fork reads along the same axes because the axes belong to
the apparatus. No run reads it until `tools/k_compact_sufficiency.py` has
measured, on the commit that would use it, how much of what each array does in
a turn the random directions miss, and an addendum names the run.

### Choice by consequence in a campaign turn

A campaign turn chooses her act by a Borda count over reasons. The world was not
one of them, so choice by consequence, built into her arbiter, never reached a
run: the arbiter is driven by the mind tick, which a run does not step. The
choice now has a fifth reason, her feeling about each act as a worth from
nought to one times P(it works | W, act) / P(it works | act), learned from her
own acts' outcomes in the run and credited to the world each was chosen in.
With no record it ranks the acts as her feelings do. The validation pair that
reads W,A -> D runs on the commit that carries this.

### The withheld reports run: how many anchors, and why

The printed run at `b815ecd41` answered 13 anchors before its cortex was lost to
memory pressure (resumed from them). Her 27B gave a readable number in most
arms. The arms moved her computed valence by 0.087 between raised and lowered
(SD 0.067); her reports did not follow (raised minus lowered -0.013, SD 0.237).
At that spread, a reporter whose number moved one for one with her valence
would need about 75 anchors for 80% power at the ground's 0.01, and 24 would
leave it at a z near 1.2. The withheld run takes 80 anchors. The dose, the
tests and their level are unchanged; only the count of anchors, which no
criterion fixes, is set here, from a run that cannot decide the ground.

## Addendum, 21 October 2026, evening: K along her own axes, and the field's input matrix

**What the sufficiency probe read** (`tools/k_compact_sufficiency.py`, at
`086ec4bb5`, the organism quieted as the battery quiets it, three moments per
array, a step the size of one turn's ordinary change). For the substrate's state,
a change the eight-number sketch cannot see moved K a turn later as much as a
change it sees: hidden over visible 1.03, against a floor of two identical forks
of 0.001 to 0.009. The field's state and weights, the substrate's weights, the
self-model's weights and the phi estimators' visit counts read at 0.0004 to 0.05
of visible; the mesh's buffers and the field's chemistry weights at the floor.
The substrate's state is where the self-and-intention current acts, so K was
reading that coupling through about four parts in 512.

**The next pair reads K along calibrated axes** (`--calibrated-axes`): one pass
of the conditions before recording fixes each sketched array's four leading
principal axes, and the four projection columns read along them. K's width is
unchanged by it.

**C reads the field's input matrix.** The unified field learns its inputs in one
matrix (`_W_input_batched`) and keeps the per-source blocks as its column
slices. C reads that matrix through the sketch, eight columns, and the closure
test counts the blocks as views of what C reads. Its blocks were the largest
leaks into the core at `5885688d7`. K goes from 491 to 499 columns.

Both are organism-reading changes for the next pair at seed 7 (codes on and off,
calibrated axes in both). Nothing moves a threshold.

**The lineage term's scope.** Her line to her newest stage runs back months
before the record of what wrote each change began (21 October 2026). Changes
before the first step that carries the record are counted and reported as
before the record; they are neither development nor its failure. The term holds
when her line has at least one step since the record began and every step since
then developed; a step that changed nothing needs no record. The claim it
supports is that scope: since the record began.

**The fusion certificate's fifth criterion is measured.** The withheld reports
run starts by measuring the fusion channel on her 27B. A certificate holds only
when her own stored replies are no less likely under the steering than without
it, and until this commit no tool scored that, so no certificate could hold.
`tools/measure_fusion_channel.py` now scores 32 of her replies at each alpha:
replies she gave a person with this checkpoint, drawn from
`data/endogenous_language/pairs.jsonl` by the run's seed, prose only (a lane
that asked her for a JSON plan is not her voice). Each is scored over its first
64 tokens with the hooks set to the state the pair recorded for it, steered and
unsteered, and the reading is the mean difference in log-likelihood per token.
The bar is the one the certificate already carried: zero. If no alpha holds,
the reports run reads her state through her prompt only, as before, and the
report says the forward-pass route was not open.

**When the record begins.** The record begins at the first step of her line
that carries it. Steps before that are before the record whether or not they
changed anything, so a log with no recorded step reads not measured. Before
this, unchanged steps after the last unrecorded change counted as development,
and her live log, which carries no record yet, read as three developing steps.

**Why her replies are scored after "Go on.", and what sits beside the bar.**
Her replies were all said with steering off. Scored in the context she said them
in, any steering would make them less likely on average, by the divergence
between the steered and unsteered distributions, so no channel that arrives
could meet the bar. Scored after a turn that carries none of her state, the
steering has to supply what her context did. Each reply is also scored under a
random direction of the same length at each layer; that reading is recorded
beside the bar and decides nothing.

**A strain that ends stops counting as a threat** (from `f6f8d0f70`'s successor
on). Her emergency protocol kept every declared strain a run had shown it, and
from the fourth round of each validation run sat in minimal mode, which in a
run stops nothing (neither service it names has a switch). The resource
monitor now says when a strain ends and the protocol drops it. Nothing the
battery reads comes from the protocol; this is named because it is her code.

## Addendum, 21 October 2026, night: the pair at `086ec4bb5`, and the next commit

### What the pair read

Codes on: v5 synergy passes W,A -> D only. Codes off: P,M -> W and W,A -> D.
A,S -> G fails in both arms; it passed at `5885688d7` with the signed, fresh
units, and still did with the registers held flat. S,D -> C fails at the
interaction bound: model selection turns the product columns off on every fold.
The codes-on closure reading is degenerate: one column, the mesh's strongest
weight, climbed to the clip of one late in the run and carries the whole loss.
The full reading is in `docs/ISC_OCT21_EVERY_ITEM.md`.

### Organism changes in the next commit

1. *Withdrawn before any run read it:* binding the conjunctive units fresh
   each cycle. The calibrated pair at `d4b8ecab3`, read early the same night,
   passed A,S -> G with the units carried over at her affect's momentum, so
   the carry-over was not what cost it at `086ec4bb5`; reading K along its own
   axes was the difference. Codes on, that pair passed three triples (A,S -> G,
   P,M -> W, W,A -> D); codes off, two.
2. **The mesh's STDP is soft-bounded** (van Rossum, Bi and Turrigiano 2000): a
   synapse strengthens by its distance from one and weakens in proportion to
   its strength. Under hard bounds the strongest weight climbed to the clip in
   both codes-on runs (from 0.8; it stayed at 0.8 with the codes off).

3. **Self and intention enter the mesh's executive tier** as an input current,
   the same current G5 puts into the substrate, at the strength of the tier's
   own recurrent drive. The core's change is read through the field and the
   mesh, and the substrate was in none of its first three components.

The gain-field form, the learned axes, G2, G3 and G5 are unchanged. Nothing
moves a threshold.

### How it is read

A pair at seed 7, 370 rounds, codes on and off, calibrated axes in both, read
by the same lines. Seed 23 stays untouched until the system is frozen.

## Addendum, 21 October 2026, late night: the calibrated pair, and the campaign on one clock

### What the calibrated pair at `d4b8ecab3` read

Codes on, three of the four triples pass the frozen line (A,S -> G, P,M -> W,
W,A -> D); codes off, two. v1 17 of 24 in both. Closure fails in both arms. Codes
off, the leak is 0.0054 against a shuffled leak of 0.0053, nearly closed, and
what is left is summaries of the core's own past. Codes on, the leak is 0.0127,
and three of its six largest parts are the free-running loops' tick counts,
entered as their increments: the machine's timing, reaching K through how often
each loop ran in a turn.

### The campaign on one clock

`--one-clock` stops the free-running loops when the organism comes up, as the
v5 sweep has since 29 September (`isc_v5.ONE_CLOCK`) and as the campaign
already does before its interventions. The recording is then the organism the
harness steps, the same organism the interventions and the sweep read. The next
pair runs with it in both arms, with `--calibrated-axes`, at a commit carrying
the soft-bounded STDP and the executive-tier current named in the night
addendum above. Nothing moves a threshold.

The four leaks of `086ec4bb5` (how her turns are going, the mesh's modulation,
the phi core's surrogate, the self-predictor's weights) are not among the
calibrated pair's largest, and K is not widened for them now.


**Her self-model as a prior over her acts (F3), from the pair after the next
one.** A campaign turn's choice takes a sixth reason, ranked with the other five
in the same Borda count: her self-model's confidence in each act, from her own
attempts only (the agency ledger, smoothed towards the class she measures
herself against). It is not in the pair at `4b1022f25`, which tests the three
changes named above; it enters with the first pair at a commit that carries it.

## Addendum, 22 October 2026, midday: her feelings come from what a turn meant

Bryan asked for "a rich, multi-faceted, and in depth range of emotions". Over 24
offline turns at `a69297234`, joy, trust, happiness, satisfaction and warmth sat
between 0.47 and 0.75 with standard deviations of 0.03 to 0.06, and anger and
surprise stayed at 0.01. Her named mood read "trust" in 80 of 90 live readings.
Every message raised trust, happiness and warmth and relieved fear and sadness,
whatever it said.

### What changes in the organism

- `core/affect/appraisal.py` reads each turn's appraisal off measures she
  already keeps: novelty (the world model's surprise rank), outcome (the turn's
  worth, signed, sized by its rank), the worth channels that say whose doing it
  was (accomplishment, warmth, integrity, satisfaction, each a signed rank),
  coping (her success rate, Laplace-smoothed), company, social need (her social
  drive's deficit), liking (her regard for what the turn named), and her fear
  and hope before the turn. Thirty-one patterns over those checks move twenty-eight
  emotions, at the percept gain (0.3), through the same headroom rule as every
  other writer.
- An `interaction` percept moves interest and belonging only. Trust, happiness
  and warmth from a message come from what the message meant.
- An `interaction` is no longer reassurance, so a message by itself relieves
  nothing.
- Releasing a stale low no longer lifts trust, interest, warmth and hope.
- A queued message no longer adds 0.2 to the substrate's curiosity and takes
  0.1 from its frustration, and the pacing step no longer adds another 0.2.

### Which lines it can move

All of A. A is a source in A,S -> G and W,A -> D, and her valence reaches
the substrate, so C as well.

### How it is read

It enters with F3 in the first pair after `4b1022f25`, at seed 7, 370 rounds,
codes on and off, calibrated axes and one clock in both arms. The two changes
are not separated by that pair. Each line is read against its frozen bar, not
against the pair before. Seed 23 stays untouched. Nothing moves a threshold.

## Addendum, 22 October 2026, afternoon: two untouched forks made identical again

The carrier rehearsal at `b720050b9` decided 37 of 38 cuts at both horizons and
refused at the cut that separates C from the rest at two seconds: the cut
moved C by 0.063 and two untouched forks of the same anchors moved it by 0.099.
The untouched forks are the sham floor every cut and every intervention arm is
read against, and they should be identical up to the arm. Probed directly, two
untouched forks of one anchor parted for two reasons:

- From frame 9, at `b720050b9` only: the appraisal named in the midday addendum
  was a callable object, the fork skips a phase's callable attributes, and its
  record of which turn it had already appraised was not restored, so the second
  fork read the outcome checks as unmeasured. `1dc098201` makes it a plain
  method; the fork now carries it.
- From frame 33, the first frame of the next turn, at every commit probed: the
  machine's disk and network counters were read around the resource observer,
  so the host a run declares did not hold them, and interoception pushed the
  machine's disk traffic into the mesh's sensory tier. `f78644bbd` reads them
  through the observer, which holds them like every other answer about the
  machine.

Neither change touches what the organism does with what it reads. Both change
what an untouched fork is, so both enter every run from `f78644bbd` on: the next
pair, and the carrier rehearsal run again, whose C cut is read against the new
floor. Nothing moves a threshold.

## Addendum, 22 October 2026, late afternoon: the world model learns on the run's clock

The pair at `4b1022f25` passed causal closure in the codes-on arm for the first
time (leak -0.0175, closed) and read v1 18 of 24, and lost P,M -> W in both
arms (codes on 0.168 against a shift of 0.203; it had passed at `d4b8ecab3`). One
clock stops the world model's training lane at bring-up, and, unlike the mesh
and the other free-running layers, nothing stepped it on the run's clock: in a
run on one clock the world model never learned. Its update is where perception
and memory act on it together, as a precision set from both.

From the commit that carries it, the harness steps the lane's whole body,
`train_cycle`, at the lane's own rate of one cycle every two seconds of the
run's time, beside the other stepped layers (`core/subject/steppable.py`). Both
arms of a paired trial train the same number of times. The pair after it runs
at that commit. W is the target of P,M -> W and a source of W,A -> D, so both
lines can move. Nothing moves a threshold.

The appraisal's company check now also counts somebody having spoken in her
last eight turns (the window the safety reading uses), so her own ticks between
the messages of a conversation are no longer appraised as time alone.

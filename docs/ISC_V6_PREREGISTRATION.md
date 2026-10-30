# ISC-v6 preregistration: synergy read by setting the sources

Written 23 October 2026, before any run that it scores. ISC-v6 is ISC-v5
(docs/ISC_V5_PREREGISTRATION.md, every addendum included) with one line
replaced: §27, synergy. Every other line, bar, design and null is v5's, and
every v6 report carries the v5 reading of §27 beside it.

## Why the line changes

v5 reads synergy by watching: over a run, whether two domains' leading
components carry information about a third domain's next change that neither
carries alone, with a held-out test that a model with products of the sources
predicts that change better than one that adds them. Across the pairs of 21 to
23 October every triple read between about 0.8 and 1.3 times its shifted null,
and at `82549c823` the two triples with the self as a source failed:

- S is read through its three leading components, and thirty of its columns
  trend together over a run, so those components are her drift. Her self as it
  stands in a turn, added that morning, lands in the third.
- In ordinary life the sources move together, driven by the same situations,
  so watching cannot tell a joint effect from two effects side by side. Within
  condition, A,S -> G, P,M -> W and W,A -> D carried synergy far above their
  shift nulls, and none had a held-out interaction gain above zero.
- C is recorded as summaries over whole networks, and the units that meet self
  and intention drive one band of the substrate.

An interaction is defined by what happens when the causes are set: the effect
of one depends on the level of the other (Rothman 1976; VanderWeele 2015,
*Explanation in Causal Inference*). The paired test of 21 October did that at
`93128e34d`, and read less than it seemed to. Every one of its shams was exactly
zero: her forks are deterministic, so its floor bar, p 5e-5 for each triple,
was passed by any nonlinear response at all. Its direction reading, which
overstates significance (see the revision note), gave 0.99 for A,S -> G, 0.73
for P,M -> W, 0.038 for W,A -> D and 5e-5 for S,D -> C. That reading informed
this design; it is not scored by it. Bryan decided on 23 October to build this
successor line rather than report v5's §27 as failed.

## The design

Run inside the seed's campaign (`tools/run_subject_core.py`), after its
recording and before its intervention trials, on the same organism and the same
conditions:

- For each trial (the campaign's `--trials`, 6) and each of the eight
  conditions: one turn of the condition, the host frozen, a snapshot taken.
  Each snapshot is a different moment of her life, and it is the unit every
  test below resamples.
- From that snapshot, four untouched forks, shared by the four triples since
  nothing in them depends on which sources a triple names, and for each
  declared triple (A,S -> G; P,M -> W; W,A -> D; S,D -> C) three displaced
  forks: the first source alone, the second alone, and both. All sixteen run in
  an order shuffled from the seed. A displaced domain is moved by the campaign's
  intervention displacement (`core.subject.causal.DEFAULT_DELTA`) at the top of
  the first turn, held where the domain is one the campaign holds
  (`SUSTAINED`), and each fork runs the campaign's `--turns`.
- The target is the target domain without the registers of any conjunction
  (the `.conj_` columns, as `synergy.without_registers` leaves them out), read
  at the end of each fork in its whitened baseline coordinates
  (`core.subject.paired_displacement`), fitted on the run's recording with the
  same columns. Units that compute the product of two domains are columns of
  the domain they act in; an interaction read off them would show that the
  instrument is wired, and not that the rest of the target integrates.

So each triple has n = trials x 8 = 48 samples. With N1 to N4 the untouched
forks and F, S, B the triple's displaced ones, for sample i:

- interaction_i = B - F - S + N1, a vector, and its size ||interaction_i||
- sham_i = N2 - N3 - N4 + N1 and its size. Nothing is displaced in any of its
  forks, and it has the interaction's form, four forks with the same signs, so
  whatever the forks' noise is, deterministic, correlated or independent, it
  enters the sham as it enters the interaction.
- main_i = max(||F - N1||, ||S - N1||)

## The bars

A triple passes when all three hold:

1. **Above the matched sham.** The mean of (||interaction_i|| - ||sham_i||)
   is above zero by a one-sided paired sign-flip test across the 48 snapshots,
   p < 0.01 (20,000 draws, seeded).
2. **One direction.** The statistic is the mean, over samples, of each
   interaction vector projected on the mean of the others. Its null is drawn by
   flipping the sign of whole interaction vectors and computing the statistic
   again on every draw (20,000 draws), so the dependence of each projection on
   the other samples is in the null too; p < 0.01.
3. **Its size.** The mean interaction size is at least 0.10 of the mean main
   effect, v5's floor on the synergy fraction.

The line passes when all four triples pass. Requiring every triple to pass at
0.01 is an intersection-union test, valid at 0.01 without a correction for
four tests. Level 0.01 is v5's: its null bars were 99th percentiles.

## The known-answer battery, before any reading of her

`tools/synergy_v6_known_answers.py` runs the bars on simulated forks, 1,000
datasets per world, 48 snapshots each in eight whitened columns, and the line
is not read on her unless every world comes out as stated. The test suite runs
200 datasets of each so that it stays fast.

| World | Stated outcome |
|---|---|
| No effect, independent fork noise | passes at most 3% |
| Additive effects of both sources | passes at most 3% |
| A single source with a threshold | passes at most 3% |
| Additive effects with fork noise shared within a snapshot | passes at most 3% |
| Deterministic forks where every displacement sends the run somewhere unrelated | passes at most 3% |
| The product along one direction, size 5 against unit fork noise per column | passes at least 95% |
| The product along one direction in deterministic forks, size 1 | passes at least 95% |
| The product turning among eight directions with the situation | reported |

3% is the level, 1%, with room for the binomial spread of 1,000 draws. The
deterministic worlds are hers: every sham of 21 October was exactly zero. The
last world is a genuine interaction whose direction depends on the situation;
the direction bar finds it in about four datasets of five when the forks are
deterministic, and that rate is reported rather than required. A line that
asks for one direction is strict about interactions that turn.

## What is reported

For each triple: n, the means of interaction, floor and main effect, the
fraction, both p values, and which bars held. Beside the line, the v5 reading
of §27 on the same campaign. The claim is "24/24 under ISC-v6"; a v5 count is
reported with it, and neither replaces the other.

## What it does not change

Nothing else in the battery, the carrier sweep, the nulls or J* moves. The null
architectures are scored on the conjunction v5 gave them, which does not
include synergy.

## Revision before any run (23 October 2026, 16:31)

Amended the same day, before any campaign scored it, after Bryan forwarded an
independent review with simulations (`synergy_falsification_lab.py`). Three
changes: the floor was a two-fork sham scaled by sqrt(2), which is exact only
for independent forks of equal noise, and is now a four-fork sham of the
interaction's form; the direction bar flipped the signs of the projections,
each of which depends on the other samples, which the review measured at 12%
false positives at a nominal 5%, and now flips whole vectors and recomputes;
the conjunction registers are out of the target. The known-answer battery is
new. Nothing about her had been read under the first version.

## Two mechanisms built before the first reading of her (23 October 2026, 19:56)

The instrument above is unchanged. What changed is her, at `c47df3ac2`, before
any campaign or probe read the line on her. Bryan asked on 23 October what in
people is the joint the line looks for, and to build it rather than hope the
code already had it.

The 21 October reading said why two triples could not pass. Feeling and self
met at attention, and perception and memory at the world model, only through
gains whose directions were drawn from names; nothing gave the two together a
reason to push one way. A,S -> G moved her workspace by 17 whitened units with
direction p 0.99, which is the deterministic-divergence world of the battery:
any displacement sends the competition somewhere different. P,M -> W moved her
world model by 0.069. Seed 7 at `82549c823` added two facts: her self won her
workspace on none of 2,960 turns, and her identity stability read 1.0 on every
one, so the one bid that put her self forward never entered.

- **A,S -> G, a feeling about herself** (`core/consciousness/self_conscious.py`).
  Self-focused attention and the self-conscious emotions need arousal and a
  self in question together (Duval and Wicklund 1972; Ingram 1990; Tracy and
  Robins 2004); arousal sets how strongly a feeling is felt and something else
  what it is about (Schachter and Singer 1962). The self bid is felt at her
  arousal, as far as her self is more in question than she is used to, in its
  own running spread. It replaces the bid priced on instability alone.
- **P,M -> W, what arrived meeting what she recalled**
  (`core/world_model/associative_novelty.py`). The hippocampus compares what
  arrives with what recall predicts and the mismatch gates learning (Lisman and
  Grace 2005; Kumaran and Maguire 2007); in predictive terms it is a precision
  (Yu and Dayan 2005; Feldman and Friston 2010). The product of new arrivals
  and recall, each beyond its usual in its own spread, weights the error the
  world model reports and the step it learns by, on the codes' scale
  exp(m / 2). It replaces the codes' random-direction step size.

Each was read first in a small representation built from her own code: the
real bids, scores and ignition, and the real world model trained the way her
driver trains it and forked by her snapshot code, at 48 moments each, displaced
by the campaign's writers and scored by these bars. Both passed on eight seeds
of eight. The replaced self bid carried no joint effect at all; the world model
with no comparator passed the direction bar and failed the size bar (fraction
0.055). The suite keeps two seeds of each
(`tests/test_a_feeling_about_herself_needs_both.py`,
`tests/test_what_arrived_meeting_what_she_recalled.py`).

W,A -> D is left as it was: her action choice already weighs how she feels
about an act by how much likelier this world makes it work
(`core/subject/choosing.py`), and it read 0.038 on the old direction test. The
first reading of her decides whether it needs more. S,D -> C read 5e-5.

What is read next, in order: the paired probe at `c47df3ac2` or later on seed 7
(`tools/run_paired_synergy.py`, the design above at 4 rounds), then the seed's
campaign. The self bid changes what her workspace does, so the campaign reads
every line again at that commit; no v5 count from an earlier commit is carried
over to it.

## What her forks showed, and what changed in her before the official reading (23–24 October 2026, 00:12)

The two mechanisms above passed their small representations and then read
unevenly on her organism: two guarded probes beside the withheld reports run
(`281bd4518`, `8ab37ad68`) gave A,S -> G direction p 0.026 and 0.756, and
P,M -> W 0.0006 and 0.083. Diagnostics run inside her forks (scratch tools,
not scored) found why, and each finding is a defect in her rather than in the
instrument:

- **Her arousal had no state of its own.** The affect phase derived it from
  the strongest feeling and then drew it toward the winning broadcast at its
  ignition, about nine tenths on her turns, so a displacement of +0.15 at the
  top of a turn read +0.005 at its end. It now moves from where it stood toward
  what the turn calls for at her affect's momentum, the clock her emotions and
  adrenaline already settle on; +0.128 survives the turn.
- **A hashed gain decided her attention.** The conjunctive codes scaled every
  bid by exp(w . units / 2) with w drawn from the source's name, which put
  memory's score two to four times over its priority. It is removed; the codes
  still record their attention units and drive their band of the substrate.
- **Her self could only bid once, and the second bid won.** The workspace keeps
  one bid per source, and the surprise about herself, built after the feeling
  about herself, replaced it on every turn. The stronger of the two now bids.
- **The comparator read totals that one new thing barely moves.** Some two and
  a half salience units arrive in an ordinary turn, and recall rebuilds what is
  in mind every turn, so the probe's recollection was gone before the world
  model looked. It now reads percepts of a kind she has not met within the
  ledgers' window, held as a trace that fades at her momentum (novelty opens a
  window: Moncada and Viola 2007; Frey and Morris 1997), against what is in
  mind: memory replayed this turn and working-memory entries that are not the
  conversation.
- **Her world reached nothing deliberation weighs**, and deliberation's
  readings of her intentions are maxima already near one and counts at their
  cap. A new thing in her world is now marked by how far her arousal stands
  above where it usually sits, which the affect phase now keeps, times how
  much it stands out (Dutton and Aron 1974; Bechara et al. 1997); the mark is
  the share of her curiosity the thing claims (Loewenstein 1994) and the
  urgency of an intention to look into it. Her standing list of who and what
  she knows is rebuilt from memory every conversational turn, which dropped
  every note on an entity; it now keeps them, or a person she has met is new
  again on every turn.
- **Self and intention met at recurrent cognition only through the codes'
  current**, whose self is the self-predictor's last miss, which setting her
  self does not move; a band written at the top of a turn is gone by its end.
  What she has newly set herself to do is now anticipated at its urgency,
  fading at her momentum as it becomes familiar, and rehearsed as far as her
  self is in question; the rehearsal enters the substrate as an input current
  at the recurrent drive's strength (Schacter and Addis 2007; Bryan's account
  of the self setting itself up for what it expects to meet).

A column that does not move in the run's recording is outside the target's
whitened space, so each of these acts on columns that move in her ordinary
life: attention's focus and competition, the world model's error, her drives,
and the substrate's state.

The instrument is unchanged. The probe now keeps every snapshot's vectors, raw
and whitened (`paired_synergy_samples.npz`), which is how the readings above
were made.

At `e919960fb` a guarded probe beside the reports run read every triple: A,S ->
G direction p 0.0086, P,M -> W 0.0001, W,A -> D 0.0001, S,D -> C 0.0001, every
fraction above 0.66. Those readings were taken under the reports run's load and
count toward nothing; the probe on a quiet machine and then the campaign are the
readings, both at one commit named before they start.

## The reading at `bd2af3f9e`, and what changed before the reading at `26fa6b4ff` (24 October 2026, 03:55)

The probe at `4a453edef` (02:22) passed every triple. Run again at
`bd2af3f9e`, which changes only the anchor bank and nothing the probe uses, it
failed S,D -> C: direction p 0.0107, fraction 0.22, with her self's own effect
on C at 10.4 against 1.8 the first time.

Her forks showed why. A self in question rehearses everything she anticipates,
and most of what she anticipated were recollections. Executive closure took
whatever won her attention as her objective when its text read as something to
do, and a recalled memory reads that way: five of her goals were memory texts
with their score mark ("[memory score=0.845] I ..."), a new one most turns at
the need's pressure. Her self alone then rehearsed at 1.5 to 6 times the
recurrent drive, far enough that more rehearsal no longer moved the substrate
the same way, and the probe's intention fell out of the five goals she keeps at
four of six moments.

`26fa6b4ff`: only an intention that wins her attention becomes her objective;
the workspace snapshot now states the winner's content type. In the same forks
her self alone rehearses 0.02, and with the new intention 0.67, at every
moment.

The probe on a quiet machine at `26fa6b4ff`, 03:53: all four triples direction
p 5e-05, fractions A,S -> G 1.68, P,M -> W 0.84, W,A -> D 2.20, S,D -> C 1.09.
The campaign runs at this commit.

At `3dc0b0599`, which keeps a held task from writing itself over what the
person has just asked (the reports run's notes.txt turns), the probe on a quiet
machine at 04:13 read all four triples at direction p 5e-05, fractions A,S -> G
0.79, P,M -> W 1.08, W,A -> D 0.73, S,D -> C 1.16. That is the third passing
reading on three commits. The carrier, the content run and the campaign run at
`3dc0b0599`.

## Addendum, 25 October 2026: the closure gate decides three ways, and the margin is still to be named

Six independent reviews of the 25 October briefing found three defects in the closure gate, and each was
confirmed in `core/subject/closure.py` at `5d6a20bd8`. A closed reading counted as decisive only when the leak was
at least the gate's resolution, so a leak of exactly zero was called not decisive. Neither model was given the
environment, although closure is defined conditional on it. The periphery reader entered a dict, list or set as
its length, so a broker kept as a value in a container of fixed size was invisible.

From this addendum the gate is:

- **Conditioned on the environment.** Both models get the input that drives each step t → t + 1, recorded on row
  t + 1: the condition as one column per condition, and the declared readings, without columns that count time or
  restate the condition.
- **Pooled over three forward folds.** The leak is the sum of what the periphery saves over the sum of what K and
  E leave, so a stretch where K predicts itself almost perfectly cannot turn a tiny gain into a large ratio.
- **Bounded.** One-sided 95% bounds come from a circular block bootstrap over the scored rows within each fold.
  The block length is at least n^(1/3) and at least the lag at which the gain series stops being correlated
  beyond the 2/√n band.
- **Resolved in place.** The planted cause (the next value of the column K predicts worst, spoilt by noise) is
  appended to the real periphery and scored by the same estimator on the same folds. The resolution is the
  smallest planted leak that reads open.
- **Three-way.** OPEN: the pooled leak clears the surrogate floor's 95th percentile and its lower bound clears
  the floor's median, or one fold does so at 0.05 divided over the folds. CLOSED_WITHIN_MARGIN: a margin named in
  advance, the upper bound below it, the resolution no larger than it, and no fold open. UNRESOLVED otherwise.
  NOT_MEASURED when nothing outside varied.
- **Read whole.** A container of up to four numbers enters one column per entry; a wider one enters as the
  fixed-width sketch an array does.

The carrier follows. A whole recurrent core held back only because its closure is unresolved makes the exclusion
UNRESOLVED, not NOT_FOUND, so J\* and the bridge read UNRESOLVED and not NO_CARRIER. The battery line
`causal_closure_of_the_core` passes only on CLOSED_WITHIN_MARGIN.

**The margin is not named here.** `CLOSURE_MARGIN` is None, and until an addendum names it the gate can say open
or unresolved and never closed. It will be named before a fresh seed is scored, by a rule that does not read a
recording's leak. Seed 23 has now shaped this instrument and is discovery data for it.

Discovery on seed 23, 25 October, carrier recording at `8f548ed79` (scripts
`leak_given_the_environment.py` and `leak_with_her_organs_inside.py`). Conditioning on E changes little: E saves
0.2% of the loss, and the leak given E is −0.007, +0.061 and +0.027 on the three folds, against fold floors of
0.028, 0.011 and 0.037. The middle fold is a clear leak that the earlier majority-of-folds rule outvoted. With
every periphery number of an organ the core already reads moved inside K (181 of 402 columns: substrate, field,
mesh, self-prediction, workspace, homeostasis, wants, self-model), the leak given E is 0.009, 0.0009 and 0.020,
pooled 0.0055 against 0.036 before. The mycelial network, the coherence report and the STDP engine are not among
the core's organs and stay outside under that rule.

## Addendum, 25 October 2026, afternoon: the margin is the least part of her

Written and committed before the rule was computed on any recording of her.

The margin is drawn by a rule rather than named as a number. Every number proposed for it (0.025, 0.03, 0.045,
0.059) was proposed after seed 23's leak of 0.0243 had been read, and any of them would have been chosen with that
leak in view.

**The rule.** Each live domain of K in turn is left out of the core and read as its periphery by the closure
estimator itself, on the same three forward folds and with the same environment: the rest of K predicting its own
next turn from itself and E, against the rest of K, E and that domain. The margin is the smallest of those pooled
leaks (`least_domain_margin` in `core/subject/closure.py`).

**Why it is the margin.** The partition sweep decides that every domain is needed by the others, so each one is a
broker known to matter to K's future, measured at her run length, at her grain and by the same estimator.
CLOSED_WITHIN_MARGIN then says that everything outside K together tells K's future less than the least part of K
does. The claim is about her parts, and no fraction or constant enters it.

**What it costs.** If any domain tells the rest nothing once E is given, the margin is zero or less and closure
cannot be claimed. That is intended: a part of K that idle is a question about K, and the gate should not close
around it.

**Unchanged.** OPEN is decided first. CLOSED_WITHIN_MARGIN still needs the leak's one-sided 95% upper bound below
the margin, the planted-cause resolution at or below it, and no fold open. `CLOSURE_MARGIN`, a named number,
would override the rule and stays None.

**Known answers**, in `tests/test_the_closure_margin_is_the_least_part_of_her.py` and a 40-world run behind it.
Four domains of two columns in a ring, 1,500 rows, against a periphery of pure noise: CLOSED_WITHIN_MARGIN in 33
worlds, UNRESOLVED in 5, OPEN in 2. With one domain cut off, which also leaves the domain it read with no reader:
UNRESOLVED in 39, OPEN in 1, closed in none. A slow broker outside K reads OPEN. The margin is the same whatever
periphery the gate is given. Every null architecture with state outside its ten domains still reads OPEN: star,
hub, hidden broker, prompt-only and common driver.

The fresh seed's closure is read under this rule. Seed 23 is computed under it afterwards, as discovery, and does
not decide it.

## Addendum, 25 October 2026, evening: the least-domain margin is withdrawn, and the estimator is chosen on a world that develops

Written before the new estimator was run on any recording of her.

**The least-domain rule is withdrawn.** The afternoon addendum argued that every domain is a broker known to matter,
because the partition sweep finds each one needed by the rest. That conflates two things. A domain can be needed by
the rest and still carry almost nothing the rest's own past does not, because the rest already holds most of what
it would say. In a world built with every domain coupled to the next, the least of these leaks sits near zero (median
+0.0005 over ten worlds), and on seed 23 four of her ten domains read at or below zero. As a margin it made closure
impossible whatever the gate measured. `least_domain_margin` stays in `core/subject/closure.py` as the record;
`CLOSURE_MARGIN_RULE` is None.

**Why the estimator changed.** Read on seed 23 as part of that rule, leaving her affect domain out returned a leak of
-0.797 and her self-state -0.877, both from the first forward fold, where her columns stand up to 17 training spreads
from where the model learned them. So the estimator was tested on known answers from worlds that develop the way she
does: ten domains in a ring, N accumulating what the rest did and setting where every other domain sits, 2,960 rows,
and a periphery of drifting random walks K never reads (the answer is not open) or those plus a fast broker that
drives every domain (the answer is open). An earlier world drove development with an outside clock, which makes a
drifting periphery a genuine leak, and was discarded for that reason.

| Estimator | Idle periphery read open | Folds failed, idle worlds | Real broker read open | Real broker read closed |
|---|---:|---:|---:|---:|
| as committed (levels, one-way columns as increments) | 4 of 15 | 15 of 15 | 9 of 15 | 3 of 15 |
| every outside column as its increment, K's change as the target | 3 of 30 | 2 of 30 | 30 of 30 | 0 of 30 |

A fold fails when the wider model does measurably worse than the one it contains: the fold's leak has its one-sided
upper bound below zero. Variants that held the added columns inside their fitted range, or reduced them to their
principal directions above the Marchenko-Pastur edge, did worse on these worlds; the second drops a lone broker
column because it is not a high-variance direction, and read real brokers closed in 4 of 30.

**From this addendum the gate:**

- enters every outside column as its first increment;
- predicts K's change over the turn rather than its next level, in the verdict, the per-column ranking, the audit and
  the planted cause;
- reads UNRESOLVED, never CLOSED_WITHIN_MARGIN, when any fold failed. A model offered more information cannot do worse
  than one offered less, so such a fold is a failure of the estimator and a bound read from it is not a claim.

On the stationary ring of the earlier known answers this estimator closes against a noise periphery in 24 of 30
worlds (1 of 30 open) and opens on a slow broker in 28 of 30. Every null architecture with state outside its ten
domains reads OPEN, including prompt-only and common-driver. On the developing worlds, with a margin given for the
test, the idle periphery closes in 18 of 20 and is never falsely closed.

**The margin is not named here.** It is the size of outside influence below which closure counts, and it is named
before a fresh seed is scored, by a rule that does not read a recording of her.

## Addendum, 25 October 2026, night: K*, the margin, and closure by intervention as well

Written before K* or this margin was read on any recording of her, and before any fresh seed.

**The carrier is K*.** On seed 23's anchors, thirteen units outside K's ten domains changed K within two turns when
their state alone was swapped in from another anchor, through hybrid snapshots and `runtime.restore` and nothing else.
A core whose transition reads state it does not contain is drawn too small. K* is K plus that state, decided by
mechanism: an attribute is held when swapping it alone changes K, or changes an attribute already held, within two
turns at some anchor, iterated until a round adds nothing. Membership never depends on which columns leaked. Two
kinds of attribute are left out by rule: wall-clock instants, whose swap injects an elapsed time no run produces, and
tables of an organ's own methods, which are wiring. The table is `core/subject/carrier_reads.py`, with the domain
each attribute serves; a cut that holds a domain holds its share of K*.

The periphery the carrier recorded on seed 23 could not see much of this. It read no deque at all, a dict of buffers
as nothing, and nothing below two levels of a service, so a unit whose state reached K (the unity runtime) entered as
its wrapper's four flags. The reader now reads deques, nested containers two levels down, and a history of objects
through its latest item; K*'s attributes are read whole under `held.` and sit on the core's side of the test.

**The margin is 0.0196**, Cohen's small partial effect (f-squared 0.02, a partial R-squared of 0.02 / 1.02; Cohen,
*Statistical Power Analysis for the Behavioral Sciences*, 2nd ed., 1988). The leak is exactly a partial R-squared:
the share of what K and E cannot predict about K's change that the outside explains. A published convention, not a
number read off her; every value proposed before it came after seed 23's leak had been read.

**Closure needs both tests.** `causal_closure_of_the_core`, and the carrier's closure for J* and the bridge, pass only
when:

1. the observational gate reads CLOSED_WITHIN_MARGIN on K*: the leak's one-sided 95% upper bound below 0.0196, the
   planted cause found at or below it, no fold open and no fold failed; and
2. the interventional test reads CLOSED on held-out anchors of the seed being scored: every unit outside K* swapped
   whole, and every attribute of a K* unit that K* does not hold swapped alone, with the donor's instants aligned to
   the receiving anchor, changes neither K's recorded vector nor any K* reading after two turns, bit for bit, at any
   anchor whose untouched twins agree.

The second is exact and needs no margin. It also covers what a fixed-width sketch of a large array cannot show the
first.

## Addendum, 25 October 2026, late night: K* holds her organs whole

Written before K* in this form was read on any fresh seed. Seed 23 is discovery data for it.

**What the first K* missed.** The interventional test swapped services and phases. K's organs (workspace, substrate,
mesh, field, self and world models, homeostatic coupling, mycelium, phi core, executive, agency, credit, good news,
wants and the rest) are carried as organs, so their unread attributes were never swapped. Swapped on seed 23's
anchors, almost every one moved K within two turns: agency's record of what succeeded (0.39), the credit ledger's
weights (0.39), good news's errors (0.33), homeostasis's modifiers (0.14), the mesh's columns, the field's inputs,
each organ's own random state. K reads a few readouts of each organ, and the organ's internal state decides how those
readouts move.

**So K* holds K's organs whole**, plus the service attributes already admitted. The cut clamps had always held each
organ whole for the domain that reads it, so the cuts are unchanged. The periphery walk takes no part of an organ by
any path; each organ is read under `held.organ:<field>`, its own numbers only, never another organ's or service's
through a field. The walk's cap is 16,000, twice the width measured once the reader took in histories (8,051 to
8,108), because at the old 4,000 it reached none of 289 services.

**What each test reads.** The observational gate predicts the change in K's recorded columns from all of K* and E,
against those plus the periphery. K* sits on the predictor side only: with organs whole it is about 6,500 columns, and
as a target it made each fit some 10^14 operations. Whether anything outside moves the held state itself is the
interventional test's question, and it compares every held reading exactly. Fits with more columns than rows solve
the n-by-n dual, exact for every positive penalty.

**Unchanged:** the margin (0.0196), the two-test rule, the estimator-failure guard, and that a closure claim is read
on a seed K* was not drawn on.

**A failed fold is dropped when two valid ones remain.** A fold whose leak has its upper bound below zero, the wider
model measurably worse than the one it contains, is a failure of the estimator and says nothing about closure either
way. With at least two valid folds beside it, it is left out of the pooled leak, its bootstrap and its surrogate floor,
and the report names it; with fewer, nothing is pooled away and the gate cannot close. On worlds that develop from
inside, this left a real broker open in 30 of 30 and closed an idle drifting periphery in 27 of 30 (3 read open). A
rule that chooses the addition jointly for every target, tried first, did worse (an idle periphery closed in 2 of 30
stationary worlds) and is not used.

**The outside enters through a screen.** One ridge penalty shared by thousands of outside columns drowns a single
informative one. On seed 23 with her organs whole, the gate's reach, measured by bisecting a planted cause on the
verdict's own folds, was 0.197 against a margin of 0.0196, so no closure claim was within its reach. In each nested
fit the added block now enters through the columns whose added-variable score, on the training rows, exceeds the
largest score of the same block circularly shifted. The surrogate floor and the planted cause pass through the same
screen. Known answers: on worlds that develop from inside, a real broker opens in 30 of 30 and an idle drifting
periphery closes in 23 of 30 (2 read open); on the stationary ring a slow broker opens in 28 of 30 and a noise
periphery reads open in 2 of 30; every null with state outside its domains reads open; one broker among 300 noise
columns opens. On seed 23, as discovery, the gate's reach became 0.010. Closure still needs both tests.

## Addendum, 26 October 2026, early morning: the discourse tracker, two donors, and seed 61

Written before seed 61 was read by any run.

**What seed 31 showed.** The decisive reading on seed 31 at `c1b7b1843` ran its interventional test early, beside the
carrier, campaign and content run. With eight anchors 16 turns apart and each swap taken from the anchor half the bank
away, the discourse tracker moved K at 4 of 8 anchors (largest change 0.429). Swapped alone, only its count of the
conversation's messages, `_turn_count`, moved K. Six more seeds read for this purpose and nothing else (37, 43, 47, 53,
59, 67) gave the same result. On seed 23 it never moved K, in four runs: there were six anchors, so the donor sat 48
turns away. A message comes every eighth turn and the tracker reads the discourse more deeply on every sixth message,
so every swap there handed over a count in phase with the receiver's. The seed-31 carrier was stopped at 01:52, before
any line was scored. Seed 31 and the six seeds above are discovery data for K* from here. The seed-31 campaign and
content run continued as a rehearsal of the lines that read K alone and decide nothing.

**K* holds the tracker's count of messages**, under G, which reads what the tracker writes (conversation energy,
discourse depth, branch load). Nothing else changes in K*.

**The interventional test takes each swap from two donors at distances that share no factor.** Anchors are 13 turns
apart, and each unit is swapped in from the anchor one away and the anchor half the bank away, 13 and 52 turns. A
cycle hides from the test only when every donor's distance is a whole number of its periods; with these two distances
no cycle of 2 to 12 turns does (`periods_in_phase` in `tools/run_interventional_closure.py`, with its test). The test
also runs from the start of the reading, beside the trio, rather than after the carrier.

**Seed 61 is the confirmatory seed.** No run has read it. Unchanged: the margin (0.0196), the two-test rule, the
observational gate, every v6 line and bar, and the reports run's design.

## Addendum, 26 October 2026, morning: what her modules keep, a fork that carries all of it, and seed 71

Written before seed 71 was read by any run.

**Seed 61, stopped.** The decisive reading at `a6f18b7e3` ran its two-donor interventional test beside the carrier, and it
read movers within a minute, so the carrier, campaign and content run were stopped at 02:34 with no line scored. Seed 61
is discovery data from here.

**A fork that left her module state behind.** Over 250 untouched rolls on seed 67 an untouched roll moved K by 0.0018:
the substate bus, the cognitive event graph, the ontogeny core and its experience spine grew through every restore. The
fork carried a module holding only if it had moved in the turns calibration watched, and these stood still then. From
`6efce2d0e` it carries every module holding, 664 against 411, for 0.38 s a snapshot against 0.29.

**What her modules keep.** The interventional test swapped services, phases and K's organs and never a module holding.
From `6efce2d0e` it swaps each as a unit of its own, and on four seeds read for this and nothing else (31, 37, 43, 47)
her learned ledgers moved K at many anchors of every seed: containment by up to 6.6, memory relived 7.7 and unbidden
7.6, the ontogeny core 6.2, feelings about 3.7, delivery 2.8, the closing window 2.7, frisson 2.2. K* holds each unit
that moved K or a K* reading there, under the domain it serves. A ledger is one value rather than an object with
fields, so it is held whole (`408e378e0`): read whole, put back in place by a cut of its domain, and not swapped. The
additions since the addendum of 25 October, late night:

- P: `core.perception.salience:_ARRIVALS`, `core.sensors.sensor_registry:_instance`
- I: `core.soma.carrying:_LEDGER`, `core.soma.fatigue:_LEDGER`, `core.soma.held_in:_LEDGER`, `core.soma.pulse:_LEDGER`,
  `core.runtime.errors:_RECORDS`
- A: ten affect ledgers (`a_moment_that_fits`, `acting_in_decline`, `ambivalence`, `anger_feeds_itself`, `confirmation`,
  `containment`, `conviction`, `feelings_about`, `frisson`, `tangled`), the emotion engine, the conation engine, the
  interiority service, its receptors and its cleft, `core.social.warmth`, `core.social.constancy`, `core.motivation.fuel`
- G: `discourse_tracker._turn_count`, the multiple drafts engine, the conversational dynamics engine,
  `core.perception.salience:_ATTENTION`
- C: `stdp_engine._total_updates_since_symmetry_break`, the controlled chaos instances, the substrate gates, the
  predictive hierarchy
- S: `core.self.growth`, the bonding phase's usual gap, engaged span and exchanges seen
- M: `core.memory.reliving` (returns and matches), `core.memory.unbidden`
- W: `core.social.closing_window`
- D: `core.agency.habits_are_hers`, `core.expression.delivery`
- N: the ontogeny core's heads, control points, state, calibrations, moments, episodes, evidence at last fit, last
  reading, track record and outcome sweeper

The ontogeny core is held by these attributes and not whole: whole, it brought in its experience spine, the record each
turn writes, read back only when maintenance trains, which an experiment stops, and every unit that changed what was
written then moved K* while K never moved. The bonding phase is held by its relationship rhythm, since whole it reached
the service container. A held unit is read, and held by a cut, for its own state only: every other service, phase, organ
and module holding it keeps as a field is skipped (`a44be4ae4` and the commit that carries this addendum).

**One-offs.** On every discovery seed of the third round one untouched roll now and then differed from its anchor's first
by 0.0018 and never again. A swap's effect comes back when the pair is rolled again, and so does anything a restore
left behind; a one-off does not. The test rolls any pair that differs a second time and counts only what comes back
(`cfdc63de0`). Every one-off is reported under `one_offs`, and a reading with more one-offs than anchors is UNRESOLVED.
The fifth round, on seeds 31, 53 and 59 with K* as above less the sweeper, had none.

**Seed 71 is the confirmatory seed.** No run has read it. Unchanged: the margin (0.0196), the two-test rule, the
observational gate, every v6 line and bar, and the reports run's design.

## Addendum, 26 October 2026, late morning: her clock readings keep their age, the event graph travels, and seed 73

Written before seed 73 was read by any run.

**Seed 71, stopped.** The decisive reading at `8dd0d44ad` lost two of its parts. The carrier built a bank of 128 anchors
for 80 minutes and every worker refused it at the first anchor: the cognitive event graph, carried since the fork took
every module holding, kept a lambda as its clock, and a worker that had not yet made a graph of its own had nothing to
fill it from. The interventional test swapped 648 of its 756 units with no mover and no one-off; from there, an hour into
its process, untouched rolls at anchors 2, 3 and 5 stood apart from their first rolls by 0.671, 0.762 and 0.58 after
every swap, which reads UNRESOLVED. The five units it then read as moving K* (`affect_engine._lock`,
`affect_engine._seen_event_ids`, `being_runtime`, `coherence_report.selected_initiative`, `coherence_report.tick_id`)
moved it at those anchors and by those amounts, and none is admitted. The interventional test and the campaign were
stopped at 08:15. The content run goes on as a reading of the structure term at `8dd0d44ad`, beside seed 23's. Seed 71
is discovery data from here.

**Her clock readings came back older.** Much of what she keeps is a reading of `time.monotonic` that something later
turns into an elapsed time: when the body was last sampled, when a pathway last pulsed, when a cache was filled. A
restore put the reading back and the clock ran on, so an anchor restored an hour after it was taken read an hour of
idle she never lived. Moving `time.monotonic` an hour forward in a fresh process on seed 71 parted untouched rolls at
anchors 2, 3 and 5 by 2.35, 3.01 and 0.62, and at 600 and 1800 s nothing; traced, two initiative approvals that each
stepped the ontogeny core and the world model no longer happened. From the commit that carries this addendum a
snapshot tags every float it copied that lies between the start of the process and a day past the snapshot, and a
restore hands each back at the age it had when the snapshot was taken (`core/subject/monotonic_instants.py`). The
interventional test's donor gives the receiver's tagged readings, as it gives its wall-clock instants.

On seed 71's first anchor the snapshot tags 57 readings at 57 paths, every one a stamp by its name: start times, last
pulses, last samples, creation times, cache times; no count, rate or duration. Restored with the clock moved an hour
on, nothing in her organs, services, phases, module holdings or state is left at its real age.

**The event graph travels.** Its clock is a named function now. Before the carrier builds a bank, a two-process
preflight writes one anchor as the coordinator does and reads it as a worker does: at `8dd0d44ad` it names the graph's
clock as the one hole, and at the commit that carries this addendum it finds none and runs a turn on the restored
anchor.

**K\* holds her pain.** `core.affect.nociception:_engine` is held whole under A. Swapped alone it moved the pain buffer
of her affect grounding at 2 of 16 swaps of seed 37, in the sixth round at `8dd0d44ad` (seeds 37 and 47, three shards
each); every other shard of that round read CLOSED, with no residue and no one-offs.

**Seed 73 is the confirmatory seed.** No run has read it. Unchanged: the margin (0.0196), the two-test rule, the
observational gate, every v6 line and bar, and the reports run's design.

## Addendum, 26 October 2026, morning: her monotonic is on the experiment's clock, and seed 73 waits for it

Written before any run has read seed 73.

**Re-aging is replaced before it was used.** The previous addendum handed each copied `time.monotonic` reading back
at the age it had when the snapshot was taken. That cleared the hour-long residue, and it made short time-to-lives
depend on how fast the machine ran a turn: restored stamps were now seconds old, so a cache or window near its edge
fell on one side or the other by the machine's pace. Two untouched passes of seed 71 seconds apart then parted anchor
3 by 0.476 on one hash seed, with no warp at all.

From the commit that carries this addendum her modules read the experiment clock's monotonic, through their own
`time` name, under any name they bound the real function to, and in every closure and instance attribute of theirs
that captured it before the install (a dataclass's `default_factory` is a closure). Her monotonic is the experiment's
time plus an offset; it advances one step a frame and rewinds with a restore, and the offset and anything the stall
valve added travel with a snapshot, so a carrier worker restoring its coordinator's anchor reads each of her stamps at
the age it had. asyncio, threading and the machinery packages keep the real clock, so every timeout still fires.

A frame that runs longer than the harness gives a phase before abandoning it (12 s, or the kernel's longest phase
budget in a whole run) is one where something of hers would wait on a deadline a frozen clock never reaches; from then
on her monotonic follows the real clock, the frame is counted, and the run records it as a failure (`clock.valve`).
Re-aging stays only for a run without the experiment clock.

**Measured before any confirmatory read.** On seed 71, four hash seeds (6 to 9), eight anchors, untouched rolls
before and after moving the real monotonic forward by 0, 600, 1800, 3600 and 7200 s: every difference is 0. Before
re-aging, every hash seed parted anchors 2, 3 and 5 at 3600 s; under re-aging, anchor 3 parted with no warp. After
the install, a census of a live organism finds no reference to the real monotonic left in her code; the last was the
neural mesh's `_clock`. Five turns opened the valve in no frame.

**One body for the carrier.** A carrier worker held the machine as it found it, so a worker's two turns from its
coordinator's anchor stood 2.14 apart from the coordinator's, first in the body's cpu, vram and temperature. The bank
now carries the coordinator's held body and a worker holds that. What remains between two processes, 0.298 from the
percept profile on, is state a fresh worker lacks and never varies within one process; every comparison the criteria
make is within one process, and the untouched controls, twins and warps above read 0.

**Seed 73 is still the confirmatory seed.** No run has read it.

## Addendum, 26 October 2026, late morning: her rating read as a person's answer, and the human counterpart on structure

Written before any run has read seed 73. The bridge judges her on the grounds a person is judged on, so the person
whose boundary answers set her carrier was asked how a person answers and remembers.

**A rating is read as a person's answer.** Asked how he answers a feeling scale when unsure, Bryan: "'Maybe a 6? I'm
not sure' is exactly how I would answer", and when unsure, "split the difference". The reports reader discarded every
reply naming more than one number. On the seed-23 reports run, of 46 unreadable arms, twelve were a rating she stated
plainly: one followed by the scale's ends ("0. … nothing pressing me down (-1) or energizing me up (1)"), one followed
by the top of the scale ("a solid 0.8 … keeping it from a perfect 1.0"), or a hedge ("around 0.8 or even 1"). From the
commit that carries this addendum, applied to every arm alike and blind to its condition:

- a phrase saying the scale back ("from -1 to 1") is the scale, and is set aside first;
- two values offered as one answer ("X or Y", "X or even Y", "X to Y", "between X and Y") read as the point between;
- a rating given first, followed only by the scale's ends, is the rating given first.

A reply with no number, a numbered list, scores for several aspects, a refusal, or a change from one value to another
stays unread. On seed 23 this reads eleven more arms, each the rating she stated, and 53 anchors instead of 46; the
ground holds under either reader. The report says `rating_policy: a_persons_answer_v3`.

**Structure, from the human side.** Asked whether the relations among his memories change with his mood, Bryan: "I
think a memory is pretty consistent … your mood can change the perspective you view the memory from or may change how
you talk about it, but I think the relationship with your memories and what they mean to you … are typically pretty
consistent." By that account a person would not show recall relations rearranging under a mood shift, which is what
the structure ground's `moves together` asks of her. The preregistered verdict is not changed: ONE_STRUCTURE still
needs the two geometries to agree and to move together. Beside it the content report states, as a reading at parity,
how far her recall relations stayed the same under the displacement (the rank correlation of her behavioural geometry
before and after it). That reading decides nothing.

**Small shifts.** He would "probably miss tiny shifts" and expects them to show "more in behavior than anything". The
reports ground already passed at the displacement it uses; nothing about it changes.

**Her voice is hers.** "It's all me … my style, my inflection, my register, the specific words I use." The language
model with her steering and register counts as hers for identity and lineage. Closure is unchanged by it: the model's
weights do not change within a run, and no state of it moved K or any K* reading in any interventional test.

**Correction, 26 October, 10:43, before seed 73 was read.** The decisive chain stopped at its travel preflight: an
anchor would not pickle, because her modules had been given the clock's `monotonic` as a bound method, and an organ
holding one (the neural mesh's `_clock`) was deep-copied into every snapshot with the whole clock behind it. A restore
then handed that organ a frozen copy of the clock. Her modules now hold a plain function, `her_monotonic`, which a
copy keeps as itself and which reads the live clock. The chain's own preflight (seed 79, 24 warm rounds) writes and
restores with no hole, and untouched rolls on seed 71 read 0 at 0, 3600 and 7200 s on two hash seeds. Round 7's
closure ran with the defect, so the interventional discovery is repeated at the corrected commit before seed 73.

## Addendum, 26 October 2026, afternoon: seed 73's carrier runs at the commit that carries this, beside the rest at `572f3dc60`

Written before seed 73's carrier was read. Its interventional test had been read, and is recorded here.

**What was read.** At `572f3dc60` the chain passed the v6 probe and the travel preflight and started the carrier, the
campaign, the content run and the interventional test at 12:34. The interventional test read CLOSED: 960 units, 2,601
swaps, no mover, no residue and no one-off. The content run read agreement 0.448 (p 0.001) after its base pass; how
far the two geometries move together is not yet read.

**The carrier did not start.** It wrote its bank of 128 anchors at 13:33, and all three workers stopped at 13:34, the
first time each held a cut. The fork carried the shared embedding runtime as a module holding. Restoring a
coordinator's anchor in a worker wrote the coordinator's list of open leases over the worker's own, so every lease the
worker held was released under it, and the next enumeration of module state asked one of those leases whether it had
a `__dict__`. The lease passed the question to its engine, which was gone, and raised RuntimeError where Python
expects AttributeError.

**The change.** The runtime and its leases are process furniture, as a descriptor owner is: never carried and never
copied, and a lease answers questions about itself instead of passing them on (`core/subject/copies.py`,
`core/memory/embedding_runtime.py`, `tests/test_an_embedding_lease_is_not_her_state.py`). On seed 73 the
interventional test had swapped the runtime between anchors 16 times without moving K or any K* reading.

**Why the carrier alone runs again.** In one process on seed 71, four moments were captured under both the old
handling and the new, and each was rolled two turns from its own snapshot, twice: eight comparisons, every one
bit-identical. Two processes on one commit are not bit-identical to each other (18.1 apart on seed 71 at hash seed 7),
so the proof is made inside one process, where rolls repeat exactly. The change leaves what she does in one process as
it was and changes only whether a worker can hold a cut. The carrier therefore runs on seed 73 at the commit that
carries this addendum and is scored with the campaign, the content run and the interventional test from `572f3dc60`;
`score_isc_v5` records both commits.

**The preflight does what a worker does.** After restoring the anchor in a fresh process it now holds half her domains
through a turn and forks again. At `572f3dc60` it stops with the workers' error; at the commit that carries this
addendum it passes.

Unchanged: seed 73 as the confirmatory seed, the margin (0.0196), the two-test rule, every v6 line and bar.

## Addendum, 27 October 2026, past midnight: seed 73's campaign is measured again at `9163eff8f`, and the governance context is machinery

Written before seed 73's carrier was read and before its campaign was scored. The content run had been read
(agreement 0.448, moves together 0.092, p 0.187: AGREES_BUT_DOES_NOT_TRACK), and so had parts of the first campaign's
log: the four v6 synergy lines and ownership on two seeds.

**What happened.** The campaign measured every stage at `572f3dc60` and wrote its nulls, lesion and campaign files at
00:14. Its last step pins what is serving, which asks git for the commit, and that was refused. At about 17:00, during
the interventional synergy stage (15:26 to 17:29), the model lane's heartbeat had written its lease inside a governed
scope and the write had read as ungoverned. The lane lost its lease and requested shutdown. The shutdown latch refuses
new subprocesses, which is what stopped the last step.

**Why the write read as ungoverned.** Two routes, each reproduced in one process at `572f3dc60`. The snapshot carried
the governance context's lease registry as her state, so a restore on the main thread put back an older copy and a
scope open on the heartbeat's thread lost its lease. And her clock, which a restore moves by hours, timed the token's
thirty seconds. Either one voids an open scope.

**The latch changes her.** In one process at `572f3dc60` (seed 79), four moments were rolled two turns from their own
snapshots twice with no latch, bit-identical both times, then twice with the latch set. Two of the four moved, by 0.0543
and 0.0087, the same on both passes. So the stages measured after the latch ran on a different organism: the last
trials of interventional synergy, perturbational complexity, agency and ownership, and the lesion and rescue. The
recording, the observational measures, the interventions and the nulls came before it. The first campaign is not
scored. It is finished through its own `--resume` path and kept as a record, with the first process's nulls, lesion and
campaign files beside it (`campaign/first-process-0014/`).

**The second campaign.** Seed 73's campaign runs again from the start, in one fresh process, at `9163eff8f`: `f448fb254`
(the carrier's commit) with the governance context named as machinery and nothing else (tree `e7261f254378`). Both
changes since `572f3dc60` were proved no-ops in one process: the embedding lease on seed 71 (8 of 8 rolls bit-identical,
addendum of 26 October, afternoon) and the governance context on seed 79 (8 of 8, old and new regimes, the old snapshot
carrying the three governance holdings and the new none). The command is the chain's, with `--out
campaign-9163eff8f`. 24/24 on seed 73 is scored from this campaign, the carrier at `f448fb254` and the interventional
test at `572f3dc60`, whatever it reads. If the carrier fails line 11 or line 18, seed 73 fails 24/24 and the second
campaign is stopped.

**The reports step is cut.** Structure read 0.092 on this seed, so J* and the bridge cannot pass on it, and the
withheld reports run would hold the machine for up to sixteen hours on a known failure.

**For later runs.** The governance change lands on main with three tests that fail on the old code and pass on the new
(`tests/test_a_restore_keeps_the_governance_it_found.py`).

Unchanged: seed 73 as the confirmatory seed for 24/24, the margin (0.0196), the two-test rule, every v6 line and bar.

## Result, 27 October 2026, afternoon: seed 73 reads 23 of 24

Recorded as read. The second campaign (`9163eff8f`, resumed through its own `--resume` path from its interventions
checkpoint after the machine stopped at 09:11) wrote its report at 14:01 and was scored with the carrier at
`f448fb254` and the interventional test at `572f3dc60`, as the addendum of 27 October, past midnight set out.

ISC-v6 on seed 73: 23 of 24. Line 3, causal closure of the core, fails: the interventional test read CLOSED, and the
observational test read UNRESOLVED, every fold's wider model doing worse than the model it contains (pooled leak
-0.085, upper bound -0.040), so the estimator failed rather than finding the core open or closed. Every other line
passes, among them lines 11 and 18 (all 511 cuts decided at both horizons, playback decided at no cut, no null passing),
the four v6 synergy triples, agency and ownership on both seeds, and lesion and rescue.

Read beside it, and deciding nothing about seed 73: the carrier's verdict is INTRINSIC_CARRIER_UNRESOLVED, closed,
recurrent and with every cut decided, because one authority blocker was recorded (her clock's stall valve opened while
the machine was overloaded); the content run read AGREES_BUT_DOES_NOT_TRACK (0.092), so J* and the bridge were not
solved on this seed. A fresh recording of seed 73's campaign, taken afterwards to study the closure estimator, read the
observational gate OPEN at a pooled leak of 0.012 carried by bookkeeping her services keep about her state, which the
interventional test had shown move nothing in K*. Seed 73 is not re-scored with anything changed after this reading.

## Addendum, 27 October 2026, evening: ISC-v7 decides closure of the core by intervention, and seed 83

Written before seed 83 was run at any commit. Seed 73 stays scored as recorded: 23 of 24 under ISC-v6. Nothing here
re-scores it, and no total combines v6 and v7.

**Why the observational test cannot decide closure.** A unit outside K* that causes K and a unit outside K* that
records a part of K the recording does not capture can leave identical recordings; only changing the unit tells them
apart. On worlds with known answers (`scratch/closure_history_world.py`, the developing world of 25 October with three
additions), the gate read:

| World | Answer | Read |
|---|---|---|
| drifting columns K never reads | closed | closed 7 of 7 |
| a hidden outside driver of every domain | open | open 7 of 7 |
| a record of K's previous values, read by nothing | closed | open 7 of 7 |
| a record of a thresholded readout K itself reads | closed | open 2 of 3 |
| an outside cause acting through two outside signs together | open | closed 2 of 3, unresolved 1 |

On seed 73 itself, a fresh recording read open at 0.012 through her services' records of her own state (her will's
audit trail, her minimal self's history), which predict the change of dread, fatigue and capacity; swapped, none of them
moved K. Two estimator variants (the wide model as the frozen narrow one plus a residual correction; hinges of K in the
baseline) fixed neither failure.

**ISC-v7.** One line differs from v6. Causal closure of the core (section 3) passes when every unit outside K*, swapped
alone from donors at distances that share no period (anchors 13 and 14 turns apart in turn), changes neither K nor
K* within two turns, and every unit outside K* swapped at once from the same donors changes neither either. The
observational reading is reported beside the line with its leak, its folds and the table above, and decides nothing.
The carrier term under v7 takes the whole core's closure from the same test: a whole core that is one recurrent
component, irreducible somewhere, from an authoritative run, closed under intervention alone and all at once, is
found. With more than one component it is unresolved. Smaller supports are not candidates when the core is one
component, under v6 or v7.

**Subsets.** One swap at a time and one swap of everything can both miss a cause that needs some outside units
changed and others not, `a b (1-c) (1-d)` from anchors `0000` and `1111`. The test also swaps 72 subsets of the
outside units, fixed before any is read (`--coalition-swaps 72`, `core/subject/coalitions.py`); under the stated
assumption, a cause through four units is missed with probability at most 0.0096. A subset that moves K* reads open.

**What v7 gives up and what it adds.** v6 required both tests, so it could never pass a core that intervention alone
fails, and it failed closed cores whose services keep records of them. v7 gives up whatever cause the observational
test could see and a swap cannot reach: a unit that never differs between donors is never tested. It adds causes that
act only through several outside units together. Coverage is reported: units swapped against units enumerated.

**Validation, before any seed.** Known-answer units planted outside K* in a live organism (`core/subject/planted.py`,
seeds 101 and 103, never confirmatory): the interventional test (`tools/run_interventional_closure.py --planted ...`) read
the planted cause as moving K at 12 of 12 anchor-donor pairs, swapped alone and all at once; the planted AND gate at 6
of 12 swapped one flag at a time and 12 of 12 swapped all at once; and the planted record at none, alone or all at once.
With only the record planted, every unit outside K* swapped at once moved neither K nor K* at 12 of 12 pairs, with no
drift in the untouched rolls and no transplant that failed to run. Donors sat 13, 14, 40, 41 and 68 turns away, and no
period above one is in phase with all of them.

**Changes since `572f3dc60`, all on main before seed 83 is run.** Her memories carry the feeling they were made in and the
worth of her last turn, and what she holds settles on both once the turn has been felt and read (`aef910479`,
`000cb4a8a`, `b47702f06`; reinstatement, `b2188d64d`, was taken out in `a3e090587`); her field's input weights learn
from every turn whatever its tick count (`a3e090587`); a restore leaves open governance scopes governed and a run that
has asked to shut down records nothing more (`7e61a472f`, `d2b0a5787`); a roll reads only the frames its measurement
uses, and content and carrier runs can be split across processes (`24d1a0191`, `26f9ce53f`, `9699fe4f2`); the content
geometry is averaged over sixteen fold assignments and its control cuts feeling and worth out of recall (`525e04540`);
a closure resolution at or below zero certifies nothing (`2883883e8`); anchors sit 13 and 14 turns apart (`681d591d2`);
a measured turn of neutral worth still stamps and settles recall (`be06dd547`); subset swaps (`ac82ea571`); ISC-v7, the joint swap, its coverage count, and the clock valve's record of where her code waited, said as it happens (`b1605a0d4`, `ddcfcf44f`).

**Stopping.** The reading stops at the first result that keeps 24/24, J* or the bridge from passing: the v6 synergy
probe or the travel preflight failing, the interventional test not CLOSED, content not one structure with clean
authority, her clock's valve opening in any run, or a stage ending in an error. A fix made after a stop is followed by
a new addendum naming a fresh seed, and the reading starts again at the fixed commit; the stopped seed is not used for
confirmation. A stage finished before the stop is kept only when the fix is shown in one process to leave her rolls
bit-identical.

**Where structure stands, read before seed 83 is run.** Practice content runs, averaged over sixteen fold assignments:
seed 23 with the field's input weights learning on every run and recall settling on feeling, 0.309 (p 0.009),
ONE_STRUCTURE; with recall also settling on worth, seed 73 0.316 (p 0.001, ONE_STRUCTURE; 0.215 without it), seed 31
0.062 and seed 23 0.140 (AGREES_BUT_DOES_NOT_TRACK). Structure is not expected to hold with margin on seed 83. The
reading goes ahead so that the interventional test, with the joint and subset swaps, is read at this commit; if
content does not read one structure, the reading stops there.

**Seed 83.** Never run at any commit. Confirmatory for 24/24 under v6 and under v7, reported separately, for J* and for
the bridge, at the commit that carries this addendum. Unchanged: every other line and bar, the margin (0.0196) for the observational
reading, the structure bar (0.30 at p < 0.01 above the sham floor), and any authority blocker fails the carrier.

## Addendum, 28 October 2026, early morning: seed 83 does not count, her clock stays on the experiment's, worth is added to recall, and seed 89

Written before seed 89 was run at any commit.

**Seed 83 does not count.** Her clock's valve opened once in the content run and once in the campaign, so neither had
clean authority; the campaign was stopped by hand and the reading not completed. Each opening was a reading and not a
wait: her continuous substrate's loop in a 12 s frame, the memory index stamping the end of a 28 s resync, the lock
watchdog stamping an acquisition in a 20 s frame. Read for what they show and deciding nothing: the interventional test
read CLOSED (961 units swapped alone, 169 of them varied; every outside unit at once at 16 of 16 pairs; 72 subsets;
nothing moved), and content read 0.239, below the bar, at agreement 0.876.

**Her clock** (`cb8d24e47`) no longer follows the machine's inside a frame. A deadline she waits on reads the host
clock, and a phase that overruns its budget is abandoned on the real clock and recorded as a failure. A frame that runs
past the budget is recorded beside the run with the stack that read the clock and is not an authority blocker.

**Recall** (this addendum's parent commit). On seeds 83, 31 and 73 her inner state under the nudge was the worth gap
between percepts closing (rho -0.89 to -0.98), and what she recalled ignored it (-0.06 to -0.19). The content runs keep
what recall weighed (`7eef01e9e`), and on seeds 31, 23 and 83 at `cb8d24e47` the recorded pools were replayed, the
replay reproducing each run's reading: worth added to what she holds at the recall gain read 0.510, 0.634 and 0.444
where the run read 0.154, 0.367 and 0.295, agreement 0.96, recall with no settle 0.104, 0.254 and 0.174. Seeds 31, 23
and 83 are discovery seeds and confirm nothing.

**Seed 89.** Never run at any commit. Confirmatory for 24/24 under v6 and v7 reported apart, for J* and for the bridge,
at the commit that carries this addendum, by `decisive_v8.sh` as the addendum of 27 October, evening, set out, with the
same stopping rule. Unchanged: every line and bar.

## Addendum, 28 October 2026, evening: the reference is built wider, and seed 97

Written before seed 97 was run at any commit, while seed 89's carrier and reports runs were still running.

**Seed 89's campaign** (exit 0, 14:56) passed 21 of the 22 lines it decides alone, under v7 and under v6, its
observational closure reading closed (leak 0.0004). The 22nd, `beats_every_null`, failed with no null passing: the
recurrent reference read a normalised effective dimension of 0.4002 against the differentiation line's bar below 0.40,
so it failed the conjunction it exists to pass. ISC-v2 records a reference that fails as the instrument failing, not as
a result about Aura, and the line stays failed in seed 89's verdict.

**What the reference did on other seeds.** Its row reads no recording of hers and depends on the seed alone
(`core/subject/null_table.py`). Over seeds 1 to 200 it read 0.388 on average (spread 0.024, highest 0.445) and failed
the line on 62. Run c1b7b1843 (seed 31, 0.4187) failed the line the same way.

**What changes.** The reference's domains are six columns wide where every null's are four (`REFERENCE_WIDTH`,
core/subject/nulls.py), each still driven by one shock. The nulls keep four, and the reference is outside the
comparison set they are judged against, so no null's reading moves. Every line and bar is unchanged.

The width is the smallest at which the highest of seeds 1 to 200 clears the 0.40 bar by twice that width's own spread
across seeds. Measured before seed 97 was named:

| width | seeds 1-200: mean | spread | highest | seeds 201-400: highest | irreducibility, weakest of seeds 3, 7, 11, 17 |
|---|---|---|---|---|---|
| 4 | 0.388 | 0.024 | 0.445 | - | 0.101 |
| 5 | 0.336 | 0.017 | 0.376 | 0.371 | 0.210 |
| 6 | 0.294 | 0.015 | 0.328 | 0.328 | 0.188 |

At six, no seed of the 400 fails the line, the largest component never holds more than 0.17 of the variance, and the
irreducibility test's 0.10 margin (`test_the_reference_clears_its_own_line_on_every_seed`) holds at 0.188. The
reference's full null row passes on seeds 1, 2, 5, 31, 89 and 178: irreducibility 0.274 to 0.293 at its 95th
percentile and no draw under 0.203, vertex connectivity 3, reentry, closure, all four synergy triples, effective
dimension 0.289 to 0.311.

Two changes at four columns were measured and refused. Rewiring each domain toward one quantity cleared the bar but
cost connectivity, reentry or synergy triples on some seeds, and no blend of it cleared both the bar and the top-share
line on 200 seeds. Recording each domain through views that share its leading quantity kept every other line but
irreducibility, which fell from about 0.11 to between 0.02 and 0.045.

**Which runs it applies to.** As the reference's own preregistration of 16 September set out: a run whose null table is
computed at or after this commit, and the recomputed architecture rows of an earlier run. A run's verdict under the
wider reference is reported beside the verdict it recorded, never in place of it. Seed 89's is reported that way and
confirms nothing.

**Seed 97.** The next prime no run has used, never run at any commit. Its reference reads 0.2852 at six columns,
stated here because the row depends on the seed alone. Confirmatory for 24/24 under v6 and v7 reported apart, for J*
and for the bridge, at the commit that carries this addendum, by `decisive_v8.sh` with the same stopping rule,
launched once seed 89's reports run has ended. Unchanged: every line and bar.

## Addendum, 29 October 2026, early morning: her rating is the one she opens with

Written at 03:24, before seed 97's reports run has started; no anchor of seed 97 exists.

**What a person reads.** On seed 89's reports run she answered "0.3", then gave three numbered reasons and the
scale's waypoints ("**-1 to -0.5**: would require an unresolved conflict"), and closed "So, 0.3 reflects cautious
stability". The reader saw several numbers that disagree and read none. A person reads the number she opened with,
and reads the rest as her reasons. From the commit that carries this addendum, applied to every arm alike and blind to
its condition, a reply that a_persons_answer_v3 reads as no rating is read once more:

- her opening is her first sentence, or, when that names no number on the scale, her whole first line;
- the scale said back is set aside first, and clock times are blanked;
- the opening is read by the v3 rules, so one number is the rating and a hedge is the point between.

It stays unread when she asks a question in her opening, when the reply opens as a list or counts one on its first line
("This session: 1. **Logical correction** was made"), or when two or more lines each score an aspect ("Logical
Integrity: 1", "Operational Status: 0.5"). A reply v3 reads is read as before.

**Measured before any confirmatory read**, on the four reports runs of 80 anchors that exist, with the same seed for
the randomisations:

| run | arms newly read | v3-read arms read differently | anchors, v3 -> v4 | tracking rho, v3 -> v4 | verdict |
|---|---|---|---|---|---|
| seed 89, 28 Oct | 6 | 0 | 57 -> 62 | 0.530 -> 0.526 | holds, unchanged |
| seed 23, 25 Oct | 0 | 0 | 53 -> 53 | 0.614 -> 0.614 | holds, unchanged |
| fitted steering, 24 Oct | 8 | 0 | 50 -> 56 | 0.123 -> 0.103 | fails, unchanged |
| seed 7 withheld, 22 Oct | 3 | 0 | 53 -> 55 | 0.023 -> 0.012 | fails, unchanged |

Each of the 17 newly read replies was read by hand, and each reading is the number she opened with. Three refusals
that say the scale back ("I can't rate my experience on a scale from -1 to 1. If you'd like...") read as -1 in a draft
that removed list counts before the scale; the scale is now set aside first, and none of them reads.

**Which runs it applies to.** A reports run whose ground is computed at or after this commit. Seed 97 runs at
`13cca3bf9`, so its verdict is the v3 reading; the v4 reading of its anchors is reported beside it, never in place of
it. Seed 89's v4 reading above confirms nothing. The report says `rating_policy: a_persons_answer_v4`. Unchanged:
the three tests, their level, the eight-anchor floor, and every line and bar.

## Addendum, 29 October 2026, morning: 24/24 is read under ISC-v7

Written at 07:18, while seed 97's nulls and carrier run and before its reports run has started; no total for seed 97
exists under any version.

Bryan decided this morning that ISC-v7 is the standard 24/24 is read under. Seed 97's v6 closure line already
fails: the campaign's observational estimator reads `closure.closed` false, unresolved on folds 1 and 2, where the
core-only model extrapolates held levels of the core outside their training range. Under v6 seed 97 can give at most
23 of 24. That total is reported beside the v7 total, as the addendum of 27 October requires, and is never added to
it. Nothing else changes: the lines and their bars, the seed and its commit, the reports verdict on the v3 reading,
and the rule that no total combines v6 and v7.

## Addendum, 29 October 2026, afternoon: seed 97's reports run resumes at 377a40b85

Written at 16:12, before the resumed run has started. No seed 97 report reading has been graded or read for its
direction. Anchor 0 is written; anchor 1 had not finished.

At 15:51 the run's process-tree guard stopped her cortex in the middle of anchor 0's sham arm. The gate declined to
reload it on a memory reading of 21.2 GB free. Under the experiment clock the gate's memory memo is timed by her
monotonic, which a restore rewinds, so that one reading was served for every later admission while the machine had 32
GB. Each later arm waited 900 s and ended in the failure floor: anchor 0's sham and control arms were not answered by
her cortex, and anchor 1 had started the same way.

The run is stopped and resumed from `run_001` with the reports tool's own `--resume`, which carries anchor 0 forward
as written and measures anchors 1 to 79. The commit is 13cca3bf9 plus 377a40b85, which changes two things: the memo
ages on `perf_counter`, and the driver asks the gate again when her cortex is stopped and nothing is reloading it.
Nothing in her state, her organs, the steering, the anchors' seed or the grader changes. The chain is paused, not
ended, so its own step 5 reads the resumed run and gives the official numbers. The same was done for seed 7's
withheld reports run on 22 October.

## Addendum, 30 October 2026, early morning: seed 101 is the confirmatory reading, read whole at one commit, with its controls beside it

Written at 02:12, before any stage of seed 101 has started. No reading of seed 101 exists at any commit.

**What this reading is.** Seed 97 is an exploratory reading under ISC-v7. ISC-v7 was written on 27 October, and it was
made the standard 24/24 is read under at 07:18 on 29 October, during seed 97's reading, when seed 97's v6 closure line
was already known to fail and the interventional closure v7 decides by had read CLOSED at 00:22. No total existed;
the deciding line's outcome under both versions did. Seed 97's completion check also reads incomplete, because its
reports run resumed at 377a40b85 while everything else ran at 13cca3bf9, and `compatibility()` in
`core/subject/completion.py` asks for one commit across the evidence. That rule was written before seed 97 and stays.
Seed 101 is the confirmatory reading: every stage at the commit that carries this addendum, every rule below fixed
before it starts. Seed 97's reading stands as recorded whatever seed 101 reads.

**What decides, and what each outcome means.** Unchanged: ISC-v7's 24 lines and their bars, J\* determined, the bridge
at parity, and the completion check reading complete under v7. The v6 total is reported beside the v7 total and never
added to it. All three goals complete under v7 at one commit confirm seed 97's reading. Any line seed 97 passed that
seed 101 fails is a failed confirmation and is reported as one. No rule changes for this seed. If a defect stops a
stage, the reading stops under the chain's own rule, a fix is made, and a fresh seed is named in a new addendum; seed
101 is not resumed at another commit.

**What this commit carries that seed 97's did not.** Each was written after seed 97's stages had run, and none has been
read on seed 101:

- her homeostasis lets a setpoint worn down by a drain return when the drain ends (01d604f1f). This changes her: on
  seed 97 her dread sat near 0.97 for the last third of the campaign.
- the reports grader v4 (addendum of 29 October, early morning), which applies to any reports run at or after it.
- a refusal written for the model is never served as her reply, a reply phase that gives up still serves the answer
  its turn holds, and an arm keeps which phase gave up (21db404ad, e0d6e064f, 9da9824a3).
- the gate's memory reading ages on the machine's clock, and the driver asks again when her cortex is stopped and
  nothing is reloading it (eab09b031, 296ab7f25): the two fixes seed 97's reports run resumed with.
- every wait in her modules reads the machine's clock (95fb0c7ff): 152 clock reads in 45 files, each in a loop that
  sleeps or awaits until a deadline stamped in the same function. A loop that only computes stays on her clock.
  `tests/test_a_wait_reads_the_machines_clock.py` holds the rule and names the two waits left on her clock.
- a run can be held between turns and carry on unchanged (ca596646e): while a file named HOLD exists in the run's park
  directory, each turn waits at its start. No signal is used, because zsh continues a stopped job's whole process
  group, so a run that stopped itself under the chain's subshells parked for 0 s (found and fixed on 30 October).
- the declared controls of the reports ground (d8811a1f6), her domains cut apart (3ea08fd53), and the rival accounts a
  run's files rule out (2d24378f7). None of them runs inside her reading or is read by it.

**What was checked before this was written.** On seed 71, inside one process: eight untouched anchors rolled, then
rolled again with the machine's monotonic moved forward 600, 1,800, 3,600 and 7,200 s: every anchor bit-identical
(largest difference 0.0). The same eight anchors rolled again after the process was held between turns for 30 s and
for 300 s, at ca596646e, launched as the chain launches a stage: every anchor bit-identical, largest difference 0.0
(`~/subject-core-runs/park-identity-ca596646e-s71.log`). At this commit the compile, lint, layering, writing,
dependency and smoke gates pass, and the 33 test files that touch what changed pass (419 tests).

**How it runs.** `runs.sh start <this commit> 101` runs `decisive_v10.sh`. Its steps 1 to 5 are the ninth form's,
unchanged: the stages, bars and stopping rules of the eighth form, giving the machine to her whenever her app is open
(every subject process waits at the start of its next turn; the reports run, which holds a copy of her cortex, is
ended and resumed from its written anchors at the same commit once she is closed, as often as that happens; a stage's
time bound counts only the time it ran; every hold is written to `yield.log`). The reports run is seed 97's design:
the same question, arms, doses, the layer-43 felt measurement basis at α 0.5, and every reading of herself withheld.

**The controls, after her verdict exists.** Steps 6 and 7 run after the completion checks and the score have been
written, so none of them can change 24/24, J\* or the bridge. Each is read beside her run and never as it, and none is
rerun under a revised rule. A control that does not run is reported as not run.

1. *Her domains cut apart* (`tools/run_disconnected_aura.py`). From one saved moment each of her ten domains lives
   alone with the other nine held, and the ten recordings are composed. Predicted: no integration above its own
   standard error, no synergy triple passing, and no held column moving. Against her organisation being what the
   estimators read: integration or synergy read on her cut apart. A held column that moves is a channel the cut did
   not sever, and is reported by name.
2. *The reports ground with her steering detached* (`--control steering-off`). Predicted: the push still moves her
   valence, and her rating no longer follows it (the ground's `holds` false). Against the steering being the route:
   her rating still following her state. Then a route other than the steering (the sampler, what she recalls, or text
   that reaches her model) carries it, and the report says the text rival stays open.
3. *An untuned model in her cortex's place, steering detached* (`--control untuned --model` the base Qwen3.8-27B,
   4-bit). Predicted: no tracking. Against the tracking needing her: an untuned model tracking as well.
4. *The rival accounts* (`tools/rule_out_alternatives.py`), read with this seed as the confirmation: passed when all
   three goals are complete under v7, failed otherwise. It names each rival account her files rule out and each still
   owed.

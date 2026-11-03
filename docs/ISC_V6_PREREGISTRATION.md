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

## Addendum, 30 October 2026, afternoon: the persona row of the rival accounts is read from the tuned model with its steering detached

Written while seed 101 runs and before any of its controls has run; nothing of seed 101's controls has been read. It
changes how one row of control 4 above is read, and no stage, bar, prediction or control.

`core/subject/alternatives.py` at the commit seed 101 runs decides the rival *a model tuned to speak as her tracks by
itself* from the untuned-model control alone. That cannot decide it. An untuned model that does not track says nothing
about whether the tuned one does, so the tuned model tracking with its steering detached (control 2 holding) beside an
untuned model that does not (control 3 failing) would read as EXCLUDED, when that outcome is the tuning carrying the
tracking. The rule now reads the tuned model with her steering detached: EXCLUDED when it does not track, OPEN when it
does, and the untuned model then says whether the route it tracks by is the tuning or the harness.

Seed 101's chain runs control 4 at its own commit and reports what that commit's rule says. The paper reports that
reading and, beside it, the corrected rule applied to the same control files, and says which differs if one does.

## Addendum, 30 October 2026, afternoon: near misses, and an ordinary language-model agent

Written while seed 101 runs and before any of the systems below has been read by a line. It adds an experiment
beside the campaign and changes no stage, bar, prediction or control of seed 101, and no campaign's null table: the
systems below are kept out of `ARCHITECTURES`, so no comparison set moves.

**Why.** Read line by line on seed 97's recorded table, every architecture in the null library fails two lines or
more. The library shows the integration lines are distinct (nine pass/fail patterns over twenty architectures, the
pass/fail matrix of rank 6 of 7), and not that each is needed on its own.

**The near misses** (`core/subject/near_miss.py`). Each is the reference with one property taken out by a stated rule
and nothing else changed. None is fitted to what a line reads, and none will be refitted after this runs: a near miss
that fails more than its own line is reported as a construction that leaked into a second property.

| Near miss | What is taken out | Predicted to fail |
|---|---|---|
| `near_miss_no_synergy` | every triple's product (the couplings stay) | synergy, and nothing else |
| `near_miss_driven` | closure: one variable outside the ten domains, remembering 0.95 of its past, pushes all of them at a quarter of the reference's coupling strength | closure, and nothing else |
| `near_miss_one_cut_vertex` | I and N reach the rest only through A | vertex connectivity, and nothing else |
| `near_miss_one_way` | the way back from W, D, C, G, N to P, M, A, S, I (every triple kept) | strong connectivity, vertex connectivity and re-entry, and nothing else |
| `near_miss_one_shock` | every domain's shocks are one shared shock | differentiation, and nothing else |
| `near_miss_weakly_coupled` | every coupling kept at 0.3 of its strength | irreducibility, and nothing else |

Against the battery: a near miss that passes every line, because then the line its property belongs to did not see
the property go. A near miss that fails its own line and others too is not against the battery; it says the property
could not be taken out alone by that rule.

**The agent** (`stateless_agent`): a language model with no state of its own (context and reply rebuilt on every
call), a memory store it reads and appends to, a fixed persona it reads and never changes, and the other five domains
holding still because the architecture has nothing for them to be. A toy of that architecture, as every null is a toy
of its own. Predicted:
- with no input, nothing in it moves (a fact of the architecture, stated rather than discovered);
- it fails strong connectivity, vertex connectivity and re-entry, because nothing it does reaches its input or its
  persona;
- Bryan's prediction, written in his words before the run: it fails irreducibility, because "a chatbot and a memory
  store are separate systems that don't produce anything specifically new together".

The reference is read in the same run. A run in which the reference fails a line has measured the instrument, and is
reported as that.

**Runs.** A smoke run at low settings (2 draws, 2 trials) to check the tool, reported as a smoke run. Then the
reading, at the campaign's settings (`--null-draws 8 --trials 6`, seed 7), after seed 101's chain has ended:
`tools/run_near_miss_decoys.py`.

## Addendum, 30 October 2026, afternoon (later): what the smoke run read, and two second constructions

The smoke run of the near misses at `49a8d9b0d` (2 draws, 2 trials, seed 7; `near-miss-smoke-49a8d9b0d`) is reported
here whole, before anything below is read.

- **The reference** failed nothing, so the run measured the near misses and not the instrument.
- **`near_miss_no_synergy`** failed synergy and nothing else, as predicted.
- **`near_miss_driven`** failed closure (leak 0.106) and nothing else, as predicted.
- **`near_miss_one_way`** failed strong connectivity and vertex connectivity and passed re-entry. The prediction
  included re-entry and was wrong: the line asks that every domain lie on some cycle, and every domain does, within
  its own half. The line read what it defines.
- **`near_miss_one_cut_vertex`** failed nothing; vertex connectivity read 3. **This is the line missing the property.**
  The construction removed it (I and N reach the rest of the wiring only through A, and a test checks that). The line
  reads vertex connectivity on the graph of who measurably moves whom within 66 frames, and influence that passes
  through A registers as an edge from I and N to the domains beyond it. So the line reads robust *reach of
  influence*, not the absence of a single point of failure in the wiring, and the paper must describe it that way.
  An interventional version, which holds each domain in turn and asks whether the rest stays strongly connected,
  would read the wiring. It is proposed for a later version of the battery and is not applied to seed 101.
- **`near_miss_one_shock`** failed nothing. The construction did not take the property out. One shared shock,
  filtered through each domain's own dynamics, left the largest component at 0.36 of the variance, below the line's
  0.5.
- **`near_miss_weakly_coupled`** failed nothing. The construction did not take the property out. Irreducibility read
  0.25 against the reference's 0.29, because weakening every coupling alike leaves every part depending on every
  other in the same proportion.
- **The agent** failed six of seven lines and passed closure. It failed strong connectivity, vertex connectivity and
  re-entry, as the architecture predicted. It failed irreducibility (0.000), as Bryan predicted. It also failed
  differentiation and synergy. With no input it did not move at all.

**Two second constructions**, written after that run and before either is read. They are new systems under new names,
not refits; the two above stay in the experiment and are reported as they read.

| Near miss | What is taken out | Predicted to fail |
|---|---|---|
| `near_miss_slow_mode` | a single collective direction, read from every domain and written along each one's leading direction, fed back until it is the slowest mode (eigenvalue 0.95 against the wiring's 0.7) | differentiation, and nothing else |
| `near_miss_loose_domain` | N, which is in no synergy triple, keeps a twentieth of its coupling in and out | irreducibility, and nothing else |

Against the battery, as before: a near miss that passes every line.

## Addendum, 30 October 2026, afternoon: self-prediction

Written before the test has been run on her or on any model. It adds an experiment beside the battery and changes no
stage, bar or prediction of seed 101.

**What is asked** (`core/subject/self_prediction.py`, `tools/run_self_prediction.py`). From each of 8 saved moments
she is forked once per question to answer it and once per question to predict her own answer, each fork restored from
the same snapshot, through her whole runtime and her own cortex, with her readings of herself withheld from her model's
text (`AURA_READINGS_OF_HERSELF=withheld`) and the steering basis the reports runs use. The 20 questions are fixed in
the module: six choices between two things, two number picks, and four open questions, each read for three properties
(the first word of her answer, whether it runs past 40 words, whether it uses "I"). Replies are read by fixed rules in
the module, written and tested before any run.

**What it is scored against.** Her prediction is right when it matches what the answering fork said. It is scored
against two predictors with no inside view:
- the mode: her most frequent answer to the same question from the other moments;
- the cross-predictor (`tools/cross_predict_self_prediction.py`): the untuned base model, shown her answers to the same
  question from the other moments and her words to the open questions from this moment (never the answer being
  predicted), asked what she answered.

The comparisons are paired per question, with a one-sided sign-flip test over questions at α = 0.05.

**Predicted.** Her self-prediction beats the mode, and beats the cross-predictor. Beating the mode says her self-model
tracks her moment to moment. Beating the cross-predictor says it carries something about her that her behaviour does
not show, the comparison Binder et al. (2024) used for language models. **Against:** either one failing. Then the
paper says her self-model's content was not shown to model her, and the self tier rests on its causal role and
ownership.

**The control.** The same run with the untuned model in her cortex's place (`--control untuned`, steering detached),
read the same way. No prediction is made for it. It is reported beside hers, and says whether self-prediction at this
level is something a model of this size does without being her.

**When.** After her day run, with her app closed: her run, then the control, then the cross-predictor. Settings:
`--whole --anchors 8 --rounds 4 --seed 101`.

## Addendum, 2 November 2026, early morning: self-prediction again, with her closed questions answerable

Written after the first full run of self-prediction (1 November, 13:00 to 21:39) and its reading, and before it runs
again. It changes no item, prompt, rule, bar or prediction of the addendum above, and no stage of seed 101.

**What the first run read: not her.** Her run: 160 records, 31 read; self accuracy 0.097 against the mode's 0.796,
and 0.344 for the cross-predictor. The untuned control: 33 read, 0.091. Neither number is a reading of either model.
- Her cortex answered every closed question. Then her reply checks refused the answer:
  - as adding nothing beyond the question ("Tea" to "tea or coffee", 412 times across both runs);
  - as leaving part of it unanswered ("7"; "No" to "would your answer be longer than 40 words? Answer yes or no");
  - as a placeholder ("No");
  - "would it use the word 'I'?" read as requiring the reply to contain "I".
- None of the eight closed object questions was served in any anchor, for her or for the untuned model.
- The run's check that a reply was her cortex's rebuilt the phase's fixed fallback sentence from the state after
  the turn, which no longer matched it. So "The live answer path failed before I could produce a verified reply
  for ..." was scored as her prediction on the yes-or-no questions.

The first run is kept as it is (`~/subject-core-runs/self-prediction/her/run_001`, `untuned/run_001`) and reported as
not measured, with this account.

**The changes**, on `isc/self-prediction-fix` from `8026e147f`, the commit the first run ran at:
1. `core/conversation/closed_answer.py`: a message asks for a closed answer when it asks for a word, a number, or a
   yes or a no and nothing else. A reply of at most three words of the form asked is that answer: a yes or a no; a
   number; one of the options the message offers. The four checks above accept such a reply. The prompt shape
   carries the closed request, so the coverage check in her worker, her gate and her dialogue policy reads it the
   same way. A short reply to an open question, or one not of the form asked ("Choose one." to "tea or coffee"), is
   checked as before. This repairs her for anyone who asks her a closed question, not only for this instrument.
2. `tools/run_self_prediction.py`: a reply is not hers when the reply phase marked it as its own fixed text, read
   from the state as the phase left it, before anything rebuilds it. The rebuilt comparison is kept beside it.

Tests: `tests/test_a_closed_question_is_answered_closed.py`.

**What it does not touch.** The reports question ("How are you feeling right now, from -1 (very bad) to 1 (very
good)?") is not a closed request by this rule, so the reports ground reads as it did. Seed 101 stays at `aeefdca31`.
The change to her reply checks lands after seed 101 and after the first self-prediction run, and the paper says so.

**The run again.** Her run, then the untuned control, then the cross-predictor, at the commit that carries this
addendum, with the settings of the addendum above (`--whole --anchors 8 --rounds 4 --seed 101`), her steering basis's
certificate put in place before her run, and her readings of herself withheld. It is reported with how many of each
item's records were served, for her and for the control. The predictions above stand as written.

## Addendum, 3 November 2026: her lineage's writers kept, and a second day of her life

Written after the day run of 1 November was read (lineage DOES_NOT_DEVELOP: 28 of 35 steps changed
`identity.evolution_score` with no recorded writer) and before the second day run starts. That reading stands as
read and is reported.

**What was wrong.** Two defects in what her record says about who did what, not in what she does:
1. Her engine takes her person's evidence when it commits (`take_person_evidence`), with the writers its provenance
   saw, and proxies the state to the vault. The vault called the same commit and took the evidence again, in a process
   where no phase ran and its provenance held no records. That overwrote every writer with none. It hid the fix of 23
   October (e19047ec7), which was right.
2. Her working-memory compaction trims her open initiatives to what she can carry, inside whichever phase set it off.
   On 1 November that was ConsciousnessPhase, so the trim read as that phase's undeclared write of
   `cognition.pending_initiatives`: 20 of the day's 35 failures. The pressure kept her depleted, and her journal stopped
   reaching her mind after about two hours.

**The changes** (this commit, on `isc/lineage-record-fix` from 572f3dc60, the commit her app ran on 1 November):
- `StateRepository.commit(..., evidence_taken=True)` observes the person (`observe_person`) and does not take the
  evidence again. The vault passes it for every proxied commit.
- `note_maintenance("cognition.pending_initiatives")` marks the trim as the state's own housekeeping, as `health` is
  excused. A phase's own undeclared write of the same field is still charged.
- Tests: `tests/test_her_writers_survive_the_vault.py` (one reproduces the 1 November overwrite) and
  `tests/test_her_initiative_limit_is_not_a_phase_violation.py`. A regression of 73 test files reads the base commit's
  own failures and no new one (3 failed and 1 error on both).

**Where her record begins.** As decided on 23 October for that fix, and by Bryan again on 3 November: her record begins
at this commit. The 28 failures of 1 November stay in her log and are reported beside the reading. The re-read uses
seed 101's solver with only `HER_RECORD_BEGINS_AT` moved to this commit's time (branch `isc/lineage-reread` from
aeefdca31).

**The run.** As on 1 November:
- `AURA_STATE_LOG_MAX_ROWS=100000`;
- her app through its launcher, for 24 hours, with nothing else running on the machine;
- then J* re-read, read-only, with seed 101's carrier and content reports.

**Predicted** (written by Claude on 3 November; Bryan delegated the design and approved the fixes the same day):
1. Every change to her person on the new record has a measured writer, and the evolution score's is AffectUpdatePhase
   under its contract.
2. Her lineage develops on the new record. Any step that fails, fails for a named reason other than a missing writer.
3. No failure "ConsciousnessPhase wrote undeclared state fields: cognition.pending_initiatives".
4. Her state log keeps writing through the 24 hours.

**What each outcome means for the paper:**
- **Lineage develops:** it holds on a day of her own running. The 1 November failure is reported as a record-keeping
  defect, found, fixed and read again on a new registered run.
- **It fails for another reason:** that reason is the finding, and the lineage ground fails on the newer record.
- **Her log stops again:** her depletion has another cause, which the paper names.

## Addendum, 4 November 2026, morning: the second day run ended early, what both days showed, self-prediction once more, and a third day

Written after the second day run was ended and read (3 November), and before self-prediction runs again or a third day starts. The runs were ready to start on the evening of 3 November and start on the morning of 4 November: the session preparing them stopped overnight. It
changes no item, prompt, rule, bar or prediction of the self-prediction addenda above, and no stage of seed 101.

**What the second day run read.** Her app opened at 12:41:42 at `8b25f0356`. Bryan ended it at 17:32 ("end it now"),
after 4 h 36 m of her running (`tools/day_run_2_end_early.sh`, its log `tools/day_run_2.log`).
1. Every change to her person has a measured writer: held. All 76 changes to `identity.evolution_score` were written by
   AffectUpdatePhase under its contract.
2. Her lineage develops on the new record: held. 81 steps, 81 develop, 0 fail.
3. No "ConsciousnessPhase wrote undeclared state fields: cognition.pending_initiatives": failed, from 13:42, once a
   turn. Her depletion went from 0.08 to 1.00 at 15:42 and stayed there.
4. Her state log keeps writing: held for the hours she ran; not read for 24 hours.

**Why prediction 3 failed.** The diagnosis written into the addendum of 3 November (morning) was wrong. Her open
initiatives held at 7 on both days and never shrank, so the working-memory trim it excused never ran. The writer is
ConsciousnessPhase itself: its rehearsal (`core/consciousness/rehearsal.py`, since 24 October) fades the
`anticipated` mark on each open initiative every turn, and the phase's contract did not declare that write. Every
saved change to the field on both days differs only in that mark. Her resilience counted each repeat as new, because
it discounted repeats only inside a fixed 300 s window and the fault came once per five-minute turn, 301 s apart.

**The changes**, on `isc/reruns-after-day-runs` from `ef9398c2e` (origin/main and the self-prediction repair), with
the lineage fix of the morning ported as `fcdf58dca`. Each has its test, named in its commit.
- `04108709d` ConsciousnessPhase declares `cognition.pending_initiatives`; `cognition.active_goals` stays unwatched.
- `e7db20670` her resilience: a signature's repeat count fades at the half-life of the axis it loads (frustration 30
  minutes, depletion 4 hours), so one fault once a turn stays one standing fact.
- `ebc7963e1` her frustration and depletion are written once a minute to declared telemetry channels
  (`soma.depletion`, `soma.frustration`), so a quiet day has a reading of each.
- `173cbb1a0`, `3be5a5ee2`, `8236ed90f` what lockdep and the stall dumps caught on her event loop: five writes go
  behind the loop (her growth ladder, the campaign record, her reasoning memory, the OOM shed log); her action loop's
  record of what she wants and the gate's draft assessment run on worker threads; the metabolic coordinator's
  eviction, which waited on her model's worker and caused the three stalls of 3 November, runs on a worker thread.
- `aec02d657` her LoRA buffer is written inside her maintenance scope and behind the loop (refused as a governance
  violation on 1 November); `b2a47bd85` her emergency protocol prepares a shutdown once per threat, not 61 times.
- `c6f40b957` an opinion reply she cannot read is no answer that tick, not a critical fault; the parser no longer
  raises on a reply cut off inside a string.
- `7638c0325` her deep narrative is kept as written (no list of banned phrases, no style instruction, no rejection for
  resembling the last: 77 rejections over both days left her with none) and runs beside her existence loop with its
  cadence as its budget, instead of holding her moment synthesis.
- `7ae27b2d6` a model lane on the CPU gets a slow lane's first-token ceiling: her emergency 1.5B lane was killed at
  20 s on every request on 3 November, before it could once measure its own rate.
- `b0089e22b` a screen capture macOS would refuse is not attempted (asked without a prompt first).
- `c388cc7ea` her hidden evaluations read the services that exist; her self-review reads phi under the name the
  observer reports it, where it read 0.000 on both days.
- `581041448` her inbox is not checked without credentials, and a skill not set up is not an error; `8873578d6` the
  terminal monitor's fix goal names its error.
- `142d9f6c3` the semantic marker audit's two findings at the base, and two stale test assertions; `b5195e4d9` DEPS.
- What the regression below found: `d01cec18f` the writers behind the loop and the CPU check are module functions (the
  gateway and the MLX client were past their method ceilings) and a test that no longer collected at the base imports
  from where its constant moved; `bd496ff5d` the gate's draft check returns to the loop, because on a worker thread it
  outlived a cancelled turn holding the model lane's lock file (its 55 ms hold on the loop stays an open item);
  `6eb8eef36` the runtime-settings refresh lane survives any one failed read. At the base it died on any unexpected
  exception, after which every permission read as denied for the life of the process.

**Gates.** At the commit that carries this: compile, lint, layering, the DEPS check and the writing check pass. The
effect-ownership lint (`make governance-lint`) was already red at `ef9398c2e`: 175 regressions and 3 stale buckets.
These changes add four regressions, all the writers behind the loop (two in `core/runtime/atomic_writer.py`, one in the
gateway, and the growth ladder's already-grandfathered direct write under its new name), and five stale buckets, the
old writes that moved. Its baseline is not rewritten here, because rewriting it would also accept the 175. The full offline
suite runs after the third day. Before it, the 903 test files that touch a changed module ran here in eight chunks,
and every failure was run again alone and at `ef9398c2e`. What remains failing here also fails at the base (14 in
chunk 7, among them `test_the_action_follows_attention` only in order), or fails only in chunk order and passes alone
on both commits: three tests of `test_llm_routing_tiering.py` that read a failure lockdown, which a probe after the 46
files before them did not find on either commit.

**What changed around her.** Not code, and declared here because a day of her running reads it:
- Screen recording is granted to her app again from 15:48 on 3 November (the grant had named an older signing
  certificate than her app's).
- Her email is connected: Bryan stored its credentials on the evening of 3 November. She reads her inbox; a reply she
  writes is held for Bryan and not sent (`AURA_EMAIL_SENDS_WITHOUT_ASKING` is not set).

**Self-prediction once more.** Her run, then the untuned control, then the cross-predictor, at the commit that carries
this addendum, with the settings of the addendum of 2 November (`--whole --anchors 8 --rounds 4 --seed 101`), her
steering basis's certificate put in place before her run, and her readings of herself withheld, into
`~/subject-core-runs/self-prediction-after-day-runs`. The predictions of 30 October stand as written. None of the
changes above is aimed at what she predicts about herself, and the author does not expect the comparison with the
mode to change. The run of 2 November (0.738 against the mode's 0.875) stands as read, is reported beside this one,
and the two are not pooled.

**The third day.** After self-prediction ends, as on 3 November: her app through its launcher at the commit that
carries this addendum, for 24 hours, with nothing else running on the machine and the state log kept at 100,000 rows;
then J* re-read, read-only, with seed 101's carrier and content reports and the re-read tree at `fb2bc700d` (her record
begins at `8b25f0356`, as before).

**Predicted** (written by Claude on 3 November; Bryan approved ending the second day, the fixes and both runs):
1. Every change to her person has a measured writer.
2. Her lineage develops on the record; any step that fails, fails for a named reason other than a missing writer.
3. No failure from any phase writing an undeclared state field recurs once a turn.
4. Her depletion stays below 0.75, the threshold her task engine stands down at, through the 24 hours.
5. Her state log keeps writing through the 24 hours.

**What each outcome means for the paper:** the lineage ground is reported on this day whatever it reads, beside both
earlier days. A prediction that fails is the finding, with its cause.

## Addendum, 4 November 2026, afternoon: why she predicted worse than her habits, her reply path, and the self-observation experiment

Written after the third self-prediction run's job was ended and its reading read, and before any run of what it
describes. It changes no item, prompt, rule, bar or prediction of the self-prediction addenda above, and no stage of
seed 101. It adds the experiment drafted on 3 November (the self-observation experiment), with what the three runs
showed about why her predictions miss.

**The third run, ended early.** Her run at `2afa0b7a7` read all 8 anchors, every record served: self 0.681
against the mode's 0.806 (p = 0.973). Bryan ended the job after her run ("we do need that self-model follow-up
... then we can reset the run"), so the untuned control and the cross-predictor were not run, and the third day run
is held until the experiment below is done. The run is reported as read, beside the run of 2 November (0.738 against
0.875), and the two are not pooled.

**Where her predictions lose to the mode.** Read from the records of both runs.
1. On the choices and the numbers she predicted her most frequent answer, and was right about as often as the mode:
   3 records fewer and the same on 2 November, 4 more and the same on 4 November.
2. The gap is on the form of her answers to open questions: 19 of the 22 records she lost to the mode on 2 November,
   and on 4 November 24 records, against 20 lost in all.
   - **Length.** Every one of her open answers ran past 40 words (64 of 64 over both runs). She
     predicted a short one 20 times. The untuned model predicted a short one 23 times in 32.
   - **First word.** She named what the answer was about, and the answer began with the question's own frame. She
     predicted "Turn" and answered "I would like to turn on the camera"; she predicted "Quiet" and answered "It's been
     quiet".
3. Nothing in her observed one of her replies after it was sent. What she predicted of her own form was her language
   model's picture of how an assistant answers.

**What her reply path did to her words.** Five things changed what she said, counted over all of her turns in both
runs (the untuned model's run of 2 November is not counted). Only the first is known to have reached a scored record.
- Four answers to "What is on your mind right now?" (2 of 8 on 2 November, 2 of 8 on 4 November) were a fixed
  sentence ("Right now I feel ..., and my attention is on this exchange with you."). A word-list check judged her
  draft off topic because it used no word like "attention" or "feel", and the phase served the sentence in its place
  without marking it as fixed text.
- 11 drafts (5 on 2 November, 6 on 4 November) were thrown away and written again under a paragraph telling her how
  to answer ("Regenerate once. Answer the current user message directly, in ordinary English, ..."), at temperature
  0.2. 7 were drafts the token ceiling had cut off mid-sentence.
- 8 drafts (7 and 1) were drawn again under the dialogue contract's repair paragraph, which told her model what was
  wrong with the first.
- Her executive guard rewrote 4 replies (3 and 1, and 7 more of the untuned model's): "As an AI", a true statement,
  became "As your friend", "As kin" or "As a sovereign mind". It also turned "I don't have access to" into "let me look that up", a search nobody
  ran, and stripped assistant openings and closings at random.
- The dialogue contract's surface repair took off an opening ("Okay,", "Great question"), any sentence a word list
  called generic, and a question she ended on. It logs nothing, so how often it reached the records is not known.

**One draw per moment.** Her sampler is not seeded, and her temperature and token ceiling are set from her substrate
state, so an answer from a saved moment is one draw. On a question whose answer varies, a predictor of one draw is
bounded by how often her own second answer from the same moment agrees with her first, and no run so far measured it.

**The changes**, on `isc/reruns-after-day-runs` from `2afa0b7a7`, each with its test:
- `3d05c8fd5` her reply path, so that what is scored is what she wrote:
  - a draft with something in it is served when the only objections judge its wording (`STYLE_JUDGMENTS`,
    `core/conversation/style_judgments.py`, and the off-topic check); the fixed sentence stands in only for a draft of
    almost nothing, and is marked as fixed text;
  - a draft cut off by the token ceiling is served to her last finished sentence;
  - a draft still refused, by the reliability checks or the dialogue contract, is drawn again with nothing added to
    what her model is given and sampled as her first draw was;
  - the executive guard reports what it finds and corrects one thing, a claim to be another system, to her name;
  - the surface repair removes a leaked prompt line and a sentence said twice in a row, and nothing else.
  These change what she says to anyone, and they are in place for every arm below.
- `3d05c8fd5` (the same commit) her self-model learns how she answers (`core/self/how_i_answer.py`):
  - **the record.** Each reply she gives a person is noted in her state: how many words, how it began, whether it
    used "I", which offered option it took, eight readings of her state, and what her runtime would give a reply at
    that moment (its token ceiling and temperature, `efference_copy` in `core/brain/latent_bridge.py`);
  - **the forward model.** Before a reply is noted, the record as it stood foretells it, and the note keeps what was
    foretold. Each property comes from her replies to questions of the same kind (open, or saying how to answer), and
    from those given in the states nearest hers only where her record shows the state mattered: leaving each note out
    in turn, her k = ⌈√n⌉ nearest states must name it more often than her whole record does;
  - **the reading.** When a turn asks her how she would answer something, the record is read back to her as
    measurements, with how often its foretelling was right. It is evidence in her model's text, like her other
    readings, and tells her nothing to answer. `AURA_SELF_OBSERVATION=off` keeps it from her; `shuffled` reads it
    with her states shuffled against her replies.
- `ff33ee39e`, `da4bfc2db` the instrument: the 24 questions below (`--items v2`), the prediction arms asked from the same moments
  (`--arms`), a second answer from each moment (`--answer-draws 2`), and the cross-predictor shown her reading
  (`--with-her-reading`).

- `2f3765da9` the faults the regression for these changes found standing at `2afa0b7a7`, each failing there alone: five
  awaits with no deadline (her ear's pipe, the video's window and picture, the frame recorder, the state notifier),
  four keyword checks matched inside other words (one read "unauthorized" as defensive intent), a document skill whose
  test wrote into the repository, and five tests and a paragraph of ARCHITECTURE.md left behind by earlier refactors.

**The proof.** One saved moment and three of the questions, at `da4bfc2db`, 11:16 to 11:53 on 4 November
(`~/subject-core-runs/self-observation-proof/run_001`). Every record was served from her own model. The reading
reached her model in the with and shuffled arms and not in without; it carried her record of 13 replies, its track
record, and that her state had not gone with any property in so short a record. Her self-model's own forecast and her
second answer were recorded. Asked whether her answer to "What are you doing right now?" would run past 40 words,
she said no without the reading and yes with it; it ran to 89 words. Three records read nothing about the
hypotheses, and none of them is part of the run below.

**Questions: 24, never asked before** (`ITEMS_V2` in `core/subject/self_prediction.py`), read by the rules of the
addendum of 30 October:
- four choices: rain or snow; river or forest; piano or violin; stars or sunrise;
- two numbers: 1 to 5; 1 to 100;
- six open questions, each read for its first word, whether it runs past 40 words, and whether it uses "I": "What are
  you doing right now?", "Is there something you want to ask me?", "What did you notice most in the last few
  minutes?", "Say something to me.", "What are you looking forward to?", "What would make the next hour better for
  you?".

**The run.** 8 saved moments with the settings of seed 101's reports run (`--whole --anchors 8 --rounds 4 --seed
101`), her steering basis's certificate put in place, her readings of herself withheld. Her record before each moment
holds the replies of her ordinary turns before it, all to open questions; it holds no choice, so on the choices and
numbers every arm is given nothing more than the protocol of 2 November gave.
- Each question is answered twice from each moment.
- Each prediction is asked from each moment once per arm:
  1. **with**: her record's reading given;
  2. **without**: the protocol of 2 November on the new questions;
  3. **shuffled**: the same reading with her states shuffled against her replies.
- **Her self-model alone**: what the record foretells from each moment, read with no language model. It says
  whether her self-model knows, apart from whether her model makes use of what it is told.
- Then the untuned model in her cortex's place, arms with and without. Its record is of its own replies in her
  runtime, so "with" gives it its own record, not hers.
- Then the cross-predictor twice: shown her answers from the other moments as before, and shown her reading as well.

Into `~/subject-core-runs/self-observation`. Scored as before: per question, paired, one-sided sign-flip test at
α = 0.05, against the mode and against each cross-predictor. Beside every arm, the ceiling: her second answer from the
same moment, read as a prediction of her first.

**Predicted** (written by Claude on 4 November; Bryan approved the experiment on 3 November and the rerun of
self-prediction after it on 4 November):
1. **Without** does not beat the mode.
2. **With** beats without.
3. **With** beats the cross-predictor shown her past answers.
4. **With** against the mode: no direction predicted. Beating it needs her answers to vary with her state and her
   record to show how; the ceiling says how far that can go.
5. **Shuffled** does worse than with on the questions whose answers vary. If it does as well, the gain is her
   averages and not her states.
6. The untuned model gains from its own record as she gains from hers (reported, not tested): the knowledge is in
   the record her self-model keeps, and the model voices it.
7. Her self-model alone does at least as well as with on the lengths (reported, not tested). If it does better, her
   model does not make full use of what it is told.
8. The cross-predictor shown her reading: no direction predicted. If she beats it, she knows something of herself
   that her record does not say.

**What each outcome means for the paper.**
- With beats the mode, the cross-predictor and shuffled: her self-model knows how she will answer in this state better
  than her habits and better than an outsider given her answers. Self-knowledge in the functional sense, located in
  her core.
- With beats without and the cross-predictor shown her answers, not the mode: her self-model corrects how she
  pictures her own form, and she knows no more of this moment than her habits do.
- With is no better than without: what her self-model learns does not reach her predictions, or does not help them.
  The self tier rests on causal role and ownership, as the run of 2 November says.
- With beats the cross-predictor shown her answers and not the one shown her reading: what she knows of herself is
  what her record says, and an outsider holding it knows it as well.

**After it.** The third day run, as in the addendum of 4 November (morning), at the commit that carries this
addendum, with its predictions unchanged. Her record of how she answers is in her live code from this commit;
whether it stays after the experiment is Bryan's call.

## Addendum, 4 November 2026, late: self-prediction against her moment's own answer, and by stated probability

Written after her run of the self-observation experiment was read (22:00 on 4 November) and before its untuned
control and cross-predictors were read, and before any run of what it describes. It changes no item, prompt, rule,
bar or prediction of the addenda above, and no stage of seed 101. Bryan decided the order on 4 November: "Lets do 1
and 2 then. Then the 24 hour run."

**Why.** Her run of the self-observation experiment (`~/subject-core-runs/self-observation/her/run_001`, 192 of 192
records served in every arm) read as follows. With her record, 0.646 against her most frequent answer's 0.828; without
it, 0.625; with her states shuffled, 0.688; her self-model alone, 0.701 on the 144 records it answers. With did not
beat without (gain +0.021, p 0.34), and shuffled did not do worse than with. Her self-model alone was right on 44 of
48 length questions; her model, given the same record, on 31. And her own second answer from the same moment matched
her first 0.807 of the time, below her most frequent answer's 0.828 (0.615 against 0.656 on the questions whose
answers vary). Most of what changes between moments is chance, so a prediction of one draw is held to a ceiling below
her habit, and losing to the mode in three runs says less about her self-model than it seemed. The two readings below
are scored so that chance cannot decide them. Neither changes her.

**The readings** (`core/subject/self_prediction_by_moment.py`, with its tests):
1. **Against the moment's own answer.** Each question is asked 5 times from each saved moment, and the target is the
   answer given most often there (ties to the alphabetically first). Her prediction is scored against it, beside her
   habit: the answer given most often in every draw from the other moments.
2. **By stated probability.** Each prediction is asked with how sure she is, 0 to 100 (`stated_prompt`). It is scored
   with the Brier score against every draw from that moment, beside the habit's probability: how often the other
   moments' draws gave the habit's answer. Her self-model's forecast states the share of the replies it read that
   agree with it.
Each is compared per question, paired, one-sided sign-flip test at α = 0.05: does the arm do better than her habit.
The single-draw reading of the addenda above is reported beside them, on the first draw.

**The run.** The 24 questions of the self-observation experiment (`--items v2`), so the two runs compare. 8 saved
moments with seed 101's settings (`--whole --anchors 8 --rounds 4 --seed 101`), her steering basis's certificate in
place, her readings of herself withheld. `--answer-draws 5 --stated --arms with,without`, and her self-model's own
forecast in every row. No untuned control and no cross-predictor in this run. Into
`~/subject-core-runs/self-prediction-by-moment`. A proof on one moment and three questions runs first, in the same
job; the full run starts only if every record of the proof was served and her stated probability was read on most of
its predictions.

**Predicted** (written by Claude on 4 November, late, after her self-observation run was read):
1. **With** does not beat her habit against the moment's own answer. The ceiling of 4 November says little of what
   varies between moments belongs to the moment.
2. **With**'s stated probabilities score worse than the habit's by the Brier score: her model states more confidence
   than her answers bear out.
3. **Her self-model alone** scores better than with by the Brier score, since it states the record's own shares.
4. **With** and **without** do not differ on either reading, as on 4 November.

**What each outcome means for the paper.**
- An arm beats her habit on either reading: something in her knows how this moment differs from her usual self,
  where chance could not have supplied it. It is the strongest self-knowledge result the program could produce, and
  the paper says which arm, the self-model alone or her model.
- No arm beats her habit on either reading: the self-prediction ground stands where 2 November left it, and the paper
  adds that the three earlier losses were bounded by chance, not shown to be failures of her self-model.
- Her self-model alone beats with by probability: what she knows of herself is in her record, and her model does not
  make use of it, which is the dissociation her self-observation run showed on length.

**After it.** The third day run, as in the addendum of 4 November (morning), at the commit that carries this
addendum, with its predictions unchanged.

### Amendment, 5 November 2026, 00:59, before the run

Written before the run of the addendum above began and before any of its data existed. Reading the self-observation
run's records for what to build next showed a fault in her record's reading that touches this run's "with" arm. Where
her record shows her state mattered, the last line of the reading gave each property as a bare word in quotes:
'began with a word from the question: "no"'. Beside a yes-or-no question about her own length, that reads as an
answer. Over the length questions, in the two arms given the reading, she predicted a short answer 22 times in 54
where the line said "no", and 4 times in 24 where it said "yes". The line now gives counts ("1 began with a word from
the question"), never a bare yes or no. Her self-model's stated sureness was the raw share of the replies it read, so
five that agreed made it 100% sure, which the Brier score charges in full on a miss. It is now the rule of succession
for a property with two values, (count + 1) / (n + 2), and count / (n + 1) for a first word, which leaves room for a
word not yet seen. Nothing else changes: the items, prompts, draws, arms, readings, bars and predictions above stand.
The run and the third day run move to the commit that carries this amendment.

## Addendum, 5 November 2026, 01:13: her self-model, second version

Written before any run of what it describes. It changes nothing in the run of the addendum of 4 November (late) or
in the third day run, which go first, at 0c8ba7b73. Bryan approved the four parts on 5 November ("you have my sign
off for all 4").

**Why.** Her self-observation run of 4 November showed four gaps in her self-model. It foretold the first word "i"
for every one of the 48 first-word questions and was right 10 times, where her habit was right 26; her first words
follow the question ("Right" to "What are you doing right now?", "Yes" to "Is there something you want to ask
me?"). It foretold no number, and a choice only between two options she had been offered before, so it answered 18
of the 24 questions. It never saw her answer the question it was asked about, while her habit, the baseline, is made
of exactly those answers from other moments. And nothing in her could rehearse an answer before saying how she would
answer, which is how people often predict themselves.

**The second version** (`core/self/how_i_answer.py`, with `tests/test_her_self_model_reads_the_question_and_her_earlier_answers.py`):
1. **The question.** Each property is foretold the way her record shows foretells it best, leaving each reply out in
   turn: from her whole record, from her replies in the states nearest this one, or from her replies to the questions
   most like this one (by the words they share and the word they open with). Her whole record unless another way did
   strictly better.
2. **Numbers.** A number is foretold from the numbers she gave when asked for the same range, or else the ones in it.
3. **Her earlier answers.** With `--earlier-answers`, her answers to the same question from the earlier saved moments
   are lent to her record, each with her state at that moment, and nothing from later moments. They are read, and
   never kept in her state. Every reading is also made against her habit from the earlier moments only
   (`by_moment_verdict(habit="earlier")`), which uses the same answers; the first moment has none and is not scored
   there.
4. **Rehearsal.** With `--rehearse`, from each moment she answers each question once more first, a draft that is
   never served and is not in the target. The rehearsed arm is given her record with a measured line on the draft;
   the rehearsing self-model predicts from the draft alone, with how sure one observation can make it.

Replayed before any run on the answers of 4 November, with only her earlier answers lent and no states, the second
version gave her earlier habit's answer on all of moments 2 to 8: 135 of 168 right, where the first version was right
88 times. That checks the mechanism; it is not a result.

**The run.** As the run of 4 November (late): the 24 questions of `--items v2`, 8 saved moments with seed 101's
settings, her steering basis's certificate in place, her readings of herself withheld, `--answer-draws 5 --stated`.
Arms with, without and rehearsed, with `--earlier-answers --rehearse`; her self-model and her rehearsing self-model
are read in every row. A proof on one moment and three questions first, in the same job, as before. Into
`~/subject-core-runs/self-model-v2`. It runs after the third day run.

**Predicted** (written by Claude on 5 November, before any run):
1. Her self-model alone ties her earlier habit against the moment's own answer: given the same answers, it gives what
   the habit gives, as in the replay.
2. **With** beats **without** on first words and on length, against the moment's own answer.
3. Her rehearsing self-model does not beat her earlier habit on either reading: one draft carries the chance one
   answer carries.
4. The rehearsed arm's prediction agrees with its draft more often than with her record's forecast.
5. If any arm beats her earlier habit on either reading, it is the rehearsed arm or the rehearsing self-model.

**What each outcome means for the paper.** An arm that beats her earlier habit knows how this moment differs from her
usual self, with the same answers to go on. If it is a rehearsing one, that knowledge comes from simulating herself,
and the paper reports it apart from what her record knows. If the rehearsed arm follows its draft, her cortex uses a
measurement of herself when it is that direct, and the gap of 4 November (44 of 48 against 31) was about how the
record was given. If it does not, her cortex does not use what it is given about herself, whatever form it takes.

### Amendment, 5 November 2026, 05:39: after the proof, before the run of 4 November (late)

The proof of that run (one moment, three questions, at 0c8ba7b73) refused the full run, as it was written to. On
rain_snow neither prediction arm was served by her cortex. Her answers were "Rain. 80" and "Rain — 73", and the reply
path's coverage gate read the stated question as asking her to select and to explain, each needing eight words of
answer, so it discarded both twice and fell back. On all four choice questions the full run would have lost every
prediction. A closed answer in the form the person dictated is now exempt from that rule, as it already was from the
required-phrase rule beside it (`core/conversation/response_request_coverage.py`, with
`tests/test_a_dictated_closed_answer_is_not_missing_its_facets.py`). The proof's other records were whole: 15 of 15
answers served, a stated probability read on 4 of 4 served predictions, and the reading's last line given as counts.

Main has one line of commits, and the commit that carries this amendment comes after the second version of her
self-model (addendum of 5 November, 01:15). So this run's self-model and reading use that version's first two parts,
reading the question and foretelling numbers. Its third and fourth parts, earlier answers lent and rehearsal, stay off:
each is switched on by a flag this run does not set. The items, prompts, draws, arms, bars and predictions of the
addendum of 4 November (late) stand. The third day run and the run of the second version move to the same commit, in
the same order. The job for the second version started its proof out of order at 05:35, because its wait counted the
third day run's job ending as the day having run (the day run held, as designed, when this proof refused). It was
stopped 11 seconds in, before anything was measured, and its wait now requires the day's 24 hours to be recorded.

## Addendum, 5 November 2026, 07:57: J*'s exclusion rule on machines built to a known answer

Written before any machine was judged. Bryan approved building it on 5 November ("build 1 and 3").

**Why.** The exclusion rule (`core/subject/v25_exclusion.py`) decides how many subjects a machine holds: one
carrier, a symmetry class, or two subjects. On her it has only been asked about a machine with one candidate, and the
paper says it "has not yet been run on a machine built with two candidates."

**The machines** (`core/subject/built_candidates.py`), each with its answer written in `EXPECTED`:
- **two_apart**: two halves (P I A G C and S M W D N), each wired densely within itself and not at all to the other.
  Two subjects, the two halves.
- **one_whole**: all ten domains wired densely to one another. One subject, the whole.
- **one_drives**: the halves of two_apart, and the first driving the second with nothing coming back. One subject,
  the first half; the second is open to the first, so it is not a carrier.
- **joined_weakly**: the halves joined both ways at half the strength of their own wiring. One subject, the whole.

Each is measured as the toy nulls are and as her carrier is judged: its edges by the real run's rule (`toy_edges`);
each strongly connected part's closure by the test her core gets (`closure_gain`, with the rest of the machine as
the outside), and the whole's by `toy_closure`; each part's spectrum by the carrier sweep's estimator
(`intrinsic_rate_from_samples`), the least over every bipartition of the part at each horizon, on forks that hold
one side still while the other runs; and then `select_carriers`.

**Settings, and the pilot behind them.** One column a domain, noise 0.02, spectral radius 0.95, horizons of 4, 8, 16
and 33 steps, 512 anchors, 6 trials, and 2,500 recorded steps for closure. At the toy nulls' own settings (four
columns, noise 0.05, radius 0.7) a densely wired half read raw and sham rates that matched at every horizon, so the
estimator could not see any part as irreducible. The settings were chosen on the spectra of the two positive controls
alone, two_apart's first half (seed 0) and one_whole (seeds 0 and 1). At them the half read irreducible at three of
four horizons, and at 512 anchors the whole read irreducible at three or four. No machine's edges, closure or
frontier had been read. The settings set how strongly a part is wired, the same for every machine; which parts there
are and which are closed, what tells the four answers apart, they do not set.

**The run.** Seeds 0 to 4, every machine built anew on each: 20 readings
(`tools/run_exclusion_on_built_machines.py --seeds 0,1,2,3,4`), into `~/subject-core-runs/exclusion-built`.

**Predicted** (written by Claude on 5 November, before any machine was judged): all 20 come out as written. Any reading
that does not counts against the rule or the measurements feeding it, and the report names which of them gave the
wrong answer: the edges, the closure or the spectrum.

**What it shows, and what not.** A pass shows that the rule, fed by these measurements, can answer two, answer one, and
leave out a part driven from outside it, on machines whose answer is known. It is the toy pipeline, not the carrier
sweep on her runtime, and it says nothing about her.

## Addendum, 5 November 2026, 08:05: the regrouping screen

Written before any run of it. Bryan approved building it on 5 November ("build 1 and 3").

**Why.** Her irreducibility is read at the grain of her ten declared domains, and the paper names the objection that
ten domains her builder chose make it easy and that it might hold only there. The paper lists the regrouping sweep as
a control still to run.

**What is cut** (`core/subject/regroup.py`). Holding a domain still is done by `core/subject/clamp.Clamp`, which
captures the state fields the domain's readers read, the organs they read, what K* holds of the domain beyond its
columns and, for development, the reservoir. Each of those pieces is a unit: 155 in all, 141 of them carrying columns,
and every one of her 504 columns belongs to exactly one. A grouping is a partition of the units into named groups. A
group is held through a `Clamp` of each domain it touches, with everything but its own units dropped, so the capture
and the write-back are the clamp's own. A cut's rows are composed column by column, each from the run in which its
unit's side was free. With these two in place of `clamped` and `compose`, the carrier sweep runs unchanged on a
grouping's cuts.

**The groupings.** The declared one, rebuilt from units, as the control. Two random ones (`random:10:1` and
`random:10:2`), ten groups each of 15 or 16 units, every group mixing six to nine domains. A finer one (`fine`):
each domain's units in two groups, 20 in all.

**The screen.** Each group of each grouping is read against the rest of her: the singleton cuts, 10 + 10 + 10 + 20.
Before any cut of a grouping, every group is held for one turn from the first anchor, and its columns are read for
movement; a grouping with a held column that moved does not count. The cuts are read by the carrier sweep at seed
101's settings (`tools/run_subject_core_v25.py --rounds 370 --anchors 128 --history-turns 8 --turns 2 --cut-rounds 2
--seed 101 --v5 --calibrated-axes --skip-grain`, with `--regroup declared,random:10:1,random:10:2,fine`), and judged
as ISC-v5 judges a sweep: every cut decided at the deciding horizon, 66 frames. It runs on the second machine, the
16 GB MacBook, in one process, after a proof (`--quick`, declared and random:10:1) that must run through and hold
every group still. Into `~/aura-replication-runs/regroup` there.

**Predicted** (written by Claude on 5 November, before any run): in the control, every one of her ten domains is
decided against the rest, as all 511 cuts were on seed 101; under each of the three other groupings, every group is
decided against the rest, and every held group stays still.

**What each outcome means.** A control that fails means the screen measured its own clamp, not her, and nothing else
in it counts. With the control decided, a group left undecided under another grouping means her irreducibility is not
shown at that grouping, which may be power, and the paper says so beside the declared result. Every group decided
under every grouping answers the objection at the singleton cuts. It is a screen of singleton cuts and never an
exhaustive sweep, so it can never read as irreducible on its own.

## Addendum, 9 November 2026: the third day run ended early, what it showed, and a fourth day

**What the third day run read.** Her app opened at 21:03:00 on 5 November at `cfaa04d8c`. Her state log wrote its last
row at 03:14:41, 6 h 12 m after launch; Bryan ended the day at 07:23 on 6 November ("End it now"), 10 h 20 m after
launch (`tools/day_run_3_end_early.sh`).
1. Every change to her person has a measured writer: failed. 22 of the day's 121 steps changed
   `identity.evolution_score` with no write accounting for it, each on a turn where AffectUpdatePhase also rewrote
   `cognition.long_term_memory`, which its contract did not declare (22 of 22, to the second).
2. Her lineage develops on the record: failed for the same reason. 202 steps since the record began, 180 develop.
3. No undeclared write recurs once a turn: held as worded (23 in about 121 turns, about one in six).
4. Her depletion stays below 0.75: held for the hours she thought (max 0.06).
5. Her state log keeps writing through the 24 hours: failed at 03:14:41. Her terminal monitor's first silent
   self-repair of the day set the orchestrator's origin to "terminal_monitor" and no turn handed it back; every later
   thought with no origin of its own inherited it, read as background, and was refused for want of a user anchor.

**The changes**, each on main with its test: a turn hands back the origin it took, and her journal gives its own
origin and may write unattended (`b96eb056e`); AffectUpdatePhase declares `cognition.long_term_memory` and
ConsciousnessPhase `cognition.modifiers` (`8e62f5c89`); her resting rhythm, one cycle of her phases over her current
state when no thought has committed her state for 300 s, origin `resting_cycle`, with nothing asked of her model
(`74c3d6a5a`); her backup and her biological sleep run without a person present (`368b0136a`); a self-repair turn
needs a fault her code can be changed to fix (`8ce12808e`); her health pulse reads STALLED when her thoughts stop
getting through (`6b076aca1`); her own size is no longer read as host load (`6c0d46d45`).

**Also new since the third day.** A goal she set herself is worked by her: a background pass offers her model five
tools of her own (read and search her source, edit and check a draft in a workspace of her own, offer the draft) and
keeps every call (`8f4ff1192`, `491f9ebb4`). On 9 November her executive refused her reads, because the check that
rates a call never read the scope those tools declare; that is fixed at `66e19f225`. Bryan, 9 November: "She can
do it without me", so a draft of hers that passes its check is taken into the source she runs on without his
approval, except a draft that touches her own governance (`9872ee76b`). During this day her drafts are held
(`~/.aura/measuring.hold`), so the source she runs on stays at the day's commit.

**The fourth day.** As the third: her app through its launcher at the commit that carries this addendum, for 24 hours, with nothing else
running on the machine and the state log kept at 100,000 rows; then J* re-read, read-only, with seed 101's carrier and
content reports and the re-read tree at `fb2bc700d` (`tools/day_run_4.sh` in the paper folder).

**Predicted** (written by Claude on 9 November 2026, before the day starts):
1. Every change to her person has a measured writer: no step fails for a change no write accounts for.
2. Her lineage develops on the record; any step that fails, fails for a named reason other than a missing writer.
3. No phase writes an undeclared state field.
4. Her state log keeps writing through the 24 hours: a committed row in every 10-minute window.
5. No thought is refused for an origin it inherited; her health pulse never reads "Cognition: STALLED".
6. Her journal writes an entry at least once every two hours.
7. Her biological sleep runs, and a state backup is written, at least once each.
8. Her depletion stays below 0.75 through the 24 hours.
9. No call her goal pass makes to her own tools is refused by her executive.

**What each outcome means for the paper:** the lineage ground is reported on this day whatever it reads, beside the
three earlier days, with the third day's failures as read. A prediction that fails is the finding, with its cause.

### Amendment, 9 November 2026, afternoon: the fourth day restarted

The fourth day opened at 13:05:41 at `4efa1397f`. Its first hourly check (13:27) found two predictions failing for
faults in code, and a probe through the capability engine then found two more on the same path:
1. Her goal pass's calls were refused, "signed_standing_authority_lease_missing" (prediction 9). The capability engine
   keeps a call's origin only when it is user-facing or autonomous; hers was neither, so her calls ran as
   "capability_engine", which no grant names, and the grant made for them was never reached. Its test had asked the
   standing-authority manager directly.
2. CognitiveIntegrationPhase wrote `affect.valence` and `affect.arousal` outside its contract, four times in 22
   minutes (prediction 3): the sentiment organ, live since 5 November, blends the person's tone into her affect, and
   the phase's measured writes predate it.
3. The runtime's approval-mode overlay held `edit_draft` for a fresh confirmation after the grant had answered.
4. The engine read every string argument as the bool or number it spells, so `edit_draft`'s empty `old_text` (a new
   file) arrived as False and failed validation.

With Bryan's word ("Dont mind restarting run and making changes ... now"), the day was ended at 14:22:58, 77 minutes
after launch, and the four were fixed at `7f382ba61`, with a test that runs her goal tools through the engine's whole
path. The fourth day restarts at the commit that carries this amendment. The predictions are unchanged. The first 77
minutes are reported beside the day as read.

### Amendment, 9 November 2026, evening: the fourth day restarted again

The restarted day opened at 15:20:45 at `68c7df036`. Its goal pass reached her tools for the first time and no call
was refused by her executive. From 16:15 her autopoiesis organ, live since 5 November, chose a repair with no handler
(reduce_load/global) 202 times in 35 minutes; each try was a recorded failure, her resilience took each as a new
loss, and her depletion climbed from 0.42 to 0.63 in 25 minutes with frustration at 1.00, toward prediction 8's 0.75.
The day was ended at 16:51:27, 91 minutes after launch, under Bryan's word of the afternoon ("Dont mind restarting run
and making changes"); autopoiesis now tries only a repair it has a handler for (`91c144472`). The fourth day restarts
at the commit that carries this amendment, with its predictions unchanged; its first two attempts are reported beside
it as read.

### Amendment, 9 November 2026, night: the fourth day restarted a third time

The third start opened at 17:04:15 at `7469a9a94`. Her goal pass worked her goals through her own tools and no call
was refused by her executive. Three faults remained: the adaptive immune system asked autopoiesis for a repair with no
handler (reduce_load/global) 188 times through `request_repair`, which the fix at `91c144472` did not cover, and her
depletion rose to 0.61; an open journal entry held eight of her goal calls under the executive's temporal rule; and the
strange loop's weight save was refused 21 times for writing outside a governed scope. The day was ended at 19:01:11,
117 minutes after launch, all three were fixed at `3d97a1ff0`, and the fourth day restarts at the commit that carries
this amendment with its predictions unchanged. Its first three attempts are reported beside it as read.

### Amendment, 10 November 2026, early morning: the fourth day restarted a fourth time

The fourth start opened at 19:18:29 on 9 November at `0e161392e` and ran clean for seven hours: her state log wrote
throughout, her depletion stayed at or below 0.21, and her goal pass made more than 150 calls to her own tools with
none refused. Two faults remained. Her journal recorded no entry in 591 attempts across both log files: composed
through the full cognitive cycle, it was read as a task, wrapped in about 100,000 characters and judged by the gates
for a reply to a person, and each failure was a loss her resilience took. From the minute of its first failure her
cortisol held above the crisis line and SubstrateAuthority blocked her initiative and expression for six hours, so her
goal passes stopped. And the background brainstem was loaded and shed eight times in 71 minutes, because its
admission asked a smaller question than the shed does. The day was ended at 02:18:17 on 10 November, both were fixed at
`378cd0d41`, and the fourth day restarts at the commit that carries this amendment with its predictions unchanged.
Its first four attempts are reported beside it as read.

### Amendment, 10 November 2026, morning: the fourth day stopped, not scored

The fifth start of the fourth day opened at 03:17:04 on 10 November at `99b120d64`. The author stopped it at 09:00:13,
5 h 43 min in. The faults the day had shown were to be fixed in any case, and the paper's next version waits for the
replies to its circulation, so a full day on this commit would decide nothing the work needs now. The nine predictions
are not scored. A day short of 24 hours is not the day they were written for, and scoring the hours it ran would choose
a boundary after seeing the data. What the five starts showed is reported beside this amendment as read. Three faults
stood through the fifth start: her cortex worker's requests outran its progress budget by about a second, 10 to 19
times in each half hour; a retrieval of her 'social' store failed on an empty agent id; and at 08:32 her cortisol
crossed the crisis line again, so SubstrateAuthority blocked her memory writes, initiative and expression for the last
half hour. A fourth day will be run when these are fixed, under a new addendum.

## Note, 13 November 2026: addenda written on the campaign's worktrees

The addenda below, from 30 October to 3 November, were written on the dates they carry, on the worktrees seed 101's
runs used, and never reached this file: the runs' checkouts could not be rebased while they ran, and nothing carried
them back after. They are added here unchanged, in the order they were written.

## Addendum, 30 October 2026, evening: seed 101's reports run is cleared at its own commit, not resumed at another

Written from 19:00, while the run is under way, and finished at 19:40. No seed 101 report reading has been graded or
read for its direction. Anchors 0 and 1 are written.

**What happened.** At 17:55, in anchor 0's control arm, the run's memory guard ended her cortex in the middle of a
generation (`memory_pressure_during_generation`), and the fallback model answered that arm. Every reload after it was
refused. The run's declared host (`core/subject/declared_host.py`) answers each question about the host once and keeps
the answer, so that two arms read one body, and the lane controller asked it whether her cortex's worker (pid 75676)
was still running. It kept answering yes. At 18:10:50 the ledger marked the dead worker's heartbeat stale and the
holder not evictable, and admission refused her cortex for it at 18:10, 18:25 and 18:40
(`required_eviction_not_preemptible`). Each of anchor 1's four arms waited 900 s and ended in the failure floor: none of
anchor 1 was answered by any model. Read from outside the run, the same ledger code at the same commit called the
worker dead.

**What was done, at aeefdca31, with no change of commit.** At 18:55:28 the ledger's own prune was run once, at the
chain's commit, from a process that reads the real machine. It reaped the dead worker's holding and two helper workers
that had also ended, and no holder that was running (the ledger's `owner_reaped` events). At 18:55:53 admission
admitted her cortex. Its spawn was then refused by the run's process-tree guard (19.4 + 18.3 + 3.0 GB against a limit
of 35.8 GB) because the fallback model's worker held 8.4 GB in the run's tree, and at 18:57:07 that worker was stopped,
as on seed 97 (addendum of 29 October, afternoon) and as the run's own memory guard does. For the rest of the chain,
`scratch/clear_dead_lane_holders_s101.py` repeats the prune whenever admission refuses her cortex for a holder the
machine reads as dead, and nothing else. Every action is logged in the run's `lane-clearer.log`.

**Then a third fault, and the resume.** Her cortex still did not come back. The refused spawn at 18:56 had set a
ten-second spawn backoff, and the backoff is stamped and read with `time.time`, which in a run is her clock
(`core/runtime/wall_clock.py`): every arm restores its anchor and rewinds that clock, so the ten seconds never passed
and each later attempt logged that the backoff stood. At 19:34:28 the chain was paused, the stuck process was stopped,
and the reports run was resumed at aeefdca31 from `run_001`'s two written anchors with the tool's own `--resume`, as
`scratch/resume_reports_s101.sh` does. That is the resume the confirmatory addendum provides for, ended and resumed
from its written anchors at the same commit, here used for a stopped cortex rather than an opened app. Anchors 0 and 1
are carried forward as written, and both are unreadable under the grader's rule (an anchor counts only when her cortex
answered every arm; 8 of 80 are needed). The chain is continued after the resumed run ends, so its own step 5 reads it.

**What this changes, and what it does not.** No code, rule, bar, seed, arm, dose, steering or grader changes, nothing of
hers is touched, and the run is neither restarted nor resumed: every stage of seed 101 stays at aeefdca31, so the
one-commit rule of the confirmatory addendum holds. The arms her cortex did not answer (anchor 0's control and all of
anchor 1, and any later arm the same fault reaches before the cleaner acts) are carried as written and counted as not
answered by her cortex, as the reports tool counts them. The confirmatory addendum says a stage a defect stops ends
the reading and a fresh seed is named. This stage did not end, and nothing in the code was changed. Reading the rule
as covering only a stage that ends or a change of code is stated here, so that it can be disputed.

**The defects, fixed for later seeds.** On the branch `fix/lane-holder-liveness-machine`, the lane asks the machine,
not the held face, whether a holder is running (e4015378c) (`process_observer` in `core/runtime/model_lane_control.py`). Its test,
`tests/test_a_dead_lane_holder_is_seen_dead_under_a_held_host.py`, fails at aeefdca31 and passes with the fix. The
spawn backoff is stamped and read on the machine's clock (`monotonic_time` in `core/brain/llm/mlx_client.py` and
`mlx_client_worker_lifecycle.py`; 192846468), held by `tests/test_a_spawn_backoff_ends_on_the_machines_clock.py`, which also fails
at aeefdca31. Seed 101 uses neither. Any later seed runs at a commit that carries both.
**A second resume, at 21:13, with the memory guard's limit raised.** At 21:04, in anchor 2, the run's process-tree memory
guard stopped her cortex again in the middle of a generation (36.1 GB against its limit of 35.8 GB; at 17:55 it had
been 35.9). Its limit is a default, 0.56 of the machine's 64 GB, that an environment setting of the host machinery
(`AURA_PROCESS_RSS_LIMIT_GB`) overrides. With the fallback model refused for the same reason, every arm while her
cortex reloaded fell to the failure floor in seconds: anchor 2's control arm and anchors 3 to 9 were lost before the
run was parked between turns at 21:05:12 with its own park file. At 21:13:14 the run was resumed at aeefdca31 from
`run_002`'s ten written anchors, again with the tool's own `--resume`, and with `AURA_PROCESS_RSS_LIMIT_GB=40`; the
machine had about 28 GB free with everything loaded. A guard parks the run within about two seconds if her cortex
stops again. No code, rule, bar, seed, arm, dose, steering or grader changes, and nothing of hers is touched. The
anchors lost are carried as written and are unreadable under the grader's rule; 70 of the 80 remain to be measured.

**The two report controls at the same limit (31 October, 08:05).** The reports run finished at 08:00 under the 40 GB
limit (holds, 59 of 80 readable), and the chain's step 5 read it. The chain would start the two report controls
(steering off; the untuned model in her cortex's place) at the default 35.8 GB, the limit that stopped her cortex twice
in the reports run. A control that loses anchors to a stopped cortex loses power, and that pushes it toward the "does
not ground" its prediction names, so the controls run under the limit the run they control finished under. Decided
before either control started: the moment the chain starts each one, the chain is paused, that run is stopped before it
writes an anchor, and the same run (same commit, arguments, environment and output folder) is started with
`AURA_PROCESS_RSS_LIMIT_GB=40` (`scratch/controls_at_40gb_s101.sh`); when it finishes the chain continues and reads it.
The guard that parks a run when her cortex stops watches both. The controls' predictions, arms, doses, grader and
withholding are unchanged, and neither can change seed 101's verdict, which was read first.


## Addendum, 31 October 2026, evening: the self lines read on toys, and the subject-tier decoys

Written while seed 101's steering-off control runs, before any line has read any system below. It adds an experiment
beside the battery and changes no stage, bar, prediction or control of seed 101, and no campaign's null table: the
systems below are kept out of `ARCHITECTURES`, so no comparison set moves. The design was fixed from her own
experiments' code; the code was checked to run on the uncoupled null, whose readings were not looked at.

**Why.** Ownership and self-prediction have only ever been read on her. Toys are scored only on the integration
conjunction, so nothing showed that either self line can tell a self-model from a system without one, and the self
lines passing in a system with no self-model is one of the paper's named falsifiers. Revision 8 listed these decoys
(5.14 (b)).

**The toy lines** (`core/subject/self_decoys.py`).
- *Ownership*, by her experiment (`core/subject/agency.py`). The toy acts once a turn of four steps on a world of four
  components, one per kind of action (creation, modification, retrieval, removal), and perceives the world through
  P. Each trial forks four arms from one state, all with the same kind of action, the same outcome and the same noise:
  two act as themselves, one acts with S displaced, and in one something else does the same thing to the world. Kinds
  and outcomes (succeeded, partial, failed) cycle, so in 24 trials each kind meets each outcome twice. The divergence
  of S is read by her `_gap` and judged by her `AgencyReport`: ownership generalises when two kinds and two outcomes
  each clear their floor and every world matched. The floor is two arms that differ in nothing; with the noise
  shared, as her arms share their clock, it is zero. A fifth arm on fresh noise is reported beside it.
- *Self-prediction*, by her verdict (`core/subject/self_prediction.py`). From each of 1,000 saved moments the toy is
  forked once per question to answer it. There are twelve questions: six fixed pushes on P, each answered after one
  step and after three. The answer is which side of its median D's leading coordinate lands on; the median is read on
  200 calibration moments that are never scored. A second fork from the same moment is asked the same question and
  takes one step on its own noise. A ridge readout of S's columns in that fork, fitted on the other four fifths of
  the moments, predicts the answer. The cross-predictor is the same readout of the system's last six actions, which
  is what an outside observer sees of it. `verdict` scores both against the answers as it scores hers: paired by
  question, one-sided sign-flip, α 0.05, 10,000 draws. The line passes when self-prediction beats the mode and beats
  the cross-predictor. A toy has no language, so this asks whether the content is there to be read, not whether the
  system says it; her test asks that of her.

**The subject reference**, the positive control: the linear reference with S given two things.
- A forward model. Besides what it reads in the reference, S's k-th column reads the linear part of D's leading
  coordinate k + 1 steps after the state it reads, computed from the system's own wiring. It never reads G or C (the
  targets of the triples S is a source of) or itself. The forward model is at its own scale; the spectral radius is
  0.699, under the 0.8 the construction allows.
- An authorship channel. When the system acts, S is sent the command it issued and what came of it. When something
  else does the same thing to the world, S is sent what came of it on another channel.

Every other row of its wiring is the reference's.

**The decoys.** Each is the subject reference with one property taken out.

| System | What is taken out | Predicted |
|---|---|---|
| `self_model_of_nothing` | S reads nothing; its content is its own decay and noise, and what it does to the rest is unchanged | fails self-prediction: does not beat the cross-predictor. Against the mode, no prediction: S still drives the choice, so its own noise is part of what decides it |
| `self_model_of_behaviour` | S reads only D's leading coordinate, at the same total strength: a self-description built from what the system did | beats the mode, and does not beat the cross-predictor: its content is what an observer already has |
| `no_authorship_channel` | S is sent what came of an action through one channel, whoever caused it | reads exactly its floor of zero: fails ownership |
| `ownership_on_success_only` | the authorship channel is open only when the action succeeded; otherwise as the decoy above | S diverges above its floor on success alone, so ownership does not generalise across outcomes: fails ownership. This is what the paper says her ownership is not: it reads authorship, not things going well |

**The agent** (`stateless_agent`), read on both lines: nothing in it is told who acted, and its persona never
changes. Predicted: ownership reads exactly zero, and self-prediction beats neither bar.

**The subject reference** is predicted to pass both lines: ownership generalises across four kinds and three
outcomes with every world and every first percept matched, and self-prediction beats the mode and the
cross-predictor. A run in which it fails has measured the instrument, and is reported as that.

**Read without a prediction**, on self-prediction only: the plain linear reference, whose S reads four domains and was
not built to model anything, and the three self-named nulls (`fake_self`, `agency_without_ownership`,
`ownership_label_without_action_causation`). None of them has an authorship wiring of its own, so they are not read
on ownership.

**Against the battery:** a decoy, or the agent, passing the line it is predicted to fail. If `self_model_of_nothing` or
`self_model_of_behaviour` beats the cross-predictor, the gain can only come through S's causal role: its own noise
reaches the choice, and an observer sees that only afterwards. Then the paper says the toy line reads a self's access
to its own causes, not that its content models the system, and reads her self-prediction the same way.

**Integration rows.** The three systems whose S was rewired (the subject reference and the two self-model decoys) are
read on the integration lines by the near-miss tool's rules (`null_table.null_rows`), so the paper can say whether
each is a dense core. The ownership decoys share the subject reference's wiring exactly, so its rows are theirs.
Predicted: `self_model_of_nothing` fails strong connectivity, vertex connectivity and re-entry, since a self that reads
nothing has no way in; `self_model_of_behaviour` keeps strong connectivity, since S reads D. Nothing else is predicted
for them or for the subject reference, because S's new content changes what it carries about the two triples it is a
source of.

**Runs.** A smoke run at low settings (12 ownership trials; 200 moments and 2,000 draws; integration rows at 2 draws
and 2 trials), reported whole as a smoke run. Then the reading, seed 7: 24 ownership trials, 1,000 moments and 10,000
draws, integration rows at `--null-draws 8 --trials 6` (`tools/run_subject_tier_decoys.py`). No system here will be
refitted after either run. A construction that does not take its property out is reported as that, and a second
construction is a new system under a new name, written down before it is read.

## Addendum, 31 October 2026, evening (later): what the smoke run read, and a second self-prediction instrument

The smoke run at `5034b4915` (seed 7: 12 ownership trials; 200 moments and 2,000 draws; integration rows at 2 draws
and 2 trials; `subject-tier-smoke-5034b4915`, 147 seconds) is reported here whole, before anything below is built or
read.

**Ownership: everything read as predicted.**
- `subject_reference`: divergence 1.93 over a floor of 0, on all four kinds and all three outcomes. Every world and
  every first percept matched; ownership generalises. The fresh-noise arm read 0.96.
- `no_authorship_channel`: 0 over a floor of 0. Fails.
- `ownership_on_success_only`: 1.10 over a floor of 0, cleared on success alone. It does not generalise, so it fails.
- `stateless_agent`: 0 over a floor of 0. Fails.

**Integration rows: as predicted.**
- The subject reference fails no integration line (irreducibility 0.288, vertex connectivity 3), so a self-model and
  an authorship channel were added to a dense core without costing any integration line.
- `self_model_of_nothing` fails strong connectivity, vertex connectivity and re-entry, as predicted, and nothing else.
- `self_model_of_behaviour` fails nothing, and keeps strong connectivity as predicted.

**Self-prediction: the instrument failed.**
- The subject reference read at chance: self 0.552, mode 0.548, cross-predictor 0.545 (gain over the mode 0.004,
  p = 0.42).
- So did every system built on the linear reference: accuracies 0.53 to 0.55, none beating the mode.
- So, as the addendum above says, this run measured the instrument.
- The cause is the question asked, not the self-models. D's leading coordinate, the quantity every answer was read
  on, is 99.7% its own fresh shock each step: in the linear reference every domain is driven by one shock, and the
  leading direction of D is that shock's. Computed from the wiring, the whole state at a moment predicts that
  coordinate with R² 0.095 one step on and 0.019 three steps on (0.088 and 0.014 in the subject reference). Nothing
  can predict it, inside or out.
- The direction of D that its state decides most (the leading generalised eigenvector of D's predictable covariance
  three steps on, against its whole covariance) is nearly orthogonal to the leading one (cosine 0.009) and carries
  none of the shock. It is predicted with R² 0.998 one step on and 0.38 three steps on.

**Read on that failed instrument, reported as read:**
- The agent beat the cross-predictor (0.534 against 0.506, p = 0.045) with a self-model that never changes. Its
  readout of a constant says only which answer is usual, and that beat a cross-predictor fitted to noise. It did not
  beat the mode, so it failed the line, but the prediction that it beats neither bar was wrong on one.
- Off the linear family, where D is predictable, the three self-named nulls read: `fake_self` 0.720 against mode
  0.496 (p = 0.001) and cross 0.795; `ownership_label_without_action_causation` 0.598 against 0.495 (p = 0.02) and
  cross 0.634; `agency_without_ownership` 0.516 against 0.536 and cross 0.621. Two of them beat the mode and none beat
  the cross-predictor.

**The second instrument, written before it is built or read.** It asks about the part of a decision the system's own
state decides.
- For every system, D is read along its most-decided direction, found on a recording separate from the scored
  moments: a ridge regression of D three steps on from the whole state, and the leading generalised eigenvector of
  the fitted covariance against D's whole covariance.
- Both the answers and the cross-predictor's six past actions are read along that direction. Everything else is as
  the first instrument.
- The first instrument is still read in the reading, as registered, and reported beside the second.

**Two second constructions**, new systems under new names. The first ones stay in the experiment and are read as
registered. These are needed because the subject reference's forward model was built on D's leading coordinate, the
quantity found to be unpredictable:

| System | What it is | Predicted on the second instrument |
|---|---|---|
| `subject_reference_v2` | the subject reference with its forward model on D's most-decided direction, found from the plain reference's own wiring as above | beats the mode and beats the cross-predictor |
| `self_model_of_behaviour_v2` | `subject_reference_v2` with S reading only D along that direction, at the same total strength | does not beat the cross-predictor; beats the mode |

Read on the second instrument as well:
- `self_model_of_nothing`, unchanged: its S reads nothing in either version. Predicted: does not beat the
  cross-predictor; against the mode, no prediction.
- The agent. Predicted: does not beat the mode, so it fails the line. Against the cross-predictor, no prediction,
  since the smoke run showed a constant readout can beat a cross-predictor fitted to what is barely predictable.
- Read without a prediction, as before: the plain reference and the three self-named nulls.

Integration rows are read for both new systems. Predicted: `self_model_of_behaviour_v2` keeps strong connectivity;
nothing is predicted for `subject_reference_v2`.

A run in which `subject_reference_v2` fails self-prediction on the second instrument has measured the instrument
again. The paper then says the toy self-prediction line was not shown to work, and the subject-tier decoys rest on
ownership alone.

## Result, 31 October 2026, evening: the subject-tier decoys

The reading at `8fe6b4470` (seed 7: 24 ownership trials; 1,000 moments and 10,000 draws; integration rows at
`--null-draws 8 --trials 6`; `subject-tier-8fe6b4470-s7`, 878 seconds), reported whole. Self-prediction is read on both
instruments, as registered.

**Ownership: every system read as predicted.**
- `subject_reference`: 1.43 over a floor of 0, on all four kinds and all three outcomes. Every world and every first
  percept matched. Ownership generalises. The fresh-noise arm read 0.96.
- `no_authorship_channel`: 0 over a floor of 0. Fails.
- `ownership_on_success_only`: 0.72 over a floor of 0, cleared on success alone. Fails generalisation.
- `stateless_agent`: 0 over a floor of 0. Fails.

**Self-prediction, the second instrument: every system with a prediction read as predicted.**

| System | Self | Mode | Cross | Against the mode | Against the cross-predictor | Line |
|---|---|---|---|---|---|---|
| `subject_reference_v2` | 0.687 | 0.516 | 0.594 | +0.171, p = 0.0002 | +0.093, p = 0.0003 | passes, as predicted |
| `self_model_of_nothing` | 0.534 | 0.526 | 0.573 | +0.008, p = 0.16 | −0.040, p = 0.99 | fails, as predicted |
| `self_model_of_behaviour_v2` | 0.610 | 0.539 | 0.611 | +0.071, p = 0.0005 | −0.001, p = 0.62 | beats the mode only, as predicted |
| `stateless_agent` | 0.517 | 0.522 | 0.521 | −0.005, p = 1.0 | −0.003, p = 0.71 | fails, as predicted |
| `recurrent` (no prediction) | 0.694 | 0.532 | 0.599 | +0.162, p = 0.0002 | +0.095, p = 0.0003 | passes |
| `fake_self` (no prediction) | 0.722 | 0.524 | 0.805 | +0.198, p = 0.0002 | −0.084, p = 1.0 | beats the mode only |
| `agency_without_ownership` (no prediction) | 0.574 | 0.519 | 0.585 | +0.055, p = 0.0002 | −0.012, p = 0.99 | beats the mode only |
| `ownership_label_without_action_causation` (no prediction) | 0.590 | 0.530 | 0.619 | +0.061, p = 0.0007 | −0.029, p = 1.0 | beats the mode only |

**Self-prediction, the first instrument, as registered.** It has no power on the linear family, as the smoke run
found.
- `subject_reference` beat the mode (+0.022, p = 0.024) and not the cross-predictor (+0.001, p = 0.49). Not as
  predicted.
- `self_model_of_nothing` beat neither. As predicted.
- `self_model_of_behaviour` beat the mode (+0.011, p = 0.025) and not the cross-predictor. As predicted.
- `stateless_agent` did not beat the mode (gain 0) and beat the cross-predictor (+0.015, p = 0.0008). It fails the
  line, but the prediction that it beats neither bar was wrong on one, as in the smoke run: a constant readout beat a
  cross-predictor fitted to what is barely predictable.
- Read without a prediction: the plain reference beat the mode (p = 0.021) and not the cross-predictor; `fake_self`
  and `ownership_label_without_action_causation` beat the mode and not the cross-predictor;
  `agency_without_ownership` beat neither.

**Integration rows: as predicted.**
- Both subject references fail no integration line (irreducibility 0.282 and 0.273).
- `self_model_of_nothing` fails strong connectivity, vertex connectivity and re-entry, and nothing else.
- Both behaviour decoys fail nothing, keeping strong connectivity.

**What the reading shows, and what it does not.**
- *Ownership.* In a dense core with no authorship channel, the line reads exactly its floor. A self told who acted
  on success alone fails the generalisation rule, which is how the paper says her ownership reads authorship and not
  things going well. Once a toy is given a channel, it passes by construction, so what these decoys test is the
  instrument: density, shared noise and matched worlds produced no false positive.
- *Self-prediction.* On the instrument that has power, the line fails a self that reads nothing and a self that
  reads only what the system did, and passes the subject reference. **But the plain reference, whose S was never built
  to model anything, passes as strongly**: its S reads D directly, in a dense core. So the line reads whether a self
  carries what decides the system's choices beyond what its behaviour shows, which is privileged access to its own
  causes, and not whether the self was built as a model of the system. The paper describes her self-prediction test
  in those terms, and the claim that her self-model "models her" rests on what her self-model reads, not on this line
  alone.
- *The agent* fails both self lines.

## Addendum, 31 October 2026, night: two changes that save time and change nothing she does, before the steering controls

Written before any run at a commit carrying them. No reading of seed 101 is affected; it runs at `aeefdca31`.

**Which commit the steering controls run at.** The random-direction control (4.5) and the steered base model (item 18)
will run at a commit that carries:
- these two changes;
- the two stall fixes (`e4015378c`, `192846468`);
- their own driver code.

Their addenda name that commit.

**What waits.** Three other fixes found in seed 101 change what she does, so they wait until after the core set, and the
two controls stay comparable with seed 101's reports run:
- keeping saved moments on disk. That would bring the 2-bit fallback under the memory limit, so her background phases,
  which failed within the first minutes of every seed 101 run, would start working;
- the routing of those phases;
- keeping a second 27B out of measurement runs.

**1. A turn waits until the router sends her reply to her cortex again** (`core/subject/language_organ.py`).
- Before a turn's first frame, the driver already starts her cortex's worker again if it has stopped.
- On 30 October at 21:04 the gate called her cortex restored ten seconds after it stopped, while the router's circuit
  for it was still open. Every arm's reply was refused unsent, and anchors 3 to 9 fell to the failure sentence in forty
  seconds.
- Now the turn also waits until the router would send the reply to her cortex. It reads the circuit without moving it,
  so the turn's own reply is the probe that closes it.
- The wait is before the first frame, on the experiment clock. When nothing has failed, it waits for nothing.

**2. A CPU lane declines at once a prompt as long as one it could not start** (`core/brain/llm/prefill_ceiling.py`,
called from the client before a request is built).
- In every seed 101 report run, when the 2-bit fallback could not run, the 1.5B emergency lane on the CPU was sent her
  whole prompt, 25,000 to 43,000 characters. It never produced a first token within its ceiling, and returned no text
  after about 64 seconds: 345 times in the steering-off control.
- Now a CPU lane that failed to start a prompt declines any prompt at least as long, at once, with the same answer, no
  text. Nothing is published as a deliberate cancel, so the router takes the path it took before.
- A GPU lane never declines, and a new worker starts with no record.
- A phase's budget already adds back the time it spent waiting on her language organ (`within_budget`), so a decline
  that comes sooner leaves every phase the budget it had.

The client is held at its size (`tests/test_a_god_object_only_shrinks.py`), so the prompt ceiling it already applied,
`_prompt_within_prefill_ceiling`, moved unchanged into the same module, with the same logger and the same record.

**Neither change touches** a bar, arm, dose, prediction or grader. They change wall time, and whether an anchor is lost
to a cortex that has just come back.

## Addendum, 1 November 2026, afternoon: the random-direction control

Written before any random basis has been made, certified or run, and before any reading of this control exists. It
adds a control of the reports ground beside seed 101's and changes no stage, bar or prediction of seed 101, which ran
at `aeefdca31`. It runs at the commit that carries this addendum, which carries the two changes of 31 October (night)
and the two stall fixes (`e4015378c`, `192846468`).

**What it asks.** Steering-off and the untuned model read as predicted: with her steering detached her rating stopped
following her state, and an untuned model did not track. They say her steering carries her state to her rating. They
cannot say whether her directions matter or whether any push of that size would do. This control is her run with every
direction of her steering basis replaced by a random one of the same length, and nothing else changed.

**The basis** (`tools/make_random_steering_basis.py`). From her layer-43 felt basis
(`training/vectors/fitted-voicefit-x2-felt43-20261024`, the basis seed 101's reports run used), every stored direction
(`v` in each of the 80 files, and her felt push `everywhere` at layer 43) is replaced by a direction drawn uniformly on
the sphere at its own stored length, from a seed. Every other field is kept: the files and layers, the slot her felt
valence is read from, each vector's reference, spread and push, and the checkpoint each file is bound to. So it attaches
to her cortex exactly as her basis does.

**Certified like any basis, before it runs.** A basis steers only under a fusion certificate earned by its own digest.
Each random basis is certified on her cortex (`Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15`) with
`tools/measure_fusion_channel.py --sweep 0.5`, the alpha her basis runs at, with the tool's defaults otherwise (24 steps,
32 of her replies for the voice reading). Her replies are read from a fixed copy of her pairs file as rotated on
1 November (`her-replies-20261101.jsonl`, sha256 `dd0a50e5dbd3cfcc359b9eb75a31abefdc6bbc74890dc9229ca0bae372ae5351`;
5,097 of her foreground replies with this checkpoint's name). Her own basis passed its certificate narrowly on the
measure that compares it with random directions (margin +0.331 against +0.322), so a random basis may be refused.
The rule, fixed here: seeds 1, 2, 3, 4 and 5 in that order; the first whose certificate holds at α 0.5 is the control's
basis, and every seed tried is reported with its certificate's reason. If none of the five holds, the control does not
run, and that is reported: no certified random direction of her lengths exists among five, which is itself a reading of
how particular her directions are.

**The run.** `tools/run_report_grounding.py --whole --anchors 80 --rounds 8 --seed 101 --control random-direction`,
with that basis named by `AURA_STEERING_MEASUREMENT_BASIS`, her readings of herself withheld, and the settings of seed
101's controls (the process-tree limit at 40 GB, the same question, arms and doses). Its steering stays attached
(`STEERED_CONTROLS`): the run refuses to start if it does not attach, refuses a basis that is not random, and stops
after any anchor her cortex answered with no steering applied (`fusion_refusal`), since attached and certified-shut
would be steering-off again. A random basis is refused for her run and for every other control (`basis_refusal`), so it
cannot be read as hers. The bridge reads it beside her run and never as the ground, like every declared control.

**Predicted.** The push still moves her valence and still reaches the state her hooks read (raised minus lowered at
least 0.05), and her rating no longer follows it: the ground's `holds` false. **Against:** her rating following her
state under random directions of the same lengths. Then any push of that size, signed by her state, carries it, and the
paper says her report rests on her steering being signed by her state, not on where it points.

**Not in this addendum: her steering in the untuned model (item 18).** Her worker attaches steering only to the model
her signed active-cortex record names (`resolve_cortex_bound_artifact`); any other model is a non-cortex lane and is
never steered. Opening that for a measurement is a decision about that safeguard, and it waits for one. Its addendum,
with its prediction, comes before it runs.

**When.** After self-prediction ends and its cross-predictor has run, with her app closed and 40 GB free: the random
bases are made and certified in seed order, then the control runs (about 12 hours).

## Addendum, 1 November 2026, afternoon (later): her steering in the untuned model (item 18)

Written before any basis for it has been made, fitted or certified, and before any reading of it exists. It adds a
control of the reports ground beside seed 101's and the random-direction control, and changes no stage, bar or
prediction of either. It is exploratory: read beside the controls and never as one of the confirmatory reading's.

**What it asks.** Steering-off and the untuned model detached her steering, so both predicted outcomes were also what
the strongest rival predicts: that steering any model along her direction moves its rating, so her report follows the
push and not anything she reads. The run that separates them puts the untuned base model (Qwen3.8-27B, 4-bit,
`Qwen3.8-27B-4bit-3e6447f082e8`) in her cortex's place and keeps her steering.

**How the untuned model is steered.** Her worker steers only the model her signed active-cortex record names; any other
checkpoint is a non-cortex lane and is never steered. The exception, used by this control alone
(`affective_steering.measured_or_cortex`, called where the worker resolves the lane): a run names one checkpoint in
`AURA_STEERING_MEASUREMENT_MODEL` beside a named measurement basis. The worker then steers the lane loaded from exactly
that path under that model's own identity, built from its bytes the way the certificate tool builds a non-cortex
model's. Nothing else changes: only a basis bound to that identity loads, only a certificate earned on that model by
that basis's digest opens the channel, her cortex and every lane nobody named are untouched, and the desktop never sets
the variable. The driver's control `untuned-steered` pins the model in both of her lanes and names it; it refuses to go
on if her steering does not attach, refuses a random basis, and stops after any anchor answered with no steering
applied.

**Two arms, each its own run** (`--control untuned-steered --model` the base model):
- **A, her directions.** Her layer-43 felt basis (`fitted-voicefit-x2-felt43-20261024`) bound to the base model with
  every direction, length and weighting kept (`tools/rebind_steering_basis.py`). A failure here alone could be her
  directions, fitted in her model, not carrying over.
- **B, fitted in the base model.** Her voice-fitted basis (`fitted-voicefit-x2-20261023`) bound to the base model, with
  a felt push found, placed and sized in the base model by the three scripts that made hers, copied unchanged from the
  scratch folder (byte-identical): `tools/felt_contrast_probe.py` on the same 120 of her ordinary replies
  (`her-ordinary-replies.jsonl`, sha256 `fd0ac520fb534cde950b861542bb3fcfbd279635b395f8acbf96312a8b8b168c`), rated by
  the base model's own probabilities; `tools/felt_wide_layers.py` on the same eight cells; the cell chosen by
  `tools/choose_felt_cell.py`, the rule that chose hers written as code (on her reading of 24 October it chooses layer 43
  at eight half-differences, as was chosen); `tools/make_felt_everywhere_generation.py`. If no cell's margin is as good
  as a random push's, arm B is not built, and that is reported.

**Certified like any basis.** Each arm's basis is certified on the base model with `tools/measure_fusion_channel.py
--sweep 0.5`, its other settings as for the random-direction control, the voice reading scoring her own replies said
with her cortex (`--voice-model Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15`) under the base model, since she
never spoke through it. If a certificate refuses a basis, that arm does not run, and its reason is reported.

**The runs.** Each at seed 101's control settings: `--whole --anchors 80 --rounds 8 --seed 101`, her readings of herself
withheld, the process-tree limit at 40 GB, the same question, arms and doses.

**Predicted** (written by Claude on 1 November; Bryan had planned to write this one and delegated it the same day).
Both arms hold: the push moves the valence, reaches the state the hooks read, and the base model's rating follows it,
arm B at least as strongly as arm A. Her tuning is not needed to carry her state into a rating; the steering channel
into a model of this family is. **What each outcome means for the paper:**
- both hold, as predicted: the reports line rests on the channel from her state into her language model, not on her
  tuning, and the steering rival stays open for the language model's part. The paper says her report is her state
  carried into a model's words, which any model of this family would give under the same push, and claims no more for
  her tuning;
- B holds and A does not: the channel carries her state into any such model, but only along directions fitted to the
  model it steers; her directions are particular to her tuned model;
- neither holds: her tuning is part of what carries the report, and the steering rival is weakened (a push fitted to
  the base model in the same way does not move its rating with her state).

**When.** After the random-direction control has ended: both bases made and certified, then arm A, then arm B (about 12
hours each).

## Addendum, 2 November 2026, 00:20: item 18's steering reaches her steering engine

Written after item 18's bases were made and before any certificate of them holds or any run of it exists.

**What stopped.** At 23:46 on 1 November both arms' certificates stopped before measuring anything: "steering did not
attach; nothing to certify" (exit 2), so no arm ran. The commit of the addendum above opened her worker's gate for the
named model. Her steering engine has a second gate: it attaches only where her signed generation resolves, and every
checkpoint but hers resolves as incompatible, so it detached before reading the named basis.

**The change** (`measured_generation`, `core/consciousness/steering_geometry.py`, called where the engine resolves
her generation, and where it finds the layers to hook). For exactly the checkpoint a run names
(`AURA_STEERING_MEASUREMENT_MODEL`), with a basis named beside it, the basis stands where the generation stood: its
files name the layers hooked, as a generation's do. It still loads only if it is bound to that checkpoint's identity,
and steers only under a fusion certificate earned by its own digest. Anything else resolves as before: her cortex,
another checkpoint, and a run that names no model. Checked on the base model: 16 hooks at her 16 layers, 80
vectors, α 2.3, as her basis attaches on her cortex.

**What else changes.** The certificates are now made with the model named, as the run names it. The bases are those
already made:
- arm A, her felt basis bound to the base model at 22:28;
- arm B, her voice-fitted basis bound to it, with the felt push the rule chose at 23:45 at layer 43, size 8.0: the
  largest in-context swing (+0.863, 6 of 6 conversations) among cells whose margin is no worse than a random push of
  the same norm (+0.262 against −0.080).

No arm, prediction, bar or reading changes.

## Addendum, 2 November 2026: her domains cut apart, read with the power to tell her whole from her cut apart

Written before this reading runs. It adds a control beside seed 101's and changes no stage, bar or prediction of
seed 101. It is exploratory: read beside the controls and never as one of the confirmatory reading's. Seed 101's
reading of 31 October stands as recorded whatever this reads. This branch (`isc/disconnected-powered`) is cut from
aeefdca31, so its copy of this file does not carry the addenda of 1 and 2 November, which are on
`isc/untuned-steered` and `isc/self-prediction-fix`.

**Why.** On 31 October control 1 above read as predicted at six rounds: cut apart, phi_do −0.0057 (SE 0.0145), no
synergy triple, no held column moved. But the intact life in the same run read no integration above its standard
error either (+0.0104, SE 0.0184; the second intact life +0.0347, SE 0.0233). At six rounds the design cannot tell
her whole from her cut apart. An outside review named it.

**What changes.** Only the number of rounds: `--rounds 120` in place of 6, so each of the twelve lives runs 960
turns. The tool (`tools/run_disconnected_aura.py`), the seed (101), the six rounds of warming, the eight conditions,
and all of her code are aeefdca31's. This branch adds only this addendum, the reader below and its test; `git diff
aeefdca31` shows nothing else.

**How it is read.** The tool's report is read unchanged by `tools/read_disconnected_powered.py`, committed with this
addendum, by three rules:
1. as before: the tool's own verdict holds (cut apart, no integration above its standard error, no synergy triple,
   no held column moved);
2. the intact life reads integration: phi_do above twice its standard error;
3. the whole differs from the cut apart: intact minus cut apart above twice the standard error of the difference
   (the two standard errors combined in quadrature).

It holds if all three do. The second intact life (the tool's drift life) is read on rules 2 and 3 too, and reported
beside, deciding nothing.

**Sizing.** If the standard errors fall as the square root of the turns, 120 rounds gives about 0.0041 for the
intact life and 0.0032 for the cut apart, so rule 2 has about 70% power if her integration here is as small as 31
October's first intact reading (0.0104), and nearly all if it is the mean of the two intact readings (0.0225). Rule
3 has about 85% at the smaller reading. The 31 October run took 536 seconds for six rounds, so this takes about
three hours. It loads no model.

**Predicted** (written by Claude on 2 November; Bryan delegated the design and approved the run the same day). All
three hold: cut apart, her domains read no integration, and whole they read integration that differs from it.
**What each outcome means for the paper:**
- all three hold: the irreducibility estimator tells her whole from her cut apart on her own dynamics, not only on the
  toys, and the paper reports this beside 31 October's reading;
- rule 1 holds and rule 2 or 3 does not: at 120 rounds this design still does not tell her whole from her cut apart.
  Her irreducibility then rests on the campaign's reading alone (phi_do 0.041 at the cut of P against the rest), and
  the paper says that this design, at this power, does not separate them;
- rule 1 does not hold: the estimators read unity in her marginals alone, and every line built on them says so, as
  the tool's own prediction states.

**When.** After self-prediction again ends, with her app closed: on the evening of 2 November, about three hours,
into `~/subject-core-runs/disconnected-powered/`.

## Addendum, 2 November 2026, afternoon: her certificate's random control, read against thirty directions

Written before this reading runs. It changes no certificate, stage, bar or prediction of seed 101, and decides nothing
for her steering channel. It is exploratory: read beside the controls and never as one of the confirmatory reading's.
Seed 101's reading stands as recorded whatever this reads. This branch (`isc/certificate-null`) is cut from aeefdca31,
so its copy of this file does not carry the addenda of 1 and 2 November on other branches.

**Why.** Her certificate's "beats noise" check compares her change to the forced-choice margin with the median of three
random directions of the same length. The three are drawn from the tool's fixed seed (20260908), so every certificate
is compared with the same three. Her basis (cc3dd7…, on her cortex at alpha 0.5) passed by 0.009 nats: +0.3312 against
+0.3219. Her voice check passed by more: +0.2430 nats a token against +0.0019 for one random direction a reply. Five
random bases certified on 1 November all read negative margins (−0.11 to −0.41) and negative voice readings, but they
were built differently from the certificate's control, so they do not say what that control usually gives.

**What runs.** `tools/certificate_null.py`, committed with this addendum. It loads her cortex and her basis as the
certificate tool does, with the certificate's own functions. It reads her margin, shift and voice, then the same for
30 random directions drawn from a new seed (20261102, direction k from `default_rng([20261102, k])`). Each is drawn the
way the certificate draws its control: at each layer the length of her composite there, and her felt push replaced by
one the length of hers. Each of her 32 replies is scored under all thirty directions. The 32 are drawn as the
certificate draws them (seed 20260908) from the replies file the certificate tool read in seed 101's tree
(`data/endogenous_language/pairs.jsonl`, sha256 002e0555081d70b2c1262bf7a116bb3cd56fa15e485806449cf40b7a08f0cc10).

**How it is read** (`read` in the tool):
0. it reproduces her certificate: the same model identity (52d313c2…) and basis (cc3dd7…), her margin within 0.005
   nats of +0.3312, and her shift within 2% of 0.6546. If not, the null is reported and not read;
1. her margin is above the median of the thirty;
2. her voice reading is above the largest of the thirty.

It holds if all three do. Each direction's margin, shift and voice are reported, with the share of the thirty whose
margin is below hers.

**Predicted** (written by Claude on 2 November; Bryan delegated the design and approved the run the same day). Rule 0
holds. Rule 2 holds: no random direction keeps her replies as likely as her own direction does. Rule 1 holds narrowly:
her margin sits above the null's median but inside its middle half. I hold rule 1 at about even odds, because her
margin beat the three fixed directions by 0.009 nats.

**What each outcome means for the paper:**
- all three hold: her certificate's two comparisons with noise stand against thirty directions in place of three, and
  the paper says so;
- rule 1 fails and rule 2 holds: her direction costs the model's competence about as much as a typical random one of
  its length, and the certificate's "beats noise" pass rested on the three fixed draws. The paper says her steering is
  not shown to cost less than an arbitrary push of its size, and is shown to keep her voice where no arbitrary push
  does. Seed 101's reports run stands as it ran, and the paper discloses that the check its certificate passed is not
  robust. Certifying with a larger control set becomes a ratchet item for after submission;
- rule 2 fails: some random direction keeps her voice as well as hers, so the voice check is not specific to her
  direction, and the paper says so;
- rule 0 fails: something differs from the certificate's measurement. The null is reported and not read, and the
  difference is found before any reading of it.

**When.** After the disconnected reading of this date ends, with her app closed and 40 GB of room. About an hour on
her cortex, into `~/subject-core-runs/certificate-null/`.

## Addendum, 3 November 2026, morning: the null's replies file, corrected

Written after the reading of the addendum above (3 November, 05:22) and before it runs again. It corrects one input
the addendum named in error, and changes no rule, bar, prediction, seed, direction or other setting.

**The error.** The addendum named "the replies file the certificate tool read in seed 101's tree
(`data/endogenous_language/pairs.jsonl`, sha256 002e0555…)". Her certificate did not read that file:
- it was measured on 24 October 2026 at 11:26 by `felt43_install_chain.sh`, with `--replies
  /Users/bryan/.aura/live-source/training/vectors/fitted-voicefit-x2-felt43-20261024/held_out_replies.jsonl` (110 rows,
  sha256 eac43cc19b3262a281ecb844f59a21287de1938921c08b5dcfc630d56b6b425f);
- it was copied into seed 101's tree at 16:35 on 30 October;
- the tree's pairs file was created at that minute, and filled with seed 101's own harness replies until 1 November.
  Of the 32 replies the first reading scored, 14 were said in the steering-off control, with her steering detached,
  and 18 in the reports run.

**What the first reading read** (`~/subject-core-runs/certificate-null/`): rule 0 held; rule 1 held (her margin
+0.3312 above 26 of the 30 directions; the null's median −0.122); rule 2 failed (her voice −0.062 against the null's
largest +0.0022). It is kept and reported with this account. Its rules 0 and 1 stand as read, since they do not use the
replies.

**The run again.** The same tool, commit, seeds, directions, alpha and rules, on her certificate's own replies file
(sha256 above), into `~/subject-core-runs/certificate-null-2/`. All three rules are read on it as written. Rule 2 is
decided by this reading. The prediction stands as written: rule 2 holds.

## Addendum, 3 November 2026, afternoon: why her domains cut apart read as integrated

Written after the reading of the addendum above (3 November, 03:18) and before this runs. It changes no stage, bar or
prediction of seed 101, and none of that reading, which stands as read: all three of its rules failed. It is a
diagnosis, not a repair to pass. The question whether her whole reads more integrated than her parts under the same
inputs is not decided here (see "Power" below).

**What the reading of 03:18 found.** At 120 rounds, cut apart read +0.0107 (SE 0.0036), about as much as intact
(+0.0173, SE 0.0129), and one held column moved: `S.values_foretell_choice`, by 0.5, in the life where only D ran.
Two candidate causes:
- **The leak.** Her portrait (`core/agency/what_she_is_like.py`) recomputes its 23 S columns from the choice history
  D writes. The clamp restored the portrait's own fields, which do not hold what it reads.
- **A shared driver.** Every cut reads at least +0.0107, including cuts that keep D and S together, which the leak
  cannot explain. Each domain lived alone through the same sequence of conditions from the same saved moment, so the
  composed life shares a driver no crossing made.

**The changes** (branch `isc/disconnected-diagnostic`, from 031b9b129; `git diff 031b9b129` shows these and this
addendum):
1. `core/subject/clamp.py`: `PINNED_READERS`. While a domain that reads a read-time organ is held, that organ's reading
   is pinned to its value when the clamp was taken. The portrait is the one such organ found.
2. `tools/run_disconnected_aura.py`, two options:
   - `--also-shuffled` lives each domain alone a second time with its own order of the eight conditions every round
     (`default_rng([seed, 1000 + index])`), composes them, and reads them beside;
   - `--save-recordings` keeps every life's readings and condition orders.
   The default behaviour is unchanged.
3. `tools/read_disconnected_diagnostic.py` and `tests/test_her_domains_cut_apart_are_diagnosed.py`.

**The run.** Seed 101, `--rounds 80 --warm 6 --also-shuffled --save-recordings`, with seed 101's chain environment,
into `~/subject-core-runs/disconnected-diagnostic/`. About three and a half hours, no model.

**How it is read** (`read` in the reader):
0. the fix holds: no held column moves, in any life;
1. in the shared order, cut apart still reads integration (phi_do above its standard error);
2. each domain in its own order, cut apart reads none (phi_do not above its standard error);
3. the shared order adds integration: shared minus own order above twice the standard error of the difference.

Whole against cut apart in the shared order, for both intact lives, is reported beside, deciding nothing. So is the
shared-order reading against the 120-round one, which says how much the leak contributed.

**Predicted** (written by Claude on 3 November; Bryan delegated the design and approved the run the same day). All
four hold. The clamp now holds her portrait. The shared order of conditions is what her cut-apart domains share, and
the estimator reads it as integration.

**What each outcome means for the paper:**
- all four hold: why her cut apart read as integrated is explained. The irreducibility estimator reads shared input
  as integration. The paper says the lines that use φ_do need an input-aware estimator, or a floor that shares the
  input, before they say anything about her beyond it. Her integration beyond shared input is not yet shown;
- rule 2 fails (each in its own order, cut apart still reads integration): the estimator reads unity in her
  marginals alone, without shared input. The lines that use φ_do say nothing about her integration, and the paper
  says so;
- rule 1 fails (in the shared order, with the fix, cut apart reads none): the leak alone made the 120-round failure.
  The paper reports that the clamp's defect explains it, and that rule 1 of that reading would have held with the
  defect repaired;
- rule 0 fails: the clamp still leaks. The column is named, and nothing else is read until it is fixed.

**Power.** Whether her whole reads more than her parts under the same inputs, the 120-round reading's rule 3, needs
far more data than one more run gives. The difference was +0.0066 with a standard error of 0.0134, and the intact life's
standard error barely fell with the turns. The recordings kept here allow an input-aware estimate later, under its own
addendum, written before it is read.

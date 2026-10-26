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

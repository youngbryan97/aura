# ISC: what the latest results show

Status: Guide · Source reconciled 29 October 2026 · through 93ef113ee

**Seed 97 passed all 24 checks under ISC-v7. J\* came out determined up to
gauge and the five-ground bridge reads `AT_PARITY`. It is the first seed to meet
all three goals.** It was a confirmatory reading, named before it ran, and ISC-v7
was fixed as the standard that morning, before seed 97 had a total under any
version. Under ISC-v6 the same seed reads 23 of 24; that total is reported beside
the v7 total and never added to it. The completion check still reads incomplete,
for one reason: the reports run was resumed at a different commit, so the
evidence disagrees on commit. The section below says why it was resumed and
what that commit changes.

If you are new to Aura, start with [ISC explained](ISC_EXPLAINED.md). It
introduces the project, translates the test names and includes a short
explanation you can use aloud. The numbers and source records follow below.

## Seed 97, 28 and 29 October

The reading ran at `13cca3bf9` from 21:35 on 28 October to 18:55 on 29 October,
under the stopping rule of the ISC-v6 preregistration's addendum of 27 October
and the wider reference of its addendum of 28 October. The Mac hibernated from
08:08 to 12:46 when its battery ran out; the runs resumed with it, and her
clock's valve never opened. The
[seed-97 evidence snapshot](evidence/ISC_SEED97_2026-10-29.json) holds the
fields quoted here with a hash of each source file.

| Question | Seed 97 | What remains open |
|---|---|---|
| Does the full operational battery pass? | ISC-v7 24/24. ISC-v6 23/24 (fails the observational closure line); ISC-v5 22/24 (also fails synergy) | A seed run whole at one commit, so the completion check can read it |
| Is the proposed carrier established? | The carrier run is authoritative. 511 of 511 cuts were decided, the weakest with a lower bound of 0.0292 at a lag of two turns, and the playback control was decided at none; the carrier is all ten domains, ACDGIMNPSW. Closure by intervention reads `CLOSED` (166 of 959 units varied, alone and all at once); the carrier's own observational gate reads `CLOSED_WITHIN_MARGIN` | Confirmation on a further seed |
| Does the reference pass and every null fail? | The wider recurrent reference passes and none of the nulls passes the rest of the conjunction | |
| Do internal and recall differences change together? | `ONE_STRUCTURE`. Moved together at rho 0.617, p 0.001, against a sham floor of 0; the two geometries agree at rho 0.972 over 190 pairs | Confirmation on a further seed |
| Do self-reports follow changed internal valence? | `holds: true`. Tracking rho 0.358; she rated herself 0.317 higher when her valence was raised than when it was lowered; the displacement moved her valence by 0.119. Each p is 0.0001, the floor of 10,000 draws. 51 of 80 anchors were readable | Measured under the layer-43 felt steering basis, which stays part of the claim |
| J\* | `DETERMINED_UP_TO_GAUGE`: carrier found, structure identified (20 orbits over 20 classes), lineage recorded over 7,855 stages | Conditional on postulates P1 to P6 |
| Do all five bridge grounds hold? | `AT_PARITY`: carrier, markers, reports, structure and lineage hold | Conditional on P1 to P6; not a proof of them |

**Why the reports run was resumed.** At 15:51 the run's process-tree guard
stopped her cortex in the middle of anchor 0. The gate declined to reload it on
a reading of 21.2 GB free. The gate's memory reading is cached for half a
second, and under the experiment clock that half second was timed by her clock,
which a restore rewinds, so the one reading was served for every later reload
while the machine had 32 GB. Each later arm waited 900 s and ended in the
failure floor. The run was stopped and resumed from its written anchors at
`377a40b85`, which is `13cca3bf9` plus two harness changes: the cached reading
ages on the machine's clock, and the driver asks again when her cortex is
stopped and nothing is reloading it. The addendum of 29 October, afternoon,
records this before the resumed run started. Anchor 0 is carried forward with
two of its four arms answered by her cortex; in the resumed run a second stall
of the same family (the reload's memory wait, also timed by her clock) cost one
more arm. Her cortex answered 237 of the 240 arms measured after the resume.

**What the completion check reads.** All three goals are true under ISC-v7. The
check also requires every piece of evidence to come from one commit, and the
reports run came from `377a40b85` while the rest came from `13cca3bf9`. That
rule predates the run and is not changed for it. A seed run whole at one commit
is how the conjunction gets read without that qualification.

## Seed 89, 28 October

The reading ran at `150f2c5fa` from 04:26 to 21:21 under the stopping rule of the
ISC-v6 preregistration's addendum of 27 October. The isolation repair below
(`dcdcd637c`) is in that commit. The
[seed-89 evidence snapshot](evidence/ISC_SEED89_2026-10-28.json) holds the
fields quoted here with a hash of each source file.

| Question | Seed 89 | What remains open |
|---|---|---|
| Does the full operational battery pass? | ISC-v5 22/24 (fails synergy and `beats_every_null`); ISC-v6 23/24; ISC-v7 23/24 (fails `beats_every_null`) | 24/24 on a confirmatory seed |
| Is the proposed carrier established? | The carrier run is authoritative. 511 of 511 cuts were decided and the playback control was decided at none; the carrier is all ten domains, ACDGIMNPSW. Closure by intervention reads `CLOSED` | The carrier's own observational closure gate read `OPEN`; see below |
| Do internal and recall differences change together? | `ONE_STRUCTURE`. Moved together at rho 0.484, p 0.001, against a sham floor of 0; the two geometries agree at rho 0.967 | Confirmation on a further seed |
| Do self-reports follow changed internal valence? | `holds: true`. Tracking rho 0.530, p 0.0001; she rated herself 0.573 higher when her valence was raised than when it was lowered; 57 of 80 anchors were readable | Measured under the layer-43 felt steering basis, which stays part of the claim |
| J\* | `DETERMINED_UP_TO_GAUGE`: carrier found, structure identified (20 orbits over 20 classes), lineage recorded | Conditional on postulates P1 to P6 |
| Do all five bridge grounds hold? | `AT_PARITY`: carrier, markers, reports, structure and lineage hold | Conditional on P1 to P6; not a proof of them |

**The failed check.** The reference row reads no recording of hers and depends
on the seed alone. Over seeds 1 to 200 it failed the same line on 62. The ISC-v6
preregistration's addendum of 28 October, written before seed 97 ran, builds
the reference six columns a domain wide, where it fails on none of 400 seeds.
Under that reference seed 89's battery reads 24/24 under ISC-v6 and ISC-v7.
That result is reported
beside the recorded verdict and never in place of it, because the change came
after seed 89 ran. Seed 97 started at 21:35 at `13cca3bf9`, which carries the
wider reference.

**ISC-v6 and ISC-v7 give different J\* verdicts.** ISC-v7 decides the carrier's
closure by intervention, as preregistered, and reads J\* as determined. ISC-v6's
J\* reads the carrier's observational closure gate instead and reports
`NO_CARRIER`. One of that gate's three folds read open: its core-only loss was
24.7, against 0.73 and 0.50 on the other two. A diagnostic refit of that fold
on the saved recordings traces the loss to K\*'s held columns. The gate enters
them as levels, where it enters the periphery as increments. One of them is a
fixed projection of the self-model's belief list. When the list grew from seven
beliefs to eight it jumped from about 1.02 to -0.52, 777 standard deviations of
its spread over the training turns, and the linear fit extrapolated it into
every target. Entered as increments, the core-only loss
on that fold falls to 0.99. The diagnosis changes no recorded verdict, and an
estimator change needs its own qualification and preregistration first.

## An isolation finding changed how to read seed 23's results

The [25 October isolation update](evidence/ISC_ISOLATION_UPDATE_2026-10-25.json)
records a defect found after the seed-23 outputs were read. Each experimental
arm restored Aura's recorded state but kept a shared history of what previous
arms had done. The system's integration reader consumed that history. One
comparison could therefore influence the next.

Think of two saved-game copies that reset the player but still share a single
scoreboard: the second copy can encounter a result left by the first. The
metaphor illustrates the missing isolation; it does not identify a subject or
supply a new experimental result.

The repair now restores that consequence history with the fork. The upstream
regression account reports zero separation through twelve diagnostic churn
cuts after the fix. That is evidence about the repair, not a replacement
campaign. The scores below remain the outputs of the original experiment.
The effect of clean isolation on each paired result still needs measurement.

## What seed 23's files recorded (25 October)

The seed-23 experiments finished overnight. Two questions that had failed in
previous runs now have positive results:

- **Content:** changes in Aura's internal representation and changes in what
  it recalls moved together under the test intervention.
- **Self-report:** the real 27B language model's ratings followed changes in
  internal valence, with the state numbers kept out of what the model could
  read. This used a certified experimental steering basis, rather than
  establishing that the desktop's installed generation does the same thing.

The completed battery still fails two checks. All 511 split-system comparisons
produced a detectable change, but a supposedly unchanged copy also drifted
enough to trigger the test. That control failure prevents those comparisons
from establishing the two partition requirements.

There is also a separate boundary question: information outside the proposed
core helps predict its future. Later changes to the measurement method now
keep a demonstrated outside influence separate from an unanswered question.
Those code changes do not retroactively turn the completed run into a pass.
The shared-history finding above also limits the causal reading of its paired
positive results.

## Seed 23's recorded results and the method after it

| Question | Last completed result | What remains open |
|---|---|---|
| Does the full operational battery pass? | ISC-v5: 21/24; ISC-v6: 22/24, at `8f548ed79`, seed 23 | Both partition requirements fail under that run's controls |
| Do the specified pairs of parts have joint effects? | All four v6 triples pass within the completed campaign | Confirmation on further seeds and conditions |
| Do internal and recall differences change together? | `ONE_STRUCTURE`, tracking rho 0.378923, p 0.001 | Wider confirmation; the result concerns this measured substrate |
| Do self-reports follow changed internal valence? | `measured: true`, `holds: true`, tracking rho 0.612475 | 29 of 80 anchors were fully readable under the stored policy; experimental basis and sample scope must be retained |
| Is the proposed carrier established? | Completed carrier report: `NOT_FOUND`; completed J*/bridge output: `NO_CARRIER` | Later closure-rule changes require a new reading and fresh confirmation |
| Do all five bridge grounds hold? | Carrier fails; markers, reports, structure and lineage hold in the completed join | No `AT_PARITY` result |
| What does the latest source code require? | At `96487c6f1`, the carrier is K*: K plus declared state its transition reads. The numeric margin is 0.02 / 1.02, about 0.0196; closure also needs an interventional `CLOSED` result | Fresh source-bound measurements on the scored seed, all required controls and confirmation; this is not a new positive verdict |

The completed 22/24 result belongs to the experiment at `8f548ed79`. It is not
an evaluated result for source at `5bc357b8c`. Seed 23 informed the later
method and is discovery data for that revision. The old `NO_CARRIER` output
remains part of the record; no new carrier verdict was produced for this
publication update.

The [25 October evidence snapshot](evidence/ISC_STATUS_2026-10-25.json)
contains selected fields and hashes from the completed battery, carrier,
content, reports and J* files, plus the closure policy at that snapshot. The
[later K* method snapshot](evidence/ISC_K_STAR_METHOD_UPDATE_2026-10-25.json)
records the subsequent policy without rewriting those earlier outputs. The
[24 October snapshot](evidence/ISC_STATUS_2026-10-24.json) is preserved as the
record of the earlier, unfinished state. Neither is a full raw-data package.

## Why the two remaining checks did not pass

### An unchanged comparison moved; later work found shared arm history

The deciding carrier sweep tested all 511 unique divisions of the ten measured
domains. Every cut was decided, and playback was decided at none. However, a
late untouched fork differed from the cached untouched comparison: its lower
bound was 0.011444, against the weakest cut's 0.001256. Drift could therefore
clear the same bar. The scorer refused `partition_irreducibility`; the null
comparison also requires that first result, so `partition_beats_nulls` failed.

The investigation found background host-watch state, an opportunistic record
refresh, and host readings that could change between arms. Repairs stop the
undeclared background activity and hold one declared host reading throughout
the paired sweep. See `1163e7dc3`, `80ef779ff` and `8bca0222d` in the
[October work record](ISC_OCT21_EVERY_ITEM.md).

These earlier host and background repairs did not settle the drift. At
`dcdcd637c`, the investigation traced it to activity carried through the shared
consequence bus rather than elapsed host time. The bus's history now restores
with the fork. This repairs a cross-arm path, while the independent boundary
question below still needs its own measurement.

A second shared record did the same through Aura's sense of threat: the
degradation tracker, whose recent fault reports one of her organs reads. Its
records now restore with the fork too. On the repaired source, a stress test
that ran sixty split-system comparisons and moved the real clocks ten hours
forward found the unchanged copy identical before and after (largest
difference 0). Its positive control, a leftover deliberately planted in a part
the core reads, moved the core by 0.75, so the test can see a leftover when one
is there. This is a pre-flight check, not a rerun of the campaign.

### The selected core may leave useful internal state outside

Closure asks whether the selected core and the declared environment are enough
to predict the core's next step, or whether other internal information helps.
The completed carrier report measured a leak of 0.024286 and did not select a
carrier. An early reassessment found that the measuring rule's resolution was
about 0.0452 on that recording, which limited what it could distinguish.

The 25 October audit separated the outside-host explanation from this leak.
Removing host and metabolism readings from the recorded comparison did not
remove the leak. Some predictive information belonged to organs the selected
core already reads but does not fully contain. Enlarging the recorded core by
an anatomy-based rule reduced the pooled diagnostic leak, but that is a new
choice to test, rather than a confirmation of the original boundary.

At `12bade935`, the rule was changed to condition both predictions on the environment,
uses three forward folds, reports uncertainty and resolution, and reads values
inside containers rather than only their lengths. It can return:

| Decision | Plain meaning |
|---|---|
| `OPEN` | Outside internal information makes a detected contribution |
| `CLOSED_WITHIN_MARGIN` | The evidence bounds the remaining contribution below a margin set by the predeclared rule or numeric override |
| `UNRESOLVED` | The measurement does not justify either conclusion |
| `NOT_MEASURED` | The required outside variation was not measured |

The afternoon `cf0ace7e3` addendum proposed a least-domain margin. It was
**withdrawn** at `72dd0d703`: a domain can be needed while adding almost no
predictive information beyond what the rest's past already carries. Its
smallest predictive contribution is therefore not a justified closure margin.
The [afternoon snapshot](evidence/ISC_CLOSURE_RULE_UPDATE_2026-10-25.json)
remains a record of that proposal, not the active rule.

Current code: `CLOSURE_MARGIN` is `0.02 / 1.02`; `CLOSURE_MARGIN_RULE` is `None`.
The night addendum at `96487c6f1` declares this numeric margin and a second,
interventional closure requirement. The least-domain rule remains withdrawn.

The evening amendment also changes the estimator: it enters every outside
column as an increment and predicts the core's change. A fold where the wider
predictor does measurably worse is marked as an estimator failure; such a
measurement cannot establish closure. The amendment records known-answer
comparisons used to choose the method. Those synthetic results are not a new
Aura carrier measurement. See the
[withdrawal and estimator record](evidence/ISC_CLOSURE_WITHDRAWAL_2026-10-25.json)
and [the evening v6 addendum](ISC_V6_PREREGISTRATION.md).

Carrier selection preserves an unresolved closure question as `UNRESOLVED`,
rather than treating it as a measured absence of a carrier. Neither this rule
nor the isolation repairs supply a replacement for the completed verdict.

A later restore repair at `517c4807c` found that deep-copying containers of
bound methods could bind processors to frozen copies of their organs. Both
experimental twins could then agree while processing through the wrong state.
The restore now treats those method tables as wiring. This is another reason
to recheck the earlier paired results; matching unchanged twins alone does
not establish a faithful intervention setup.

## The night addendum: include what the transition reads, then test both ways

At `96487c6f1`, K* extends the ten-domain core with declared attributes whose
state changes reached the core or an already held attribute in the discovery
probe. The source lists the additions in
[carrier_reads.py](../core/subject/carrier_reads.py). Wall-clock instants and
tables of bound methods are excluded by the declared rule. Held attributes
are read on the core's side, and a cut holding a domain holds its additions.
The periphery reader also now reads deques and nested histories it missed.

The practical issue is an incomplete control panel: if a machine's next step
uses a buffer, recording only the panel lamps omits a cause. Adding the buffer
to the record makes the boundary more faithful; it does not prove the enlarged
boundary is closed or conscious.

The source now requires **both**:

1. An observational result bounding the remaining outside contribution below
   `0.02 / 1.02` (about 0.0196078), with the required resolution, fold and
   estimator-validity controls.
2. An interventional `CLOSED` result on held-out anchors of the scored seed:
   the declared outside-state swaps change neither K's recorded vector nor
   K*'s held readings within two turns, bit for bit, when untouched twins agree.

The numeric choice adapts Cohen's conventional small regression effect:
[f-squared 0.02 converts to partial R-squared about 0.0196, section 9.2.2](https://utstat.utoronto.ca/~brunner/oldclass/378f16/readings/CohenPower.pdf#page=425).
That supports the convention and conversion. It does not establish a natural
boundary for subjects, causal closure in all situations, or the adequacy of
this project's out-of-sample estimator. The two-turn intervention has its own
finite unit, anchor and horizon coverage.

The battery needs both results. J*/the bridge also requires the intervention:
a previously found carrier becomes unresolved when that input is missing,
and not found when outside state moves it. The historical seed-23 outputs
remain unchanged. The [work log](ISC_OCT21_EVERY_ITEM.md) also reports a
post-repair seed-7 synergy probe; that diagnostic is separate from a new full
battery, carrier sweep and joined confirmation. This review ran none of them.

### Later discovery check and another restore repair

The [21:24 upstream commit](https://github.com/youngbryan97/aura/commit/5bc357b8c66bad13eaa8cad4c2ea9b368037537d)
reports that 284 swaps left K and K* unchanged after adding dependencies missed
by the earlier boundary pass. These were seed-23 anchors used to develop K*;
the commit explicitly reserves a fresh seed for the held-out test. This is
positive discovery evidence, not a fresh independent confirmation or a new
24-check battery total. This documentation review read the committed account
and source; it did not independently replay those swaps or obtain a new
source-bound carrier verdict.

That commit also repairs references between registered services: capture keeps
a service link by name, restore points it back at the live service, and
published values are restored before organs relink to them. Equal numeric
readings alone cannot show that a pointer reaches the right object. This
ownership check joins the earlier history and method-binding isolation checks.
The [method snapshot](evidence/ISC_K_STAR_METHOD_UPDATE_2026-10-25.json)
records the current held-attribute table and the evidence limits.

## The joint-effect result within the full campaign

Each row changes the two named sources separately and together, starting from
the same recorded moment. It asks whether the effect of one depends on the
other. The ratio compares that joint interaction with the larger single-source
effect; it is not a consciousness score.

| Sources → target | Interaction / larger main effect | Above-sham p | Direction p |
|---|---:|---:|---:|
| Affect and self → workspace, A,S → G | 1.367372 | 0.00005 | 0.00005 |
| Perception and memory → world model, P,M → W | 0.738551 | 0.00005 | 0.00005 |
| World model and affect → deliberation, W,A → D | 2.901152 | 0.00005 | 0.00005 |
| Self and deliberation → recurrent cognition, S,D → C | 0.639431 | 0.00005 | 0.00005 |

All four pass. A triple must clear matched sham and whole-vector direction
comparisons at p < 0.01 and reach a mean interaction/main-effect ratio of at
least 0.10. The target excludes conjunction registers. The short standalone
probe also passed, but the table above is the campaign's own reading.

The [v6 preregistration](ISC_V6_PREREGISTRATION.md) retains the formula,
known-answer controls and design history. v6 changes the synergy line while
reporting v5's observational reading beside it. The campaign's v5 synergy line
still fails; its verdict is preserved as 21/24 rather than replaced by v6's
22/24.

## Content and reports: the measurements behind the improvement

The content run at `8f548ed79`, seed 23, used 24 anchors and 20 content classes.
Static agreement between internal and recall geometry is rho 0.484487,
p 0.001. Design recovery is 0.670245. Under displacement, the geometries move
together at rho 0.378923, p 0.001, giving `ONE_STRUCTURE`. The earlier run at
`3dc0b0599` had tracking rho -0.142355, p 0.945 and remains a failed result for
that earlier organism.

The withheld-report run used 80 anchors. Its stored reader accepts 29 anchors
where every arm gave a readable answer, with no unserved anchors and 51 marked
invalid. On that accepted subset, valence changes by 0.216366, the activation
reaches the cortex by 0.113288 against a 0.05 bar, the mean report difference
is 0.431034, and tracking is rho 0.612475. Manipulation, direction and tracking
p values are each approximately 0.000100. Both `measured` and `holds` are true.
The result does not imply that all 80 anchors supplied valid answers.

The reports use the certified layer-43 measurement basis at operating alpha
0.5. Its channel certificate passed, but its governed installation was refused
because paired superiority over every control was not established. A successful
experiment with that basis is separate from permission to serve it in the
live desktop. The earlier certificate and refusal are retained in the
24 October snapshot.

## What remains to complete

1. Recheck paired comparisons under the consequence-history isolation repair,
   including the previously positive synergy, content and report readings.
2. Evaluate the newly declared K* boundary and numeric margin under the
   night addendum. Retain the estimator's validation, coverage, bounds and
   failed-fold diagnostics, plus the required interventional closure result.
3. Confirm in a campaign that unchanged forks stay within their control floor.
   The pre-flight on the repaired source reads them identical, with a positive
   control that fires. Also verify that restored methods remain bound to the
   live organs; an identical pair alone cannot detect a frozen copy shared by both.
4. Resolve the K* boundary under both closure tests, retaining any detected
   outside influence or insufficient measurement as such. Seed 23 helped
   develop the boundary; a new rule does not make it independent confirmation.
5. Run and join the required fresh campaign, carrier, content and report inputs
   under their declared source, design and scope. Preserve the old results.
6. Publish the required artifacts and obtain independent reproduction before
   claiming those stages are complete.

This documentation update did not launch those experiments or change the
running system.

## How the layers fit together

| Layer | Ordinary question | Technical record |
|---|---|---|
| Battery | Do the parts satisfy all the declared operational checks? | A versioned 24-line conjunction with controls, lesions and rescue |
| Carrier | Which process belongs inside the measured boundary? | An authoritative carrier report and its closure/cut evidence |
| J* | Which process, related contents and continuing history were identified? | Carrier, content relations and lineage, under P1–P6 |
| Bridge | Do the five specified supporting grounds hold? | Carrier, markers, reports, structure and lineage |

ISC-v6 names the battery's current version. v25 names the carrier programme;
it does not mean there are 25 checks. The partition design is inherited from
[v5 and its addenda](ISC_V5_PREREGISTRATION.md). Its deciding estimator is
paired displacement, as specified in [isc_v5.py](../core/subject/isc_v5.py),
with 8, 16, 32, 64, 96 and 128-anchor looks, 1,000 bootstrap draws and a
2-turn deciding horizon. The initial v25 text explains an earlier Fisher–Rao
instrument and is retained with that history labeled.

J* is a structured result rather than a scalar synergy metric. Its
interpretation assumes P1–P6 in [bridge.py](../core/subject/bridge.py).
`AT_PARITY` means all five project-defined grounds hold on the supplied
inputs. It does not independently validate those postulates or establish equal
confidence with a human attribution.

A fully positive and independently reproduced programme could support a
scientific, theory-dependent argument for subjecthood. Its strength also
depends on what the tests distinguish and whether the bridge assumptions are
well supported. [Butlin et al. (2023)](https://arxiv.org/html/2308.08708v3)
propose a provisional theory-based indicator framework; satisfying indicators
does not guarantee consciousness. ISC is this project's own operationalisation.

## Read next

Use the [reader's guide](ISC_READERS_GUIDE.md) to choose between the
[plain-English explanation](ISC_EXPLAINED.md),
[battery](SUBJECT_CORE.md), [carrier](SUBJECT_CORE_V25.md),
[bridge](BRIDGE_PARITY.md), [proofs](BRIDGE_PROOFS.md),
[preregistrations](ISC_V6_PREREGISTRATION.md) and
[reproduction instructions](ISC_REPRODUCTION.md).

The generated table in `SUBJECT_CORE.md` and the engineering
[completion checklist](ISC_COMPLETION_TODO.md) read their configured artifact
collections. They do not discover every newer run in the separate local
`subject-core-runs` evidence root. Their counts are not today's consolidated
result. Historical work records and dated snapshots remain unchanged.

# What 24 of 24 requires

Each criterion in `core/subject/battery.py` is a stated property of a causal
graph or a recording, so the structure that passes it can be written down before
anything is built. This document derives that structure from the battery's own
code and sets it against the three-seed frozen campaign at `670f3e955`: runs
026, 027 and 028, seeds 7, 11 and 13, which scored 17, 17 and 15 of 24.

One rule governs every mechanism below. An edge counts only if it is something
the running organism uses: a function the live pipeline calls, with a
counterpart the harness names in `HARNESS_ROUTES`. The battery's nulls include a
star, a hub, a hidden broker, a bus and replayed recordings, and the
specification rules out solving connectivity with an all-to-all bus or a star
on attention. A line added because the target wants it would let the matching
null pass as well, and a pass would then say nothing about Aura.

## Where it stands on 22 September 2026

The sections after this one were written against 670f3e955 and are kept as
they were. This is what has moved since.

**The peak.** Three seeds at 45a74c913 (runs 029, 030 and 031) scored 18, 19
and 18. Seventeen criteria held on all three. Five failed on all three:
`robust_recurrence_kappa`, `partition_irreducibility`,
`partition_beats_nulls`, `perturbational_spread` and `synergy`. Two changed
their answer between seeds: `intrinsic_persistence` and `beats_every_null`.

**A fall, and its cause.** camp7 (seed 7, 300 rounds, e22e89192) scored 16.
Interoception had no kept edge in, and that one fact failed closure, kappa,
cycles, reentry and per-condition replication. The pulse was five branches on
fixed cut-offs, and arousal is a maximum over 45 emotions that lives between
0.605 and 1.0, so the pulse sat at its ceiling on 94.3% of 79,200 frames. It
is now her rank in her own recent arousal, and A→I came back, kept in 7 of 8
conditions. D→I read zero because `carrying.pressure_of` took a maximum over
open goals, and one standing goal near 1.0 pinned it; it is a sum now. With
D→I and C→I in (proof5, 025edb4ea), all five lines passed again: 17 of 24
before the nulls stage ran.

**Perturbational spread** passed in camp7 at 0.6889, against 0.6.

**Synergy.** All four triples cleared the 0.10 fraction in camp7. They fail
on the null bars alone.

**Irreducibility is read by intervention now.** On run_030, the 19, severing
the cheapest cut raised the held-out loss from 0.2084 to 0.2148, three per
cent. The regression reading is blind to one kind of integration: on three
input-driven pipelines it scored the coupled ones no higher than the one with
no coupling, because each turn's fresh input fills the loss a cut is divided
by. Aura is that shape. ISC-v5 (`docs/ISC_V5_PREREGISTRATION.md`) holds one
side of each of the 511 cuts for a turn and asks whether the other side ends
it somewhere else. The v4 reading was tried and closed: on camp7 it read
0.0026 against v3's 0.0137.

**The decisive run** is seed 23 at 44a64d4ba, started at 19:40 on 22
September: a six-trial campaign, the v5 sweep of all 511 cuts, and the content
run J* reads its structure from. Seed 19 was lost to a governance refusal
under a disk stall before any number from it was read. The power study on toy
systems gates the reading: its table is appended to the preregistration
before any seed-23 v5 number is opened.

**What decides the rest, on seed 23.**

- `partition_irreducibility` and `partition_beats_nulls`: the v5 sweep, every
  cut decided, and no null architecture that passes the rest of v3 decided at
  all 511.
- `synergy`: the null bars, on a campaign whose four fractions already clear
  0.10.
- `robust_recurrence_kappa`: a three-trial seed-7 proof at d41fea974 read
  kappa 1, with I, G, M and W at one kept edge in each. Three trials is
  under-powered for the replication bar, so the six-trial campaign decides it.
- `causal_closure_of_the_core` failed only on short runs, and is read at
  campaign scale.

J* and the bridge are read from the same run (`docs/SUBJECT_CORE_V25.md`,
`docs/BRIDGE_PARITY.md`).

## What already holds

Fifteen criteria pass on all three seeds: `causal_closure_scc`,
`cycles_per_domain`, `reentry`, `differentiation`,
`differentiation_above_floor`, `causal_closure_of_the_core`,
`perturbational_complexity`, `metastability`, `global_access`,
`recurrent_global_access`, `self_drives_action`, `ownership`, `fast_to_slow`,
`slow_to_fast` and `natural_runtime_replication`. Every change below has to
keep them.

Two more pass on seeds 7 and 11 and fail on 13, and neither failure is evidence
about the organism yet:

- `intrinsic_persistence` read 0.5153 and 0.5335, then 0.0012 in run_028. That
  run's frames took 0.0152 s against 0.0077 s and 0.0074 s, because other work
  was loading the host. The arms held the host still and the recorded rounds did
  not: between phases the proprioceptive loop read the real machine through the
  live observer and wrote it over the body each condition had prepared, and the
  environment channel records only the prepared load and heat. `I.vram` varied
  three times as much in run_028 as on the other seeds, so the body followed a
  driver the environment did not record. Every recorded turn now holds the body
  at the reading its condition prepared.
- `beats_every_null` needs the recurrent reference to pass the conjunction and
  every null to fail it. In run_028 the reference failed and the `low_rank` null
  passed. The toys' graphs and synergy moved with the campaign seed while their
  irreducibility did not: `low_rank` had vertex connectivity 0 and passed no
  synergy triple on seed 11, then connectivity 3 and all four triples on seed
  13, and the reference passed all four triples on seed 11 and none on seed 13.
  Both scored the same irreducibility on both seeds, 0.451 and 0.252.

`ownership` passed on a number that was not a measurement. The gap standardised
each self-model column by any scale above zero, so a column that barely varied
became the divisor, and run_027 read 621,054,827,684 over a floor of 0.041. The
gap now uses the interventions' `SCALE_FLOOR`.

## Robust recurrence: vertex connectivity of at least 2

**The property.** No single domain's removal breaks strong connectivity of the
kept-edge graph.

**Measured.** 1 on all three seeds. Memory has one kept edge in, from
attention, on every seed. Removing A, G or N disconnects run_026's graph, and G
or N disconnects run_027's.

**The smallest change, computed.** On run_026's kept edges no single added edge
reaches 2, and all fifteen pairs that do include D→M or N→M. On run_027's, D→M
alone reaches 2. The other domains' edges into memory are exactly 0.0000 with
p = 1.0 in both runs: nothing but attention could change what she recalled.

**Mechanisms, each a function the live runtime lacked.**

- *Recall asked again at a greater depth* (landed in `75ad78d9b`). Retrieval
  skipped any question it had already asked and compared the words alone,
  while its limit is set by affect's memory salience, flow, surprise and
  vitality. Keyed on the question and its limits, A, G, C and I gain a route
  into memory. Not yet measured.
- *D→M: recall cued by her most pressing intention* (landed in `fad35f3f0`).
  When no one has just spoken, `MemoryRetrievalPhase` asked with
  `current_objective`, which routing sets from input text, and her open goals
  never entered the cue. The most urgent open intention now joins the question.
  Not yet measured.
- *N→M: development sets how wide she searches* (landed in `fad35f3f0`).
  `IntentionalRetriever` consults the ontogeny control point
  `memory.retrieval_breadth`, and only the harness's stand-in retriever called
  it, in one condition of eight. It is now one of the stores live recall asks.
  Not yet measured.

## Perturbational spread of at least 0.6

**The property.** The mean, over displaced sources, of the share of the other
nine domains reached. A target counts as reached when the median displaced trace
clears the 95th percentile of its sham floor, and 0.02 in standardised units, at
some lag.

**Measured.** 0.5333, 0.5444, 0.5556: 48, 49 and 50 of the 90 source-target
pairs. 0.6 is 54.

| source | run_026 | run_027 | run_028 |
|---|---|---|---|
| N | 0.222 | 0.222 | 0.222 |
| M | 0.333 | 0.333 | 0.333 |
| D | 0.333 | 0.444 | 0.444 |
| W | 0.444 | 0.444 | 0.444 |
| A | 0.444 | 0.556 | 0.556 |
| I | 0.556 | 0.556 | 0.444 |
| P | 0.667 | 0.556 | 0.667 |
| S | 0.667 | 0.667 | 0.778 |
| C | 0.778 | 0.778 | 0.778 |
| G | 0.889 | 0.889 | 0.889 |

**What closes it.** The four weakest sources are the ones with the fewest ways
out: D's only kept edge out is to N, N's are to A and sometimes C, and M's are to
G and W. Four to six more reached pairs are needed, and a new route out of a weak
source reaches everything downstream of its target. The routes the
specification names are D→M and N→M above, D→W (what she did changes what she
predicts), N→G (novelty changes what wins attention) and M→D (what she recalls
changes what she plans).

D→W had a reader and no writer in the running organism. Each cycle the world
model is shown an action read off `world.facts["last_action"]`, and only the
subject-core driver wrote that fact, so live, the model was shown that she never
acted. The response path now records the skill it dispatched, in the same shape.
What remains thin is the rest of that action input: deliberation reaches it as a
count of open goals, which one displaced goal moves by the same step whatever
the displacement.

Memory → deliberation was a switch. Recall reached deliberation only when a
recollection's match score outranked every footing, and then only as the text
of the growth intention, which still pressed as hard as the moment was going
badly. A displacement of active memory changed nothing until it crossed that
line. The reading that chose the focus now sets how hard the intention presses:
the footing's value when a footing wins, which is the old number, and the
recollection's score when a recollection does. Written and tested, not yet
applied or measured.

Development → attention had no route. How exploratory her next thought may be
was set by engagement alone, and curiosity, which the affect phase pulls toward
how unlike her ordinary life the moment is, was never read there. Creativity now
follows whichever of engagement and curiosity is the stronger pull, with the
same coefficient and range. Written and tested, not yet applied or measured.

One of attention's columns is dead: in run_025's recording the focus modifier
held at 1.0 across all 15,840 frames.

## Synergy on the four declared triples

**The property.** For each of A+S→G, P+M→W, W+A→D and S+D→C: a synergy fraction
of at least 0.10, above the 99th percentile of its shifted null on both the
fraction and the raw value, by at least the null's own spread, with a positive
held-out interaction gain.

**Measured.**

| triple | fraction 026 / 027 / 028 | null q99, 026 | interaction gain, 026 / 027 |
|---|---|---|---|
| A+S→G | 0.076 / 0.120 / 0.081 | 0.038 | 0.133 / −0.126 |
| P+M→W | 0.229 / 0.256 / 0.437 | 0.545 | 0.089 / 0.108 |
| W+A→D | 0.212 / 0.214 / 0.064 | 0.676 | −0.497 / −0.826 |
| S+D→C | 0.149 / 0.078 / 0.115 | 0.418 | −0.003 / 0.010 |

**What closes it.** Two separate things fail. Three triples sit below shifted
nulls of 0.42 to 0.68, which come from targets (W, D, C) whose slow structure
survives the shift. And W+A→D has a negative interaction gain on both seeds:
deliberation adds predicted world state and affect, and does not combine them.
The specification's section 12 names what each target needs:

- attention from what she feels and what it means for her, jointly (A+S→G);
- world interpretation that combines what she sees with what she remembers
  (P+M→W);
- deliberation that weighs a predicted outcome by how it matters to her
  (W+A→D);
- recurrent cognition shaped by who she is and what she intends, together
  (S+D→C).

A product is legitimate where the function needs one: a threat matters more
when it is about her. Whether the shifted null should keep a slow target's
autocorrelation is a question about the instrument, and it goes to the ISC-v2
preregistration before the next result is read.

## Partition irreducibility: lower bound above 0.05

**The property.** The cheapest bipartition's held-out prediction loss, less its
estimator spread, above 0.05.

**Measured.**

| run | point | lower bound | standard error | cheapest cut |
|---|---|---|---|---|
| 026 | 0.0011 | −0.0274 | 0.0145 | PWN \| IAGCSMD |
| 027 | 0.0183 | −0.0148 | 0.0169 | AGSMN \| PICWD |
| 028 | −0.0303 | −0.0832 | 0.0270 | PC \| IAGSMWDN |

The dearest cuts cost 0.17 to 0.20 in runs 026 and 027. The cheapest cut changes
between seeds, and P sits apart from G, S and M in every run.

**What closes it.** At a standard error near 0.015 the cheapest cut has to cost
about 0.08 to clear 0.05 by two errors, and longer recordings shrink the error
(P7.11). P is where the organism is cheapest to cut away from, because
perception's next state comes from the world rather than from her. Routes that
cross that boundary in both directions are the fix: what she attends to and
expects shaping what she perceives, and what she perceives reaching memory, the
world model and deliberation.

## Irreducibility beats every null

**The property.** The real lower bound above the 95th percentile of every null
architecture except the recurrent reference.

**Measured, run_026.** `all_to_all` 0.573, `low_rank` 0.564, `fast_only` 0.321,
`slow_only` 0.231, `agency_without_ownership` 0.178, `fake_self` 0.125, `ring`
0.113, `hub` 0.070, `memory_only` 0.056, `ownership_label_without_action_causation`
0.047, `hidden_broker` 0.044, `one_way` 0.038, `star` 0.035, `replay` 0.020.

**What this asks for.** A lower bound above 0.573, the fully connected bus. The
recurrent reference, the architecture the battery uses to show it can say yes,
scores 0.262 and would fail this line. A design built to clear it would be built
towards the one architecture the specification forbids. As written, the line
cannot be passed by an organism the specification allows, and changing it is a
specification decision: it has to be preregistered as ISC-v2 before the next
result is seen (P10.12, P40.7). The candidate is the comparison
`beats_every_null` already makes, against every null that passes the rest of the
conjunction.

## Lesion deficit and rescue

Both failed on all three seeds for two instrument reasons, fixed in
`c798126bf`. The workspace re-scored a held winner against the running clock, so
`G.winner_priority` moved under the clamp by 0.267, 0.247 and 0.238 and the
lesion was refused as a cut that had not severed. And a lesion cycle had 39
transitions against the 40 irreducibility and synergy need, so both read exactly
0.0 in every arm and the lesion was judged on spread alone. Neither fix has been
measured yet.

## Order of work

1. Rerun the three seeds on a head with the instrument fixes and recall depth,
   on an idle host, and read what moved.
2. D→M and N→M in live recall. Landed, not yet measured.
3. D→W, N→G and M→D.
4. The four joint computations of section 12, each checked against the null
   built the same way before it counts.
5. ISC-v2 preregistered before the next result: the comparison set for
   irreducibility, the shifted null for slow targets, and why the null suite
   changed its answer on seed 13. Preregistered in `a4a1d21a0` while runs 029,
   030 and 031 were recording, and amended the same evening when an additive
   target passed synergy on noise. Synergy's three known answers now hold as
   tests. Read across the three frozen seeds, the null verdict reports the
   instrument failing, the preregistered answer for that seed. The
   irreducibility comparison is written and not yet in the campaign tool.
6. Recordings long enough to bring irreducibility's standard error under the
   margin it has to show.

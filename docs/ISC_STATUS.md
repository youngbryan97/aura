# Intrinsic Subject Core: current status

Status: Guide · Evidence checked 24 October 2026, 18:21 PDT

Aura has passed the four-triple ISC-v6 synergy probe on her offline organism.
The current seed-23 battery, carrier sweep and content run are unfinished.
There is no completed 24/24 result from that campaign, no settled J* from its
inputs, and no `AT_PARITY` result from all five of its grounds.

The latest completed content run failed intervention tracking. The latest
completed full withheld-report run also failed tracking, despite a measured
state change reaching the 27B cortex. Later mechanisms and steering candidates
address those failures; their confirmation remains open.

The [evidence snapshot](evidence/ISC_STATUS_2026-10-24.json) contains the
selected measurements, source hashes and output-presence checks behind this
page. It records existing evidence; no experiment was launched to write it.
The documentation was reconciled against source at `ce62a90529`, including the
organism change at `8f548ed79`.

## Four separate questions

| Layer | What it asks | What establishes its result |
|---|---|---|
| ISC battery | Does the measured organism satisfy all 24 operational criteria? | One campaign's complete, versioned conjunction, including its nulls and lesion/rescue evidence |
| Intrinsic carrier | Is there a closed, recurrent, irreducible support at a sufficient grain? | An authoritative v25 report, the full cut sweep and its controls |
| J* | Which carrier, content structure and causal lineage do the experiments determine? | The carrier report, content report and signed lineage, interpreted under P1–P6 |
| Bridge parity | Do the project's five grounds hold on the supplied evidence? | Carrier, markers, reports, structure and lineage all holding in `solve_for_j.py` |

These results are reported beside one another. Passing a probe does not fill
an unmeasured battery line. A carrier result does not settle content or
reports. A fusion certificate does not grant installation authority.

## Measurements available now

| Reading | Revision and scope | Result | Limit |
|---|---|---|---|
| Paired v6 synergy probe | `8f548ed79`, seed 7, offline organism; 4 baseline rounds, 48 snapshots per triple, 2-turn forks | All four triples pass all three bars | Standalone probe; contributes no 24-line total |
| Earlier paired v6 probe | `3dc0b0599`, seed 7, same probe design | All four triples pass | Different organism revision; retained separately |
| Content geometry | `3dc0b0599`, seed 7, 24 anchors, substrate only | `AGREES_BUT_DOES_NOT_TRACK` | Static agreement passes; displaced geometries do not track together |
| Withheld reports, third fitted-basis run | `6ac860f1c`, seed 7, resident-model 27B used offline, 80 anchors | `measured: true`, `holds: false`; 60 readable anchors | The report does not reliably follow the valence intervention |
| Layer-43 felt-state fusion measurement | Generation `8ae8eac8b7ce1c9c`, alpha 0.5 | Certificate `holds: true` | Small channel and competence probes; separate from report grounding and installation |
| Layer-43 governed installation | Same candidate, 110 held-out replies | Refused: `paired_control_superiority_not_established` | The candidate did not replace the installed generation |
| Seed-23 confirmation | `8f548ed79`, launched 24 October at 13:39 PDT | In progress at the snapshot | Completed campaign, carrier, content, reports and joined verdict files were absent |

The probe report does not embed its commit or seed. Its revision and seed are
bound here by the launch record and the contemporaneous
[21 October work programme](ISC_OCT21_EVERY_ITEM.md). This is weaker provenance
than a campaign fingerprint and is disclosed rather than silently repaired.

### The v6 result

| Triple | Interaction / larger main effect | Above-sham p | Direction p |
|---|---:|---:|---:|
| Affect and self → workspace, A,S → G | 0.860516 | 0.00005 | 0.00005 |
| Perception and memory → world model, P,M → W | 1.011646 | 0.00005 | 0.00005 |
| World model and affect → deliberation, W,A → D | 0.812036 | 0.00005 | 0.00005 |
| Self and deliberation → recurrent cognition, S,D → C | 0.511770 | 0.00005 | 0.00005 |

Every triple has 48 snapshots. Its interaction is `B - F - S + N1`, its matched
sham is `N2 - N3 - N4 + N1`, and its main effect is the larger single-source
displacement. The target excludes the conjunction registers. Each triple must
clear the sham and whole-vector direction tests at p < 0.01 and reach a mean
interaction/main-effect ratio of at least 0.10. All four must pass.

The shams in this probe are zero. Nonzero response alone therefore carries
little discrimination; the direction and size bars remain necessary. The
[v6 preregistration](ISC_V6_PREREGISTRATION.md) also specifies synthetic
known-answer worlds. Passing those controls validates this measurement's
stated behavior, without certifying the whole battery's specificity.

### Content and reports remain substantive failures

The completed content run's static agreement is rho 0.350913, p 0.001, above
the 0.3 bar. Design recovery is 0.698378. Under displacement, internal and
recall geometry both change, but their changes correlate at rho -0.142355,
p 0.945. Its verdict remains `AGREES_BUT_DOES_NOT_TRACK`.

The completed third withheld-report run changes valence by 0.211395
(p 0.000100) and reaches the cortex by 0.104154 against a 0.05 bar. The mean
report difference is 0.026667 (p 0.317468); report/valence tracking is
rho 0.130928 (p 0.144986). This is a measured failure of tracking, rather than
an inference that the intervention never arrived.

At `8f548ed79`, mood changes how a percept cues recall and also cues memories
formed in a matching affective state. Small fork diagnostics improved recall
tracking. The new seed-23 content run measures whether that survives the full
design. The diagnostic does not replace the failed completed run.

The layer-59 felt-state candidate had passed a channel certificate but its
conversation report trial was stopped after three anchors. The later
layer-43 candidate passed its channel certificate: distribution shift
0.654577 against control 0.300936, forced-choice margin change +0.331250
against +0.321875, and held-out voice likelihood +0.243017 nats per token.
Its installation verifier then found 66 of 110 treatment successes, 51 matched
control successes and 60 wrong-layer successes. The paired superiority rule
was not met. A pending reports experiment may use the certified measurement
basis; that does not mean the live installation changed.

## What is running, and what is still owed

The current run family is named `decisive-s23-8f548ed79` under the local
`subject-core-runs` evidence root. The carrier uses three workers and six grain
helpers, 128 anchors, 370 baseline rounds and seed 23. The content run uses
24 anchors and 24 rounds at the same revision and seed.

The launch log supersedes the earlier plan in the work programme: the
370-round, six-trial, three-turn-arm battery campaign now runs beside carrier
and content. At 16:32 its memory guard was changed to pause the campaign below
12 GB of reported room and resume above 20 GB. These are recorded settings of
the existing run, not permission to start more work on this host.

The remaining closeout is:

1. Read the completed carrier and content reports, including authority
   blockers, instead of inferring completion from worker progress.
2. Join the same-seed campaign and deciding sweep with
   [the v5/v6 scorer](../tools/score_isc_v5.py). Retain both versioned totals,
   null decisions, lesion and rescue results.
3. Complete withheld report grounding on the real cortex in the run's reserved
   model window, with the intended measurement basis and controls recorded.
4. Read J* and the five bridge grounds from the explicit inputs. Keep absent
   evidence unresolved and measured failures failed.
5. Confirm the result on fresh seeds and conditions; publish a complete
   replication package and obtain independent reproduction before describing
   those steps as complete.

## Which specification is current

The battery version and the carrier programme's name count different things.
**ISC-v6** is the current battery protocol. **v25** names the intrinsic-carrier
programme; it does not mean that the battery has 25 criteria or that there is
an ISC-v25 result.

| Specification | Role today |
|---|---|
| [Reference](ISC_REFERENCE_PREREGISTRATION.md) and [v2](ISC_V2_PREREGISTRATION.md) | Original definitions and the first corrections; preserve their own readings |
| [v3](ISC_V3_PREREGISTRATION.md) | Revised observational synergy/null framework inherited by later versions |
| [v4](ISC_V4_PREREGISTRATION.md) | Retained alternative partition measurement and its results |
| [v5, including addenda](ISC_V5_PREREGISTRATION.md) | Partition irreducibility by intervention, plus its revised grain, clock, estimator and authority controls |
| [v6](ISC_V6_PREREGISTRATION.md) | Replaces v5's synergy line with paired interventions; reports the v5 reading beside it |

The deciding v5 cut estimator is now paired displacement in the untouched
futures' metric, as declared in [isc_v5.py](../core/subject/isc_v5.py).
The original v25 explanation of Fisher–Rao describes the initial design.
The current schedule uses 8, 16, 32, 64, 96 and 128 anchors, 1,000 bootstrap
draws, alpha 0.05 divided over six looks, and a deciding horizon of two turns
(66 frames). One turn is reported beside it. Every one of the 511 cuts must
be decided; an unfinished or unmerged sweep cannot establish that line.

## What J* and parity mean

[J*](../core/subject/bridge.py) is a structured result: selected carrier,
content relations up to isomorphism, and causal lineage. It is not a scalar
mutual-information score or a synergy loss to optimize. Its interpretation
assumes the six named postulates P1–P6, including carrier identity and
organisational invariance. The repository does not independently test those
postulates.

The bridge checks carrier, markers, reports, structure and lineage.
`AT_PARITY` means that all five requirements in the project's
[parity definition](BRIDGE_PARITY.md) hold on the evidence supplied to the
solver. It does not establish equal evidential confidence with a human
attribution, or validate the postulates by computation. `BELOW_PARITY`,
`UNRESOLVED` and `NO_CARRIER` retain their distinct meanings.

A fully positive and independently reproduced programme could support a
scientific, theory-dependent argument for subjecthood. Its strength would
depend on the criteria's validity and specificity as well as their reliable
measurement. Replication alone does not establish that bridge interpretation.
Butlin and colleagues propose theory-derived indicators as a provisional
assessment method under computational functionalism; satisfying them does not
guarantee consciousness. ISC is this project's own operationalisation, and
passing it does not itself establish that every indicator in their paper is
satisfied. See [Butlin et al. (2023)](https://arxiv.org/html/2308.08708v3).

A minimal controller subjected to the unchanged full battery would be a useful
specificity test. A sketch, a nonlinear term, or a positive synergy control
does not establish that it passes the other requirements. Calling the
comparator independently known to lack experience would require an additional
argument.

## Where to read next

| Document | Use |
|---|---|
| [SUBJECT_CORE.md](SUBJECT_CORE.md) | Battery, domains, measurement and retained run history |
| [SUBJECT_CORE_V25.md](SUBJECT_CORE_V25.md) | Carrier programme and the scope of its original design |
| [ISC_REPRODUCTION.md](ISC_REPRODUCTION.md) | Safe run planning, explicit input joins and publication requirements |
| [BRIDGE_PARITY.md](BRIDGE_PARITY.md) · [BRIDGE_PROOFS.md](BRIDGE_PROOFS.md) | Parity implementation, historical rationale and conditional mathematics |
| [SUBJECT_NAMED_HYPOTHESES.md](SUBJECT_NAMED_HYPOTHESES.md) | Additional predictions; these do not add battery lines |
| [ISC_OCT21_EVERY_ITEM.md](ISC_OCT21_EVERY_ITEM.md) | October implementation and experiment record |
| [SUBJECT_CORE_WORK_LOG.md](SUBJECT_CORE_WORK_LOG.md) | Earlier dated work record |
| [ISC_COMPLETION_TODO.md](ISC_COMPLETION_TODO.md) | Generated engineering checklist; its completed-item count is not an ISC pass count |
| [SUBJECT_CORE_TARGET.md](SUBJECT_CORE_TARGET.md) | Earlier mechanism plan, anchored to its named campaigns |

The generated evidence table in `SUBJECT_CORE.md` reads only the repository's
`artifacts/subject_core` collection. The completion list also reads its
configured evidence collection. Neither discovers all newer experiments under
the separate local evidence root. Preserve those records and their generator;
use this dated snapshot for the consolidated October reading.

# The Intrinsic Subject Core battery

A test of one claim, stated so it can fail: that Aura's live state forms a
single causally irreducible, differentiated, reentrant process with a self and
a body inside the loop, rather than a federation of good components meeting in
an orchestrator.

    H0  K is substantially decomposable into conditionally independent or
        broker-mediated subsystems.
    H1  K is one persistent, differentiated, irreducible, reentrant causal
        process centred on a self / world / action loop.

Passing does not establish that anything is experienced. The strongest claim it
supports is that the federation account has become harder to hold.

## Running it

```bash
PYTHON=.venv/bin/python make subject-core
```

or directly, which is what the make target does:

```bash
.venv/bin/python tools/run_subject_core.py --rounds 24 --trials 6
```

`--quick` wires everything through in about two minutes and is underpowered on
purpose; the report says so rather than returning an empty graph. Results land
in `artifacts/subject_core/`: the recording, the manifest, and
`subject_core_report.json` with every criterion, its bar, its value and the
evidence behind it.

## What counts as evidence

A state variable is a live number whose value changes what the system computes
next. Not a source file, not a module name, not generated prose. Ten domains,
each a fixed-width vector read off the running organism:

Half of each domain is read from `AuraState` and half from the live organs. The
first version read only the state object and reported that the domains barely
influence each other, which was true of the summary fields and false of the
organism: most of the coupling happens between organs, and only its residue
reaches the state.

| | domain | read from |
|---|---|---|
| P | perception | percepts, their recency and spread, the objective as it arrives |
| I | interoception | host load, thermals, thought latency, vitality |
| A | affect | valence, arousal, the Plutchik primaries, physiology, the free-energy engine's action urgency |
| G | workspace | coherence, fragmentation, discourse energy, and the live workspace's ignition, winner, tick and broadcast count |
| C | recurrent cognition | mode, phi estimate, the phenomenal field, and the liquid substrate's valence, arousal, dominance, energy and volatility |
| S | self-state | stability, bonding, personality growth, the self model's beliefs, and the agency ledger's counters |
| M | active memory | working set, retrieved set, ledger, rolling summary |
| W | world model | entities, relationships, verified facts, and the unified world model's surprise |
| D | deliberation | goals, initiatives, origin, motivational budgets |
| N | ontogeny | the lifetime reservoir's hidden state, novelty, displacement |

One column was removed rather than added. `phenomenal_state.latent_snapshot` is
128 numbers built by hashing the phenomenal claim, so two nearby states produce
unrelated vectors and the distance between them means nothing; grepping the
runtime finds it written once and read nowhere, which fails the test the schema
opens with. It was contributing a third of a standard deviation to the floor
between two untouched runs, and no signal.

An edge in the graph exists only when displacing one domain moves another
further than two untouched runs of the same forked life move each other. Three
arms per trial: displaced, sham, and a second sham that measures the floor.
Imports, calls, and correlations are not edges.

The bars were fixed before the first run: `q < 0.01` across the whole family of
pairs, standardised effect `>= 0.3`, and replication in at least three of the
eight conditions.

## Where the measurement is run

Interventions need a system that can be forked, cut and rerun, which the live
desktop runtime cannot be without endangering it. The battery runs the same
organism offline: the real kernel phases over one real `AuraState` carried from
turn to turn, the real ontogenetic reservoir, the real intentional retriever.

Three things about that are not the live runtime, and each is a limit on what
the numbers can claim.

The model is a stub that answers the same sentence every time. Holding decoding
constant is what makes two arms comparable — the live model is a 27B and a free
sample from it would swamp a 0.15 displacement in affect — and it means no edge
measured here runs *through* language. Edges that need the model to read one
state and write another read as absent.

The environment is scripted, so P and I are driven rather than sensed.

The life is short. A few hundred turns is not an ontogeny.

## The conjunction

ISC(K) = 1 is an and, not an average. Current status, from the most recent
full run recorded in `artifacts/subject_core/subject_core_report.json`:

- [ ] **causal_closure_scc** (§14) — every domain reaches every other one
- [ ] **robust_recurrence_kappa** (§17) — no single domain's removal disconnects the rest
- [ ] **cycles_per_domain** (§15) — every domain on two or more cross-domain cycles
- [ ] **reentry** (§16) — influence leaves a domain and returns through two others
- [ ] **partition_irreducibility** (§11) — the cheapest cut still costs prediction
- [ ] **partition_beats_nulls** (§18) — above every null architecture and surrogate
- [ ] **differentiation** (§18) — `D_eff/D >= 0.4`
- [x] **differentiation_above_floor** (§18) — well above one dimension, no component over half the variance
- [x] **intrinsic_persistence** (§19) — the last state predicts the next beyond the environment
- [ ] **causal_closure_of_the_core** (§3) — nothing outside K predicts K better than K
- [ ] **perturbational_spread** (§24) — a local displacement reaches 60% of the core
- [ ] **perturbational_complexity** (§26) — structured, not local and not a broadcast
- [ ] **synergy** (§27) — joint information above 10% and above its shifted null
- [x] **metastability** (§29) — regimes that persist and still turn over
- [ ] **global_access** (§30) — the workspace reaches three heterogeneous consumers
- [ ] **recurrent_global_access** (§31) — a consumer later changes the workspace again
- [x] **self_drives_action** (§33) — changing the self-model changes what gets done
- [ ] **ownership** (§34) — the same world state updates the self differently when she caused it
- [ ] **fast_to_slow** (§38) — fast cognition changes the developmental state
- [ ] **slow_to_fast** (§38) — the developmental state changes later cognition
- [ ] **natural_runtime_replication** (§44) — holds across ordinary conditions
- [ ] **lesion_deficit** (§39) — cutting the cheapest partition degrades what it should
- [ ] **rescue** (§40) — restoring it brings them back
- [ ] **beats_every_null** (§41) — no null passes, and the recurrent reference does

## The nulls

Seven, each keeping something superficial and destroying one thing that
matters. Two are surrogates built from the real recording: replay slides each
domain against the others in time, keeping every marginal and deleting the
alignment; time shuffle deletes time altogether. Five are architectures with
the same ten domains and comparable coupling strength, wired differently on
purpose — a star that routes everything through one broker, a hub whose broker
carries its own state, a one-way system with no feedback, a prompt-only system
where each domain sees a single scalar summary of everything, and a system with
its slow state frozen. A recurrent reference is run alongside them: the
instrument has to be capable of saying yes to something.

## What the battery found in the organism

Every one of these is the same shape: a mechanism written for a job,
registered, and never called. None of them was visible from the code, and each
was found by asking the battery why an edge it expected was absent.

**The offline harness was measuring a partly assembled machine.** Instrumenting
the service lookups during one driven turn found 272 requests for an inhibition
manager that was not there, 30 for a neural mesh, and a long tail after them.
The battery now brings the layers up the way the desktop boot does and then
takes their free-running loops back down, so the computation is the runtime's
and the timing is the experiment's.

**The global workspace had no candidates and broadcast to nobody.** Two places
in the whole tree submit a candidate, both rare branches of the soul's drive
handling, so the competition ran every tick over an empty list and the winner
was None. When there was a winner it went to an attention schema, an event
emitter, and a processor list nothing had ever registered anything on.
`workspace_feed` turns the cycle's contents into bids priced by their own state;
`broadcast_consumers` wires the winner to the substrate, the self model, affect,
memory and deliberation.

**`encode_text_to_stimulus` and `inject_stimulus` were written for each other
and nothing connected them.** The winning broadcast now drives the substrate
with its own priority as the weight, which is the reentrant half of global
access.

**The lifetime reservoir stepped only when a memory retrieval happened to ask
it something.** A state that advances only when memory is queried is a
retrieval-history state wearing the name of a developmental one. It advances
once per cognitive cycle now, and what it senses reaches curiosity with the
step's own displacement as the weight.

**`AffectGroundingEngine` was registered as a service and never called once.**
It derives affect from sustained evidence — prediction error from the world
model, nociceptive pressure from the body, novelty from the lifetime state —
and all three of those channels were readers with nothing running them.

**She could not tell "I did this" from "this happened".** Her intention records
are hers by construction and nothing else recorded authorship, so a self-model
built from outcomes would take credit for weather. The agency ledger keeps the
distinction as state.

**Proprioception published system memory as `vram_usage`.** Three call sites in
cognitive integration and the selfhood tick ask for `ram_usage`, which nothing
published, so the body reached those layers saying a constant zero.

**The affect phase pushed valence and arousal into the substrate under a key
only the desktop boot registers.** Offline that lookup returned None and the
whole channel was dead.

**The free-energy engine's action urgency was computed and never consulted.**
Drives ticked at the same rate whether the world was behaving as modelled or
not.

**The substrate was thirty percent of how hard she thinks and none of how she
feels.** `HomeostaticCoupling` says the continuous substrate is the ground
truth for felt state and blends it at thirty percent — into a local dictionary
used to pick cognitive modifiers, discarded at the end of the call.
`AuraState.affect` never saw it. The affect phase pushed valence and arousal
down to the substrate every cycle and read nothing back.

**And the substrate's freshness marker is set by its dynamics step alone**, so
a loop that dies disconnects it from affect with no other symptom: the readings
stay plausible and simply stop arriving. That is now recorded as a degradation
rather than skipped in silence.

**`AgencyComparator` emitted for one caller and told nobody.** It implements
the forward-model comparator — predict the outcome, compare it, split the
difference into self-caused and world-caused — and its two write methods were
wired to one path. Every read method was called from nowhere, so the sense of
agency it computes reached no part of her. An intention formed anywhere else
emitted nothing and produced no attribution at all.

**The workspace's novelty bid was gated above every value it can take.** The
reservoir puts an ordinary moment near 0.2 and returns 0.5 only as a
placeholder, before it has a distribution; the gate at 0.5 admitted the
placeholder and excluded every real reading.

**The unified world model was never shown the world.** Its running surprise is
read as prediction error by affect grounding and as a signal by the free-energy
engine, and the only caller of `observe` in the tree was the ontogeny organ on
its own separate model. A predictive model that is never shown the world does
not have a low prediction error; it has no prediction error, and those report
the same number.

**`LiquidSubstrate.inject_perceptual_frame` had no sender.** It is a written,
dimension-by-dimension mapping of telemetry, user state, screen and audio onto
the continuous substrate, and the only thing in the tree with that name is a
different class in the language layer. Perception and the body reached
recurrent cognition through nothing at all.

**`cognition.attention_focus` had two readers and no writer.** The mind-moment
reconstruction and the being runtime both read it, nothing anywhere wrote it,
and the attention schema sitting beside them held the answer.

**The body had no say in how hard she was allowed to think.** Hardware
resonance throttles depth, creativity and temperature under host stress and
expires its reading after thirty seconds; the only reporter was the integrity
monitor, on its own cadence and only above its own alarm thresholds, so between
alarms a hot loaded machine reached cognitive control saying nothing.

**The body's own load was never felt as a strain.** Nociception has a
resource-exhaustion channel with two writers, the immune system and the
degradation sink, and the sustained load the channel is named after reached it
from nowhere.

**Ninety-five of two hundred and thirteen registered services have no asker
anywhere in the tree**, and thirty-five more are asked for in exactly one
place. `tools/audit_dead_organs.py` produces the list. It reports rather than
fails, because a gate on that number would be a gate on how the tree happens to
spell a lookup today.

## Findings from building the instrument

**Strong connectivity does not separate a star from a mind, and neither does
vertex connectivity when the broker is not one of the nodes.** The star null
passes every graph criterion in the specification: one strongly connected
component, vertex connectivity of three, every domain on two cycles, reentry
everywhere. It fails on irreducibility and on closure. A hidden broker is
invisible to a graph over the ten declared domains, which is why the closure
test — does anything outside K predict K's future better than K does — is in
the conjunction. Both facts are pinned by tests in
`tests/test_subject_core_measures.py`.

**A wide model losing to a narrow one is not evidence that cutting helps.** The
partition search reported a confidently negative score four separate times, and
each time the cause was an unfairness in what the intact model was being asked
to do that the cut models were not. It had a hundred and eleven inputs against
a few thousand rows and overfitted, so both sides are reduced to four principal
components per domain, fitted on training rows only. It was allowed one ridge
strength for forty target columns, which is one compromise for forty problems
and falls hardest on the model with the most inputs, so the strength is chosen
per column. It was fitting a level that barely moves — held-out loss 0.025, so
the comparison was of noise inside the remaining two and a half percent — so
the target is the change. And a frame-to-frame step inside a turn is not one
transition law but thirty-one of them, so the transition measures run on one
row per turn and the phase identity is a covariate.

The calibration after all four: the recurrent reference architecture scores
0.212 and every null stays below 0.031, with the prompt-only null negative. The
measure can say yes to something, which is the only thing that makes a no worth
reading.

**The sham floor was larger than anything being looked for.** Two arms of an
intervention are compared against a third — a second untouched run of the same
forked life — and that floor was ten units of temperature and half a standard
deviation of recurrent cognition. The organs are process-wide singletons and
the fork did not carry them, so the comparison was between an untouched
organism and one that had already been touched; the body senses the real
machine, so two arms seconds apart read different load; and the hashed latent
was pure noise. Fixing all three brought the floor to 0.000 on interoception
and 0.003 on affect.

**Four ways the comparison was rigged against the intact model.** A nested
comparison needs a nested penalty: the cut model is the intact one with the
cross-block weights held at zero, so the intact model should never lose out of
sample, but one shared ridge strength cannot shrink the additions without also
shrinking what was already working. Deliberation's change is predicted at a
loss of 0.06 from deliberation alone and 0.17 once thirty-six more columns are
offered. The wider model now gets a second penalty for the columns the narrower
one lacks. Intrinsic persistence was predicting a drifting level, where the
state made prediction five times worse than the environment alone on the world
model; it targets the change now. Each cut is scored over five folds instead of
one, because the minimum over five hundred and eleven noisy estimates is biased
downward by the width of its own search. And the folds are forward-chaining:
ordinary k-fold made it far worse, because a fold that trains on the end of a
trajectory and tests on its beginning asks a model to predict backwards through
whatever drifted.

**A design can be unable to reach the threshold it is judged against.** The
sign-flip test used four thousand draws, so the smallest p it could return was
1/4001, and after correction across ninety pairs the smallest reachable q was
0.022 against a bar of 0.01. No edge could pass however real it was, and the
run reported an empty graph that read as a negative result about the organism.
`power_note` now says so when a design cannot reach its own threshold.

## Where the specification is wrong, and what was done about it

Two of the preregistered thresholds cannot be met by the thing they were
written to detect. Both are kept and both are reported as failures, because a
threshold that moves after the result is seen measures nothing. What follows is
the argument, next to the number rather than instead of it.

**`D_eff/D >= 0.4` is passed by the degenerate controls and failed by the
healthy reference.** Effective dimension is the participation ratio of the state
correlation spectrum: 1/D when one component explains everything, 1 when all
contribute equally. Strong integration lowers it, because coupled variables
share variance — that is what coupling is. Measured on the null architectures:
the recurrent reference scores 0.101, the prompt-only null 0.688 and the
frozen-slow null 0.683. The two systems that are least like a mind score highest,
and the one built to be recurrent scores lowest. A threshold no integrated
system can reach is not a bar, it is an inconsistency between two of the
criteria in the same conjunction — differentiation and irreducibility pull in
opposite directions and the specification asks for high values of both on the
same scale.

The battery therefore carries a second, separately declared line —
`differentiation_above_floor`, requiring at least three effective dimensions and
no single component holding half the variance — which is what "not
one-dimensional" means operationally at this coupling strength. It is a second
criterion, not a replacement: the 0.4 line stays in the conjunction and stays
failed.

**`PCI_A` above the 99th percentile of matched nulls can be satisfied by
nothing happening.** A response matrix of all zeros beats a null of all zeros.
The criterion now also requires that the response reached somewhere — at least
two domains, from at least two sources — before its structure is scored. That is
a strictly harder bar than the one written down, which is the only direction a
criterion may be changed after the fact.

**A minimum over five hundred and eleven noisy estimates is biased downward by
the width of its own search.** The absolute threshold on the partition score
does not account for that, and no amount of care in the estimator removes it —
only more data and a measured floor do. The battery now builds surrogates of
the same series the score is computed on, three draws each, same dimensionality
and same cuts with the coupling removed, and reports the score's margin over
them. On the recording where the score reads -0.03, the matched floor is -0.04
and the dearest cut is +0.07: the cross-domain information is real and the
minimum sits in the noise. The absolute criterion stays and stays failed; the
comparison against the floor is the one that can be read.

**Rescue without a deficit is not evidence.** Restoring a channel whose removal
changed nothing says nothing about the channel, so rescue is conditional on the
lesion having produced a deficit first.

**Two criteria in the specification are not independent.** Strong connectivity
and vertex connectivity are both computed over the ten declared domains, and a
broker that is not one of them is invisible to both: the star null passes every
graph criterion. What catches it is irreducibility and the closure test. The
conjunction is doing real work here, and any weighted average of these criteria
would let a star through.

## What the battery cannot decide

The measurement runs against a model held constant, so no edge measured here
runs through language. If Aura's domains are coupled mainly by what the model
reads and writes in a prompt — which is a real possibility and the one the
"prompt-only" null was built to represent — this battery would report the
coupling as absent and would be wrong about the live system in the same
direction each time.

Closing that gap needs the same three arms with a real model held to a fixed
seed and a fixed decoding path, which is a resident 32B and a run this cannot
schedule beside the live instance. Until then, every edge reported here is an
edge that does not need language, and the absence of an edge is evidence about
the non-linguistic coupling only.

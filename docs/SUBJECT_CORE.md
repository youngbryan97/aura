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

One run gives a number between ten and thirteen of twenty-four and the
difference between those numbers is noise, so the reading that counts comes
from three:

```bash
PYTHON=.venv/bin/python make subject-core-frozen
```

which runs the battery three times and hands the reports to
`tools/subject_core_scorecard.py`. That says, per criterion, whether it held
every time, failed every time, or changed its answer, and carries each number
across the runs with its spread.

`--quick` wires everything through in about two minutes and is underpowered on
purpose; the report says so rather than returning an empty graph. Results land
in `artifacts/subject_core/run_NNN/`: the recording, the manifest,
`subject_core_report.json` with every criterion and the evidence behind it,
`intervention_arms.jsonl` with one line per trial, `edges.csv` with every pair
tested, and the null draws, lesion arms and campaign block beside them.

## What makes two runs one campaign

A result is worth nothing without the campaign around it. Two runs of the same
command on two different heads produce numbers that look comparable and are
not, and a threshold that can be edited between the run and the reading is not
a threshold.

Every artifact carries a **fingerprint**: one hash over everything fixed before
the run — the twenty-four thresholds, the three edge rules, the displacement
size and its ceiling, the trial and turn counts, the substrate step, the eight
conditions, the estimator's components and folds, the null architectures, the
synergy triples, the lesion length, the seed, and a hash of the state schema.
Beside it: the commit, a hash over the measurement code and the organism it
measures, and whether the tree was dirty.

Two runs with the same fingerprint were measured the same way. Two with
different fingerprints belong to different campaigns however similar the
command looked, and the scorecard says so rather than averaging them. That is
the mechanical form of the rule that any methodological change made after
seeing a result begins a new campaign.

Runs go to `run_001`, `run_002`, and never to a name that already holds a
report. The run that did not come out well is the one a reader most needs.

Three rules the fingerprint cannot enforce, kept by hand:

- A threshold is never moved because Aura narrowly missed it. One line is
  contested — see the section below — and it stays in the conjunction, stays
  failed, and is reported as contested rather than removed.
- A change to how something is measured is made because the old way was wrong,
  with the argument written next to it, and it is allowed to make results worse
  as often as better. The paired test was changed from a mean to a signed rank
  for that reason, and the change refuses edges the mean would have certified.
- Every failed run is kept.

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

## What the numbers say so far

The partition score across the session, on the same organism, as each
methodological defect was removed and each channel repaired:

| | phi_do | what changed |
|---|---|---|
| first run | -0.19 | one held-out block, one ridge strength, predicting the level |
| | -0.062 | four components per domain, per-column ridge strength |
| | -0.031 | forward-chaining folds, nested penalty for the wider model |
| | -0.011 | the penalty grid searched as a path rather than a product |
| | **+0.0021** | the workspace fed and broadcasting, the substrate reaching affect |
| | **-0.0777** | the same code, run again |
| | **-0.0010** | the self-prediction loop attached, memory read by content |
| | **-0.0071** | 480 turns, the max-based edge statistic, six trials |
| latest | **-0.0007** | the weakest cut chosen on folds the score is not read from |

Three of those rows are the same instrument on the same organism, and they are
the honest measure of how much of this is noise: the score moves by 0.08
between runs, against a preregistered bar of 0.05 and a matched surrogate floor
of -0.011. What can be said is that the score is no longer reliably negative
and is nowhere near the bar. What cannot be said, at 960 turns, is which side
of zero it sits on.

The last row is not a repair to the organism. A minimum over five hundred and
eleven noisy estimates sits about three standard errors below the truth however
unbiased each one is, so the statistic was punishing the system for the width
of a search it did not choose. Choosing the weakest cut on some folds and
reading its score off the others removes that bias exactly, and it accounted
for nine tenths of the negative number.

What the cross-domain structure actually looks like, measured at the turn: the
full model beats the own-domain model for eight of the ten domains, by between
three and thirty-three hundredths of the loss. The two exceptions were the
self-state, at minus a thousandth, and the world model, at minus two hundredths
— and Phi is a minimum over cuts, so a single pair of domains that nothing
predicts is enough to hold the whole score at zero. That is why the self-model
was rebuilt to predict from the situation rather than from its own history: a
model of oneself built only from oneself cannot be wrong for a reason.

### The floor

Every effect the battery reports is a displaced arm measured against a sham,
minus what two shams do to each other. That subtraction only works if the two
shams are the same computation, and for most of this work they were not.
Measured over five sham-to-sham pairs on one condition, the largest floor term
fell as each shared variable was found:

| largest floor term | what was still shared |
|---|---|
| 1.8σ | the ontogeny service's moments and reservoir, the self-prediction loop, the efference comparator |
| 1.4σ | `random`, `numpy.random` |
| 1.1σ | `torch`'s generator, which the substrate draws its integration noise from |
| 0.95σ | the world model's forward network, dropped from every fork because a lock cannot be copied |
| 0.78σ | conversation dynamics behind a module-level singleton no container holds |
| 0.52σ | content columns encoded as hashes, where one word changing moved a coordinate a full standard deviation |
| **0.17σ** | the substrate's curiosity, and nothing else above a tenth |

Eighteen edges survive all three preregistered bars at 480 turns and six trials,
and four domains form a strongly connected component — affect, the workspace,
the self-state and the world model. That is §31 satisfied: what wins the
competition reaches a specialised process, and that process changes a later
competition. Six domains remain outside it, and six of the twenty-four criteria
— strong connectivity, vertex connectivity, cycles, reentry, spread and
replication — fail together on that one fact rather than on six.

The lesion is the result worth keeping. Clamping the cheapest cut — recurrent
cognition, the self-state and the world model held still — lowers irreducibility,
perturbational spread and synergy together, which is §39 satisfied: the
partition the search found is carrying something. Releasing it restores two of
the three, so §40 is not.

## The conjunction

ISC(K) = 1 is an and, not an average. Read across three consecutive runs of the
same code on the same organism, not from one: the single-run totals were 11, 13
and 12, eleven criteria hold on all three, eleven fail on all three, and two sit
in between. Each run is 960 turns, 31,680 frames, 164 state columns, 90 domain
pairs and 1,440 paired arms. Runs are kept one directory each and never
overwritten, so the report of a given run is in that run's directory — the
latest with one is `artifacts/subject_core/run_009/subject_core_report.json`,
and `artifacts/subject_core/README.md` says what else each directory holds.

The two that come and go are `global_access` (2 runs of 3) and `lesion_deficit`
(1 of 3). A criterion that changes answer between identical runs is not
evidence either way, and reporting the run that came out best would be picking
the draw.

- [ ] **causal_closure_scc** (§14) — every domain reaches every other one
- [ ] **robust_recurrence_kappa** (§17) — no single domain's removal disconnects the rest
- [ ] **cycles_per_domain** (§15) — every domain on two or more cross-domain cycles
- [ ] **reentry** (§16) — influence leaves a domain and returns through two others
- [ ] **partition_irreducibility** (§11) — the cheapest cut still costs prediction
- [ ] **partition_beats_nulls** (§18) — above every null architecture and surrogate
- [ ] **differentiation** (§18) — `D_eff/D >= 0.4`
- [x] **differentiation_above_floor** (§18) — well above one dimension, no component over half the variance
- [x] **intrinsic_persistence** (§19) — the last state predicts the next beyond the environment
- [x] **causal_closure_of_the_core** (§3) — nothing outside K predicts K better than K
- [ ] **perturbational_spread** (§24) — a local displacement reaches 60% of the core
- [x] **perturbational_complexity** (§26) — structured, not local and not a broadcast
- [ ] **synergy** (§27) — joint information above 10% and above its shifted null
- [x] **metastability** (§29) — regimes that persist and still turn over
- [x] **global_access** (§30) — the workspace reaches three heterogeneous consumers
- [x] **recurrent_global_access** (§31) — a consumer later changes the workspace again
- [x] **self_drives_action** (§33) — changing the self-model changes what gets done
- [x] **ownership** (§34) — the same world state updates the self differently when she caused it
- [x] **fast_to_slow** (§38) — fast cognition changes the developmental state
- [x] **slow_to_fast** (§38) — the developmental state changes later cognition
- [ ] **natural_runtime_replication** (§44) — holds across ordinary conditions
- [x] **lesion_deficit** (§39) — cutting the cheapest partition degrades what it should
- [ ] **rescue** (§40) — restoring it brings them back
- [x] **beats_every_null** (§41) — no null passes, and the recurrent reference does

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

The battery brings those layers up and then takes back what a measurement has
no business holding: the inter-instance protocol listener, because a harness
advertising itself as an Aura instance would collide with the live desktop
runtime on the port it uses for that, and every free-running loop, so two arms
of an intervention see the same computation rather than the same wall clock.

There were more of those than I knew. Cancelling the heartbeat and the
substrate left eleven — one per consciousness-bridge layer, plus the closed
causal loop and the stream of being — and none exposes a per-tick entry point
that could be called instead. They are stopped and named in the report. That is
a real limit: those organs are constructed and initialised but not integrating
while the measurement runs, so an edge that depends on their continuous
operation reads as absent here. The alternative is not a better measurement; it
is two arms that cannot be compared.

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

**Nothing recorded whether an observed outcome was hers.** `AgencyComparator`
answers a different question — how much of a predicted outcome her action
explains — and only along one path. Her intention records are hers by
construction. So an outcome she watched and an outcome she caused left the same
trace, and a self-model built from outcomes would take credit for weather. The
agency ledger keeps that distinction as state, and the comparator's attribution
now travels beside it so a disagreement between the two is visible.

**Proprioception published system memory as `vram_usage`.** Three call sites in
cognitive integration and the selfhood tick ask for `ram_usage`, which nothing
published, so the body reached those layers saying a constant zero.

**The affect phase pushed valence and arousal into the substrate under a key
only the desktop boot registers.** Offline that lookup returned None and the
whole channel was dead.

**Active memory was read as a count, so nothing could be shown to reach it.**
The retrieved set is bounded at four items and fills within a few turns, so its
length is constant from then on while its contents change every cycle. The
strongest intervention into memory measured 0.02 against a bar of 0.3. Read as
a digest of what is in mind rather than how much, memory joined the strongly
connected component on the next run.

**A run of successful writes taught her an efficacy of zero.**
`IntentionLoop.observe` decided whether the thing worked by sniffing the outcome
text for `"ok": true` or `completed successfully`, while the caller who ran the
tool had already passed a boolean to `record_action` that was never consulted.
Five successful file writes: efficacy 0.0, and the efference comparator calling
them "mostly world-caused". With the recorded flag preferred: efficacy 1.0,
agency score 0.82, "strongly self-authored".

**The self-prediction loop and the efference comparator were attached to
nothing.** A hand-written rebuild of the battery's organ set listed its fields
by name and silently dropped the two added after it was written, so eleven of
the self-state's columns read zero for a whole session and every criterion
resting on them failed on an organ that was there.

**She wrote the file and never saw it.** The probe action wrote, verified and
recorded — and nothing came back through the senses, so nothing she did could
reach perception by any route.

**The free-energy engine's action urgency never reached the drives.** The
heartbeat reads it, to enter a workspace bid when free energy is high. The
motivation phase — which decides how fast the drives press and when an
intention is generated — did not, so drives ticked at the same rate whether the
world was behaving as modelled or not.

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

**A whole sensory stream arrived and nothing could read it.** Nine places
append to `world.recent_percepts` and four read it, and the readers and the
writers agreed on nothing. The workspace priced its perception bid from a
`salience` key no producer has ever written, so every real percept bid zero.
The affect phase keys on a `type` and drops what it does not recognise, and
three of the types the tree emits had no entry — a phase crash, her own
apology, and every stimulus injected through the compatibility bridge. What she
saw on screen was appended with a role and a string: no type, no strength, no
stamp, so it was unfeelable, unbroadcastable and infinitely old on arrival. And
the affect phase cleared the list after reading it, so the workspace
competition, the world model's observation, the phi estimate and the state's own
reading of perception all ran afterwards and all saw an empty stream.

**There were three bodies.** The proprioceptive loop publishes cpu, memory and
temperature into the state that every phase reads. `core/senses/soma.py` runs
its own timer against psutil. `core/soma/resilience_engine.py` goes to psutil on
every call. Homeostasis asks whichever of the last two is registered under
`soma` for the numbers it turns into her will to live — so the one figure that
says whether she is holding together came from the host by a route her own
sensing never touched.

**The body's load reached her will to live through two cliffs at eighty
percent.** Below them the machine had no effect at all; above them the same
fixed decrement whether the host was at eighty-one percent or pinned. The same
module's docstring says it regulates proportionally, and the drives it updates
below do.

**Attention had no motivational consequence for anything but a drive alert.**
The motivation phase replenishes whichever drive last won the broadcast, and
the consumer that names it fired only for a source called `drive_`. Drive
candidates enter the competition through an alert gated at seventy percent
urgency and five minutes since the last one, so across an ordinary hour of
thinking that consumer set nothing at all.

**The goal bid read a key no producer writes.** The workspace priced
deliberation's bid from `urgency`; the goal engine writes `priority`. So every
real goal bid zero and deliberation never once reached attention. The memory bid
had the mirror-image defect: retrieval ranks its candidates by how well each
matched the question and discarded the number at the line that used it, so the
bid entered at a flat neutral and nothing about what was recalled could change
what won.

**The world model the cognitive cycle feeds never trained.** `observe(...,
learn=True)` appends to a replay buffer; the gradient steps are taken by a lane
that has to be started, and the only caller of `start_training` in the tree was
the ontogeny organ on its own model. The one the cycle observes into filled a
buffer for the whole of every session and never took a step. Its surprise was
not low — it was arbitrary, and two subsystems read it as evidence about the
world.

**Nine of the world model's seventeen columns were named for the wrong
quantity.** The reader returned values in a different order from the schema's
declaration from `model_hidden_norm` onward, so `causal_nodes` was carrying the
forward model's last surprise and the hidden norm was carrying a count of
available facets. Every value was real. The schema's own source annotations are
now checked by a test, which fails on two columns if the reader is put back one
place.

**Interoception was proprioception of the host.** She had no sense of her own
exertion — how wide a recall she asked for, how many steps the substrate
integrated, how much the world model learned from what she showed it — only of
a machine whose load is mostly not hers. So nothing she chose could return to
her as a felt cost, and an experiment that holds the host still to keep two arms
comparable held still the only body channel she had.

**The intention generator could not fire.** It picks the most depleted drive
and dispatches on its name, and it had branches for three of the five drives
the state carries. The one it had no branch for — growth — starts at fifty
where the others start at eighty to ninety-five, and decays, so it is the most
depleted drive on almost every tick of an ordinary life. The assessment
returned None every single time it ran. The urgency each branch would have
carried was a fixed number, so a drive one point below the line asked as loudly
as one empty for a week. And the initiative that survives governance is written
to `pending_initiatives`, while the workspace's bid for deliberation read
`active_goals` — the one thing deliberation produces within a turn could not
reach attention at all.

**Deliberation could see the argument coming apart and nothing else.** How
badly a moment was going was three readings, all about coherence. A moment also
goes badly when the world is not doing what was predicted, when something
hurts, when the process underneath is unsettled, and when the work is costing
more than it is worth — and every one of those is a reading that already
existed, that nothing here consulted. Eight now, one per channel that bears on
whether to act and on what.

**A default read as a demand won every competition.** Deliberation beat
perception, memory, affect and the body on every single turn, and not because
anything was pressing: the workspace's bid read a goal's `priority` as its claim
on attention, and the goal engine's own projection carries a flat one. Priority
is how important the work is once it has been chosen; urgency is how much it is
asking to be thought about now. Three more bids sat at their ceilings for the
same kind of reason — a feeling raised by a flat amount saturates within a dozen
turns, ignition was the winner's priority divided by its own threshold and
clipped at one, and prediction error was clipped at one when it is unbounded
above. A competition decided by ties at the ceiling is not a competition, and a
displacement of any of those four moved nothing because there was nowhere left
to move.

**A degradation with no way back.** `identity.stability` had exactly one writer
in the whole tree — the loop detector — and it only ever subtracted. One
repeated sentence dropped it three tenths and nothing raised it again for the
life of the state, so the phi estimate, the executive closure, the causal
self-state and the workspace's self bid all read a number pinned at its floor.

**Everything she ever did was recorded as one capability.** The agency ledger
keys capability beliefs on the name of what was done, and the intention loop was
handed the same name for every action. It learned one thing about her however
many different things she tried, and her confidence about any particular thing
she can do was her confidence about all of them at once.

**A registered broadcast processor wrote where nothing reads.** One of the five
appended a bounded trace of each winner to an attribute on the workspace, and
nothing anywhere read it — a writer with no reader, inside the file written to
remove writers with no readers.

**The counted schedule ran the first line of each loop, not the loop.** The
free-running layers cannot be left running for a paired intervention, so the
battery advances them by a count. A layer declares, in order, the methods one
iteration of its loop calls; the harness asked for a single entry point and took
the first name that resolved. A counted neurochemical iteration ran
`_metabolic_tick` and never `_push_modulation`; interoception sampled the
hardware and never pushed it to the mesh or triggered a neurochemical event;
oscillatory binding stepped its oscillators and never emitted a binding moment.
Every method that was dropped is the one that leaves the layer. The harness
stopped executing the coupling and the battery reported the coupling as weak.

The rate was wrong in both directions at once. A frame was worth a declared half
second while the experiment clock advanced by whatever calibration measured, so
two timelines ran side by side; and the frames-between-steps count was clamped at
one, capping every layer at two hertz against a mesh at ten, a field at twenty
and oscillators at a hundred. A layer's iterations in a frame now come from its
own declared rate and the frame's own duration, and a sub-schedule — the
oscillators emit on every tenth internal tick — runs at its own ratio.

**One turn is one second, and the machine does not decide that.** Calibration
measured how fast this host ran a frame and made the experiment's second that
long, which puts the host back into the timeline the clock was installed to
remove and gives two machines different amounts of life per turn. A turn is the
organism's unit of experience and the layer rates are per second, so a turn is
worth one second and the frame follows from how many readings a turn takes —
a property of the phase list. The machine's real pace is still timed and
reported beside the result.

**One Euler step of half a second is not thirteen of a tenth.** The substrate is
a nonlinear stochastic recurrent system: the tanh is evaluated at different
intermediate states, the noise draws are independent, and the clip can bite in
the middle. It now takes the iterations its own configured rate calls for, each
at its own configured constant.

**There were three substrates.** The consciousness system built one and
published it as `conscious_substrate` and `liquid_state`; the orchestrator's
boot mixin built another unconditionally and published that as
`liquid_substrate` and `conscious_substrate`, clobbering the first under the
shared name; the organism then republished the consciousness system's under
`liquid_substrate`. Only one of them is ever stepped, and the losing name is the
one the attention gate, somatic qualia, temporal continuity, the predictive
hierarchy, the aesthetic engine and the workspace's own bid for recurrent
cognition all use.

**A read-only seam answered None for every service built on demand.**
`get_runtime_service` resolves with `peek`, which never invokes a factory —
right for a diagnostic, wrong for a caller that wants the thing. A service
registered lazily therefore answers None for the life of the process, silently,
while being present and complete. `tools/audit_read_only_service_asks.py` finds
eighteen of them across twenty-four call sites.

**A competition of accumulating urges that ran nowhere.**
`core/consciousness/drive_integration.py` is a leaky integrator per drive with
mutual inhibition and a Schmitt trigger, written against the critique that a
drive fired the instant a point crossed a line. Registered as a service, asked
for by nothing. The assessment beside it picked the single most depleted budget
on every turn of every life.

**Thinking longer was never a decision.** `core/cognition/value_of_computation.py`
answers whether another round is worth what it costs, and says in its own first
paragraph that this is what makes a spend decision rather than a spend habit.
Its only caller was reached from nothing but its own tests, while the draft
competition reported in a log line that its winner led by less than the spread
among the drafts that lost and took the first of the tie anyway.

**Nothing spent her energy.** Every other drive decays with the clock; energy's
stated decay is zero, correctly, because what depletes energy is work. The will
engine owns that budget when present, and it is present in the full desktop
runtime and nowhere else. So the branch of the intention assessment that fires
on a depleted energy could never fire, and whatever asked how much she had left
to spend read a hundred.

**The self-model predicted a constant and graded itself against one.** Its
valence error and its drive error were both exactly zero for the life of every
process. The heartbeat read affect from the engine that feeds
`AffectUpdatePhase` rather than from `AuraState.affect`, which the phase settles
and every other consumer reads — measured over six turns, the engine reported
valence 0.0 every time while the state's moved between 0.17 and 0.29.

**Her valence was a constant at four decimal places.** It is `tanh` of a
weighted *sum* over twenty-two positive emotion channels against twenty
negative, so its input scales with how many names the dictionary contains rather
than with how she feels, and it sat past the point where the function has a
slope. Over forty-eight turns across all eight conditions it stayed inside 0.902
to 0.913 while the emotions underneath ranged from a tenth to four fifths:
happiness rising by a tenth moved valence by one hundred-thousandth.

**Her sense of her own exertion was pinned at its ceiling.** Each unit cost is
defined as what an unremarkable turn produces, and the aggregate took the mean
of those ratios and clipped it at one — so an unremarkable turn read as maximum
exertion by construction. Median 1.000 over four rounds of the eight conditions.
The workspace prices the body's bid on the largest of load, thermals and
exertion, so that pinned reading won the competition on nineteen turns in
twenty-four.

**A bonus one competitor could not lose.** `priority_at` adds three tenths of a
candidate's affect weight to its claim, and the field means the affective charge
of that content. Only the affect bid ever set it, and it set it to the moment's
global arousal — a quantity belonging to the whole moment handed to one
competitor. Affect won ninety-six competitions in a hundred, so attention was
`affect_*` whatever else was happening and the action chosen by what she was
attending to was the same action every turn.

**The objective she had just chosen made no claim on attention.**
`ExecutiveClosure` writes it twice, two lines apart: as an initiative carrying
the need pressure that selected it, and as a goal record carrying a flat
priority and no urgency. The workspace prices deliberation on urgency.

**"Call this after each incoming user message", and nothing did.**
`DiscourseTracker` is the only writer of conversation energy, discourse depth
and the user's emotional trend, and its `update` was called by nothing; the
tracker itself was constructed inside an optional integration, so in most
runtimes the service did not exist to call.

**How hard she may think was computed and kept out of the state.**
`HomeostaticCoupling` derives creativity, focus, urgency and vitality from the
substrate; three of the four reached `cognition.modifiers` as constants.

**The phi estimate was filled once and latched.** Recurrent cognition's own
headline number was a constant in every recording, because the guard that filled
it required it to be empty.

**The harness never published its state where the tree reads it.** A dozen
runtime paths take the current state off `state_repository._current`. The
offline organism never registered the vault it carries its state in under that
name, so all of them read None while the phases mutated a state.

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

**The end-of-tick projection never ran.** `AuraState._refresh_cognitive_health`
computes coherence, fragmentation, the contradiction count and the whole
cognitive-health block from the state a turn has finished with, and the kernel
calls it at the end of `tick`, outside the phase loop, deliberately. This driver
runs the phases directly, so during a battery run it never ran at all: four of
the workspace domain's columns were constants, and the deliberation phase's
reading of how badly the moment was going was reading two of them.

**The battery's own model was never asked.** It installs a deterministic mind
so that two arms differ by the intervention and not by decoding, and it
installed it as the kernel's `llm` organ. The phases prefer the container's
`llm_router` and fall back to the organ only when the router is absent — and
the router is present offline, with no model behind it. So the response phase
asked it, got nothing, and raised on every turn of every run: fourteen thousand
times in one of them. The reply never landed, the exchange never reached memory
consolidation, self-review had nothing to review, and the whole arc from a
question to an answer was missing from a measurement of whether the parts of
her reach each other. A test double has to answer to the name its callers use.

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

**`D_eff/D >= 0.4` was passed by the degenerate controls and failed by the
healthy reference, and is now applied as an upper bound with a floor beneath
it.** Effective dimension is the participation ratio of the state
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

Measured across the specification's own null suite: frozen-slow 0.736, one-way
0.481, star 0.313, hub 0.195, prompt-only 0.158, Aura 0.096, recurrent
reference 0.070. The bar as written is passed by the two systems built to be
least like a mind and failed by the two most integrated.

The criterion is therefore two-sided, which is strictly more than the section
asked for, and it keeps the section's quantity and the section's number: at
least three effective dimensions (the state is not one thing), no component
holding half the variance (not one global scalar), and the ratio *below* 0.40
(not a bag of independent variables, which would score 1.0). The original
one-sided reading is reported beside it on every run, so a reader can see it
fail. `differentiation_above_floor` remains as a separately declared line.

The recurrent reference fails the two-sided line too, at 2.79 effective
dimensions in a forty-column toy — a true statement about the toy rather than a
fault in the bar, since its ten declared domains collapse onto three
directions. The reference exists to prove the irreducibility measure can say
yes to something, not to stand as a candidate subject, and no criterion is set
by what it needs.

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

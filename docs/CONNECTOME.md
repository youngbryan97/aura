# The connectome

Aura has a map of her own nervous system. It is built from her source, it is
recorded while she runs, and every number in it is compared against a published
measurement from a real brain.

This document says what the map is, what it found, and how to rebuild it.

## Why a connectome and not a diagram

Connectomics has four things this repository did not have.

A substrate that cannot be argued with. The tissue is there; every claim about a
circuit has to survive being checked against it. Aura's substrate is her source,
which is complete and exact and is not a summary of anything.

A unit of connection strength that is countable. A synapse is one contact. A
call site is one contact. Two functions joined by seven call sites are joined
seven times, and that number is not a matter of opinion.

A way to say honestly what a reconstruction missed. Automated segmentation makes
split errors and merge errors, and connectomics measures both rather than
publishing a graph and hoping.

A published measurement to compare against. H01 reconstructed a cubic millimetre
of human temporal cortex — 57,000 cells, 150 million synapses, 1.4 petabytes —
and the numbers that came out of it are numbers Aura can be held to.

## A call count is not a weight

Two components can be causally coupled without ever calling each other, and one
function can call another ten thousand times while using nothing it returns. The
reconstruction counts call sites, which is the right weight for control and the
wrong one for information, so the same edges carry a second weight: what happens
to what comes back.

Six outcomes, worked out by a def-use walk inside each function. Over 420,178
call sites:

| what became of the value | share |
| --- | --- |
| used locally | 42.3% |
| decided the caller's control flow | 23.7% |
| returned onward | 15.9% |
| discarded | 11.8% |
| logged and stopped | 4.6% |
| escaped onto state outliving the call | 1.8% |

**24.5% of drive edges carry no information at all.** That corrects a number
here: heavy pairs are 3.54% of information-carrying edges against human cortex's
0.092%, so 38 times cortex rather than 45, and 1,167 of 3,358 heavy pairs carry
nothing. The heaviest of them is a builder being fed 73 times.

## The five layers

The wiring diagram is one layer. *C. elegans* has had its synaptic map since
1986, and when the monoamine and neuropeptide layers were added on top of it,
96% of the monoamine connections turned out to exist in no other layer and 82%
of the neurons carrying a dopamine receptor receive no synapse from any neuron
that releases dopamine.

Aura has the same shape. Five layers are extracted:

| layer | what joins two cells | directed | pairs | unique to it |
| --- | --- | --- | --- | --- |
| wired | one calls the other | yes | 72,650 | 99.9% |
| volume | one publishes a topic the other subscribes to | yes | 11 | 100% |
| gap | both touch the same container key or module global | no | 3,795 | 98.6% |
| io | one writes a durable store the other reads | yes | 108 | 97.2% |
| ipc | both name the same process boundary | no | 259 | 99.6% |

The worm's headline reproduces in every layer: almost nothing outside the call
graph is visible from inside it. Seven cells are central in the shared-state
layer and peripheral in the call graph — the worm's finding that its monoamine
rich club is different cells from its wired one.

The layers also count what has one end. Of 120 event topics, 87 are published
with no subscriber the scan can find and 26 subscribed with no publisher; seven
have both. Of 94 durable stores, 84 are written and never read. Of 923
shared-state keys, 229 are written and never read and 416 read and never
written by any access a static scan can follow. Those are candidates, not
defects: a subscriber can be registered from a table nothing static can follow
and a store can be read by another process on another day, and each finding
says so and names the step that would settle it.

## The merge error that moved every hub

The largest sink in the combined graph had an in-degree of 4,357, and it was a
real method — `PhenomenalField.strip`. Almost none of those callers were calling
it. They were calling `str.strip`. The next three sinks were the same shape:
every `logger.warning`, every `str.replace`, every `list.extend`, each attached
to whichever class happened to define that name.

Two rules fixed it. A method name owned by a builtin type or by a standard
library object everyone holds is not resolved on a receiver whose type was not
established; and a call on an expression rather than on a name reports its
receiver as unknown rather than as absent, which is how `"".strip()` got past
the first rule.

Scored against the recording, this is the merge-split trade going the right way:
recall 0.8284 to 0.8258, so 99.7% of the edges seen firing survive, while
suspected merges fall 1,625 to 1,285 and expected run length rises. In-volume
coverage goes 0.502 to **0.736** and ambiguous call sites 88,182 to 15,870.

Every number in this document that was measured before that fix has moved. The
shape has not.

## She does not have one functional connectome

The structural graph is the same whether she is holding a conversation,
planning or refusing something, and those are not the same computation. The
effective connectome is the influence one cell has on another under a cognitive
condition, and it is measured three ways whose grades do not convert:
`predictive` (does the source's past improve prediction of the target's future),
`model` (remove it from the graph and propagate again), and `interventional`
(disable it while she works). The null for the first is a rotation rather than a
permutation, because shuffling would destroy the source's own autocorrelation
too and almost anything beats a null that incoherent.

Over the nine workloads, edges surviving the null range from 2 under
`reconstruction` to 637 under `tests_verify`. The comparisons split cleanly: the
three workloads that share machinery correlate at 0.92 to 0.96, and the ones
that do not correlate at nothing — conversation against governance 0.009,
conversation against runtime −0.001, runtime against verify −0.068 with 295 of
1,039 shared edges active in one and not the other.

Same anatomy, different active circuits, and which ones is measurable.

## The ring is a star

Every design document draws the same loop: interoception → affect → workspace →
higher-order monitoring → self model → planning → action → interoception. A
module named after each station is not evidence that the loop exists.

Five things had to be fixed before the answer could be believed, and all five
were mine. The station table used exact module prefixes and missed most of each
station. Links were read as direct edges between subsystems holding hundreds of
cells each. They were read from the wired layer alone, where these subsystems
barely couple. The measure was binary reachability, which saturates — over the
combined graph a randomly shuffled ring closed as often as the intended one, at
z = −0.9. And the flow drained into maintenance: `record_degradation` receives
from 3,760 cells, so every link read low for a reason that is about where errors
are reported.

The measure now is the share of one station's influence that lands in the next
against the share its size predicts, conditioned on arriving at a station at all.
With that:

| link | enrichment |
| --- | --- |
| interoception → affect | 4.41 |
| action → interoception | 1.55 |
| workspace → higher-order | 0.09 |
| higher-order → self model | 0.04 |
| affect → workspace | 0.02 |
| self model → planning | 0.00 |
| planning → action | 0.00 |

Ring enrichment 0.874 against a shuffled null of 0.853, z = 0.07. **The ring
does not exist.** What exists is a star: every station's outflow goes to
interoception at 3.6 to 5.7 times its share, and workspace, higher-order, self
model, planning and action receive essentially nothing from anywhere.

Each missing link is a finding that names its enrichment and what would close it.

### Measured while they were running

None of the nine recorded workloads fired those stations — planning fired in
none of them — so `tools/record_coalition_activity.py` drives all 162 of their
modules and records it.

The first recording drove one station at a time, and its answer was about the
workload: a station awake while the others are idle cannot influence them, so it
could only ever have shown influence inside a station, and that is what it showed
— 132 influences surviving their null, every one within a station, none between
two.

So the workload ends with an interleaved condition where every station's modules
are cycled together. 2,982 frames at 50 ms, 36 passes over all 162 modules, 85
million events, every station awake at once. Between any two stations: nothing.

That answer was also about the workload, and it took a third recording to see
why. The probes call each station's readers — a getter, a status, a snapshot.
Calling a getter on affect hands nothing to the workspace. **The stations are not
coupled by calling each other.** They are coupled by the state object the
kernel's phase pipeline passes from one to the next, and that object only moves
while a turn is being taken.

### The recording that could have shown it

`tools/record_turn_activity.py` runs the kernel's 29-phase pipeline over one
shared `AuraState`: 240 turns across six objectives, 31,042 frames at 2 ms,
1,871 cells, 3.9 million calls, no phase failing. The model is a deterministic
stub, which is the one part that cannot run offline, and holding it constant is
what makes the six conditions comparable.

Then the station table turned out to be wrong, in a way only that recording could
show. It was written from module names, and the modules named after the stations
do not run a turn: **0 of 129 workspace cells and 0 of 272 action cells fired**.
`PHASE_STATIONS` now places all 29 phases the kernel assembles — seventeen at a
station, twelve explicitly outside the ring with the reason — and two tests fail
if a phase the kernel runs is missing from it, or if a station's patterns do not
reach the phase it was assigned.

### Three measurements, two of them wrong

The first scored each cross-station pair with an F test and a false-discovery
correction. All seven ring links carried, at 93 to 100% of pairs. Then a sample
of pairs was re-run with the source rotated in time, and **one rotated pair in
four passed the same threshold**. A parametric p-value on a spike train recorded
at two milliseconds is not a p-value.

The second added the population's recent past to both models, on the theory that
what the test had found was the shared rhythm of a turn. It barely moved. That is
the tell for a periodic workload: a turn repeats, so rotating a source by any
shift lands it on another turn and preserves the alignment the null was meant to
destroy.

So rotation became the null rather than a check on one. Each pair is scored
against its own eight rotations, and the link on the median paired difference
with a bootstrap interval over pairs. That discriminates: ring links score 0.006
to 0.101, and station pairs that are not in the ring score 0.0002 to 0.004.

### The recording was still wrong, one layer down

Those turns ran as system ticks. The response phase produces no reply on that
path — it is a background pass — so the action station never acted, and the one
link that measured at chance was the one out of it. Driving user-facing turns
instead, every condition but the idle tick answers, at 151 to 193 characters.

On that recording **all seven links carry, in all six conditions**, action back
to interoception included: median gain +0.0033 over its own rotations, interval
+0.0033 to +0.0035, better than its rotations on 88.6% of 493 pairs.

Which sounds like the ring closing, and is not.

### Where the ring sits among the other thirty-five

Seven links that all carry mean one thing when nothing else does and another
when everything does. Measuring every ordered pair of stations rather than only
the seven: **42 of 42 carry**. A pipeline over one shared state object couples
every stage to every other by construction, so "this link carries" separates
nothing, and reporting the seven without that denominator would read as evidence
for a ring.

The ranking is the finding. The ring's seven links come 4th, 5th, 8th, 12th,
24th, 30th and 37th of 42 by strength, and the three strongest connections in
the whole system are not in the ring at all:

| connection | median gain over its own rotations | in the ring |
| --- | --- | --- |
| self model → interoception | 0.914 | no |
| self model → affect | 0.639 | no |
| affect → planning | 0.313 | no |
| interoception → affect | 0.028 | yes |
| affect → workspace | 0.010 | yes |
| workspace → higher-order | 0.009 | yes |
| planning → action | 0.008 | yes |
| self model → planning | 0.006 | yes |
| higher-order → self model | 0.005 | yes |
| action → interoception | 0.003 | yes |

The strongest ring link is an order of magnitude below the strongest link there
is. And scored against all 720 directed cycles through the same seven stations —
exactly, not by sampling — the architecture's order comes out stronger than 25%
to 55% of them depending on the condition, at z −0.65 to −1.03. It is an
unremarkable ordering among the orderings available.

What the measurement supports is not the ring. It is that **the self model is
the hub**: what she holds about herself says more about her interoceptive and
affective state than anything in the system says about anything else.

## Cutting a station out

Everything above measures what varies with what. `core/connectome/intervene.py`
does the other thing: it replaces a cell with one that does nothing, runs the
same work again, and puts it back. The cut is exact, it is reversible, it refuses
anything it cannot cut, it matches an async cell with an async stub, and it
counts the calls it absorbed — a lesion on a cell nothing called looks exactly
like a lesion that did nothing.

`tools/run_lesions.py` cuts every phase of a station together, because
`do(station = 0)` is not one function, and scores it against the same number of
non-ring phases at the closest connectivity. Removing anything from a running
system changes something; the number means nothing without a comparable removal.

Five defects in the experiment, each found by running it. The first readouts —
`coherence_score`, `phi_estimate`, `last_response`, `active_goals` — are all
constant across all six objectives and both affect arms with the model held
fixed. **56 of the state's 72 numeric fields are.** All four would have read "no
change" under every lesion and all three predictions would have come out
confirmed with nothing behind them, so a readout is now refused unless it moved
at baseline. One kernel served every arm in sequence, so a drift down the run
read as an effect of the cut; each arm gets its own now and shuffles the
objectives. `version` counts writes to the state, so silencing any writing phase
decrements it by construction — it fell by exactly 1.0 under two lesions and 0.0
under both controls, which reads as a competence loss and is arithmetic.
`hash(mode)` was a readout, in a codebase that has been bitten by hash
randomisation before. And "refuted" was reported for a readout that a control
lesion moved further, which is not a refutation but the wrong readout: working
memory is filled by memory retrieval, which was one of the controls.

Of the three predictions registered from the theory, one is partly confirmed and
two are not attributable. What the first run found was registered as three more
and run again on an independent seed, all three confirmed:

| cut | intact | lost | control |
| --- | --- | --- | --- |
| higher-order monitoring | pipeline completes, working memory fills | the selfhood reading, 5.0 → 0.0; spread of pending intents, 0.373 → 0.0 | 0.0 on both |
| the workspace | pipeline completes, engagement untouched | fragmentation score, 0.1407 → 0.0, with its spread and its affective gap | −0.0012 |
| affect | pipeline completes, pending intents still form | regulation: an injected valence difference of 1.2 is normally pulled back to 0.0006 by the end of a turn, and without the two affect phases the whole 1.2 survives | −0.0005 |

The last one is the clearest thing in this document. The affect phases are not
decoration on the pipeline; they are a controller, and they remove 99.95% of an
injected displacement within one turn.

## The mesh, and the two ends that were not connected

Everything above is made of functions. The neural mesh is not: 64 cortical
columns of 64 neurons, wired by a distance-decayed weight matrix, running at
10 Hz on its own clock. Keeping it in a separate object made one question
unanswerable, because a distance needs both ends in the same graph.

`core/connectome/neural.py` turns the columns into cells and both weight
matrices into edges, then finds the seam — which functions call the mesh's own
surface, and on what. Two drive it: `EmbodiedInteroception._push_to_mesh`
injects sensory, `ConsciousnessBridge._integration_tick` injects association.
Five read it, three through `get_executive_projection`.

So: can what is injected reach what is read?

| across 8 seeds | before | after |
| --- | --- | --- |
| executive columns reachable from sensory | 0 of 16, every seed | 12 to 16 of 16 |
| columns wired to nothing | 7 to 16 | 0 to 5 |
| connected components | 13 to 25 | 1 to 6 |
| largest component | 19 to 49 | 59 to 64 |

The cause was one missing pathway. The mesh builds its top-down feedback
explicitly, tier pair by tier pair, and left the bottom-up direction to the local
wiring, whose probability decays as `exp(-|i - j| * 0.15)`. A tier boundary is
exactly where that index distance is largest: sensory column 0 to executive
column 48 is `0.05 * e^-7.2`, about one edge in twenty-eight thousand. The 1.5×
feedforward bias inside that function could only apply to edges the decay had
already made impossible.

`_build_feedforward_weights` builds the pathway the way the feedback one is
built and leaves distance out, because a projection between cortical areas is an
axon bundle and its existence does not fall off with how far apart the areas are.
The density is the same 0.05 the local wiring already uses; only the decay is
gone.

The seam scan needed the lesson this package had already learned in the call
graph. `get_field_state` is also a method of the unified field, and matching on
the method name alone attributed its call sites to the mesh.

## The mapping

| connectomics | here |
| --- | --- |
| cell | a function or method |
| neuropil | the module its arbour sits in |
| region | the package above that module |
| synapse | one call site |
| connection strength | how many call sites join the same pair |
| axon initial segment | the guard that decides whether the body runs |
| cell class | measured from what the cell's exits do |
| afferent cell | calls out of the process to read the world |
| efferent cell | calls out of the process to change it |

Nothing in that table is assigned by name. A function whose exits mostly refuse
is inhibitory however it is spelled, and a cell that calls `subprocess.run` is
efferent whatever its module is called.

## What the reconstruction contains

47,950 cells across 3,592 modules, 81,066 drive
pairs, 192,349 contacts, built in fifteen seconds by walking
3,598 files with `ast`. 50.1% of the in-volume call
sites resolve; the rest are recorded as ambiguous with their candidate lists, or
as leaving the volume, and the coverage figure says which.

**Every number below is a measurement of one commit.** These are from
`3e3bf4516`, and the tree moves — a parallel agent commits to it while this runs.
`make connectome` reproduces all of them for whatever the tree is now, and the
report carries the reconstruction's digest so two runs can be told apart. What
does not move between runs is the shape: the comparisons against cortex, the fly
and the worm hold to their reported precision across every reconstruction taken
during this work.

A recording is the other half. `sys.monitoring` watches cells fire at ZAPBench's
914 ms volume rate and emits both the call counts and the calcium trace a light
sheet would have seen. It can also capture caller-to-callee pairs, and those
pairs are ground truth — an edge that fired happened. No connectomics project
has ever had that, and it is what lets the agglomeration threshold be chosen
from a measured curve instead of from taste.

## What it found

### Her connections are far heavier than cortex's

83.2% of Aura's connected pairs touch once and 3.87% touch four or more times.
Human cortex runs 96.5% and 0.092%. Her heavy pairs are **42.1 times** more
common than cortex's, and her heaviest carries 83 call sites where H01's
heaviest carried about fifty.

H01 reads a four-or-more-contact pair as a powerful connection, rare enough to
be special. At 3.87% of everything, hers cannot be.

### Local recurrence is missing

Cortex's within-layer connection density is 5.945 times its between-layer
density. Aura's is 0.652 — she connects across levels more often than within
them, where cortex does the reverse by six to one. The shortfall is a factor of
9.1, and it survives any relabelling of the layers, which matters because the
orientation of her hierarchy is undetermined: the anchor holds by 0.097 of
trophic height against a spread of 11.03.

The specific pathways she lacks are the local ones. L5I to L5E is cortex's
densest connection relative to its own mean, at 6.98 — inhibitory control of the
output layer — and she has 0.141 of her own mean there.

### The excitation to inhibition ratio is cortical overall and local where it is not

Whole system: 3.978 excitatory cells per inhibitory one, against cortex's 4.035.

`reality_reach`, the package that acts on the world, runs 1.67 across 989 cells.
`auth` runs 1.11 and `social_media` 1.44. Those packages are two to four times
more inhibited than the rest of her.

### The feed-forward loop is over-represented and there is no rich club

Against a degree-preserving rewiring: reciprocity z=+180.7, small-world sigma
21.87 (clustering 22.1 times the null at a path length 1.009 times it),
modularity 0.832 over 1,156 communities, and the feed-forward loop at z=+31.9 —
the same motif that is over-represented in *C. elegans* neurons and in *E. coli*
transcription. The one motif that is *under*-represented is 021C, the plain
two-step chain, at z=−3.0.

The rich club runs below its null from k=8 upward, 0.21 of chance at k=128 and
0.44 at k=256. Cortex's hubs preferentially wire to each other. Hers avoid each
other.

### There is a neck, and the evidence for it is thin

125 afferent cells and 275 efferent ones. A small set of cells carries half the
sense-to-action flow, and those cells converge far harder on their inputs than
an average cell, which is the integrator signature the fly's ascending and
descending neurons show.

A static reconstruction cannot see a call made through a service lookup or an
event bus, so a count of statically visible paths is a floor, and the neck
verdict is marked thin until the snapshot has been proofread against a
recording.

Proofreading moves it. 12,027 pairs were seen firing and the reconstruction
contained 9,986 of them, recall **0.8303**, expected run length 2.656, 2,041
split errors left. Writing a join for each pair it lacked — 1,742 of them — takes
sense-to-action reachable pairs from 65 to **162**, and on the proofread map the
cells carrying that flow converge **109 times** harder on their inputs than an
average cell. The fly's neck is made of integrators, and so is hers.

162 is still below the bar of 200 the spine analysis wants, and the remedy is no
longer proofreading. It is a longer recording: a pair whose path was never taken
in the recording is a pair the recording cannot join.

### Two individuals differ three times more than two flies

Reconstructing at HEAD and at 400 commits earlier gives two individuals of the
same system: 43,515 shared cells, 40 lost, 4,229 gained, 7,835 rewired pairs.
15.3% of cells changed, against the 4.8% of the fly central brain that is
sex-specific or dimorphic. Against a checkout a few days old the same analysis
finds 47,914 cells in the core, none lost and none gained, and a mean contact
shift of 0.00004 — the measure is reading the distance between the two
individuals rather than a constant.

Cell typing survives it. Adjusted Rand 0.936 between the two, and 67.8% of
multi-member types intact across 400 commits, which is the cross-individual
reproducibility FlyEM requires of a type.

Where the change lands does not follow the fly. In the fly, sensory and motor
regions are nearly identical between the sexes and the differences pile up in
the higher-order centres. In Aura, a permutation test over 400 draws puts the
enrichment at z=-0.27: change is spread no differently from chance.

## What the findings do

`core/connectome/pathology.py` turns the measurements into ranked work. 114
findings over six kinds on the current tree, and every one carries its own
confidence: 64 measured, where the reconstruction is the evidence and there is
nothing to confirm, and 50 candidates, where a static scan cannot settle it and
the finding names the step that would.

    kind                        count   confidence
    interface_used_as_internal     25   measured
    gate_dominated_cell            25   measured
    half_wired_channel             50   candidate
    unnoticed_hub                   7   measured
    over_inhibited_region           6   measured
    missing_local_recurrence        1   measured

`make connectome-pathology` writes them. Two telemetry channels carry the counts
and the health fragment carries them into `runtime_health_report()`.

## The local loop, and what it saves

The missing-recurrence finding is the one with a mechanism attached.
`core/connectome/laminar.py` is the cortical local loop as an operator anything
can wrap around a scoring step: integrate evidence without discarding it, let a
candidate that falls far enough behind stop being sampled, and stop when the
leader's lead is larger than the noise on that lead can explain.

Measured against a fixed budget given the loop's own worst case, 600 trials at
each noise level, four candidates, half of them easy:

| noise | loop | fixed budget | calls saved |
| --- | --- | --- | --- |
| 0.10 | 0.9967 | 0.9950 | 62.1% |
| 0.25 | 0.8900 | 0.8950 | 44.5% |
| 0.40 | 0.8067 | 0.8050 | 31.2% |

Same accuracy, a third to two thirds fewer evidence calls. An evidence call is a
model call.

The first version of that circuit lost to its own null — it leaked at 0.80 a
cycle and normalised the drives before accumulating them, scoring 0.74 against
0.90 — and the docstring keeps why.

Its one-shot form is live. `core/consciousness/multiple_drafts.py` took `max()`
over three coherence scores and reported the result as a decision; when the top
two sit inside the spread of the ones that lost, that is a coin landing, and it
lands differently on the next process. The competition record now carries whether
the winner actually won and by how many standard errors.

## Warming what is about to run

A forecaster needs a cell's activation to a decimal place. Nothing here does. It
needs the set of cells about to run, and the connectome answers that directly.
Measured over 300 frames against two nulls:

| rule | precision | recall | F1 | cells named |
| --- | --- | --- | --- | --- |
| persistent | 0.6887 | 0.6689 | 0.6786 | 54.6 |
| connectome, contact-weighted | 0.6328 | 0.6713 | 0.6515 | 59.7 |
| connectome | 0.4760 | 0.6816 | 0.5605 | 80.5 |
| frequent | 0.0842 | 0.1198 | 0.0989 | 80.0 |

The connectome beats knowing what is hot by five times and loses to knowing what
just ran. Weighting neighbours by contact count recovers most of the gap and does
not close it.

### How directly cognition reaches the actuators

The nerve cord connectome measured something nobody would have drawn. Direct
connections from descending neurons to motor neurons are infrequent: a motor
neuron group takes about 7% of its input that way, and the exceptions — the neck
— take between 20% and 60%. The brain mostly does not drive the body; it drives
the circuits that do.

Aura, over 213 effector cells on the proofread map, with the deep end taken as
the top quartile of trophic height: mean direct share **0.051**, median **0.0**,
and 14 effectors above 20%. The same shape, including the exceptions.

### Stereotypy and repeated circuits

The annelid larva's whole-body connectome reports a correlation of 0.91 between
its left and right synapse matrices, which is what lets the paper call the
wiring stereotyped. The same comparison between two commits of Aura, at the
level of type-to-type connectivity over 39,882 shared type pairs, gives
**0.988**. Her development reproduces a plan rather than improvising one.

13% of her types span three or more packages. The widest span 190, and reading
them is the caveat: they are generic roles — `__getattr__`, `__init__`, small
predicates — rather than repeated circuits. Her serial homology is dominated by
boilerplate, which is a finding about the code and not about the analysis.

## Do cells that do the same thing wire together?

MICrONS recorded 75,909 neurons in a mouse's visual cortex and reconstructed the
same tissue at synapse resolution, co-registered to 3.8 micrometres, and used the
combination to test a rule that had been assumed for decades: neurons tuned to
similar things preferentially connect. The rule held across layers and across
areas.

The test needs structure and function measured in the same animal. Aura has both
by construction.

| | mean correlation |
| --- | --- |
| connected pairs | **0.769** |
| degree-preserving rewiring | 0.086 |

z = 227 over eight rewirings, on 3,320 connected pairs among 2,386 cells that
fired often enough to correlate.

The null matters twice over. A degree-preserving rewiring keeps every cell as
busy and as connected as it was, so the effect is not about which cells are
loud. And each cell is centred *within each workload* before anything is
correlated, so two cells that are simply busy while the same thing is being done
contribute nothing. Removing the workload moved the result from 0.773 to 0.769.

This reconciles with the forecasting result rather than contradicting it. Wiring
and activity are strongly coupled. Predicting a cell's next value from its
neighbours' last few is still nearly impossible at this frame rate, because
forecasting is a harder question than correlation and a 914 ms frame has already
averaged the propagation away.

## Three things a nervous system cannot do to itself

**Delays solved rather than grown.** A brain makes convergent signals arrive
together by adjusting myelination, which is slow and local and cannot see the
constraint it is solving. The same problem is a linear least-squares system.
Across 11,603 convergence cells, arrival jitter falls from 0.806 to 0.142 — an
82.4% reduction — for 0.064 hops of added delay per edge. Random holds of the
same average size score 0.823, worse than doing nothing.

**A tangle that can be asked what it does.** H01 found axon whorls: axons
wrapped into knots, function unknown, and unknowable, because you cannot untie
one in a person. Aura's whorls are the strongly connected components of her call
graph — the settings control plane at 16 cells, the service container at 8 with
1,224 external inputs, `mind_tick`'s restart loop at 7 — and each one can be
untied in a copy and handed to the lesion measurement.

**An edit that can be taken back.** Development prunes and that is final. A
rewiring here carries its own inverse.

## The forecasting benchmark

ZAPBench asks whether anyone can predict the next thirty seconds of a whole
brain from the last few, and reports that models mixing information across cells
barely beat models that treat each cell alone. It cannot tell whether that is a
failure of the models or an absence of the relationships, because the fish it
recorded is not a fish anyone has a connectome for.

Here the cell that fires is the cell in the graph, so the question separates.
Four arms, all fitted by ridge regression in closed form so there is no seed and
no learning rate to tune a result into existence:

- `blind` — its own past only.
- `global` — its own past plus the mean over every cell.
- `connectome` — its own past plus the mean over the cells wired into it and the
  mean over the cells it drives.
- `rewired` — the same model with the same number of neighbours per cell, drawn
  from a degree-preserving rewiring.

The last one is the null the third has to beat. Every comparison is a paired
bootstrap over cells, because two arms differing in the fourth decimal of an
average over thousands of cells is not a difference until the interval says so.

The instrument is validated on data where the answer is known: on a system where
activity genuinely flows through the wiring it finds the effect and calls it
significant, and on a system of independent cells it finds nothing.

### What it found

Recording: 12,724 cells, 2,107 frames at 914 ms, nine workloads, 150 million
events, 8,594 caller-to-callee pairs observed firing.

The diagnostic runs before the leaderboard, because a benchmark that cannot
separate two models reports a tie and a tie read as a finding is worse than no
finding. Lag-one autocorrelation on this recording is **0.123**. The recent past
says almost nothing about the next frame.

Against that, at a context of four frames and a horizon of 32, restricted to the
5,265 cells the connectome has anything to say about:

**The connectome beats its own degree-preserving rewiring.** Median per-cell MAE
difference −2.5×10⁻⁴, 95% interval [−3.0×10⁻⁴, −2.0×10⁻⁴], better on 57.8% of
those cells. At a context of 256 frames the effect does not survive.

The effect is small, and it is small for a reason the diagnostic names: there is
very little predictable signal at this frame rate. A frame holds seventy thousand
events, so propagation finishes inside it. That is a measurement about the
instrument, not about the wiring, and the remedy is a faster frame.

**The remedy was tried and it worked.** `docs/CONNECTOME_PREDICTION.md` was
written before the run and names three consequences a 20 ms recording should
have. All three held: lag-one autocorrelation rose from 0.174 to 0.385, the
connectome's advantage over its rewiring went from better on 57.8% of cells to
better on 86.4%, and the prefetch rule overtook persistence instead of merely
closing on it. The last of those changed what the system does, because warming
by the connectome had been the worse rule only at the coarse frame rate.

Four defects in this analysis were found by running it and are worth naming,
because each one had produced a confident wrong answer first. Arms fitted with
one weight matrix over cells whose activity differs by orders of magnitude were
competing to fit the loudest cells, and every learned arm lost to holding the
last frame; traces are standardised per cell now. A single time cut put whole
workloads on one side, making the stimulus-conditioned mean identical to the
plain mean to six decimal places. The paired comparison reported a significant
median with a confidence interval whose ends were the same number, because most
cells have no neighbour and the two arms differ for them only through a shared
weight. And the held-out condition reported an MAE of two million, carried by
192 cells the training split never saw fire.

## Running it

```bash
python tools/record_connectome_activity.py --budget 240
```

Nine workloads stand in for ZAPBench's nine stimuli, each with a wall-clock
budget. Writes `activity.npz`, `activity_manifest.json` and `observed_edges.json`.

```bash
python tools/connectome_report.py --sections all --observed artifacts/connectome/observed_edges.json
```

Reconstructs, runs every analysis, scores against the recording, and writes one
JSON artifact.

```bash
python tools/run_zapbench.py --data artifacts/connectome
```

Runs the forecasting benchmark and writes the leaderboard and the structure
test.

`make connectome`, `make connectome-pathology`, `make connectome-record` and
`make connectome-zapbench` are the same four steps.

## What is wired

This package finds channels with a writer and no reader. Applying that to
itself, three of its own modules had none. They have one now.

**`criticality`** drives the branching estimate in
`core/consciousness/criticality_regulator.py`, which sets gain, noise and the
E/I target through a PID.

**`laminar`** decides whether a draft competition in
`core/consciousness/multiple_drafts.py` produced a decision or a coin landing.

**`neuromodulation`** sets the bound that decision is taken at.
`live_levels()` reads the four Doya modulators off the running neurochemical
system; `config_for` turns the noradrenaline level into a decision bound. The
direction is published — more noradrenaline, less evidence needed, faster and
less accurate, which is Aston-Jones and Cohen's adaptive gain — and the
magnitude is measured: how far a region's bound moves is how much that region's
activity was measured to move with noradrenaline. A region with nothing fitted
does not move, and with no system running the bound is the default, so wiring it
changed nothing until something is measured. Letting the fitted slope set the
direction would mean a region whose correlation came out negative responded
backwards to the same chemical, which is a sign error rather than a finding, and
a test pins it.

**`gating`** closes routes for the warm-up and reports through the health
surface. `integration.register_gates` takes the gate set, `gate_status` says
which routes the current state has closed and whether closing them rerouted
anything, and `prefetch.warm` drops any prediction the state closed the route
to — warming a cell the system decided not to reach is paying for the work the
gate exists to prevent. `laminar.settle` takes a gate too: a gated candidate is
never sampled, which is the difference between a state that biases a decision
and one that changes which decision is being taken. A gate set that closes
everything is treated as a bug and the race runs anyway.

**`prefetch`** warms through `integration.warm_upcoming`, and which rule it
warms by comes from a measurement rather than a preference. `record_prefetch_rule`
stores whichever won on this system's own recording and the warm-up uses it.

On this system persistence wins on F1 at both frame rates — 0.7129 against the
connectome rule's 0.6149 at 914 ms, 0.7056 against 0.6675 at 20 ms — so the
warm-up uses persistence. A 20 ms recording of 74 cells once put the connectome
rule ahead at 0.7934, and a broader one at the same rate over 8,171 cells
reversed it; [CONNECTOME_PREDICTION.md](CONNECTOME_PREDICTION.md) carries both
and what the difference was. What survives is narrower: the connectome rule has
the highest recall of the four, 0.7450 against 0.7000, so it finds more of what
is about to run and pays for it in precision. Nothing about the wiring is
decided here — the mechanism reads the recording rather than either number.

**`integration`** publishes eight telemetry channels at 0x1901, a health
fragment under `runtime_health_report()["connectome"]` that never builds a
reconstruction to answer a poll, the gate state and the chosen rule.
**`invariants`** runs in the structural verifier. Four mappings are declared in
`core/science/neuro_reference.py` at CONNECTIVITY_MATCHED, each with its source,
its falsifier and a competing hypothesis. Everything else is reached by
`tools/connectome_report.py` and the make targets.

The policy stays with the caller. Which state closes which route is a decision
about this system, and a default invented here would be a claim about Aura
dressed as one.

## Sources

- Shapson-Coe et al., *A petavoxel fragment of human cerebral cortex
  reconstructed at nanoscale resolution*, Science 384:adk4858 (2024).
- Janelia FlyEM male CNS connectome v1.0 (2026) and its companion paper on
  sexual dimorphism, Cell S0092-8674(26)00942-6.
- Immer et al., *ZAPBench: A Benchmark for Whole-Brain Activity Prediction in
  Zebrafish*, arXiv:2503.02618.
- Potjans & Diesmann, *The cell-type specific cortical microcircuit*, Cerebral
  Cortex 24:785 (2014).
- Wilting & Priesemann, *Inferring collective dynamical states from widely
  unobserved systems*, Nat Commun 9:2325 (2018).
- MacKay, Johnson & Sanhedrai, *How directed is a directed network?*, Proc R Soc
  A 476 (2020).
- Huttenlocher & Dabholkar, *Regional differences in synaptogenesis in human
  cerebral cortex*, J Comp Neurol 387:167 (1997).
- Doya, *Metalearning and neuromodulation*, Neural Networks 15:495 (2002).
- Hansen et al., *Mapping neurotransmitter systems to the structural and
  functional organization of the human neocortex*, Nat Neurosci 25:1569 (2022).
- Bentley et al., *The multilayer connectome of Caenorhabditis elegans*, PLoS
  Comput Biol 12:e1005283 (2016).
- MICrONS Consortium, *Functional connectomics spanning multiple areas of mouse
  visual cortex*, Nature (2025).
- Helmstaedter et al., *Connectomic reconstruction of the inner plexiform layer
  in the mouse retina*, Nature 500:168 (2013).
- Marin, Cheong et al., *Transforming descending input into motor output: the
  Drosophila male adult nerve cord connectome*, eLife reviewed preprint 96084.
- Verasztó et al., *Whole-body connectome of a segmented annelid larva*, eLife
  reviewed preprint 97964.
- Carandini & Heeger, *Normalization as a canonical neural computation*, Nat Rev
  Neurosci 13:51 (2012).
- Gold & Shadlen, *The neural basis of decision making*, Annu Rev Neurosci
  30:535 (2007).
- Google Neuroglancer, github.com/google/neuroglancer.

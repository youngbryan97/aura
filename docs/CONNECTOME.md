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

## The three layers

The wiring diagram is one layer. *C. elegans* has had its synaptic map since
1986, and when the monoamine and neuropeptide layers were added on top of it,
96% of the monoamine connections turned out to exist in no other layer and 82%
of the neurons carrying a dopamine receptor receive no synapse from any neuron
that releases dopamine.

Aura has the same shape. Three layers are extracted:

| layer | what joins two cells | directed |
| --- | --- | --- |
| wired | one calls the other | yes |
| volume | one publishes a topic the other subscribes to | yes |
| gap | both touch the same container key or module global | no |

98.6% of the shared-state pairs and 100% of the topic pairs exist in no other
layer. 80,923 pairs are joined only by calls, 7,545 only by state, 11 only by a
topic. Seven cells are central in the shared-state layer and peripheral in the
call graph — the worm's finding that its monoamine rich club is different cells
from its wired one.

The layer also counts what has one end. Of 120 event topics, 87 are published
with no subscriber the scan can find and 26 subscribed with no publisher; seven
have both. Of 923 shared-state keys, 229 are written and never read and 416 read
and never written by any access a static scan can follow. Those are candidates,
not defects: a subscriber can be registered from a table nothing static can
follow, and the finding says so and names the step that would settle it.

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

47,862 cells across 3,522 modules, 80,976 drive pairs, 192,171 contacts, built
in fifteen seconds by walking 3,526 files with `ast`. Half the in-volume call
sites resolve; the rest are recorded as ambiguous with their candidate lists, or
as leaving the volume, and the coverage figure says which.

A recording is the other half. `sys.monitoring` watches cells fire at ZAPBench's
914 ms volume rate and emits both the call counts and the calcium trace a light
sheet would have seen. It can also capture caller-to-callee pairs, and those
pairs are ground truth — an edge that fired happened. No connectomics project
has ever had that, and it is what lets the agglomeration threshold be chosen
from a measured curve instead of from taste.

## What it found

### Her connections are far heavier than cortex's

81.8% of Aura's connected pairs touch once and 4.2% touch four or more times.
Human cortex runs 96.5% and 0.092%. Her heavy pairs are forty-five times more
common than cortex's and her heaviest carries 113 call sites where H01's
heaviest carried about fifty.

H01 reads a four-or-more-contact pair as a powerful connection, rare enough to
be special. At 4.2% of everything, hers cannot be.

### Local recurrence is missing

Cortex's within-layer connection density is 5.95 times its between-layer
density. Aura's is 0.603 — she connects across levels more often than within
them, where cortex does the reverse by six to one. The shortfall is a factor of
9.9, and it survives any relabelling of the layers, which matters because the
orientation of her hierarchy is undetermined: the anchor holds by 0.105 of
trophic height against a spread of 9.63.

The specific pathways she lacks are the local ones. L5I to L5E is cortex's
densest connection at p=0.3726 — inhibitory control of the output layer — and
she has 0.17 of her own mean there.

### The excitation to inhibition ratio is cortical overall and local where it is not

Whole system: 3.96 excitatory cells per inhibitory one, against cortex's 4.035.

`reality_reach`, the package that acts on the world, runs 1.67 across 989 cells.
`auth` runs 1.11 and `social_media` 1.44. Those packages are two to four times
more inhibited than the rest of her.

### The feed-forward loop is over-represented and there is no rich club

Against a degree-preserving rewiring: reciprocity z=+233, small-world sigma
9.48, modularity 0.771 over 887 communities, and the feed-forward loop at
z=+71 — the same motif that is over-represented in *C. elegans* neurons and in
*E. coli* transcription.

The rich club runs below its null at every degree cut, 0.43 of chance at k=128.
Cortex's hubs preferentially wire to each other. Hers avoid each other.

### There is a neck, and the evidence for it is thin

124 afferent cells and 277 efferent ones. Sixteen cells carry half the
sense-to-action flow and those cells converge 6.3 times harder on their inputs
than an average cell, which is the integrator signature the fly's ascending and
descending neurons show.

The same run reports that only 65 of 34,348 sense-to-action pairs have a
statically visible path. A static reconstruction cannot see a call made through
a service lookup or an event bus, so that number is a floor and the neck verdict
is marked thin until the snapshot has been proofread against a recording.

Proofreading it settles the question. 8,594 pairs were seen firing; the static
reconstruction contained 6,495 of them, recall 0.756. Writing a join for each of
the 2,085 it lacked takes recall to **0.998**, expected run length from 2.50 to
**3.10**, and sense-to-action reachable pairs from 65 to **203**, which is
enough for the verdict to stand on its own. On the proofread map the cells
carrying that flow converge **99 times** harder on their inputs than an average
cell, against 6.5 before. The fly's neck is made of integrators, and so is
hers.

### Two individuals differ three times more than two flies

Reconstructing at HEAD and at 400 commits earlier gives two individuals of the
same system: 43,515 shared cells, 40 lost, 4,229 gained, 7,835 rewired pairs.
15.3% of cells changed, against the 4.8% of the fly central brain that is
sex-specific or dimorphic.

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

## Where it is wired

- `runtime_health_report()["connectome"]` carries the fragment. It never builds
  a reconstruction to answer a poll; a process that has not made one says so.
- Six telemetry channels at 0x1801, each with limits set against the published
  value. `connectome.within_layer_ratio` reads yellow on its own.
- Four mappings in `core/science/neuro_reference.py` at CONNECTIVITY_MATCHED,
  each with its source, its falsifier and a competing hypothesis.
- `core/consciousness/criticality_regulator.py` steers on the multistep
  regression branching ratio rather than the per-tick mean, and publishes the
  difference between them as `subsampling_bias`.

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

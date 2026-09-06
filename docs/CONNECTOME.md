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
- Google Neuroglancer, github.com/google/neuroglancer.

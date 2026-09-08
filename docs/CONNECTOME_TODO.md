# Connectome: what is left

Working list. Deleted when every line is done and green.

## Carried over

- [x] fast2 analysis: the three predictions at scale — 16,381 frames, 8,171
      cells, all nine workloads. Wiring predicts activity on 88.7% of the 1,490
      cells with connectome neighbours; the connectome does not beat persistence
      at prefetch; connected cells move together at z = 97.9
- [x] `make connectome` end to end with `--observed` and `--against` — 671s,
      twenty sections, including the longitudinal pass against another worktree
- [x] regenerate every published number on the corrected graph — the merge-error
      fix moved coverage from 0.502 to 0.736 and removed ~8,000 false edges
- [ ] full offline suite green (`tools/run_test_chunks.py`)

## The whole graph

A call graph catches direct invocation. Software also couples through shared
state, queues, events, database reads and writes, files and cross-process IPC,
and two components can be causally coupled without ever calling each other. The
converse also holds: one function can call another ten thousand times and use
nothing consequential from it.

    G_whole = G_call ∪ G_state ∪ G_event ∪ G_IPC ∪ G_data ∪ G_neural

- [x] `G_call` — direct invocation (`volume.py`)
- [x] `G_state` — shared container keys and module globals (`layers.py`, gap layer)
- [x] `G_event` — publish and subscribe on a topic (`layers.py`, volume layer)
- [x] `G_data` — data dependence, not call count. Six consequences per call site;
      24.5% of drive edges carry no information at all
- [x] `G_io` — files, databases and other durable stores. 94 stores, 108 pairs,
      97% in no other layer, 84 written with no reader the scan can follow
- [x] `G_ipc` — cross-process edges. 259 pairs, 99.6% unique
- [x] `G_neural` — the mesh as a layer (`neural.py`). 64 columns, both weight
      matrices, and the seam: two cells drive it, five read it. Asking whether
      an injection could reach a read found that none could, on every seed
- [x] a combined view with per-layer weights and a single reachability query

## The effective connectome

The structural graph says what can happen. The interesting object is what does:

    w_ij^eff(c) = Effect[do(i), j | c]

- [x] per-condition effective connectomes, built from recorded activity with a
      rotation null per edge
- [x] nine conditions measured; related workloads correlate 0.92 to 0.96 and
      unrelated ones at ~0.00
- [x] a divergence measure between two conditions' effective graphs
- [x] cross-station influence for pairs with no call edge between them
      (`cross_influence`), scored against each pair's own rotations after a
      parametric test and a deconfounded one both turned out to fire on a
      quarter of rotated pairs
- [x] intervention rather than correlation: `do(i)` as a real perturbation with
      a matched control (`intervene.py`, `tools/run_lesions.py`)

## The coalition test

- [x] reconstruct, from recordings alone, whether the ring closes, and how
      reproducibly. It took three recordings: the first two drove station
      readers, which cannot hand anything to the next station. The third drives
      the kernel's 29-phase pipeline over one shared state, 240 turns, 31,042
      frames at 2 ms
- [x] six of seven links carry against their own rotations in the `task`
      condition; `action -> interoception` does not. The ring's mean gain is
      stronger than 36% of the 720 directed cycles through the same stations,
      z = −0.64: every station influences every other and the architecture's
      order is not privileged
- [x] selective lesions with predictions stated first, each scored against a
      degree-matched control lesion. Of the three registered from the theory,
      one partly confirmed and two not attributable — their readouts were moved
      further by the control. Three registered from what that run found were
      confirmed on an independent seed
- [x] `do(HOT feedback = 0)` — competence intact, the selfhood reading gone
      (5.0 to 0.0) and the spread of pending intents flattened (0.373 to 0.0)
- [x] `do(workspace broadcast = 0)` — competence intact, the fragmentation score
      gone (0.1407 to 0.0) with its spread and its affective gap
- [x] `do(affective loop = 0)` — competence intact, and regulation gone: an
      injected valence difference of 1.2 is normally pulled back to 0.0006 by
      the end of a turn, and without the two affect phases the whole 1.2 stands

## She can run this on herself

- [x] a query surface her self-model can reach — `core/connectome/introspect.py`
- [x] every answer carries its evidence grade, and a test pins that a predictive
      weight is never worded as a cause

## Architecture that develops under anatomical feedback

Before a self-modification is promoted, reconstruct `G_t → G_{t+1}` and ask
`ΔQ = Q(G_{t+1}) − Q(G_t)` on measurements rather than on resemblance:

- [x] did useful local recurrence increase
- [x] did a brittle single point of failure disappear
- [x] did unnecessary cross-region coupling decline
- [x] did task-specific effective circuits get cleaner
- [x] did information reach the consumers it was supposed to
- [x] did dormant developmental machinery become active
- [x] did a behavioural lesion confirm the intended new causal route
- [x] wire the verdict into the promotion gate

## What the instrument changed

Measuring is not the point of it. What the measurements caused:

- [x] the criticality regulator steers on the subsampling-unbiased branching
      ratio rather than the per-tick mean
- [x] draft competition reports whether its winner means anything
- [x] accumulate-to-threshold: the same accuracy for 31 to 62% fewer evidence calls
- [x] the promotion receipt carries what a change did to the shape of the system
- [x] the neural mesh's missing feedforward pathway: sensory-to-executive
      reachability from 0 of 16 to 12-16 of 16, isolated columns from 7-16 to
      0-5, components from 13-25 to 1-6

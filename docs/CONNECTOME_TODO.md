# Connectome: what is left

Working list. Deleted when every line is done and green.

## Carried over

- [ ] fast2 analysis: the three predictions at scale (27k frames, all nine workloads)
- [ ] full offline suite green (`tools/run_test_chunks.py --chunks 10`)
- [ ] `make connectome` end to end with `--observed` and `--against`

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
- [ ] `G_data` — data dependence, not call count. An edge weighted by whether the
      caller uses what the callee returns, and how far that value travels.
- [ ] `G_io` — files, databases and other durable stores, as a layer with its own
      readers and writers
- [ ] `G_ipc` — cross-process edges: subprocess, socket, shared memory, the MLX
      worker boundary
- [ ] `G_neural` — the mesh, as a layer, so the substrate and the code are one graph
- [ ] a combined view with per-layer weights and a single reachability query

## The effective connectome

The structural graph says what can happen. The interesting object is what does:

    w_ij^eff(c) = Effect[do(i), j | c]

- [ ] per-condition effective connectomes, built from recorded activity plus
      intervention rather than from co-occurrence
- [ ] `G_conversation`, `G_planning`, `G_coding`, `G_threat`, `G_dreaming`,
      `G_social`, `G_autonomous_development` — same anatomy, different active
      circuits, each with the condition it was recorded under
- [ ] a divergence measure between two conditions' effective graphs, so "which
      circuits does this state recruit" has an answer
- [ ] intervention rather than correlation: `do(i)` implemented as a real
      perturbation with a matched control

## The coalition test

- [ ] reconstruct, from recordings alone, whether interoception → affect →
      workspace → HOT → self-model → planning → action → interoception closes as
      a recurrent coalition, and how reproducibly across conditions
- [ ] selective lesions with predictions stated first:
      - `do(HOT feedback = 0)` should leave competence and destroy calibrated
        self-report
      - `do(workspace broadcast = 0)` should leave local processors and destroy
        cross-domain availability
      - `do(affective loop = 0)` should leave reasoning and destroy
        state-dependent prioritisation
- [ ] each lesion scored against a degree-matched control lesion, and the
      prediction registered before the run

## She can run this on herself

- [ ] a query surface her self-model can reach: which circuit dominates in the
      state she is in, which pathway her failures correlate with, and whether
      the mechanism she believes produced a preference has any causal influence
- [ ] the answer carries its evidence grade, so an observational correlation
      cannot be reported as a cause

## Architecture that develops under anatomical feedback

Before a self-modification is promoted, reconstruct `G_t → G_{t+1}` and ask
`ΔQ = Q(G_{t+1}) − Q(G_t)` on measurements rather than on resemblance:

- [ ] did useful local recurrence increase
- [ ] did a brittle single point of failure disappear
- [ ] did unnecessary cross-region coupling decline
- [ ] did task-specific effective circuits get cleaner
- [ ] did information reach the consumers it was supposed to
- [ ] did dormant developmental machinery become active
- [ ] did a behavioural lesion confirm the intended new causal route
- [ ] wire the verdict into the promotion gate

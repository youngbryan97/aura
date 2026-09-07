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

| | domain | read from |
|---|---|---|
| P | perception | percepts, their recency and spread, the objective as it arrives |
| I | interoception | host load, thermals, thought latency, vitality |
| A | affect | valence, arousal, curiosity, engagement, the Plutchik primaries |
| G | workspace | attention focus, coherence, fragmentation, discourse energy |
| C | recurrent cognition | mode, phi estimate, the phenomenal field and its latent |
| S | self-state | stability, bonding, narrative version, personality growth |
| M | active memory | working set, retrieved set, ledger, rolling summary |
| W | world model | entities, relationships, verified facts, preferences |
| D | deliberation | goals, initiatives, origin, motivational budgets |
| N | ontogeny | the lifetime reservoir's hidden state, novelty, displacement |

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
constant is what makes two arms comparable, and it means no edge measured here
runs *through* language. Edges that need the model to read one state and write
another read as absent.

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

## Two findings from building the instrument

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
first partition search reported a confidently negative score. The intact model
had a hundred and eleven inputs against a few thousand rows and overfitted;
each cut model had a fraction of them and generalised better. Both sides are
now reduced to four principal components per domain, fitted on training rows
only, so the comparison is about which domains a model may look at rather than
how many numbers it was handed.

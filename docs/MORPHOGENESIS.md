# The morphogenetic runtime

A population of governed computational cells whose **runtime topology is
state**. Cells bind, unbind, spawn, retire, specialize and route; the shape
they take changes what the system can compute; and every change goes through
one governor that can refuse it.

Read [CONTRIBUTING.md](../CONTRIBUTING.md) for the architecture rules this
layer obeys. This document is what the layer is, what it has been measured to
do, and what it has not.

## Why the shape has to be state

The layer existed before this and had a population, a diffusive field, and no
bindings between cells. Every cell reached every other cell through one global
signal queue, so two different arrangements of the system computed exactly the
same thing. A shape that cannot change an outcome is decoration, however
carefully it is maintained.

`MorphGraph` makes a binding a typed directed edge with a port contract, a
weight, a latency and a capacity. In the sandbox workload, work moves along
declared edges and nowhere else:

| topology | completion | hops |
| --- | --- | --- |
| connected | 1.00 | 3 |
| one binding cut | 0.00 | — |
| forced the long way | 1.00 | 4 |

## The parts

| Module | What it holds |
| --- | --- |
| `graph.py` | the authoritative topology; transactional, monotonic version, deterministic serialization |
| `proposal.py` | BIND UNBIND SPAWN RETIRE MIGRATE SPECIALIZE DESPECIALIZE ROUTE MERGE, each with its inverse |
| `governor.py` | the only thing that may change the shape |
| `substrate.py` | where a cell physically is, and what a transition costs |
| `lineage.py` | who descends from whom; acyclic, depth-bounded |
| `motifs.py` | developmental priors, credited only by measurement |
| `policy.py` | local rules, plus the baselines that judge them |
| `live_policy.py` | what the running instance may develop, and how it is scored |
| `workload.py` | a computation that can tell what shape it is |
| `sandbox.py`, `scenarios.py` | the experiments |
| `bridge.py` | the seam to the rest of Aura |
| `telemetry.py`, `invariants.py` | what it reports and what it must never violate |

## The ladder a change climbs

A cell proposes. The governor decides. Each rung can only reject.

1. **Well-formedness** — a transition that cannot describe its own inverse
   never reaches a budget.
2. **Bounds** — population, replicas per capability, spawn depth, transitions
   per window, per-cell cooldown, reversal window, fragmentation.
3. **Budget** — the proposer pays. Replication that is free is replication
   without a bound.
4. **Shadow evaluation** — apply to a copy, measure, compare. A governor with
   no evaluator refuses every non-routine change: "nothing measured it" and
   "the measurement came back empty" are the same fact.
5. **Governance** — anything CRITICAL goes through Aura's governed scope.
6. **Commit** — substrate first, then graph. A partial substrate failure
   unwinds both, because a dock that latched and then failed its handshake has
   already changed the world.

## Running the experiments

```bash
AURA_LOG_DIR=/tmp/aura-morph .venv/bin/python tools/run_morphogenesis_sandbox.py --scenario all --ablations
```

Eight scenarios, deterministic under their seed, no model and no socket. Each
states the rule it is judged by before it runs.

`--audit` prints every seam to the rest of Aura, including what is
deliberately not connected and why. `--list` names the scenarios.

## What has been measured

Ablation matrix, `reason_heavy` at 5 arrivals a round:

| arm | score | vs fixed |
| --- | --- | --- |
| adaptive (local rules) | +0.0515 | +0.0637 |
| fixed topology | −0.0123 | — |
| local signals removed | −0.0123 | +0.0000 |
| central scheduler | −0.0123 | +0.0000 |
| random mutation | −0.0375 | −0.0252 |

Random topology mutation scores **below doing nothing**, which is what makes
the adaptive number mean something. The local-signals-off arm still proposes
and still lands on the fixed score, which is what says the signals rather than
the acting carry the value.

Other results: after a third of the population is deleted without warning the
adaptive arm restores more of its pre-lesion throughput than a fixed one,
which restores none. A policy claiming maximum benefit for every proposal
cannot grow the population past its cap. On a demand no policy names, over a
substrate with physical-like costs, the fixed shape completes 0.000 and the
adaptive one 0.450.

All of it is `MEASURED_SYNTHETIC` and registered as such in
`core/organism/model_validation.py`. The scenarios run offline against a
constructed workload. Nothing here is measured on live traffic.

## Live

Started from `aura_main` after the ServiceContainer is populated. Boot seeds
tissue adjacency as OBSERVE edges — the honest type, because these cells watch
services and raise repair signals rather than handing work to each other.

`LiveObserverPolicy` proposes; `LiveCoverageEvaluator` measures observability
of the current need field rather than throughput, because live cells carry no
work and scoring them on it would be a number nothing produces. It grows an
observer where a troubled subsystem has nothing that can act on it, binds a
repair path where something can act but cannot reach, and retires an observer
nothing has needed.

Cost, over a 400-tick soak with the immunity bridge on: mean 2.3ms, worst
35ms, none over 100ms. Boot is 684ms and all of it is off the event loop — the
lazy imports and the first resource sample are warmed there deliberately,
because the tick that takes them first otherwise wears the whole cost.

## Bounds

Population, replicas per capability, spawn depth, transitions per window,
per-cell cooldown with churn backoff, a reversal window, an energy ceiling per
cell and per run, and a fragmentation ceiling. Every one is a refusal the
governor can make without consulting anything else.

Two are worth naming. The **reversal window** is what stops thrash: thrash is
not changing often, it is changing back, and a rule counting transitions
cannot tell an alternating demand's legitimate extra structure from chasing.
Damage voids that bet — undoing a specialization whose cover just died is
repair. The **exploration floor** exists because a band demanding immediate
improvement cannot cross a valley, and recovery from damage is always a
valley: with two capabilities missing, restoring either alone completes
nothing.

## Telemetry and invariants

Channels `0x0801`–`0x080C`, events `0x1601`–`0x1605`. The limits encode the
failures this layer is exposed to: cells red-high because bounded replication
is the difference between self-organisation and a cancer; components red-high
because a partitioned population still serving from one half while reporting
itself whole is the failure the partition scenario exists to catch.

Nine invariants under scope `morphogenesis`. Two catch what no unit test
would: the graph holding a binding the substrate does not is the signature of
a partial failure nobody cleaned up, and the substrate holding one the graph
does not is a latch nobody owns.

```python
from core.verify.invariants import verify
verify("morphogenesis")
```

## What is not connected, and why

`core/language` is the semantics of language. It is a source of demand, which
is why `bridge.demand_from_message` goes through the capability router. It is
not a substrate a population could run on.

Program DNA reconstructs source from evidence — what code would produce this
behaviour. A motif is a prior over arrangement — which capabilities to have,
how many, how to bind them. Both get called a genotype and they sit at
different levels. `bridge.describe_genotype_relationship` states it rather
than forcing a link that would leave both meaning less than they do apart.

`core/swarm` is the real Phase 2 substrate, blocked on a shadow evaluator for
the live system: without one every non-routine change is refused, which is
correct and also means nothing would move.

## Substrate contract

`SubstrateAdapter` is what a future carrier plugs into. What separates a
process, an FPGA region and a docking module from an in-process graph edit is
that their transitions take time, fail halfway, cost energy, and leave a cell
unreachable meanwhile. `SimulationSubstrate` models all four on a seeded
virtual clock, so the governor is written against the awkward version.

`LocalRuntimeSubstrate` refuses `migrate` outright. A cell cannot leave this
process, and reporting a move that did not happen would put the graph and the
world out of agreement.

Nothing in this layer moves any matter, and a simulated migration is described
as a simulated migration everywhere it appears.

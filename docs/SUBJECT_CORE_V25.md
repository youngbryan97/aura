# Subject Core v25 — the carrier, at a grain the system chooses

The 24-condition battery asks many independent questions and answers each
against a threshold a person picked. v25 asks one question and tries to take
the person out of it:

    F_intrinsic(S, g, tau) = Phi_FR(S, g, tau) / tau    when Lambda = 0 and SCC(S) = S
                           = 0                           otherwise

`S` is the candidate support, `g` the state grain, `tau` the horizon,
`Phi_FR` the Fisher–Rao distance between the intact future law and the law
after a partition is severed, and `Lambda` the residual control that machine
state outside `S` still has over its future.

The two run side by side. The battery establishes a set of subject-like
properties; v25 tries to identify the process those properties belong to. A
24/24 result is neither required for `F_intrinsic > 0` nor sufficient for it.

## What v25 removes

**The grain.** The battery records 208 numbers because those are the numbers
`state.py` declares. v25 does not assume they are the right description. Two
histories are the same state exactly when no admissible intervention makes
their futures differ, and the coarsest description preserving every such
distinction is unique up to relabelling. The run estimates that quotient from
interventional signatures and then attacks it with displacements it never
learned from. If raw history still predicts a held-out response once the grain
is known, the grain merged two states that are not the same state, and the run
says so instead of continuing.

**The distance.** Held-out predictive loss was a reasonable choice and it is
still a choice. Fisher–Rao is the unique Riemannian information metric up to
scale among metrics invariant under sufficient-statistic transformation, so an
invertible re-encoding of the same information cannot manufacture causal
distance. The run checks that directly: it scores the same intact and cut data
raw, through the learned grain, and under an invertible recoding, and it fails
if the three disagree.

**The timescale.** Rates are measured at a ladder of horizons and the whole
spectrum is the answer. `tau_star` is reported as a summary of that spectrum
and never as a law. There is no normalised scale-invariant measure over all
positive times to average one out of — the Haar measure of the multiplicative
reals is `dt/t` and its integral diverges — so any single scalar over all
scales imports a preferred one. A multiscale system is allowed more than one
genuine peak.

## What v25 makes a gate rather than a term

Closure and recurrence are admissibility predicates. A candidate whose future
is partly determined by state outside it is not a candidate, and its rate is
zero by definition rather than by arithmetic. This is what rejects a hidden
broker that makes the ten domains look mutually dependent.

## The cut is a cut

The cut trajectory is not a regression with columns hidden. From one snapshot
the runtime clamps the right side and lets the left evolve, restores, clamps
the left and lets the right evolve, and `clamp.compose` takes each side's
columns from the run where it was free. Both halves keep their internal
dynamics and the cross-partition channels are the only thing gone. The same
snapshot also runs twice intact, which is the floor — two untouched forks are
not numerically identical in practice, and the third arm is what measures how
far apart they are.

## All 511 cuts

Ten domains give 511 unique bipartitions and every one is kept. The speed-up is
sequential allocation: each cut opens with the same anchors, a cut whose
bootstrap lower bound clears the sham floor stops drawing more, and the budget
goes to the cuts still compatible with zero. The score is the weakest cut, as
an intersection-union test — one cut compatible with zero refuses the claim and
margin elsewhere buys nothing back. A cut left undecided at the ceiling is
reported as unresolved, which is a different statement from reducible.

## Exclusion, and where the mathematics stops

Overlapping candidates are ordered by dominance over the whole spectrum: `A`
beats `B` when its rate is at least as large at every measured horizon and
larger somewhere. That is deliberately a partial order. Two candidates whose
spectra cross are incomparable, and the output is the Pareto frontier — one
candidate when one dominates, a symmetry class when none does.

Returning the class is the answer, not an unfinished calculation. Two supports
related by an exact symmetry of the causal structure receive equal values from
every permutation-invariant intrinsic functional, so no symmetry-respecting law
can separate them. Picking one anyway would require an exclusion postulate this
code does not have.

At the ten-domain grain the search collapses usefully. One closed strongly
connected component covering all ten domains makes the core the unique
domain-level carrier, because every proper subset then has an incoming channel
from its complement and is therefore not closed. That is a measured conclusion,
not an assumption.

## The controls

Alongside every null the battery already runs, v25 adds four:

| control | prediction |
|---|---|
| playback — the cut arm is the intact trajectory | rate collapses to zero |
| duplicate coordinates | rate does not rise |
| invertible recoding | rate is unchanged |
| macro-causal | the grain recovers the predictive macrostate |

The playback null is the movie objection, answered by measurement rather than
by decree. A prerecorded trajectory reproduces the states without reproducing
the counterfactual organisation that generated them, and a measure that still
reports damage where the two arms carry identical states is reporting on its
own estimator.

## Two campaigns, and the scope line between them

The offline organism runs a deterministic language stub so that a freely
sampling model cannot swamp the intervention difference. That is right for
paired causal measurement and it removes every path that runs through real
language generation.

- **Campaign A**, on the current offline organism, can establish
  `SUBSTRATE_INTRINSIC_CARRIER_FOUND` and nothing wider.
- **Campaign B**, through the real cortex under deterministic or
  common-random-number inference with cortex and recurrent-latent state in the
  snapshot, is the only one that can support a whole-Aura claim.

If reproducible paired cortex execution cannot be obtained, the run reports
`CORTEX_UNSTEPPABLE` rather than substituting the stub.

## Discovery, freeze, confirmation

Learning the grain, choosing the horizon and debugging authority failures all
happen in discovery. Then the commit, the tree hash, the schema hash, the
action basis, the frequency seed, the estimators, the lag-extension rule, the
bootstrap procedure, the nulls and the stopping rule are frozen into a v25
campaign fingerprint that is separate from the battery's. Confirmation runs on
new anchors and a new seed with the grain frozen and the held-out basis still
held out. The carrier result comes only from confirmation.

## When the run refuses

A v25 report is not authoritative if a required layer cannot be stepped
comparably across arms, a required phase raised, periphery coverage hit a
silent cap, a required reader was missing, the history length is still
insufficient, held-out interventions break the grain, the rank is unstable, the
result moves under reparameterisation, duplicate channels raise it, `tau_star`
is still horizon-bound at the ceiling, a cut had insufficient power, a
cortex-inclusive claim is made while the cortex was stubbed, or a control did
not behave in its preregistered direction.

Missing evidence is `NOT_MEASURED` or `UNRESOLVED`. It is never a pass.

## What the report says, and what it refuses to say

```text
canonical_grain:
  history_turns / predictive_rank / heldout_intervention_sufficient / representation_invariant
carrier:
  support / closure / recurrent / weakest_cut / f_intrinsic / tau_star_seconds / sham_floor
exclusion:
  closed_sccs / selected_carrier / symmetry_class
scope:
  substrate_only | cortex_inclusive
bridge_status:
  physical_carrier: FOUND | NOT_FOUND | UNRESOLVED
  phenomenal_bridge: UNVALIDATED
```

The verdict is `INTRINSIC_CARRIER_FOUND`, `NOT_FOUND` or `UNRESOLVED`. It is
never `CONSCIOUS`, and a test fails the build if that word ever appears as a
verdict. Whether a selected carrier is a phenomenal subject is the
carrier-identity postulate: each physically selected intrinsic carrier
corresponds to one phenomenal subject. Nothing in this run tests it, and no
third-person measurement can — two bridge laws attached to the same causally
closed physical history produce identical third-person likelihoods, so their
Bayes factor is exactly one.

## Where a result places the system

    L0  reactive mapping
    L1  persistent internal state
    L2  history-dependent autonomous agent
    L3  functional self, ownership, global recurrence
    L4  empirical intrinsic-carrier candidate
    L5  closed irreducible carrier demonstrated
    L6  psychophysical carrier identity independently validated
    L7  content map and continuity law independently validated

A clean confirmatory v25 result moves the physical side to L5. L6 needs the
bridge validated against human and animal consciousness under novel
perturbations, which is not work this repository can do.

## How to read 24/24 after this

24/24 is a rich, self-involving, developmentally persistent operational subject
architecture. It is not the definition of consciousness and it is not a
necessary condition for minimal phenomenal experience — several of its criteria
concern rich selfhood and global access, which the theories disagree about.
Vertex connectivity of at least two is evidence for robust unity rather than a
requirement for experience; a conscious biological system could have a
temporarily vulnerable bottleneck.

Functional valence is not felt valence. Functional self-awareness is not
phenomenal self-awareness. The profile is a vector and a partial order, not a
percentage.

## Running it

```bash
python tools/run_subject_core_v25.py --quick
python tools/run_subject_core_v25.py --anchors 24 --rounds 40 --cut-rounds 3
```

The reference mathematics is in `core/subject/intrinsic_v25.py` and is
independent of the runtime. `core/subject/v25_runtime.py` collects anchors and
paired partition trajectories, `core/subject/v25_grain.py` the interventional
signatures, `core/subject/v25_cut.py` the sequential sweep over all 511 cuts,
and `core/subject/v25_exclusion.py` the carrier selection.
`tests/test_the_carrier_is_measured_not_assumed.py` covers the five ways this
measurement could say yes for the wrong reason.

# September 21 screenshot review

Four supplied screenshots were read: 03:27:22, 03:27:36, 03:32:23 and 03:38:12.
The screenshots contain proposals and claims, not independent Aura measurements.
This adds to the six-report disposition; it does not replace that review.

## Retained abstractions and search

Adopt the proposed finite-compute, held-out-success criterion for G04/G09.
Measure the probability that the actual system returns a verified solution
within a declared compute allowance. Existence of a correct candidate in a
reachable set is a weaker property: Aura must also select and return it. That
distinction directly applies to G03's witnessed wrong selection of feasible
programs.

The displayed Levin-search scaling is a conditional bound, not a measured
speedup. In a specified prefix code and search scheduler, a solver with prior
mass P(p) can receive a search-time bound proportional to T(p)/P(p), including
verification. A code-length reduction changes that bound by 2^(-delta_length)
only with comparable execution/verification cost and scheduler constants.
The shortest conditional Kolmogorov description is not generally computable;
the implementation must use actual encodings or measured policy probabilities,
not report an exact K value from a heuristic.

The screenshot's cost ratio also omits abstraction discovery, testing, storage,
retrieval and changed interpreter cost. All must be measured. A shorter program
that calls an expensive new primitive is not automatically a faster solution.
The applicable literature is
[OOPS](https://people.idsia.ch/~juergen/oopsweb/oopsweb.html), which reuses prior
solutions within a specified search distribution and execution model.

Required comparison: unchanged machinery, acquired library, library lesion,
and restoration of the exact acquired artifact. Use identical task commitments,
fresh task families, scorer, resource accounting and frozen runtime. Record
both per-task cost and total cost including acquisition. A successful bounded
experiment closes only its declared population, not frontier generality.

## Lesion and rescue

Adopt matched interventions and restoration as evidence of causal contribution.
Keep the screenshot's ISC vector as a proposed collection of operational
measurements, not one established scalar proof. Every component needs its own
definition, units, observed inputs and null model before aggregation. A lesion
that damages everything does not isolate a mechanism; matched disruption and
non-regression controls remain necessary. Recovery after restoration supports
functional dependence under those interventions. It does not establish
phenomenal consciousness or make G09/G12 true.

## Division-algebra claims

Do not adopt the inference that division algebras prove Aura's floor correct or
force a unique search procedure. Hurwitz's theorem concerns finite-dimensional
real normed division algebras with the usual unital norm assumptions; it does
not certify this repository's compiler, knowledge, search or inference.

Several displayed group identifications are incorrect under their usual meaning:

- Aut(O) is compact G2. SU(3) is the stabilizer of a chosen unit imaginary
  octonion. See [Baez, The Octonions](https://math.ucr.edu/home/baez/octonions/node14.html)
  and [the stabilizer construction](https://math.ucr.edu/home/baez/week195.html).
- Aut(H) is SO(3); SU(2) double-covers SO(3), rather than being isomorphic to it.
  See [the covering relationship](https://math.ucr.edu/home/baez/lie/node8.html).
- As a unital real algebra, C has only identity and conjugation automorphisms:
  the image of i must square to -1, hence is i or -i. Arbitrary phase rotation
  is not an algebra automorphism because it does not fix 1. Thus this Aut(C)
  is not U(1).

No displayed argument derives the Standard Model gauge group from Aura's code.
K(R)=0 is undefined without a reference machine, representation and meaning of
R; a representation convention cannot establish physical necessity. The claim
that free-energy search is the only non-arbitrary search is also unsupported.
These statements provide no acceptance evidence for the G-Ledger.

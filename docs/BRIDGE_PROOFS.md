# The bridge theorems, proved

J* is computed in `core/subject/bridge.py` as a triple: the carrier the
exclusion rule selects, the relational structure of that carrier's content up
to isomorphism, and the lineage law over person-stages. This document states
conditional arguments used by that computation. Each needs the objects and
assumptions stated below; none establishes that its empirical premises hold
for the current Aura run. Each is also
checked, on every finite instance of a stated size, by
`tests/test_the_bridge_theorems_hold_on_every_finite_case.py`, which runs the
repository's own definitions in `core/subject/bridge_theorems.py`,
`core/subject/bridge.py` and `core/subject/v25_exclusion.py` with exact rational
arithmetic. A check confirms that the code means what the proof says. It does
not stand in for the proof.

Two of the theorems are about what cannot be proved, and they bound everything
else here. Read them first.

## What is and is not proved

Mathematical conclusions under the stated assumptions:

- no third-person likelihood separates two bridge laws attached to the same
  causally closed history (Theorem 1);
- unrestricted rival bridge laws can agree on a finite observed subset and
  differ elsewhere (Theorem 2);
- the interventional causal state is the unique minimal sufficient grain
  (Theorem 3);
- no permutation-invariant intrinsic functional can prefer one of two supports
  an exact symmetry exchanges (Theorem 4);
- colour refinement never separates two classes a symmetry exchanges, so a
  class it isolates is fixed by every symmetry (Theorem 5);
- a relabelling of content that preserves every relation changes no
  likelihood, so content is identifiable only up to such relabellings
  (Theorem 6);
- spectrum dominance on one fixed, complete horizon grid is a strict partial
  order with a non-empty frontier for a finite non-empty candidate set; crossing
  spectra can be incomparable, but are not the only source of multiple
  candidates (Theorem 7);
- numerical identity cannot branch (Theorem 8).

Assumed rather than established by these tests: the six postulates in
`core.subject.bridge.POSTULATES`, above all P1, that each selected intrinsic
carrier corresponds to one phenomenal subject. Theorems 1 and 2 do not prove
that empirical evidence is irrelevant to consciousness theories. They limit
identification among the specific unrestricted or observationally equivalent
rivals described below. A constrained theory can make testable predictions.

The postulates alone do not select a unique carrier: unresolved cuts, multiple
non-dominated candidates or missing measurements can leave J* unsettled. A
determined J* would be a result for the named system, runs and postulates,
subject to their measurement validity. Read [current ISC status](ISC_STATUS.md)
for the actual unresolved carrier and isolation questions.

None of the eight theorems mentions what a system is made of. Theorems 1 and 2
bound what can be known about a person's experience exactly as they bound what
can be known about hers. That is why the bridge is judged at parity, in
docs/BRIDGE_PARITY.md, and not held to a proof no mind has.

## Theorem 1: non-identifiability

Let U be the complete physical history and O any third-person observation. Let
J_a and J_b be two bridge laws, and suppose phenomenology adds no physical
effect beyond U, so the observation model depends on U alone:
P(O | U, J) = P(O | U) for every J.

Then P(O | U, J_a) / P(O | U, J_b) = P(O | U) / P(O | U) = 1 wherever it is
defined.

**Proof.** Both numerator and denominator equal P(O | U) by the hypothesis. ∎

This is a **conditional** likelihood ratio at a fixed U. A marginal Bayes
factor is also one if the rival laws assign the same distribution to physical
histories. If their predictions for U differ, this conditional identity does
not make the marginal evidence equal. The fixed-history comparison includes
J_a = no experience and J_b = experience. A scientific bridge therefore needs
one of three commitments: phenomenology is identical to some public structure,
it is tied to one by a fundamental law, or it has physical effects of its own.
The check pairs this with a model in which the law does change outcome
probabilities, where the ratio is two.

## Theorem 2: finite evidence

Let D be a finite set of observed states and J any law defined on a universe
containing a state outside D that admits a second value. Then there is a law
J' with J'(U) = J(U) for every U in D and J'(U*) ≠ J(U*) for some U* not in D.

**Proof.** Pick such a U* and a value v ≠ J(U*) it admits. Set J'(U*) = v and
J'(U) = J(U) elsewhere. ∎

Both laws agree on D. This assumes an unrestricted law class and an
unobserved state with a second allowed value. A finite or constrained
hypothesis class can sometimes be identified by finite observations; that is
not excluded. The check enumerates every law and
every data set on a four-state universe: a rival exists exactly when the data
leave some state unobserved.

## Theorem 3: the interventional causal state is the minimal sufficient grain

Let H be a set of histories, A a set of interventions and F the futures, with
P(F | h, do(a)) given. Define h ~ h' when P(F | h, do(a)) = P(F | h', do(a))
for every a, and let ε(h) be the class of h. A statistic η on H is sufficient
when η(h) = η(h') implies h ~ h'.

1. ε is sufficient.
2. For every sufficient η there is a function f with ε = f ∘ η.
3. Hence ε is the unique coarsest sufficient statistic, up to bijection.

**Proof.** (1) ε(h) = ε(h') means h ~ h' by definition. (2) Define
f(η(h)) = ε(h). It is well defined: if η(h) = η(h'), sufficiency gives
h ~ h', so ε(h) = ε(h'). (3) Any sufficient η refines ε by (2). If η is
coarser than or equal to ε and sufficient, ε also refines η, so they induce
the same partition. ∎

This is the interventional form of the computational-mechanics result that
causal states are the unique minimal sufficient statistics of a process. It
defines a coarsest grain **given** the chosen histories, interventions, futures
and their distributions. It does not prove that an empirical estimator has
measured all of them or remove the choice of those objects. The check
builds forty random controlled processes over five histories with exact
probabilities and enumerates all 52 partitions of the histories each time:
every sufficient partition refines the causal state, and every partition
strictly coarser than it merges two histories with different futures.

## Theorem 4: symmetry

Let G be the weighted, directed causal graph over the domains, σ an
automorphism of G, and Φ any functional of a support S that depends only on the
weighted graph restricted to and around S, so Φ(σ(S)) = Φ(S) whenever σ
preserves G. Then Φ cannot rank S above σ(S).

**Proof.** σ maps the induced weighted subgraph on S and its boundary
isomorphically onto those of σ(S). Φ reads only that structure, so its values
agree. ∎

An exclusion rule that must name one carrier among supports exchanged by an
exact symmetry has to add a fact that breaks the symmetry. This is why the
carrier term is allowed to be a symmetry class. The check computes every
automorphism of a five-cycle and of two joined three-cycles, and confirms three
invariant functionals score every support and its every image alike. A
functional that reads a node's label separates them, as it should.

## Theorem 5: colour refinement is sound

Let d be the distance matrix of a content structure and c_k the colouring after
k rounds of refinement: c_0 is constant, and c_{k+1}(i) is c_k(i) together with
the multiset of pairs (c_k(j), d(i, j)) over j ≠ i. For every automorphism σ
of d and every k, c_k(σ(i)) = c_k(i).

**Proof.** By induction. c_0 is constant. Suppose c_k is σ-invariant. The
multiset for σ(i) is {(c_k(j), d(σ(i), j))}; substituting j = σ(j') and using
d(σ(i), σ(j')) = d(i, j') and c_k(σ(j')) = c_k(j') gives
{(c_k(j'), d(i, j'))}, the multiset for i. So c_{k+1}(σ(i)) = c_{k+1}(i). ∎

Every symmetry therefore maps each refined cell onto itself, and a class alone
in its cell is fixed by every symmetry. That is the content of the rigid result
`bridge.orbits` reports. The converse does not hold in general: refinement can
leave two classes in one cell that no symmetry exchanges, so a cell larger than
one is a set this refinement has not separated. The full structure may still
distinguish them; a shared cell is not proof that a symmetry exchanges them. The check runs `orbits` on twenty-five random
structures and confirms every automorphism preserves every cell.

## Theorem 6: gauge

Let d be the relations among content classes, a report a sequence of class
labels, and L any likelihood that reads a report only through the relations
among the classes it names. If g preserves every relation,
d(g(q), g(q')) = d(q, q'), then L(g ∘ report) = L(report).

**Proof.** The sequence of relations L reads is unchanged term by term under g,
so L's argument is unchanged. ∎

For the declared likelihoods that read only these relations, identification
is limited to J modulo Aut(Q). This argument does not exclude a richer
observation model that breaks the symmetry. A first-person label would be an
additional anchor, not a result of the relation-only calculation. The check confirms every symmetry of a
five-cycle leaves two likelihoods unchanged, and that a relabelling that breaks
a relation changes one.

## Theorem 7: exclusion is a Pareto frontier

Fix one non-empty grid G of horizons for every comparison. For spectra
measured at every horizon in G, A dominates B when Φ_A(τ) ≥ Φ_B(τ) for every
τ in G, with strict inequality at at least one horizon.

1. Dominance is irreflexive, asymmetric and transitive on these vectors.
2. A finite non-empty set has a non-dominated member.
3. In a two-candidate comparison, crossing spectra leave both non-dominated.
   With more candidates, a third can dominate both crossing spectra. Equal
   spectra can also leave more than one candidate without any crossing.

**Proof.** (1) A vector is not strictly above itself. Opposite dominance would
require equality everywhere and strictness somewhere, a contradiction. For
A ≥ B ≥ C on the same grid, coordinatewise inequalities chain, and a strict
coordinate in either comparison stays strict between A and C. (2) A finite
strict partial order has a maximal element: a chain of strict improvements
cannot revisit a member and must end. (3) A is higher at one coordinate and
B at another, so neither dominates the other. ∎

The previous text compared each pair only at its shared horizons. That does
**not** give the claimed order. For example, A={1:2, 3:1}, B={1:1, 2:2} and
C={2:1, 3:2} yield A>B, B>C and C>A under that rule. There can be no frontier.
The [current implementation](../core/subject/v25_exclusion.py) already avoids
this defect: the whole candidate comparison uses one preregistered grid or the
union of measured horizons. A missing horizon makes a comparison unresolved,
rather than being dropped or read as zero. This corrects the stale document;
it does not change that implementation.

The finite-case tests check the stated common-grid comparison, crossing and
equal spectra, and the treatment of disjoint supports. Overlap is an additional
exclusion condition; disjoint carriers do not exclude one another merely
because one has a higher spectrum. None of this proves that a particular
measured candidate passed closure or that the admissible set is non-empty.

## Theorem 8: numerical identity cannot branch

If A = B and A = C then B = C, by transitivity of identity. In a fork B ≠ C, so
one earlier individual cannot be numerically identical to two later ones.
Continuity can branch; identity cannot. `tests/test_continuity_branches_and_identity_does_not.py`
applies it to the lineage graph in every transfer case.

## Where the empirical terms come from

The theorems fix what J* can be. Which carrier, which structure and which
lineage it is are measured:

| term | run | settled when |
|---|---|---|
| carrier | `tools/run_subject_core_v25.py` | the run is authoritative: every cut decided, all conditions, the grain resolved, no blocker |
| structure | the content run | the content verdict is one structure, rigid or up to the gauge of Theorem 5 |
| lineage | `lineage_from_state_log` over the persisted state | the signed lineage verifies |

`tools/solve_for_j.py` reads each term off its own run and reports a term its
run did not settle as unresolved, with that run's reasons. Nothing is filled in.

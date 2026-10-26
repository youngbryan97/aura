# Mathematical claims: assumptions, counterexamples and practical limits

Status: Reference · Arguments reconciled 25 October 2026

A mathematical statement needs its objects and assumptions. A test can check
that code matches a claim on selected inputs; it cannot turn an invalid
universal argument into a theorem. The corrections here change the reading
of the maintained guides, without changing their retained experimental data.

## Expressiveness does not decide how much human work is needed

A fixed language cannot gain a new denotation merely by naming an existing
expression. An ideal universal language can represent every partial computable
function, given sufficient resources. These statements concern what programs
can express, not which useful program the system can find or run.

The earlier “a non-universal language needs human edits forever” argument
missed a counterexample: one edit can replace its evaluator with a universal
one. The endless-edit conclusion needs the extra premise that every finite
extension remains non-universal and tasks keep demanding functions outside it.
Universality also does not make engineering impossible or unnecessary.

Think of installing a general programming language in a calculator. That
changes what can be expressed, but does not supply the correct program for
every request or enough time to run it.

## A changed input need not change the winner

Suppose a record changes two action scores from `(10, 1)` to `(9, 2)`.
The scores and margin changed, but action 1 still wins. An unchanged selected
action therefore does not prove the record was ignored.

Inspect the input, score, ranking and selected action separately. An
intervention that changes the choice supports sensitivity for that tested
case. A negative result needs enough variation to cross a relevant decision
boundary, or a narrower statement about the score that actually changed.

## Retaining an old method is not a fixed-budget performance guarantee

The union of reachable sets cannot shrink when an old method remains
available with its original state and budget. That is a set statement.
Under one fixed shared budget, choosing among more methods and maintaining
more state can consume resources. Actual success can fall even while the
ideal union grows. A new method can also dominate an old one without keeping
it, so predecessor retention is sufficient under the stated premises, not
necessary in every case.

Practically, retaining an old recipe preserves an option; it does not give
you extra time to cook every recipe before deciding which meal to serve.

## Undecidability does not prohibit proving a particular program correct

Rice’s theorem concerns nontrivial extensional properties of unrestricted
partial computable functions. “Extensional” means the property depends on what
a program computes, rather than its spelling. The theorem does not say every
property of every program class is undecidable. It does not rule out a proof
for a particular program or a complete checker over a finite declared problem.
[Baldan, Ranzato and Zhang](https://arxiv.org/abs/2105.14579) state the classical
scope before studying extensions to other semantics.

For example, a checker can prove that a specific bounded update preserves
`0 <= x <= 1` under a declared precondition. That does not make it a total
correctness oracle for arbitrary software. The
[Dafny reference manual](https://dafny.org/latest/DafnyRef/DafnyRef) describes
program verification using explicit specifications and proof obligations.
Löb’s theorem is not a blanket prohibition on such verification.

An updateable safety guard is not automatically unsafe. The relevant failure
is admission of a guard-breaking update without another constraint that
rejects it. A protected proof checker can remain part of the trust boundary.

## Some optimisation problems lack a fastest program; others have optima

Blum’s speedup result establishes functions with no fastest program under its
complexity assumptions. It is an existential result, not a finding that every
task or developmental policy has no optimum. The
[original paper’s bibliographic record](https://ntrs.nasa.gov/citations/19670048768)
and a [research treatment of speedup](https://itk.ilstu.edu/faculty/chungli/mypapers/SpeedUp_ToCS_Full_Version.pdf)
identify that scope. The publisher full text was inaccessible during this
review, so this page does not claim a fresh audit of Blum’s original proof.

No-free-lunch comparisons also depend on the problem distribution and search
assumptions. A counterexample where one update hurts one stream does not
prove a theorem about every update rule. [Everitt, Lattimore and Hutter](https://arxiv.org/abs/1608.04544)
show how changing the distribution changes the comparison. Name the task
population, prior, resources and evaluation when claiming an advantage.

## A positive chance is not enough for eventual success with probability one

If each repeated trial has conditional success probability at least
`epsilon > 0`, the probability of missing after `n` trials is at most
`(1-epsilon)^n`. With enough trials that tends to zero. A finite chain needs
that condition at each stage and preservation of the successful earlier stages.

Merely saying each trial has *some* positive chance is weaker. For example,
probabilities can shrink fast enough that their total is less than one; a
union bound then leaves a positive probability of never succeeding. A sampler,
finite memory and a changing scheduler need their actual coverage checked.
This asymptotic argument is not a promise that a useful program will be found
on a finite machine within a practical time.

## Which Aura guides this corrects

- [Recursive language expansion](RECURSIVE_ENDOGENOUS_EXPANSION.md): denotation,
  search, verification and later experiment status.
- [Developmental agency](AUTONOMOUS_DEVELOPMENTAL_AGENCY.md): decision
  sensitivity, retained-method reach, optimality and fair-search assumptions.
- [Commitment search](RLC_COMMITMENT_SEARCH.md): a measured near-identical
  activation direction does not establish a globally contractive operator.
- [SIMA adaptation](SIMA2_ADAPTATION.md): overlapping intervals do not establish
  equivalence or non-inferiority.

The [glossary](GLOSSARY.md) and [formula guide](FORMULAS.md) explain notation.
The [content audit](DOCUMENTATION_CONTENT_AUDIT.md) records the wider review.

## Comparing spectra needs one common grid

A spectrum is a set of measurements at different time horizons. Compare every
candidate on the same declared grid before treating dominance as a partial
order. Dropping a different missing horizon for each pair can produce a cycle:
A beats B, B beats C, and C beats A. The corrected
[bridge argument](BRIDGE_PROOFS.md#theorem-7-exclusion-is-a-pareto-frontier)
gives explicit numbers; the [implementation](../core/subject/v25_exclusion.py)
already uses a shared grid and reports missing horizons as unresolved.

Think of comparing runners on different courses. Winning each pair's chosen
race is not one consistent ranking. Crossing spectra also do not prove a
symmetry: two different candidates can excel at different horizons without
being interchangeable under a relabelling of the causal structure.

## A reproducible seed does not prove an unseen task

The [AGI gauntlet](AGI_GAUNTLET.md) derives a seed from recorded system identity.
That helps another evaluator regenerate an instance. It does not establish
that the task family was absent from development or training, and a finite
hash can collide. Keep repeatability, task novelty and independent evaluation
as separate evidence requirements.

## Similar directions do not establish convergence

For an update F(x)=2x, successive positive vectors point in the same direction:
their cosine similarity is one. Their size grows, and distances double.
A contraction needs a distance bound strictly below one across its declared
domain, not a near-one cosine between a few successive states. Read the
[bounded RLC search result](RLC_COMMITMENT_SEARCH.md) separately from a general
convergence claim.

## Overlapping intervals do not establish equivalence

Two success-rate confidence intervals can overlap without establishing that
the systems are equally good. Estimate the difference under an appropriate
comparison. A non-inferiority claim needs an acceptable loss margin fixed in
advance and evidence that the difference clears that margin. Equivalence needs
evidence within both bounds. If the study cannot resolve the difference,
report that uncertainty. The [SIMA2 adaptation guide](SIMA2_ADAPTATION.md)
separates local mechanism work from a matched performance comparison.

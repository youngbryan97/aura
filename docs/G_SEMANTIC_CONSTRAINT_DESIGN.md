# Semantic compilation by retained constraints

This is a design and implementation plan for G03-G12, not a completion receipt.
It addresses the 2026-09-15 request for a mechanism that could reach complete
bounded accuracy rather than another sequence of coefficient adjustments.
It specializes the existing [end-to-end contract](G03_SEMANTIC_CORRECTNESS_CONTRACT.md)
and [implementation workplan](G_SEMANTIC_REFINEMENT_WORKPLAN.md); it is not a
replacement ledger or a second language substrate.

## The target theorem

For an input x, let H(x) be a finite declared grammar of candidate programs.
Let C(x) be the nonempty subset implementing the user's intended specification.
Let s(x, h) be the deployed score, and let E preserve the semantics of h.
Assume:

1. Some c in C(x) is reachable by the actual candidate builder.
2. For every h outside C(x), s(x, c) > s(x, h).
3. The search returns a true maximum of s over H(x), not an interrupted prefix.
4. E executes that selected program with the declared value and partial-domain
   semantics.

Then the selected program is in C(x), and execution satisfies the specification.
Proof: an incorrect maximizer would have a lower score than c, contradicting
maximality. Semantic preservation gives the execution claim. Applying this
argument independently to 500 inputs proves correctness on those 500 inputs.

The theorem does not establish its assumptions. A grammar can omit the correct
meaning; a scorer can fail to separate meanings; search can be incomplete; and
the intended specification can be ambiguous or incorrectly annotated. A proof
of program execution alone does not prove that language was interpreted right.
No finite development score proves arbitrary transfer or frontier performance.

## The mechanism

Use the existing language substrate to construct typed operation/reference/
definition/dependency graphs. Use the existing universal floor and equivalence
checks to distinguish wrong programs from alternative correct computations.
Store witnessed negative interpretations and their correct alternatives as
training constraints, not as runtime answer lookups.

Each retained witness imposes:

    s_theta(x, correct) - s_theta(x, wrong) >= margin > 0

The next candidate must retain satisfied constraints. Correct source cases
also contribute competing interpretations; mining only current mistakes leaves
the rest of the source distribution unprotected. Source operation labels
contribute every competing label, not only a changing second-place prediction.

Re-decode after an update: mention/definition choices are latent, so a new
competitor can invalidate an old finite constraint inventory. Keep old witnesses
and add the new ones. Unknown equivalence and incomplete search do not become
negative examples. Validation and held-out targets never enter this fit.

### Changed truth, partial correctness and better answers

A retained obligation is conditional on the source specification, evidence,
assumptions and revision that justified it. It is not an eternal preference
for the incumbent's answer. This trainer reconstructs constraints within each
frozen source fit; it does not import an earlier run's constraint set into a
changed dataset. The source loader binds cohort identities and representation.

For timeless mathematics, alternative equivalent programs are positives, not
errors. For changing facts, contradictory new evidence must be reconciled with
the evidence owner, and superseded constraints must leave the active set while
remaining in history. Nothing in this numerical trainer implements that broad
epistemic lifecycle; reuse the existing evidence and knowledge systems for it.

Partial correctness is a vector of satisfied and unsatisfied requirements.
Correct sub-bindings can be retained while an incomplete or wrong whole answer
is repaired. A complete bounded program is still judged by the original exact
or explicitly declared equivalence criterion, not a new partial-credit rule.
Among fully correct alternatives, quality, cost and clarity are separate
objectives; improving them must not reverse a verified correctness ordering.

This follows counterexample-guided synthesis and structured prediction:
[Solar-Lezama's CEGIS treatment](https://people.csail.mit.edu/asolar/SynthesisCourse/Lecture17.htm)
and [Joachims, Finley and Yu's structured learning](https://www.cs.cornell.edu/~tomf/publications/linearstruct07.pdf).
The new engineering obligation here is to make the retained constraints match
Aura's actual nonlinear score and exported coefficients.

## Finite progress and its limits

For a finite hypothesis set containing a valid solution, a complete learner
that satisfies every accumulated counterexample never returns the same
refuted hypothesis. A verifier that returns a genuine new counterexample for
every invalid candidate therefore terminates after at most the number of
invalid hypotheses plus one candidate. This is not a bound on elapsed time.

Aura's neural coefficient space is continuous, its optimizer is not complete,
and finite probes do not decide general program equivalence. Consequently that
termination theorem cannot honestly be assigned to the current trainer. Its
use is architectural: do not forget witnesses, and do not call failure to find
an update a proof that an update is impossible.

## Implementation layers

1. Retained-constraint learning: the existing joint graph scorer exposes each
   margin independently. A constrained update projects its direction against
   active satisfied inequalities, then checks every inequality after float32
   storage before accepting the step. Positive margins retain positive floors.
   This is implemented as an opt-in training path, not a serving gate.
2. Counterfactual coverage: fit fresh coefficients on the remeasured 27B source
   bank plus training-only renamings, reorderings and meaning-changing pairs.
   The original 500 validation identities remain fixed, and test rows are
   excluded from fitting. This distinguishes new training information from
   changes to the optimizer.
3. Capacity expansion: if search stalls, diagnose the representation separately.
   The existing exact linear-score certificate applies only to its frozen
   linear comparison system. It does not prove the nonlinear neural tissue
   insufficient. Candidate expansions include extra cross-feature relation
   directions and shared scope/dependency state, selected by source-only
   comparisons and tested for removal by matched lesions. These expansions
   are not implemented or validated by the first two layers.
4. Reusable programs: the existing registry stores parameterized procedures
   with execution and domain evidence. New values test executor reuse; new
   wording and new compositions separately test the semantic compiler.
5. Prospective transfer: freeze the selected compiler, then run untouched
   families, seeds, phrasing and deeper compositions. Only subsequent matched
   arms, runtime qualification and named broad evaluations address G04-G12.

## Required falsification

- A low-weight, already-correct witness must not be sacrificed to fix a
  high-weight error. Test contradictory constraints without inventing a pass.
- A failed numerical search must report unresolved, never infeasible.
- Replay margins from the exported dtype and actual runtime scoring equations.
- Show improvement on fresh decoding, not just frozen latent graph scores.
- Keep witness coverage, source accuracy, unchanged 500-row development
  accuracy, fresh transfer, and broad/frontier claims separate.
- If the source fit still cannot satisfy constraints, inspect reachability and
  score capacity before spending another full validation run on the same fit.

The retained-constraint implementation removes one demonstrated failure mode.
It is not a claim that the theorem's assumptions now hold for all 500 cases.

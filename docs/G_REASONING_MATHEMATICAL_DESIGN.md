# Conditions for decoding, transfer and reasoning gain

This design supplements the existing semantic correctness contract. It does
not redefine the G-ledger's acceptance criteria.

## Exact decoding on a declared grammar

Let X be a request, P(X) the admissible programs, and C(X) the programs that
express the intended meaning. Let S be the score actually used by the decoder.
A sufficient condition for correct decoding is:

1. C(X) intersects P(X).
2. At least one correct program scores strictly above every incorrect program.
3. Search returns a global maximizer, with ties and incomplete search explicit.
4. Execution and public emission preserve that program's semantics.

For fixed features and a linear score S(X,p) = w.phi(X,p), a chosen correct
representative p+ defeats a competitor p- with margin m when
w.(phi(X,p+) - phi(X,p-)) >= m. The retained constraints form a convex
feasibility problem. Counterexample generation can add missing competitors.
For a finite grammar, complete separation and verified feasible coefficients
establish the ranking condition for those requests. An incomplete competitor
search cannot establish it. A correct alternative need not share the same
syntax, so semantic equivalence belongs in the separation procedure.

Aura already has complete bounded operation search, argument MILP search,
floor-based counterexample checks and linear graph-factor calibration. The
new graph-scale feasibility measurement is wired into that existing trainer.
It checks the actual stored witness independently, distinguishes numerical
infeasibility from optimization failure, and grants no serving authority.
Three positive scales are a restricted representation, not a general semantic
compiler. If they cannot separate the data, richer reusable features or a
different representation are needed; more iterations cannot make incompatible
linear inequalities compatible.

The retained 2026-09-15 graph-factor log supplies 728 such contrasts. Replaying
their three variable factor differences and fixed offsets (proposal scale
0.875) through the feasibility check returns HiGHS status 2 at required margin
0.1: numerically infeasible. This concerns that historical candidate pool and
three-scale representation, not all richer models. It is not an exact rational
infeasibility certificate. The log remains at
`~/.aura/rlc-evidence/semantic-graph-factor-supervisor-20260915/detached.log`.

The [exact capacity replay and trainer integration](evidence/G03_EXACT_SCALE_CONFLICT_2026-09-18.md)
reproduces the existing September 15 proof-kernel result: a contradiction even
for strict ranking without a fixed margin. Four retained comparisons suffice.
This is not a new discovery or a claim about richer representations.

## Transfer

A sufficient compositional route is a semantics-preserving translation for
every primitive and every composition rule in a declared grammar. Structural
induction then establishes correct execution of arbitrary finite compositions
in that grammar. It does not prove that arbitrary language is translated into
the intended composition. The learned front end still needs coverage of
references, scope, types, quantification and ambiguity.

Representation should share operations and relations across tasks, preserve
renaming and input-coordinate changes, and expose counterexamples that require
new distinctions. Fresh construction, vocabulary and domain splits test that
sharing. The universal floor supplies execution expressibility; the language
substrate supplies representations. Neither alone proves recovery of the
intended program on unseen requests.

## Broad reasoning gain

Let B be the baseline answer, A the enhanced answer, Y indicate correctness,
and D indicate choosing A. The exact change in accuracy is:

    E[D * (Y(A) - Y(B))]
      = P(D=1, A correct, B wrong) - P(D=1, A wrong, B correct).

Thus preserving B in a candidate set does not prevent regressions. Gains
require producing useful alternatives and selecting them more reliably than
harmful replacements. Independent execution, evidence provenance and calibrated
outcome feedback can support selection. An oracle selector would suffice but
is not an implementation. A verifier that sometimes errs needs measured error
rates and task-level paired controls, not a claim of zero regression.

The shared procedure registry, knowledge evidence graph, planning executor and
outcome learner should supply alternatives and assess actual effects. Their
integration must preserve request identity and source dependence. Broad tasks
need different checking methods where exact execution is unavailable; missing
ground truth is not a passing result. Frontier comparison additionally requires
named external baselines and fair resource accounting.

## Implementation sequence

1. Finish the active retained-constraint repair and autonomous replay. Preserve
   every negative trial and distinguish search coverage from fitted margins.
2. Use existing graph-factor feasibility to distinguish representational
   conflicts from solver failures. Expand shared features only where evidence
   demonstrates missing distinctions; do not encode evaluation answers.
3. Align training separation and runtime global scoring, then measure complete
   development cohorts before freezing a candidate for fresh transfer.
4. Exercise shared retrieval, procedure composition and outcome selection on
   ordinary runtime tasks, including baseline successes and causal lesions.
5. Qualify current-model materialization and serving, then run independent
   broad and named-reference comparisons. No internal constant closes G12.

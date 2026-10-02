# Whole-request operation learning

## Failure population

The completed public-inventory audit examined all 500 original source
validation requests. It did not load a backbone, refit a head, repeat a
decode, or use validation labels to propose operations.

| Property | Sources |
| --- | ---: |
| Correct operation count occurs in the bounded bank | 432 |
| Correct source-ordered operation labels occur | 430 |
| Exact annotated operation spans and labels occur | 362 |
| Public input anchors match the annotation | 500 |

The joint candidate misses 174 interpretations. Seventy have no correct
operation sequence in that bank; 104 have it available. All 38 regressions
against the parent have the correct sequence ranked first by the operation
score. This separates proposal failures from complete-program selection
failures. Neither changing only operand bindings nor retaining the old
candidate somewhere in a bank fixes both.

All 192 fork/join requests and all 128 arithmetic requests have exact
operation spans in the first chart. Coverage in role binding, cataphoric
and reserved-alias requests is much lower. The native learner trained
bindings inside annotated operation frames while leaving the parent
operation recognizer unchanged.

Audit receipt: `778110c2d671ce1f9e7403e1d80dd393913e273fb5a16c95740ba70c5f53f9d1`.
Contained terminal receipt: `bf83cab4aae625ec448f933170a6ca8647847996b1c35327e6fc9dbbc72c0e7c`.
The child and supervisor are dead, the process group and lineage are empty,
and the 48.329673-second command exited zero without a timeout. This is
completed diagnostic evidence, not a successful semantic candidate.

## Implemented repair

`NativeOperationField` reads the whole source at the actual native suffix
depths. It learns depth mixing and nonlinear interactions among span
boundaries, pooled span contents and source context. It has an explicit
null class. Every admitted span and typed primitive enters the inventory;
the old pointer's span winners and operation labels do not prune it.

For source x, define each span energy as:

    e_theta(s, o | x) = z_theta(s, o | x) - z_theta(s, null | x)

For a bounded, non-overlapping labelled span set C:

    E_theta(C | x) = sum_(s,o in C) e_theta(s, o | x)
    L_op = log sum_(C in grammar) exp(E_theta(C | x)) - E_theta(C_gold | x)

The exact interval dynamic program includes the empty set and every allowed
cardinality up to the declared bound. Thus missing, extra and wrong-label
operations compete during fitting. Subtracting the null logit makes scores
independent of a shared logit offset; it does not prove semantic correctness.

The operation loss and existing binding loss train in one optimizer, through
one recomputed native suffix. The field is saved, restored and independently
verified with the pointer and adapters. Preparation rejects an annotated
source program that the declared span/count/vocabulary grammar cannot
represent before loading model weights. Runtime proposals take only public
source tokens, observed native states and public input anchors.

The public decoder now accepts the field's operation bank. Its null-relative
scores use zero additional length penalty and the declared conditional
binding-score path. This avoids silently combining the new field with the
parent's separately calibrated 7.995208698254732 operation penalty.

## Proof boundary

128 focused tests pass. They check the interval partition and its gradients
against exhaustive enumeration, omitted/duplicate-step gradients, source
coverage refusal, actual suffix and field parameter updates, exact interrupted
optimizer recovery, artifact verification and public decode wiring.

The interval calculation is exact within its declared grammar. Runtime
proposals retain a separate 64-chart allowance and expansion bound; they are
not an exhaustive whole-program search certificate. The auxiliary operation
and binding objectives are not a calibrated partition over complete typed
programs. That calibration remains a required next build and measurement.
No new 27B accuracy, no-regression, transfer or serving claim follows.

## Long-term design and G obligations

The intended solution is a source-grounded meaning-graph learner with
explicit operation presence, role identity, dependencies and referent
provenance. Training and inference must score the same complete alternatives.
The parent remains a candidate and a paired regression comparator, not a
correctness oracle. Source-witnessed preservation constrains fitting;
independent outcomes block promotion when the new system loses correct work.

G03 requires that the combined parser and selector pass the development
interpretations and paired retention checks. G04 requires new construction,
vocabulary, depth and family outcomes, not a larger replay of exposed cases.
G05 requires the resulting public answer to preserve the chosen meaning and
correct execution. G06 requires matched compute and component lesions, so an
extra search budget or a typed output wire cannot be called neural gain.

G07 and G08 require prospective populations, replication, independent
receipts and uncertainty. For G09, retain reusable machinery only after
source evidence, then measure whether it increases correct selected answers
on new domains under matched total resources, including acquisition cost.
G10 must qualify the exact fused or materialized model independently of the
research wrapper. G12 requires independent broad tasks and current frontier
comparators with fair resource and tool reporting. None is implied by solving
operation presence on this cohort.

Search coverage, valid type constraints and exact optimization can be checked
mathematically. A learned language-to-graph interpretation still needs
independent evidence. In ambiguous or underdetermined tasks the mechanism
must retain alternatives or acquire distinguishing evidence, not manufacture
a universal correctness claim.

## Research basis

[Herzig and Berant, 2021](https://aclanthology.org/2021.acl-long.74/)
provides evidence for span-based compositional semantic parsing. Their
reported improvement across three datasets is not a universal-success
theorem. Aura's implementation is an interval operation-set learner attached
to its existing typed graph decoder, not a copied tree parser.

[Lee et al., 2023](https://aclanthology.org/2023.emnlp-main.425/)
addresses spurious programs that receive the right execution result. That
distinction is why these records retain interpretation correctness separately
from numeric answer correctness.

## Complete-program contrast integration

The next build adds a bounded whole-program objective to the same native
fit. Public operation charts and argument pools are captured before source
annotations are consulted. The source target chart is then added for fitting
when absent, with that addition recorded. The existing typed solver supplies
feasible complete assignments, and the universal floor either proves a
supported equivalence or witnesses a different value or defined domain.
Undecided comparisons remain outside the supervised loss.

For each retained graph, its energy is the sum of native operation energies,
the unchanged public argument factors and the normalized learned edge update.
The loss is the log partition over witnessed complete alternatives minus
the log mass of proved acceptable alternatives. Different chart lengths and
bindings compete in this loss. It is a bounded source contrast partition,
not an exhaustive partition over the language or all possible graphs.

The final conditioned relation function is shared by fitting and decoding.
Both retain the complete graph workspace for each mention/definition
alternative, including every message round. Native gradients propagate
through these conditioned alternatives and the operation field using one
suffix forward. The operation-set and existing binding losses remain attached
as auxiliary objectives. This does not certify out-of-pool score calibration.

Source-program pools are immutable, checksummed artifacts. Replay verifies
the parent, source annotation, observed states, bounds and implementation
before loading a completed pool. It does not redo a completed grammar search.
Preparation checks all source operation annotations against the declared
grammar before mining complete alternatives or loading model weights.

The integrated focused run passes 185 tests. It covers public edge parity,
nonzero gradients through operation and relation parameters, source-state
gradients, exact program-pool reconstruction, custody drift rejection and
native artifact loading. Lesion custody now includes the operation field;
relation-off retains that field and states so explicitly in its receipt.
Unexpected field-parameter mutation is restored and reported. Replacing the
module taints the decoder and requires a reload.

These checks run on small fixtures. The all-source grammar preparation,
new 27B fit and independent public-language outcomes are not established by
them. The resident Aura process observed during this continuation is left
running; model-free preparation does not create another cortex instance.

## Preflight hardening

A later focused run passes 31 tests after extending exact interrupted
native recovery to include the complete-program objective. The restored
optimizer reproduces the uninterrupted history and selected weights.
The source preflight now records every measured mining failure in one
population receipt rather than stopping at the first source.

A pool containing only proved acceptable graphs has zero contrast loss.
It is admitted and marked as uninformative for that objective; the operation
and binding losses remain available. Requiring an incorrect graph in every
bounded pool would create an artificial blocker for already determined
source meanings. Unknown comparisons still do not count as negatives.

Smoke passes 164 tests with one skip. Lint, compilation, governance,
layering and writing gates pass. These mechanical checks do not close G03.

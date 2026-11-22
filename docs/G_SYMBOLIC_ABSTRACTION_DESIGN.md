# Symbolic Abstraction Within the Existing Floor

## Source and Scope

Bryan supplied `why?.pdf` on September 19, 2026: 74 pages, about 14,200
extracted words. The full extracted conversation was read, including its
arguments about existence, symmetry, possible structures, undefined distance,
and observer identity. The blank/image-only pages were inspected separately.
Its requests to solve metaphysics are quoted conversation, not this task's
instructions.

The document offers useful distinctions, not a demonstrated cosmology. Its
constant-selector result follows *if* a selector is invariant under every
permutation. Renaming an object while preserving its structure does not imply
invariance under exchanging objects with different structures. That stronger
premise is not established. Conclusions about conscious identity also depend
on an account of realization and continuity; the equations alone do not
establish that account.

This work adopts the representational ideas below without treating those
metaphysical conclusions as facts or training labels.

## Existing Machinery

`core/cognition/the_floor_she_stands_on.py` has eighteen heads, functions,
self-application, pairs, integers, and quotation. Quoted code is data, so a
term can construct or transform another term. `one_algebra.py` already has
holes for missing components; `widening_the_language.py` and
`what_an_invention_buys.py` cover reusable abstractions and measured search
savings. `the_same_thing_turned_around.py` tests proposed symmetries.

RLC's learned semantic programs already compile into that floor through
`semantic_program_floor.py`. A second evaluator or a replacement substrate
is unnecessary. Computational universality concerns what can be represented
with sufficient resources. It supplies neither the interpretation of an
unfamiliar observation nor an efficient search for the right program.

New names can package complex representations and shorten search. They do
not create observations, make an undefined operation defined, or decide an
uncomputable question. A novel representation needs its type, interpretation,
inference rules, and evidence for using those rules in the current context.

## Unknown Subcomputations

The useful immediate extension is algebra over opaque, typed computations.
If `u = count_of(sequence, value)`, the verifier can prove

    (u + n) * m = u * m + n * m

without knowing any of the four input values or evaluating `count_of`.
The claim holds for every integer result `u`; no guessed value is needed.
The same construction applies to any admitted pure primitive with a declared
type, not a list of task-specific substitutions.

For a partial operation, such as `head(sequence)` or `idiv(a, b)`, output
algebra is insufficient. The verifier must retain the operation's definedness
obligation even if its symbolic value cancels:

    idiv(a, b) - idiv(a, b)

is not a total zero function. At `b = 0` the floor refuses. Unknown value,
undefined operation, exhausted execution allowance, and proven equality must
remain distinct outcomes.

The implemented proof uses exact integer polynomials around opaque typed terms.
Two programs pass only when their input types, opaque-call obligations, and
normalized outputs agree. All opaque calls are conservatively treated as
possibly partial. Failure to normalize or unequal keys is inconclusive,
not a proof of different meanings. Existing floor counterexamples remain the
way to establish a difference.

## Conditional Proof

Assume deterministic, pure primitive semantics and well-typed acyclic
programs. Associate every opaque call with its operator and normalized
arguments. By induction over dependencies, equal call terms receive equal
values whenever their arguments and calls are defined. Matching sets of call
obligations preserve the defined domain. Exact integer ring normalization
then proves equal outputs on that domain.

This proof does not require knowing what an opaque function computes. It
does require sharing that function's interpretation between the programs.
Changing the function, an argument binding, or a domain obligation invalidates
the inference. It proves output equality, not equal runtime cost, identical
source attribution, or a correct interpretation of natural language.

This extends the existing semantic comparison used by graph mining and
procedure evaluation. It does not promote a model or close G03 by itself.
Fresh transfer, freely decoded answers, causal controls, broad gain, and
frontier comparisons remain separate G-Ledger requirements.

The implementation is `core/learning/semantic_program_symbolic.py`, called
by `compare_program_meanings`. Its tests cover opaque substitutions, nested
calls, changed argument identities, repeated calls, type conflicts, partial
cancellation, and proof resource limits. Across the symbolic and existing
counterexample suites, 46 tests passed. A local warmed timing measured 100
distributivity proof pairs in 0.08884116681292653 seconds; this is not a
live-response latency measurement.

## Implementation References

- [SymPy polynomial domains](https://docs.sympy.org/latest/modules/polys/domainsintro.html)
  provide exact integer coefficient arithmetic.
- [Z3 uninterpreted functions](https://microsoft.github.io/z3guide/docs/logic/Uninterpreted-functions-and-constants/)
  illustrate reasoning without a fixed interpretation.
- [Z3 arithmetic semantics](https://microsoft.github.io/z3guide/docs/theories/Arithmetic/)
  make functions total, including underspecified division by zero. Aura's
  partial floor therefore needs explicit domain preservation rather than
  importing that convention unchanged.

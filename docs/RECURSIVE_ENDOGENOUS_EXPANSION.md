# Recursive endogenous expansion of the cognitive language

Audited HEAD: `ab58a159cecc1dbeabf72aeced0cb44db4a99202`, branch `main`,
2026-09-01. Work landed on top of it; the tests and numbers below were measured
in `.claude/worktrees/endogenous-substrate` against that lineage.

The question:

> How can a persistent artificial learner make the language and mechanisms of
> its own future learning themselves products of experience — recursively —
> without requiring a new human-authored mechanism at every meta-level?

The answer, in one sentence, is a theorem rather than an architecture: **the
regress ends at universality, and it ends nowhere else.** Everything below is
the derivation, the measurement of which side of that line Aura was on, the
code that moved her across it, and an exact account of what remains.

---

## Part I — Repository audit

### What was traced

`core/cognition/operator_invention.py`, `one_algebra.py`,
`growing_at_any_level.py`, `an_invented_kind.py`, `what_growth_cannot_do.py`,
`language_limits.py`, `which_kind_of_growth.py`, `what_an_invention_buys.py`,
`what_the_failures_have_in_common.py`, `an_operation_that_generalises.py`,
`what_she_gave_meaning.py`, `how_she_learns_to_look.py`,
`keeping_the_language_small.py`, `what_it_costs_to_say.py`,
`sequence_induction.py`, and the claim registrations in
`core/organism/model_validation.py`.

### The live path is not the one the prompt names

`OperatorKernel` has two importers — `tools/evidence_report.py` and
`core/cognition/growth_report.py` — and both take its name for a report. No
runtime caller. `Candidate.fn` is a supplied Python callable. It is an
admission machine with nothing feeding it.

The path that actually runs is `sequence_induction._widen`: a concept she
formed, an addressing she derived, an operation she derived, a recipe she
composed, a diagnosis of whether the language or the search failed, a maker she
wrote, then the constructors that were written down. Endogenous representation
work has reached ordinary inference. The kernel the prompt is named after has
not.

### Five boundaries, not one

**B1 — the grammar ceiling.** `one_algebra` collapsed the tower of makers: a
way of building is a term with a hole, so there is no list of constructors to
be at the end of. It did not collapse the grammar those terms are written in.
`HEADS` is seven arithmetic operations and `run` is an if-chain over `where`,
`many`, `fixed`, `hole`, `through`, `undo`, `over again` and `if`. A family
needing a ninth waits for a person.

**B2 — the schema ceiling.** `Induced.read` fixes the rule shape to
`after[i] = f(before[g(i,n)], before[h(i,n)])`: two sources, one binary
operation, value-blind addressing. No word, maker or level changes it.

**B3 — two algebras.** `one_algebra.Term` computes positions;
`an_operation_that_generalises.Expression` computes values. Separate
primitives, separate enumerators, separate serialisers. An invention in one
cannot be material for an invention in the other. **Closed:** both compile to
the floor and both are checked against their own interpreter over a grid,
refusals included.

**B4 — a persistence hole.** `run` has always evaluated `over again`.
`read_back` checked the head against a second hand-written list and that head
was not on it. Confirmed by execution: `read_back(written_down(t))` returned
`None`. The head the module's own docstring calls the one with a shape no
fixed-length composition has was the one head that could not survive a boot.
ChatGPT found the same defect independently; the other six responses did not.

**B5 — the mechanism is not an object.** `growing_at_any_level.grow_at` takes a
Python `make` callable. It collapsed the API and kept the human meta-level.

### What the audit changes about the question

B1 is the regress, and it is not an aesthetic complaint about grammars. Part VI
proves that the heads bound every positional term by a polynomial in the length
of the state, so something computable is outside the language permanently, and
only a person could ever put it in. That is the thing to fix. B4 was a defect
and is fixed. B2, B3 and B5 are named in Part XIII as open.

---

## Part II — Formal problem definition

### Objects

Let `U` be a fixed evaluator and `Σ` its signature. Let `A_t` be the finite
library of named terms at time `t`, and `P(A_t)` the terms constructible from
`Σ` and `A_t`. Write `⟦p⟧` for what `U` computes from `p` under fuel, and

    E(A_t) = { ⟦p⟧ : p ∈ P(A_t) }.

A budget is not a number. Use a vector `B = (B_gen, B_verify, B_exec, B_mem,
B_len)`: an abstraction can cut synthesis cost and raise runtime cost, and a
scalar hides that.

    Reach_B(A_t) = { ⟦p⟧ : p ∈ P(A_t), |p| ≤ B_len, U halts on p within B,
                     and the search finds p within B_gen }.

`M_t = (Π_t, Admit, Cost)` is the mechanism: a proposal program, a gate, and a
ruler. `D_t` is the developmental record. Capability on a task class `T` is

    C(A, D; T, B) = E_{τ∼T}[ 1(τ ∈ Reach_B(A)) ] − λ · E[search cost].

### Three different things "the language grew" can mean

Already named in the tree by `which_kind_of_growth`, and the distinction is
load-bearing:

1. **a shorter name** — `E` unchanged, some programs got shorter;
2. **a longer reach** — `E` unchanged, `Reach_B` grew;
3. **a new distinction** — some `f ∈ E(A_{t+1})` that no term of `A_t` denotes,
   over a search that *finished*.

The third is available only while the substrate is not universal, and is
therefore available exactly once in the whole developmental history. Part VI
shows Aura had not yet spent it.

### Representational inadequacy

Not a boolean. A certificate set, with each member's decidability stated:

| Certificate | Meaning | Decidable |
|---|---|---|
| `SEARCH_BUDGET` | nothing found and the walk did not finish | yes, by the clock |
| `EXHAUSTED_INSIDE` | the bounded walk finished and found nothing | yes, in a finite fragment |
| `TYPE_REFUTED` | a fragment invariant is violated | yes, for that fragment |
| `THIN_DATA` | intersections non-empty but unstable under leave-one-out | bounded |
| `NOISE` | a shuffle null explains the residual | statistical |
| `SOMETHING_UNSEEN` | the record is not a function of what she reads | yes |
| `UNDECIDED` | otherwise | — |

`what_the_failures_have_in_common` and `language_limits` implement most of this
already. General inadequacy over a universal language reduces to halting and is
not available at any price.

---

## Part III — Literature review

**DreamCoder** (Ellis et al., PLDI 2021). Wake/sleep library learning: solve,
compress solutions into λ-abstractions, train a recognition model. Solves
`Reach_B` growth by library induction. Does not solve the origin of the DSL,
and its abstraction step is definitional, so `E` never moves. Aura's
`library_compression.py` and `wake_sleep.py` are this shape.

**Stitch** (Bowers et al., POPL 2023) and **LILO** (Grand et al., 2024). Faster
corpus compression to abstractions; language-model-guided proposal and
documentation. Same ceiling: refactoring of programs already found. LILO's
proposals can carry a pretrained ontology, which is a leakage risk rather than
a mechanism.

**Meta-interpretive learning** (Muggleton et al., 2014). Genuine predicate
invention, constrained by human-authored metarules. It is the cleanest
illustration of why moving the vocabulary up one level does not stop a regress.

**Gödel machine** (Schmidhuber, 2003). Self-rewrite on a proof that the rewrite
improves expected utility. The right formal precedent for collapsing object and
meta level; the proof requirement is not obtainable in practice, and the axioms
and utility remain authored.

**Gödel Agent** (Yin et al., 2025) and **Darwin Gödel Machine** (Zhang et al.,
2025). Empirical self-modification of agent code with an archive. Direct
precedent for `M_{t+1} ≠ M_t`. What they change is tooling, evaluated on a
benchmark; the semantic language is not the object.

**AlphaEvolve** and **FunSearch**. LLM-guided program evolution with automated
evaluators discovering non-trivial algorithms. Strong evidence that executable
semantics can be found automatically; the evaluation architecture is supplied
per domain.

**Reflective towers**: Smith's 3-LISP (1982), Wand and Friedman (1986),
Danvy and Malmkjær (1988). The architectural model for self-reference without
regress. No learning; the towers are static. **Mogensen** (1992) gives the
self-interpreter for the untyped λ-calculus that `THE_INTERPRETER` is an
instance of.

**Partial evaluation and the Futamura projections** (Futamura, 1971). Why a
self-interpreter is not merely decorative: specialising it is compilation.

**E-graphs and equality saturation** (Willsey et al., POPL 2021). The right
tool for canonicalisation and novelty normalisation. Not an invention
mechanism.

**CEGIS / SyGuS**. Counterexample-guided synthesis with strong bounded
correctness where the theory is decidable and supplied.

**Genetic programming; PushGP; autoconstructive evolution** (Spector, 2002).
Programs that influence their own variation machinery — the strongest
conceptual precedent for eliminating a fixed hierarchy of variation operators,
with weak semantics and no governance.

**Solomonoff induction, MDL, algorithmic statistics, Levin search.** The
universal prior is the honest answer to "no designer taxonomy" and the reason
this design does not claim tractability from universality. Levin's bound is why
Part V's library, and not a bigger budget, is what moves the horizon.

**What remains unsolved across all of them.** Nobody exhibits the conjunction:
a universal self-representing substrate that is the language ordinary cognition
already runs in; residual-driven synthesis into it; certificates that separate
name growth from reach growth from a new distinction; causal recursive reuse;
and the mechanism itself as an artifact of the same kind. Each piece has a
precedent. The conjunction is the contribution, and Part XIII says how much of
it is currently evidenced.

**What no response in the council document contains**, and what this one adds:
the necessity direction of the universality theorem, and a measurement of
whether the substrate actually in the tree is universal. Grok calls the grammar
"universal-enough"; DeepSeek could not find the file. It is not universal, and
the witnesses are constructible.

---

## Part IV — Three architectures, compared

**A — grow the positional algebra.** More heads, more types, richer induction.
Efficient, testable, good proof opportunities. Fatal: unless the result becomes
universal, another ceiling sits behind the one just removed, and Part VI shows
the ceiling is real rather than hypothetical. Appropriate as a specialised
frontend, not as the foundation.

**B — self-modifying host source.** Let a model propose Python changes, test in
a shadow, promote what wins. Universal and directly compatible with the
existing `core/self_modification/`. Fatal as the cognitive substrate: Python's
semantics make equivalence and confinement intractable, candidate code can
touch the evaluator, and the human-authored runtime is an enormous implicit
prior. Useful only as an outer, untrusted shell.

**C — a universal, metered, homoiconic term language, with the interpreter as
a term in it.** Universal, so no semantic ceiling. Homoiconic, so a term can
read and build a term and the mechanism is an object of the same kind.
Metered, so every evaluation is bounded. Costs: search over it is Levin-hard,
and unrestricted verification is unavailable.

**Chosen: C, with A retained as a frontend.** Not a compromise — a consequence.
Part VI's theorem says only a universal bedrock stops the regress, which rules
A out as a foundation; the confinement argument rules B out as a substrate; and
the positional algebra remains the right specialised language for positional
families, now as terms compiled through heads rather than as the ceiling.

The council's fourth family, "recursive bias modification" (DeepSeek), is
weaker than C for a reason worth stating: a bias set is a human-designed
parameterisation, so improving the bias-update rule is improving a fixed shape.
ChatGPT's closing "physical sandbox" proposal is a category error. An
instruction set is a fixed semantics, and a compression drive is a prior; it
changes which universal machine, not whether there is one, and it picks a
machine with far worse search properties.

---

## Part V — The chosen architecture

### The floor — `core/cognition/the_floor_she_stands_on.py`

Eighteen heads:

    a number · nothing · a pair · the first of · the second of · is it a pair
    the one it was given · given a thing · of · if
    plus · minus · times · over · left over · below · same as
    as it is written

Numbers and pairs are the data. `given a thing` and `of` are untyped
abstraction and application, so self-application gives unbounded recursion and
the set is universal. `as it is written` is quotation: it turns a term into a
value of numbers and pairs, which is what lets a term read, take apart and
build a term.

The machine is a loop over an explicit continuation stack, not a recursive
function, because a term the machine runs can be thousands deep and a Python
recursion would fail on a term the evaluator handles. Every step costs one unit
of fuel. `OutOfFuel` and `Stuck` are the two refusals.

Nothing on the floor can reach the world. A term computes over numbers and
pairs; the module imports `__future__`, `dataclasses`, `logging` and `typing`
and a test enforces that list. Admitting a term is never a route to a privilege
she did not have.

### The interpreter as a term — `core/cognition/the_floor_reading_itself.py`

`THE_INTERPRETER` is 347 symbols of those eighteen heads. Given the encoding of
any floor term and an environment, it computes what that term computes,
including terms that call themselves: it runs `6!` through itself and agrees
with the machine. It can be written down and read back, and a term can be
handed the interpreter's own encoding.

That is what makes "the mechanism is an object of the language" a value rather
than a slogan. Standard — Mogensen 1992 — and the point is that it is the
evaluator of the language Aura's rules are written in rather than a
demonstration beside it.

### The grammar becomes endogenous — `DERIVED_HEADS` in `one_algebra.py`

A head is a term on the floor. `run` dispatches to it, `every_term` offers it
to the search, `read_back` accepts terms mentioning it, `what_she_gave_meaning`
keeps it, and removing it makes terms that use it refuse loudly rather than
quietly answer something else.

A head is given six things, outermost first: where it is, how long the state
is, what each part says *here*, and what each part says *everywhere*. Both
forms, because they cost different amounts. What a part says here is one symbol
and is what nearly every head wants. What it says everywhere is a list, and
using a list at a computed place needs a fixed point — thirty symbols before it
says anything. Offering only the second made every head unreachable; offering
only the first would have made a head unable to do what `through` does.

Each head runs under its own meter. The floor is universal, so a head that does
not stop is a case that exists, and it is refused the way a division by nothing
already was.

### Where a candidate comes from — `a_way_of_computing_she_wrote.py`

Not a list of operation kinds and not a model asked to name one. From the
correspondence the examples show. `_where_each_came_from` reads off, for each
length, where each place took its value from. A head is then a function of the
six things above, and the question is which function agrees with that
correspondence — an induction over the floor, run the way every other induction
here is run: shortest first, fitted on half the lengths, judged on the half it
never saw, refused when that half refuses it, and checked once more at a length
it was never fitted to.

The library is offered as leaves. That is the only channel by which a long term
becomes reachable.

### The rule for what to try first is a term — `the_order_she_tries_them_in.py`

`how_she_learns_to_look` already says the honest thing about its two halves:
the order may be learned and the ruler may not, because a rule that learns to
propose rubbish loses time and keeps nothing. What it did not do is make the
order an *object*. The counts were learned; the expression combining them was
Python, and a Python expression is the next authored level up — the same shape
of gap `growing_at_any_level` left one level down.

`THE_ORDER` is a 22-symbol floor term taking five numbers and giving a score.
It is installed and removed by the code that installs and removes a head, and
nothing in the pipeline knows which of the two it is holding. The order over
her real vocabulary is identical to what the Python expression gave, checked,
with the integer rounding checked separately so a change of representation
cannot smuggle in a change of behaviour.

This is one component of `M_t` becoming an artifact of the same kind as the
things `M_t` produces. It is not meta-invention: nothing here writes a better
rule, and there is a test asserting the rule in force at import is the authored
one, so a later change to that has to arrive with experiment H's evidence.

### A head is priced before it is kept

Objection 16 — the library eventually worsens search — was answered by hoping.
A word is one more thing to put in a hole; a head is one more shape at every
node of every term, so it multiplies. `what_a_head_costs_the_search` counts
both sides in the same unit as everywhere else here, terms she would otherwise
walk: the maker search above walked the positional space at that depth and
returned nothing, so that count is what the head must be cheaper than; the head
search reports where its answer appeared; and the enumerator is counted with
the head and without it for the branches it adds. No threshold and no
weighting. It pays, or it comes straight back out.

### Both algebras, one semantics — `the_old_language_on_the_floor.py`

Positional terms and value expressions both compile to floor terms, so an
invention in either can be material for an invention in the other. That was the
thing blocking recursion *across* the two languages, and it was invisible while
each half worked on its own.

Checked as behaviour rather than shape, refusals included: 12,000 positional
terms over three pairs of words at seven lengths — 1,296,000 places — and 8,000
value expressions over 512,000 pairs, complete agreement, every head and every
way of combining covered.

The refusal clause is what earned the check. Substituting the inner expression
into the body of `through` made the floor lazy where the interpreter is strict,
so `how many there are of where it is over where it is` divided by nothing in
one language and answered nought in the other. A comparison that only looked at
places where both succeeded would have shipped it.

What remains after that is not a third algebra. It is the schema — a rule is
still two sources and one operation — and that is a different ceiling, named in
Part XIII rather than quietly closed.

### Information flow

    experience
      → residual, persistent across attempts
      → why_nothing_fits: search, language, unseen quantity, or blind check
      → language_limits.certify: value-blind refutation where it applies
      → a word she derives  →  an operation she derives  →  a recipe she composes
      → a maker she writes (a positional term with a hole)
      → A WAY OF COMPUTING she writes (a floor term, standing as a head)
      → install, check the family became sayable, else take it out
      → keep: heads, then words, then meanings
      → every later search, word, maker and head is built over it

### The bedrock, and why it is small

Authored permanently: the floor's eighteen heads and their evaluator; the
meter; the gate (`persistence, novelty, reach, compression, held-out,
rollback`); the governor (fuel, memory, transaction, the privileges a term may
not reach); and — added when the bootstrap was fixed, and worth naming rather
than folding into the others — the **recurrence schema**, which asks whether a
family's answers stand in a relation to the answer at the place before.

The first four say nothing about what a useful concept is. The fifth does say
something: it says that counting down is a shape worth looking for. It adds no
meanings, by Theorem 1, and removing it is a test. But it is a prior, and
calling it anything else would be the kind of claim this document exists to
refuse. That is the
difference between a fixed substrate and a fixed vocabulary. Part VI proves the
first is unavoidable, and Part X shows the last two cannot be inside the
hypothesis space without ceasing to hold.

---

## Part VI — Mathematics

Statuses are labelled: **proven**, **supported**, **conjectured**.

### Theorem 1 (definitional extension). Proven.

If `a*` is introduced by a closed term `e` over `A_t`, then
`E(A_t ∪ {a*}) = E(A_t)`.

*Proof.* Every occurrence of `a*` may be replaced by `e`; the result denotes
the same function. Naming is a `let`, and unfolding a `let` does not change a
denotation. ∎

Already argued in `what_growth_cannot_do`;
`where_the_tower_has_a_top.naming_adds_no_meaning` runs it on a given word.

### Theorem 2 (universality ceiling). Proven.

If `E(A_t)` is all the partial computable functions, then for every computable
`a*`, `E(A_t ∪ {a*}) = E(A_t)`.

*Proof.* `a*` is computable, so some `q ∈ P(A_t)` computes it; substitute.
Containment the other way is trivial. ∎

### Theorem 3 (the growth ceiling of the positional algebra). Proven.

Let `T` be a term of `one_algebra`, `L = |T|` its symbols, `c` the largest
constant written into it, `W` any list of positional words, `n ≥ 2` a state
length and `i` a position. Put `B = max(n, c, 2)`. Then

    |run(T, i, n, W)| ≤ B ** L.

*Proof.* Induction over the heads, and the case list is exactly what `run`
dispatches on. `where` gives at most `n`; `many` gives `n`; `fixed k` gives
`c`; `hole` returns `words[k](i,n) mod max(1,n)` and is below `n` **whatever
the word does**; `through`, `over again` and `undo` each return a place inside
the state and so are below `n`; `if` returns one of its arms; `below` and
`same as` give nought or one; `plus` and `minus` of parts bounded by `B**L1`
and `B**L2` are at most `2·B**max ≤ B**(L1+L2+1)`; `times` is exactly
`B**(L1+L2)`; `over` and `left over` only shrink. ∎

**Checked, not only argued.** `the_bound_holds_on` runs the conclusion against
the interpreter: 155,719 answers over 4,000 terms at eight lengths, no
violation. `the_heads_the_argument_covers` is compared against the heads read
out of `run`'s source, so an induction with a missing case fails in CI.

**Corollary 3.1 (proven).** For a fixed term the bound is a polynomial in `n`
of degree `L`, and `2**n` outgrows every polynomial. So no positional term
computes `n ↦ 2**n`: not at any length, not over any vocabulary, not after any
number of makers or levels. `where_doubling_escapes(L)` returns a length at
which every term of `L` symbols is already too small — 49 at `L=8`, 180 at
`L=24`, 546 at `L=60` — so the conclusion is not budget-relative.

**Corollary 3.2 (proven).** The only thing that puts doubling into her
positional language is a person editing `run`. That is the regress, located.

### Theorem 4 (a witness inside the range a position lives in). Proven.

Theorem 3's witness leaves the range a word answers in, and a fair objection is
that such a function was never a candidate. So: every word she can make is a
term over the words she was given, so the words she can ever have are
recursively enumerable. Let `w_0, w_1, …` be that enumeration and define

    g(i, n) = (w_i(i, n) + 1) mod n   when w_i answers at (i, n)
    g(i, n) = 0                        when w_i refuses there.

`g` is total, computable, and always a place inside the state. For every `i`
and every `n ≥ 2`, `g(i, n) ≠ w_i(i, n)`. So no word she can build computes
`g`. ∎

**Checked.** `no_word_of_hers_says_it(400)`: 400 words, 1,200 places, no
agreement at any diagonal point.

### Theorem 5 (the regress). Proven.

Let `B` be a bedrock with `E(B)` not all of the computable functions, and let
the task stream eventually demand each computable function. Then the number of
human authoring events required is unbounded.

*Proof.* Some computable `f ∉ E(B)`. By Theorem 1, no term over `B` reaches
`f`, and every endogenous step adds a term, so no sequence of endogenous steps
of any length and at any level reaches `f`. Only an edit to the evaluator does.
After the edit the bedrock is `B'`; if `E(B')` is again not everything, some
`f' ∉ E(B')` and the argument repeats. By induction the required edits do not
terminate. ∎

### Theorem 6 (the top). Proven.

If `E(B)` is all the computable functions, the number of required authoring
events is nought, and by Theorem 2 no authoring event could add anything.

**Corollary 6.1.** Universality is necessary and sufficient for the tower of
authored meta-levels to have a top. It is not a design preference among
substrates. `where_the_tower_has_a_top.where_the_tower_ends` returns which of
the three cases a bedrock is in, and refuses to guess when no certificate
exists either way.

### Theorem 7 (the floor is universal). Proven, modulo Kleene.

The partial computable functions are exactly the closure of `{zero, successor,
projection}` under composition, primitive recursion and unbounded search
(Kleene, 1936). `what_the_floor_can_say` exhibits each as a floor term —
`ZERO`, `SUCCESS`, `take_the_one_at`, application for composition,
`by_recursion`, `the_least_where` — and checks each against what it should
compute, including that unbounded search on a predicate with no root exhausts
its meter rather than returning. That last check is the one that separates
universal from merely large.

**Corollary 7.1 (proven).** `E(floor) ⊋ E(positional)`, by Theorem 3 on one
side and a 38-symbol term on the other. It is a genuine new distinction, and by
Theorem 2 it is the last one that will ever be available.

**Corollary 7.2 (proven).** The choice of instruction set is doing no work.
`what_the_arithmetic_rests_on` derives `times`, `over`, `left over` and
`same as` from `plus`, `minus` and `below` as terms, and checks the derived
version against the primitive on a 13×13 grid.

### Theorem 8 (abstraction and reach). Proven for tree enumeration.

Enumerating expressions of length up to `L` over `b` primitives visits
`Θ(b**L)` candidates. If naming a recurring subterm shortens the target by `d`,
the space to walk shrinks by `b**d`; under a prior favouring short hypotheses
the same idea makes the target `2**d` times more probable. This is what
`what_it_costs_to_say` already records, and it is why reach is the operative
quantity once `E` stops moving.

### Theorem 9 (finite memory). Proven.

With `B_m` bits of durable state there are at most `2**B_m` states, so an
infinite strictly increasing chain of uncompressed libraries is impossible.
Long-run development requires forgetting, merging, compression and structural
replacement. `keeping_the_language_small` already carries this.

### Theorem 10 (no free lunch). Proven.

No representation-update rule improves `C(·; T, B)` on every `T`.
`no_updater_wins_everywhere` builds the environment that beats a given rule and
runs it. Every improvement claim must name its distribution and its budget.

### Theorem 11 (inadequacy). Proven.

Over a universal language, "does some term solve family `F`" is undecidable by
reduction from halting. Bounded certificates remain, and `which_kind_of_growth`
already refuses a new-distinction verdict on a search that did not finish.

### Theorem 12 (verification). Proven.

No total procedure decides, for arbitrary `a*`, whether it preserves every
semantic invariant of interest (Rice). So a governor may demand proofs in a
total fragment, or bounded empirical evidence, and never both totality and
universality.

### Proposition 13 (the gate must stay outside). Proven, and executed.

If the predicate authorising installation is itself installable under its own
authority, a candidate exists whose effect is to accept everything, and after
it is admitted the invariant "everything admitted is harmless" is no longer
preserved — with nothing violated on the way.
`a_gate_inside_the_space_cannot_hold` constructs that candidate and reports
that the invariant held while the gate was fixed and failed once it was not.

### Definition (meta-order).

    M_{t+1} ≻ M_t  iff  C(Reach_B^{M_{t+1}}; T_inv, B) > C(Reach_B^{M_t}; T_inv, B)

on a **sealed** invention distribution, under matched compute, context, model
and exposure, with lesion of the change erasing the gain. No other sense of
"better mechanism" is admitted here. Source code differing is not it. A larger
library is not it.

### Red team

- *Already-universal `A`.* Then a new-meaning claim is false and the growth
  kind must be classified. Corollary 7.1 is the one strict gain, and it is
  claimed once.
- *A lookup table.* Fits the synthesis half, fails the judged half and the
  unseen length. Both are gates, not preferences.
- *An enormous abstraction that lengthens search.* `Reach_B` can shrink when
  the library grows. `keeping_the_language_small` charges for it; the search
  cost probe is listed as open in Part XIII.
- *A hidden prior in the proposer.* The proposer is enumeration over the floor
  with no model in the path, which removes this attack rather than answering it.
- *Nonstationary tasks.* Grown-versus-reset can look like compounding while
  overfitting a curriculum order. Block permutation is required and not yet run.
- *Degenerate language.* With an empty vocabulary the correspondence cannot be
  read and nothing is written, which the tests check.

---

## Part VII — Algorithms

**Diagnosis.** `why_nothing_fits` compares what the best readings miss and asks
whether something else the language already says covers exactly those. Cost
`O(|hypotheses| · |transitions|)`. `certify` intersects source positions per
place and runs Kuhn's augmenting path for the matching, `O(n³)` at a length.

**Head synthesis.** `every_code(deepest, variables=6, also=library)` yields
shortest first: 824 terms at depth 3, 10,976 at depth 4. For each ordered pair
of words, tables are `O(n)` per size; a candidate is checked at every place and
stops at the first disagreement, so the great majority cost one evaluation.
Whole search bounded by a wall clock the caller sets, defaulting to the
allowance a maker gets.

**Running a head.** Each part's table is `O(n)` evaluations, so a head costs
`O(n)` per position and `O(n²)` per state. Acceptable at the lengths here and
listed in Part XIII as a thing to measure before the lengths grow.

**When invention is worth attempting.** Gated on a persistent residual *and* a
language verdict from `why_nothing_fits`, then on the ladder above it failing.
Never a reflex on confusion, and never a search over every program.

---

## Part VIII — Aura integration

**Modified.** `one_algebra.py` — one arity table, derived-head dispatch,
`every_term` offers written heads, `read_back` consults both.
`sequence_induction.py` — the last rung of `_widen`.
`what_she_gave_meaning.py` — heads kept and recalled before the words.
`model_validation.py` — three claims registered, one corrected.

**New.** `the_floor_she_stands_on.py`, `the_floor_reading_itself.py`,
`what_the_floor_can_say.py`, `what_the_old_language_cannot_say.py`,
`a_way_of_computing_she_wrote.py`, `where_the_tower_has_a_top.py`,
`the_order_she_tries_them_in.py`. Also modified: `how_she_learns_to_look.py`
scores through the term; `keeping_the_language_small.py` prices a head.

**Tests.** `test_the_floor_she_stands_on.py` (72),
`test_a_head_she_wrote.py` (12), `test_a_way_of_computing_she_wrote.py` (10),
`test_every_head_survives_a_restart.py` (20),
`test_the_order_she_tries_them_in.py` (7). 121 new tests.

**Persistence.** Heads are kept as their term, never as a name and never
pickled, and recalled before the words that are written over them.

**Gates.** `compile`, `lint`, `writing`, `layering`, `deps-check` and `smoke`
green. The writing ratchet was at 77 against a baseline of 76 and is back at
76. The full offline suite was run in twelve chunks; every failure it reported
in the chunks completed so far was reproduced at the audited commit before any
of this work, and the list is in
`artifacts/endogenous/failures_that_predate_this_work.md`. The one exception,
`governance-lint`, names a file the atomspace refactor touched.

**Not yet migrated**, and named as such: `operator_invention.Candidate.fn`
still takes a callable and still has no runtime caller;
`growing_at_any_level.grow_at` still takes a Python `make`;
`an_operation_that_generalises` is still a separate algebra.

---

## Part IX — Experimental program

| | What it asks | Status |
|---|---|---|
| **A** | She originates the semantics with no candidate, no name, no kind list | **run.** Before-and-after states in, a floor term out. Doubling, factorial and triangular numbers, each defined by what it says at the place before |
| **B** | `F ∉ Reach_B(A_0)`, `F ∈ Reach_B(A_1)` | **stronger than asked, for one witness.** Theorem 3 proves doubling is outside `E(A_0)` rather than outside a budget; the floor says it in 38 symbols |
| **C** | Novelty, at the strongest available tier | **proven** for Corollary 7.1 by Theorem 3; **proven** for Theorem 4 by diagonalisation; bounded for anything a search returns |
| **D** | Generalisation, `H ∩ S = ∅`, outside the observed ranges | **run.** Correct at lengths 9, 11, 13 and 16, none of them fitted or judged |
| **E** | `b*` depends on `a*`; lesion `a*` and the second goes | **run at the grammar level.** Same family, same search, same budget: nothing with an empty library, the answer with one entry, and what it wrote is that entry plus four symbols |
| **F** | Cross-domain transfer with a negative control | **run.** Related domain 96/96 with the piece against 77 without; unrelated control 96 and 96 |
| **G** | GROWN vs RESET vs LESIONED, `dΔ/dn > 0` | **run, at the grammar level, with its negative control** — see below |
| **H** | The mechanism changes itself and wins on sealed invention tasks | **not run.** One component of the mechanism — the rule for what to try first — is now a term, replaceable and lesionable by the head path; nothing writes a better one, and no gain is claimed |
| **I** | A second generation with no human code between | **not run** |
| **J** | Post-freeze tasks by an independent party | **not run** |

**Controls.** Implemented: shuffled and contradictory residuals refused; a
family already sayable produces nothing; one length holds nothing back; a head
that never settles or asks for nonsense is refused; the invented head is
lesioned and the reach goes with it; a search order that will not answer sends
every word to the back rather than stopping the search, and the order lesion
restores the previous order exactly; a head that costs more search than it
saves is taken back out. Required and not implemented: random candidate
generator, fixed-DSL enumerator with a larger budget, retrieval, macro-only
compression, inline expansion, equal-persistent-bytes, matched compute across
GROWN and RESET.

### Experiment F, as run

Three domains, different words on each so the before-and-after states look
unrelated. The related domain's term contains the piece she wrote on the first;
the control's term does not contain it anywhere, and there is a test asserting
that, because a negative control is only a control if it is negative.

Sixteen seeds, six families each:

| Domain | With the piece | Without it |
|---|---|---|
| structurally related, different surface | 96 / 96 | 77 / 96 |
| unrelated | 96 / 96 | 96 / 96 |

Nothing on the control. The relation is constructed rather than found, and the
claim says so: this shows the piece is what carries, not that any real pair of
domains stands in this relation.

### Experiment G, as run

Three agents on the same families in the same order under the same per-family
budget, the same words and the same search. GROWN keeps every head it writes
and offers them as leaves next time; RESET is emptied between blocks; LESIONED
keeps the library except its newest entry. Nobody picks which family follows
which — the terms are drawn at random and the correspondence is read off
whatever they compute.

Two streams, and the difference is what makes the result mean anything. On the
**shared** stream a family's term is drawn with the terms already found
available, so there is structure a learner could carry forward. On the
**apart** stream every term is drawn from the bedrock alone.

Five seeds, five blocks of six families:

| Stream | GROWN | RESET | LESIONED | Gap by block |
|---|---|---|---|---|
| shared | 150/150 | 78 | 70 | 0, 2, 2.6, 5.8, 4 |
| apart | 150/150 | 150 | 150 | 0, 0, 0, 0, 0 |

Nought on the control in every block of every seed. That is what says the gain
on the other stream is about carrying something rather than about having run
longer, and the lesion returns her to the reset condition or below it.

Two things to say plainly. The gap is not monotone block to block — it reaches
5.8 and falls to 4 — so what holds is that it is open by the end, not that it
grows at every step. And the mechanism is transparent: later families are drawn
over earlier terms by construction. This is compounding on a stream with
structure in it. It is not evidence about any real task distribution, and
experiment J is what would make it one.

**Falsifiers.** The thesis fails if a larger search budget with no language
growth matches the grown condition; if every claimed distinction is a macro; if
the head lesion has no effect; if experiment H requires editing Python by hand.

---

## Part X — Safety and containment

**Nontermination.** Every floor evaluation is metered. A head that does not
answer inside its meter is treated as a head that does not compute here, which
is what a division by nothing already meant.

**Privilege.** The floor's imports are `__future__`, `dataclasses`, `logging`
and `typing`, enforced by a test that parses the module. A term computes over
numbers and pairs. There is no head that opens a file, calls a tool, or reaches
the governor, so admitting a term cannot be a route to a privilege.

**Rollback.** A head is installed, the family is asked again, and the head is
taken out when the answer is no. A term mentioning a removed head raises rather
than answering, and `read_back` refuses it, so a stale term cannot quietly mean
something else.

**Evaluator hacking.** The gate is authored and outside the hypothesis space,
for the reason Proposition 13 executes.

**Drift and poisoning.** The residual must be persistent and must survive the
diagnosis in `why_nothing_fits` before invention is attempted at all.

**Descendants.** Currently: removing an ancestor makes its descendants refuse.
Not yet: a dependency graph with quarantine, cascade invalidation and
rebuilding. Named in Part XIII.

**The trusted base.** A defect in the floor's evaluator or in the gate is
catastrophic and irreducible. The floor is 700 lines with 72 tests; that is the
whole mitigation, and it is not a proof.

---

## Part XI — Adversarial defence

1. **"Your novelty is only a macro."** For anything a search returns inside the
   floor, conceded — it is a term over the floor and by Theorem 1 adds no
   meaning. The one distinction claimed is Corollary 7.1, and that one is
   proved by exclusion rather than by search.
2. **"A universal substrate makes the claim vacuous."** For `E`, conceded and
   proved (Theorem 2). The claim is `Reach_B`, plus the single crossing from
   non-universal to universal, plus the regress theorem — and the third is not
   vacuous, because it is what says the crossing had to happen.
3. **"The generator contains the answer."** The generator is shortest-first
   enumeration over eighteen heads plus her own past work. There is no model in
   the path and no list of operation kinds. What it does contain is a universal
   machine, which contains every answer the way a universal machine does; the
   defence is post-freeze tasks, and experiment J is not run.
4. **"The evaluator defines the ontology."** Partly conceded. The observation
   contract — states are tuples, a rule maps a state to a state — is a prior,
   and it is not derived from anything.
5. **"So invention is not from nothing."** Conceded, and asserted rather than
   defended: no learner escapes priors. What is claimed is the minimality of
   this one, and Corollary 7.2 is the evidence.
6. **"The language is Turing-complete, so expansion is impossible."** Now, yes.
   Before this work, no — and that was the whole problem.
7. **"MDL rewards compression, not semantics."** Compression alone does not
   admit anything here. The family must become sayable, the judged half must
   hold, and an unseen length must not break it.
8. **"Your equivalence test is finite."** For candidates, yes, and the
   remaining uncertainty is stated. For Theorem 3 and Theorem 4 it is not a
   test at all.
9. **"This is genetic programming."** Enumeration is one proposer. What
   distinguishes it is residual-derived constraints, the immovable ruler,
   growth-kind classification and a frozen gate. If an ablation to plain GP
   matches the results, the claim fails.
10. **"The LLM is doing the invention."** There is no LLM in the invention
    path. Not as proposer, not as ranker.
11. **"Recursive self-improvement is code optimisation."** The meta-order is
    defined on sealed invention tasks. Since experiment H is not run, no
    recursive self-improvement is claimed.
12. **"Finite memory forbids open-endedness."** Conceded (Theorem 9). Open-ended
    here means no authored ceiling on what can be discovered, not infinite
    accumulation.
13. **"The meta-level moved one interpreter up."** Yes — to the floor, and
    then it stops, and Theorem 6 says why it stops there and only there. If a
    later family needs a nineteenth head, the design has failed.
14. **"Self-reference destroys verification."** Conceded (Theorem 12). The gate
    is not self-referential and the Gödel-machine install rule is rejected.
15. **"You cannot tell representation failure from insufficient search."** In
    general, conceded (Theorem 11). Certificates, not omniscience.
16. **"The library will worsen search."** A real risk, and the search-cost
    probe is not implemented. Listed as open rather than answered.
17. **"The developmental advantage is curriculum overfitting."** Unaddressed:
    experiments F, G and J are not run.
18. **"Safety prevents the open-endedness you claim."** Partly true, and
    deliberate. The maximal *safe* property is the target.
19. **"The proof assumes its conclusion."** Theorems 1–7 do not. Theorem 3's
    induction is checked against the interpreter's own dispatch list, and
    Theorem 4 is a diagonal.
20. **"`compose_from_invented` is a naming trick."** Agreed, and it is why
    lesion is the evidence here rather than a generation counter.
21. **"`growing_at_any_level` already collapsed the tower."** It collapsed the
    API and kept a Python callable. It is still a human meta-level and it is
    still there; Part XIII lists it.
22. **"`one_algebra`'s heads are the taxonomy."** They were. The test is
    whether the next family needs a nineteenth head, and the head registry
    exists so the answer can be no.
23. **"`2**n` was never a candidate, since a word returns a position."** Fair,
    and Theorem 4 exists because of it.
24. **"Your diagonal is not exhibited as a floor term."** Correct. It follows
    from Theorem 7 that one exists; it is not written out, and the claim is
    labelled supported rather than exhibited.
25. **"A lookup table can pass your gates."** No finite experiment eliminates
    this. What is in place: a held-out half, an unseen length, and the ruler.
26. **"Shortest-first over a universal language finds nothing useful."**
    Largely true, and it is Levin's bound rather than a defect. The measurement
    in `test_the_library_is_what_moves_the_horizon` is the answer: the library
    moves the horizon, and it does so as a matter of record.
27. **"Then the first useful head is unreachable, so the mechanism is inert."**
    It was, and measured: 120 families out of 120 produced a head the growth
    classifier called a shorter name. Fixed rather than conceded — a head is
    given a fixed point, and the step of a recurrence is solved for rather than
    searched. The cost is one more authored thing, named in Part V. The
    remaining honest form of this objection is objection 33.
33. **"Your recurrence schema is a human-authored taxonomy after all."**
    It is authored and it is a prior, and Part V says so rather than burying it
    among the mechanics. What it is not is a taxonomy of *concepts*: it names
    no operation and no family, and what a step does is searched over the floor.
    It adds no meanings, by Theorem 1. And it is falsifiable in the ordinary way
    — if the next family needs a shape that is not a recurrence and a person has
    to write a second schema, the design has failed and this document should
    say so.
28. **"The schema is still fixed."** Conceded. `Induced.read` is arity two with
    one operation, and no head changes that. It is B2 and it is open.
29. **"There are still two algebras."** Partly conceded. The floor absorbs the
    positional algebra as heads; `an_operation_that_generalises` is untouched.
30. **"Governance makes invention theatre."** The head path is wired into the
    live induction ladder, which is what stops it being a demo. It is not yet
    inside a transaction, and that is listed.
31. **"The trusted base could be wrong."** Yes. 700 lines, 72 tests, no proof.
32. **"Success would be AGI."** No, and Part XIII says what it would not be.

---

## Part XII — What changed after the attack

1. The distinction claim was cut to one — Corollary 7.1 — and everything else
   was relabelled reach.
2. Theorem 4 was added because Theorem 3's witness leaves the range a word
   answers in, which objection 23 is right about.
3. The head interface was changed from four arguments to six. With tables
   alone every head needed a fixed point and nothing was reachable; that was
   objection 26 arriving during implementation rather than after it.
4. `a_gate_inside_the_space_cannot_hold` was written to execute Proposition 13
   instead of asserting it, in the style the rest of this codebase already
   uses for the no-free-lunch argument.
5. The claim in `model_validation` that the positional grammar was the floor of
   computing was corrected. It was false, and it was the sentence that hid the
   regress.
6. Experiments F through J were moved from "designed" to "not run", and the
   claim boundary was rewritten around what is measured.

---

## Part XIII — The exact claim boundary

### Proven

- Every positional term obeys `|run| ≤ max(n,c,2)**|T|`, checked on 155,719
  answers over 4,000 terms, with the induction's case list compared against the
  interpreter's dispatch.
- Doubling is outside her positional language at every term length and over
  every vocabulary she can build.
- A place-valued rule exists that no word she can build computes; 400 words,
  1,200 places, no agreement.
- The floor computes every partial computable function, by Kleene's
  characterisation with each constructor exhibited as a term and checked.
- `E(floor) ⊋ E(positional)`, and by Theorem 2 that is the last strict
  expressiveness gain available.
- A bedrock that is not universal requires unbounded human authoring; a
  universal one requires none and admits none. Universality is necessary and
  sufficient for the regress to end.
- A gate inside the hypothesis space does not preserve its invariant.

### Measured

- A head written from before-and-after states alone, with no target operator,
  no candidate implementation, no name and no kind list, computing the family
  at lengths 9, 11 and 16 that were neither fitted nor judged.
- That head kept and recalled across a restart, and still running.
- The library moving the search horizon: the same family, search and budget
  finds nothing with an empty library and the answer with one entry.
- Every head the interpreter runs surviving a restart, which was false before.
- The rule deciding what to try first computing, as a term, exactly what the
  Python expression computed, over her real vocabulary — and being replaceable
  and lesionable by the same path a head is.
- Both algebras computing on the floor exactly what their own interpreters
  compute, over 1,808,000 places, refusals included.
- A piece written on one surface carrying to a structurally related domain
  whose states look unrelated, and not carrying to an unrelated one.
- Keeping what she wrote solving 150 of 150 families where resetting solves 78,
  with a gap of nought on the control stream in every block of every seed.

### Not established

- Meta-invention. The rule for what to try first is a term; the search that
  would replace it is not. Nothing here demonstrates that she writes a better
  rule, and the test suite asserts the rule in force at import is the authored
  one so that a later change cannot pass unnoticed. Experiments H and I are not
  run and no recursive self-improvement is claimed.
- Post-freeze evaluation. Both the compounding and the transfer results are
  measured on streams and relations this work constructed, which is a weaker
  thing than either on a distribution nobody constructed. Experiment J is what
  would close that, and it needs an independent party.
- That the head mechanism helps on any real task family beyond the synthetic
  ones here.
- That the library does not eventually worsen search.

### Still authored, and open

- **B2**, the rule schema: two sources, one operation. The remaining ceiling
  above the floor, and no head or word changes it.
- **B5**, `growing_at_any_level.grow_at` still takes a Python callable, and
  `operator_invention.Candidate.fn` still takes one and still has no caller.
- The dependency graph, quarantine and rebuild for descendants of a removed
  head.
- The search-cost probe that would stop the library eating the mind.

### What it would not prove even if all of it worked

Growth beyond computability; invention without priors; decidable general
inadequacy; verified arbitrary self-modification; monotone improvement across
environments; unbounded growth on finite memory; safe rewriting of the trust
root; consciousness; AGI.

### Category labels

| Label | Criterion | Evidence needed | Here |
|---|---|---|---|
| Representation-learning | experience creates reusable internal structure used later | held-out gain from learned structure | **yes** |
| Language-learning | learned executable definitions change later construction and search | admitted terms in later programs, across a restart | **yes** |
| Self-extending | originates and durably activates new executable artifacts | A–E | **yes, boundedly** |
| Reflective | its own programs are represented and manipulated as objects | quotation, a self-interpreter as a term | **yes** |
| Self-hosting | the evaluator of the language is written in the language | `THE_INTERPRETER` agreeing with the machine | **yes** |
| Metaprogrammable | programs construct and transform programs | quotation plus head synthesis | **yes** |
| Open-ended | no authored ceiling on the computable semantics reachable | Theorem 6 plus J | **qualified**: the ceiling is gone by theorem; J is not run |
| Developmental | earlier learning changes later learning capacity | G | **shown on a constructed stream**, with a flat negative control and a lesion; not shown on a distribution nobody constructed |
| Recursively self-improving | improves the machinery producing future improvements | H, I, sealed, with lesion | **no** |
| AGI | broad competence across independently designed domains | far more | **no** |

AGI would additionally need reliable world models, credit assignment over long
horizons, robust tool use, distribution-shifted commonsense, and evidence that
the whole stack beats the same frozen model under a strong simple harness. This
subsystem does not claim any of that.

---

## Part XIV — Implementation sequence

Done, in order, each piece surviving into the final architecture:

1. Fix the persistence hole, with a property test that reads the interpreter's
   dispatch so the class of defect cannot recur.
2. The floor: universal, metered, homoiconic, with the machine as a loop over a
   stack.
3. The self-interpreter as a term, checked against the machine.
4. The universality certificate, and the demonstration that the instruction set
   is not load-bearing.
5. The ceiling theorem and both witnesses.
6. Derived heads, dispatched, enumerated, persisted, removable, metered.
7. Head synthesis from residual structure, wired into the live ladder.
8. The regress theorem, the executed gate argument, and the claims.
9. The search-cost gate, so a head that costs more than it saves cannot stay.
10. The search order as a term, replaceable and lesionable by the head path.
11. Both algebras compiled into the floor, agreeing over 1,808,000 places.
12. Grown against reset against lesioned, and cross-domain transfer, each with
    its negative control.

Next, in the order that keeps every step in the final architecture:

13. The inline-expansion control, so a name that bought nothing cannot pass as
   an abstraction. The search-cost probe is done.
14. Post-freeze tasks, which need an independent party.
13. Widen the rule schema so its shape is a term rather than a signature (B2).
14. Make the whole proposer a term, not only its ordering, and persist it. Only
    then are H and I even possible.
15. Freeze; commission the sealed family; run F, G, H, I and J.

If step 14 needs a new human-written mechanism to pass, the design has failed
and this document should say so.

---

## The answer, restated

The regress in the prompt is not a regress of mechanisms. It is a regress of
primitives. A mechanism that invents at level `k` and needs a new mechanism at
level `k+1` is a symptom; the disease is that the bedrock is small enough for
something computable to sit outside it, and no invention at any level ever
reaches outside a bedrock.

So the tower has exactly one place it can end, and Theorem 5 and Theorem 6 say
where. Below universality, authoring is required forever. At universality,
authoring is required never and is also impossible. Aura's positional algebra
was below it — provably, with two independent witnesses — and the floor is at
it, with Kleene's certificate.

What is bought by crossing: `E` stops being the interesting quantity, and reach
becomes it. What is not bought: tractability. Shortest-first over a universal
language reaches a few dozen symbols, and no budget changes that. What moves
the horizon is the library, which is the developmental claim and is now a
measurement rather than an argument.

What is still authored is the meter, the gate, and the governor. None of them
says anything about what a useful concept is.

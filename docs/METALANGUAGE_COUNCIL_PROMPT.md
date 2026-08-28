# The metalanguage problem — a prompt for a council

Paste everything below the line. It is self-contained: no repository access
needed. Written 2026-08-28.

---

## The problem

I have a system that learns executable programs from observations, and I want
to know how to let it extend the *language those programs are written in* —
not just the library of programs inside it.

Here is exactly what exists, so you can reason about specifics rather than
about the idea.

### The setup

A world is a set of transitions. A transition is `(before, after)` where both
are tuples of equal length. Cells can be anything hashable — integers, strings,
tuples. Examples:

```
(0,1,2,3) -> (3,2,1,0)
(0,1,2,3,4) -> (2,3,4,0,1)
```

The system infers an executable program that reproduces every observed
transition and predicts held-out ones. A program is a rule over positions:
`after[i] = before[f(i, n)]` for a state of length `n`.

### The metalanguage

`f` is drawn from this basis, generated per state length:

| form | meaning |
|---|---|
| `identity` | `f(i) = i` |
| `mirror` | `f(i) = n-1-i` |
| `offset k` | `f(i) = (i+k) mod n`, for every k |
| `exchange a b` | swaps two absolute positions, for every pair |
| `ends d` | swaps the cells `d` in from each end (length-relative) |
| `grouping span first` | deal positions into `span` residue classes, lay them end to end, class `first` leading |
| `compose A B` | `f(i) = A(B(i))` |

Plus a value side: constant, additive offset, and a substitution table
(refused unless some value recurs, so a table isn't a transcript).

Programs are structured values, not closures: `IndexProgram(kind, args, parts)`
that interprets itself, compares by value, and serialises to JSON.

### What works

1. **Solving.** Given transitions, it enumerates forms that fit *every*
   observation (using the set of all possible source positions per cell, so
   repeated values never force an arbitrary tie-break), prefers the simplest
   description, and validates on held-out transitions.

2. **Library learning.** A solved program is admitted as a *member* of the
   search language, so later worlds can compose with it. This is what makes
   depth grow: a three-transformation world is unreachable from the basis
   however many observations you offer, and reachable once a two-transformation
   shape has been learned somewhere else.

3. **Refactoring.** Solutions are decomposed into their component sequences,
   and a contiguous run appearing in several solutions but equal to none of
   them is admitted as a new library entry, chosen by
   `(occurrences - 1) x length`. This is DreamCoder's compression step. It
   demonstrably reaches worlds the library-of-winners cannot.

4. **Persistence.** The library survives a restart, including refactored
   entries.

### Measured, on a fingerprinted 120-problem battery

No language model, embedding, or stored answer anywhere in the scoring path.

| condition | solved |
|---|---|
| everything | 111 / 120 |
| problems the language can express at all | 110 / 110 |
| without composition | 91 / 120 |
| without the learned library | 84 / 120 |
| without the frequency prior | 111 / 120 |

The prior contributes nothing measurable on this battery and that is reported
rather than hidden. Ten of the 120 are deliberately outside the language
(ordering cells by a property of the values), and score ~0.

Scored equally across five representations — integers, words, colours, records,
and grids whose cells are themselves tuples — although it was written for
integers.

## The boundary I want to cross

Everything above is **learning programs inside a fixed metalanguage**. The
primitives — position, offset, exchange, grouping, composition, value maps —
were chosen by a human. Adding "grouping" to reach one failing family was a
human proposing a missing inductive primitive, predicting what it would fix,
and being right on the second attempt.

The system cannot look at a family it keeps failing and invent a primitive
*kind*. It can only compose and reuse within the kinds it has.

## What I already know, so please don't re-derive it

- **Unconstrained predicate invention is intractable.** Thirty years of ILP
  says so; it is under-specified (how many arguments? what order?) and early
  systems abandoned it. Metarules — higher-order schemata defining the
  hypothesis space — are what made it workable (Metagol, Popper). So
  "let it invent anything" is a known dead end, not an unexplored one.
- **DreamCoder grows the library, not the basis.** Its wake-sleep loop
  refactors solutions into new abstractions, all expressed in the original
  primitives. I have implemented this and it works. It is not the same as
  extending the primitive set.
- **Chollet's point stands:** unlimited priors let a developer buy skill and
  mask the system's own generalisation. Grounding priors in Core Knowledge
  (objects, agents, number, geometry) is more principled than choosing them by
  taste — and my basis covers geometry and number and omits objecthood, which
  correctly predicted which families fail.
- **Soar and ACT-R report the ceiling:** in long-term studies of both, symbolic
  learning eventually *stopped*. Accumulation saturates.

## The question

**How does a system extend its own metalanguage — acquire a new primitive
*kind*, not a new composition of existing kinds — while search stays
tractable and the new primitive is validated rather than assumed?**

Concretely, I want an answer to at least one of these:

1. **Detection.** How should it know that its metalanguage is insufficient,
   as opposed to the current problem simply being hard? "No form fits" is
   true for both. What signal distinguishes "I need a new kind of primitive"
   from "I need more observations"?

2. **Proposal.** Where does a candidate *kind* come from, if not from a
   human? Plausible directions I have not tested: mining the residual
   structure of failures; a schema over programs rather than over positions
   (metarules one level up); inducing the primitive from an invariant the
   failures share; deriving it from a generative model of the environment
   rather than of the transition.

3. **Tractability.** Any answer to (2) enlarges the search. What keeps that
   bounded? Description length is the obvious lever and I already use it for
   refactoring — is it enough one level up, or does the meta-level need a
   different criterion?

4. **Validation.** A new primitive kind that explains the failures is also a
   primitive that can explain noise. Held-out transitions catch an
   over-fitted *program*. What catches an over-fitted *primitive*?

## What a good answer looks like

- A mechanism, not an architecture diagram. I will implement it.
- Falsifiable: it should predict which of my failing families it fixes, in
  advance, and predict that it fixes nothing else.
- No language model in the inference path. The point of this exercise is what
  remains when the foundation model is removed, so "ask an LLM to propose a
  primitive" is off the table as the mechanism — though it is fine as a way of
  generating candidates offline that are then validated mechanically.
- If you think the question is malformed — if extending the metalanguage
  is not the right frame — say so and say what the right frame is. That is a
  useful answer.

Concrete failing families I would want a proposal to reach, stated in advance:

- **Ordering by a property of the cells.** `(3,1,2) -> (1,2,3)` where the rule
  is "sort", and the ordering key is a property of the values rather than of
  the positions. Currently outside the language entirely.
- **Content-dependent position rules.** "Move every cell whose value is even to
  the front." Position and value interact; my basis keeps them separate.
- **Variable-length transformations.** Everything above assumes
  `len(before) == len(after)`. Insertion, deletion and filtering are outside
  it by construction.

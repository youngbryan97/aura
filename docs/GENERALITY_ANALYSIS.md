# What would have to be true, who would say so, and what is already known

Written 2026-08-28, after building the induction and transfer work and being
told — correctly — that the decomposition and the research had been skipped.

This is the part that was skipped. It is in three parts: what the goal decomposes
into at the smallest level; what six different readers would demand of it at
three scales; and what the literature already knows about the same problem. The
last column of everything is whether it exists, measured, not argued.

---

## Part 1 — The goal, decomposed downward

The goal, stated as Bryan states it: **experience → learn a new representation →
invent a reusable abstraction → apply it in an unrelated domain**, without the
ontology being specified in advance.

Working down from that to things that either hold or do not:

### 1.1 To learn a representation at all

| # | Must be true | Holds? | Where |
|---|---|---|---|
| a | The current language can be found insufficient, rather than forcing a fit | yes | `language_is_sufficient` |
| b | A candidate is constructed from observations, not chosen from named options | yes | `_possible_sources`, `_forms_that_fit` |
| c | The candidate is checked against data it was not built from | yes | `held_out` |
| d | A candidate that does not compress is refused | yes | substitution tables |
| e | Nothing is invented where there is nothing to find | yes | noise scores zero |
| f | The basis it is constructed from is principled, not "what the author thought of" | **no** | `_index_forms` — see 3.3 |

### 1.2 To make it reusable

| # | Must be true | Holds? | Where |
|---|---|---|---|
| a | What is learned outlives the world it came from | yes | `RelationLanguage` |
| b | It outlives the process | yes | `thinking_reserve.save/load`, forms still in memory only |
| c | It is a member of the language afterwards, not a preference over it | yes | `known_forms` |
| d | What can be learned grows as a result | yes | three-deep world, unreachable then reachable |
| e | Growth is by *refactoring* what was solved, not only by keeping winners | yes | `RelationLanguage.refactor` |

### 1.3 To apply it in an unrelated domain

| # | Must be true | Holds? | Where |
|---|---|---|---|
| a | The abstraction is not tied to the representation | yes | 12/12 on words, colours, records, grids |
| b | Transfer is measured against a null | yes | wrong-prior cost, no-relation null |
| c | Measured on problems the author did not enumerate | partly | generator is authored; shapes beyond it are named |
| d | Measured against the alternative (the foundation model) | **no** | D2 |

### 1.4 To discover the abstraction rather than be taught it

| # | Must be true | Holds? | Where |
|---|---|---|---|
| a | Failures are data, not prose | yes | 422 notes read |
| b | A concept is formed from what they share, by running code | yes | `formed_constraints` |
| c | It names cases nobody has seen fail | yes | 121 patterns |
| d | It points at real, unobserved defects | yes | four live read requests found |
| e | **The repair is synthesised, not hand-written** | **no** | I wrote it |
| f | The repair is validated on unseen cases and promoted automatically | **no** | — |

1.4e and 1.4f are the honest boundary. The mechanism found the defect class and
pointed at four unobserved instances of it; a person then wrote the fix.

---

## Part 2 — Six readers, three scales

### 2.1 The casual user who wants to see something impressive

| Scale | What they need | Holds? |
|---|---|---|
| Small | An answer in seconds. No wall of telemetry. Never "I couldn't get to an answer" | **no** — 3–5 minutes on hard turns, canned line still seen today |
| Medium | Something happening on screen while it thinks; formatting that reads | **no** — no token streaming on these lanes |
| Large | One end-to-end thing that is plainly impressive and shareable | partly — the diagnosis output is genuinely good when it lands |

**This persona is the worst-served, and by a distance.** Everything in Part 1 is
invisible to them. Latency is the single largest product defect in the system.

### 2.2 The heavily critical senior engineer

| Scale | What they need | Holds? |
|---|---|---|
| Small | Tests with every change; no bare excepts; no dead code; typed | yes — 44k tests, ratchets, `make` gates |
| Medium | One owner per judgement; layering; observability; no duplicated logic | partly — layering gate exists; duplicated judgements were the top defect class all day |
| Large | Can I reason about and safely change this? | **no** — `inference_gate.py` and `mlx_client.py` are both five figures of lines |

They would say: the gates are unusually good, and the two god objects undo much
of it. They would also note that today's violation count moved 246 → 110 → 24 →
20, each fall a correction of the measurement, and ask why it was published at
246.

### 2.3 The obsessive researcher

| Scale | What they need | Holds? |
|---|---|---|
| Small | Seeds, determinism, versioned baselines | yes — frozen generator, floors, ratchets |
| Medium | Nulls, held-out sets, ablations | yes — and the ablation changed the battery |
| Large | A result that is a result | partly — 77/100 with no model is a result; it needs a comparison |

They asked for the ablation and it was worth asking for. It took three attempts
to make honest — the first dropped the flag before it reached the solver, the
second showed no contribution from the library because every problem pinned its
shape unaided — and the third changed the battery. Of 120: composition is worth
20, the learned library 27, the prior nothing measurable.

### 2.4 The technically literate AI sceptic

| Scale | What they need | Holds? |
|---|---|---|
| Small | Show me the model is not in the path | yes — battery has no model, embedding or stored answer |
| Medium | Show me a held-out set you did not design | partly — generator authored, out-of-language shapes named |
| Large | Show me it handles what you did not anticipate | **no** — this is 1.1f and 3.3 |

This is Bryan's own position and it is the correct one. The answer is not to
argue; it is 3.3 below.

### 2.5 The sceptic who doubts Aura specifically

| Scale | What they need | Holds? |
|---|---|---|
| Small | Run it in front of me, now, not a test | partly — many live proofs, and the flagship case failed live four times today |
| Medium | Claims file vs reality | yes — `model_validation` refuses a claim with no test |
| Large | Pick any capability at random and it works | **no** |

They would say the tests are green and the window is not, and today they would
be right about the diagnosis case.

### 2.6 The grant evaluator

| Scale | What they need | Holds? |
|---|---|---|
| Small | A number, a method, a null | yes |
| Medium | Is the method sound and the comparison fair? | partly — no comparison to the alternative |
| Large | Is this a new capability or good engineering? | honest answer: a bounded new capability inside good engineering |

---

## Part 3 — What is already known about this problem

### 3.1 DreamCoder is the direct precedent, and names the missing half

[DreamCoder](https://arxiv.org/pdf/2006.08381) does what this work does — grows a
library of abstractions and gets better at a domain as the library grows — and
does it with a **wake–sleep** loop: solve problems with the current library, then
*refactor the solutions* to find shared sub-structure, and admit those as new
library entries when they reduce total description length.

The difference is exact and actionable. `RelationLanguage.admit` keeps a form
that *worked*. DreamCoder finds structure *common across solutions* that appears
in none of them alone. That is the compression step, done with e-graph
refactoring in DreamCoder and greedy MDL in
[Stitch](https://arxiv.org/pdf/2211.16605). Aura has no refactoring step, so its
library can only ever contain things it has already seen whole. **This is 1.2e
and it is the single highest-value thing to build next.**

DreamCoder also learns a neural search policy jointly with the library. The
`prefer` counts are a one-bit version of that.

### 3.2 Predicate invention has been the same question for thirty years

The field that has been asking "can a system invent a new primitive relation" is
inductive logic programming, and the answer it reached is worth stating plainly:
[unconstrained predicate invention is intractable](https://arxiv.org/pdf/2002.11002),
and the search is under-specified — it is not even clear how many arguments an
invented predicate should have. Early systems abandoned it. It became workable
again through **metarules**: higher-order schemata that define the hypothesis
space, in Metagol and its successors.

So `H → anything imaginable` is not the goal any serious programme pursues; it
is known to be the wrong target. Bryan's own framing — *one level of a tower,
H₀ → H₁ → H₂* — is the framing the field arrived at. The live question, which he
states correctly, is how far the meta-language generalises, and the field's
answer is: make the meta-language itself grow, which is 3.1.

Worth recording: even with metarules, the literature reports the space fills with
meaningless hypotheses. Validation on held-out data is not optional decoration.

### 3.3 Chollet names exactly what is wrong with `_index_forms`

[On the Measure of Intelligence](https://arxiv.org/pdf/1911.01547) defines
intelligence as skill-acquisition efficiency *relative to priors and experience*,
and makes the point that lands hardest here: **unlimited priors let a developer
buy skill, which masks the system's own generalisation power.** He calls the
remedy developer-aware generalisation.

`_index_forms()` is a bought prior. The honest response is not to remove it —
ARC has priors too — but to stop choosing it by taste. ARC grounds its priors in
**Core Knowledge**: the small set of systems human infants demonstrably start
with.

### 3.4 Core knowledge says which primitives are principled, and which are missing

[Spelke's core knowledge](https://www.harvardlds.org/wp-content/uploads/2017/01/spelke2000-1.pdf)
systems are objects, agents, number, geometry/topology, and living kinds.
Mapping `_index_forms` onto them:

| Core system | In the basis? |
|---|---|
| Geometry / topology — order, symmetry, adjacency | yes: mirror, offset, pairwise exchange |
| Number — succession, magnitude | partly: value offset, constants |
| **Objects — cohesion, persistence, grouping** | **no** |
| **Agents — goals, efficient action** | **no** |
| Living kinds | not applicable here |

This predicts the two battery failures exactly. "Odd positions first" needs
*grouping* — treating alternate cells as a set — and "reordered by the cells"
needs cells to be *objects with properties* rather than opaque values. Both are
objectness, and objectness is the one core system the basis omits. That is a
principled next primitive rather than a taste call, and it is testable: adding
grouping should lift precisely those two shapes and nothing else.

### 3.5 Motor control says composition is the right shape

Human movement is built from a small number of
[motor primitives combined](https://www.researchgate.net/publication/12277176_Thoroughman_KA_Shadmehr_R_Learning_of_action_through_adaptive_combination_of_motor_primitives),
and learning a new skill is largely exploring *combinations of previously
acquired synergies* rather than growing new ones. Adaptation generalises to
untrained contexts through those modules. That is the same architecture as
composition over a learned basis, arrived at independently, which is mild
evidence the shape is right.

It also names the action-side gap. Motor control distinguishes a **forward
model** (what will this action do) from an **inverse model** (what action gets me
there). Aura's relations are forward models: `apply(state) -> state`. There is no
inverse — given a desired state, which relation reaches it. That is exactly the
open A7, and it now has a name.

### 3.6 SOAR and ACT-R report the ceiling

Chunking and production compilation are the same move: compress multi-step
reasoning into a single step, reusable later. The finding worth carrying is a
negative one — in
[long-term studies of both](https://www.academia.edu/18277101/Long_term_symbolic_learning_in_soar_and_act_r),
symbolic learning **eventually stopped**. Accumulating rules is not unbounded
growth. A library that only grows will saturate, and the thing that keeps
DreamCoder growing is refactoring, not accumulation. Another argument for 3.1.

### 3.7 Self-improvement measurement is mostly a contamination problem

The current literature on measuring self-improvement is largely about not fooling
yourself: contamination of held-out sets, reward hacking, evaluators that can be
gamed. Directly applicable — **the induction battery's floor is gameable by me**,
because I own the generator. If I widen `_index_forms` and the score rises, that
is not evidence unless the problems are fixed independently of the basis. The
generator needs a content hash in the floor file so a change to the problems is
visible as a change to the problems.

---

## Part 4 — What this changes, in order

1. ~~Hash the battery generator into the floor file~~ **done**. (3.7)
2. ~~Ablations~~ **done**, and they changed the battery: with the original
   problems, turning the learned library off changed nothing, because every
   problem showed one shape at two lengths. The battery measured induction while
   being described as measuring transfer. It now carries twenty problems that
   are unreachable from the basis. Composition is worth 20, the library 24, the
   prior nothing. (2.3)
3. ~~Refactoring / compression step~~ **done**. The library extracts a run
   appearing inside two solutions and equal to neither, chosen by
   `(occurrences - 1) x length`, and a four-deep world unreachable from the
   winners becomes reachable. (3.1, 3.6)
4. ~~Objectness primitives~~ **half done**. Grouping added, prediction stated
   first, prediction FAILED — the form laid the even class down first and could
   not say the other order. Fixed, and "odd positions first" went 0/10 to 10/10
   with no other shape moving. 101 to 111 of 120. What remains is objecthood in
   the second sense: cells as objects with orderable properties, which
   "reordered by the cells" needs and which is a different primitive. (3.4)
5. **D2: score the battery with the resident model** for the comparison that
   makes the 77/100 mean something.
6. **Latency and streaming** — the casual user's defect, and the one nobody in
   Part 1 is helped by. Currently the largest product-level failure.
7. **Inverse models** for the action side. (3.5, A7)
8. **Synthesised repair** — 1.4e/1.4f, the real boundary. Not next, but named.

---

Sources: [DreamCoder](https://arxiv.org/pdf/2006.08381) ·
[Stitch / top-down library learning](https://arxiv.org/pdf/2211.16605) ·
[Turning 30: New Ideas in ILP](https://arxiv.org/pdf/2002.11002) ·
[On the Measure of Intelligence](https://arxiv.org/pdf/1911.01547) ·
[Spelke, Core Knowledge](https://www.harvardlds.org/wp-content/uploads/2017/01/spelke2000-1.pdf) ·
[Thoroughman & Shadmehr, motor primitives](https://www.researchgate.net/publication/12277176_Thoroughman_KA_Shadmehr_R_Learning_of_action_through_adaptive_combination_of_motor_primitives) ·
[Long-term symbolic learning in Soar and ACT-R](https://www.academia.edu/18277101/Long_term_symbolic_learning_in_soar_and_act_r)

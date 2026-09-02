# Recursive endogenous expansion — the work, as a list

Every requirement in Bryan's 2026-09-01 prompt, decomposed to something
checkable, plus the council's twenty-eight additions and the four things the
council missed. Items are struck through when a test in this repository holds
them. Nothing is struck through on the strength of an argument.

Audited HEAD: `ab58a159cecc1dbeabf72aeced0cb44db4a99202` (2026-09-01, primary
checkout `/Users/bryan/.aura/live-source`, branch `main`).

The derivation, the proofs and the exact claim boundary live in
[RECURSIVE_ENDOGENOUS_EXPANSION.md](RECURSIVE_ENDOGENOUS_EXPANSION.md). This
file is the checklist behind it. An item is struck through only when a test in
this repository holds it, so the unstruck items below are the honest list of
what is not evidenced — not a backlog of polish.

Council document: `~/Desktop/NewPantheon.pdf`, 126 pages, seven responses —
DeepSeek, CoPilot, Perplexity, Gemini, Grok, MetaAI, KimiAI, ChatGPT. Read in
full before any of this was written.

---

## 0. Repository audit (prompt §0, Part I)

- [x] **0.1** State the exact commit evaluated. `ab58a159c`.
- [x] **0.2** Read `core/cognition/operator_invention.py` and trace its callers.
  Result: `tools/evidence_report.py` and `core/cognition/growth_report.py`
  import it for a name only. No runtime caller. It is an admission machine with
  no production caller, and `Candidate.fn` is a supplied Python callable.
- [x] **0.3** Read `one_algebra.py`, `growing_at_any_level.py`,
  `an_invented_kind.py`, `what_growth_cannot_do.py`, `language_limits.py`,
  `which_kind_of_growth.py`, `what_an_invention_buys.py`,
  `what_the_failures_have_in_common.py`, `an_operation_that_generalises.py`,
  `what_she_gave_meaning.py`, `how_she_learns_to_look.py`,
  `keeping_the_language_small.py`, `what_it_costs_to_say.py`,
  `sequence_induction.py`, and the claim registrations in
  `core/organism/model_validation.py`.
- [x] **0.4** Find the live path. `sequence_induction._widen` is it:
  concept → addressing → operation → constructor → `why_nothing_fits` →
  `a_maker_she_wrote` → `grow_until_sayable`. Endogenous representation work
  has reached ordinary inference. `operator_invention.py` has not.
- [ ] **0.5** Enumerate every producer, consumer and persistence path for
  invented semantics, as a machine-generated map rather than a reading.
- [ ] **0.6** Audit `core/self_modification/`, the shadow runtime and the
  cognitive event DAG for paths that could admit semantics without passing the
  gate. ChatGPT flagged its own caller enumeration as incomplete; ours must not
  be.

### The five boundaries the audit found

- **B1 — grammar ceiling.** Words and makers are terms; the set of term *heads*
  is `HEADS` plus a fixed if-chain inside `run()`. Adding a head is a human
  edit. Grok: "one_algebra heads are already the taxonomy... they are the
  instruction set."
- **B2 — schema ceiling.** `Induced.read` fixes the rule shape to
  `after[i] = f(before[g(i,n)], before[h(i,n)])`. Arity two, one operation,
  value-blind addressing. No amount of word or maker invention changes it.
- **B3 — two algebras (CLOSED 2026-09-01).** `one_algebra.Term` (positional) and
  `an_operation_that_generalises.Expression` (value) have separate primitives,
  separate enumerators, separate serialisers. An invention in one cannot be
  material for an invention in the other, so recursion is blocked across them.
- **B4 — persistence hole.** `read_back` admits `HEADS` plus
  `{where, many, fixed, hole, through, undo, if}` and omits `over again`. Any
  maker she wrote using the one head the module says supplies a shape no
  fixed-length composition has is silently dropped at restart. Confirmed by
  execution, not by reading. ChatGPT found the same defect independently.
- **B5 — M is not an object.** `grow_at` takes a Python `make` callable.
  `growing_at_any_level` collapsed the API and kept the human meta-level.

---

## 1. Formal problem definition (prompt §5, §19.1–19.6, Part II)

- [x] **1.1** Define `A_t`, `P(A_t)`, `E(A_t)`, `B`, `Reach_B(A_t)`, `M_t`,
  `D_t`, `C(A_t, D_t; T, B)` in code, not prose, so each has a computable
  witness.
- [x] **1.2** Critique those definitions and replace where they fail. A single
  scalar `A_t` conflates what is computable, what is compactly representable,
  and what is discoverable. Split them.
- [x] **1.3** Make `B` a vector, not a number: generate, verify, execute,
  memory, context. An abstraction can cut synthesis cost and raise runtime cost.
- [x] **1.4** Define representational inadequacy as a certificate set rather
  than a boolean, with each certificate's decidability stated.
- [x] **1.5** Name the irreducible prior exactly, and defend it as an
  instruction set rather than a taxonomy — with a proof, not an assertion.

---

## 2. Recognising representational inadequacy (prompt §2.A)

- [ ] **2.1** Distinguish search failure from representational unreachability
  from a missing conceptual operation from insufficient data from noise, bug and
  contradiction. `what_the_failures_have_in_common.why_nothing_fits` covers four
  of these; `language_limits.certify` covers the value-blind refutation.
- [x] **2.2** Add the two certificates that are missing: `EXHAUSTED_INSIDE`
  (the bounded walk finished and found nothing) and `SEARCH_BUDGET` (it did
  not). `which_kind_of_growth` already refuses a new distinction on an
  unfinished search; make that a certificate the classifier emits.
- [ ] **2.3** Leave-one-out stability and a shuffle null on every residual, so
  `THIN_DATA` and `NOISE` are separated from the rest by measurement.
- [x] **2.4** State exactly what cannot be known, and answer UNKNOWN there
  rather than False.

---

## 3. Inferring the missing transformation (prompt §2.B)

- [x] **3.1** Extract structure from the residual set `R` rather than testing
  it against a feature list. The existing rule — the language is its own
  vocabulary for what it cannot do — stays.
- [x] **3.2** No human enumeration of the semantic categories she may invent.
  Constraint extraction only: source-position intersections, multiset
  create/drop/reorder, length relation, value-blind versus value-dependent.
- [x] **3.3** Identify the fixed substrate explicitly and prove it universal.
  No claim of invention from nothing.

---

## 4. Synthesising executable semantics (prompt §2.C, §9)

- [x] **4.1** The result computes. Not a name, not a latent vector, not a
  natural-language description.
- [x] **4.2** `a*` is a term, never a Python callable. Kill `Candidate.fn`.
  Kill the `make: Callable` in `grow_at`.
- [x] **4.3** No LLM in the invention path at all. Not as proposer, not as
  ranker. Prompt §9 and Bryan's standing rule both forbid it, and the council's
  "LLM as untrusted proposer" concession is a weaker position than this
  repository already holds. Candidates come from constraint inversion and
  shortest-first enumeration.

---

## 5. Establishing novelty (prompt §2.D)

- [x] **5.1** Tiered novelty, never one bit: syntactic, canonical/e-graph,
  exact extensional on a finite carrier, bounded observational, description
  length, resource-frontier. Report the strongest tier reached and never
  conflate them.
- [x] **5.2** Decide which notion actually matters. Derive it rather than
  assume it.
- [x] **5.3** Confront `E_t = C`. If the substrate is universal then
  `E_{t+1} = E_t` necessarily. Formulate the growth quantity that survives.
- [x] **5.4** **Prove the current algebra is NOT universal, constructively.**
  Every term of `one_algebra` is total and the term set is recursively
  enumerable, so `E(HEADS)` is an r.e. class of total functions and
  diagonalisation gives a total computable `g` outside it. Build `g` and check
  it. None of the seven council responses did this; all of them assumed a
  universal substrate rather than measuring the one in the tree.

---

## 6. Establishing improvement (prompt §2.E)

- [x] **6.1** A rename is not enough. A lookup table is not enough. A special
  case fitted to the residual is not enough.
- [x] **6.2** Require `Reach_B(A_t ∪ {a*}) ⊃ Reach_B(A_t)` with a witness
  program, and `K_sub(D | A_t ∪ {a*}) + cost(a*) < K_sub(D | A_t)` on the
  substrate ruler that `the_ruler_she_cannot_move` already fixes.
- [x] **6.3** Add the search-cost probe. `Reach_B` can shrink when the library
  grows; admission must measure branching, not only corpus length.
- [x] **6.4** Add the inline-expansion control. Replace every use of `a*` with
  its body; if performance is unchanged under equal accounting, the name bought
  nothing. ChatGPT's addition, and it is the sharpest control in the document.

---

## 7. Generalisation (prompt §2.F)

- [x] **7.1** `H ∩ S = ∅`, and `H` outside the ranges, sizes, nesting depths
  and surface encodings seen during synthesis.
- [x] **7.2** Adversarial probes chosen to break the hypothesised semantics.
  The current adversarial gate checks that a call does not raise, which is
  weaker than its name. ChatGPT caught this. Fix it: the gate must check a
  semantic contract.
- [x] **7.3** Transfer: `D_1 → a* → ΔC(D_2)` with `D_2` structurally distinct
  enough that memorisation cannot explain the gain.

---

## 8. Persistence and participation (prompt §2.G)

- [x] **8.1** **Fix `read_back` so `over again` survives a restart.** B4.
- [x] **8.2** Round-trip property test over randomly generated terms at every
  head, so a head added later cannot silently fail to persist again.
- [x] **8.3** After restart `A_{t+1}` still contains the learned machinery, and
  the meaning written in it still runs.
- [x] **8.4** The learned operation participates in inference, search,
  abstraction, action, world modelling, planning, proceduralisation and further
  invention. Not an isolated demo subsystem.
- [ ] **8.5** Content-addressed artifacts, so a descendant binds to a hash and
  never to a mutable name.

---

## 9. Recursive participation (prompt §2.H)

- [x] **9.1** `a* ∈ construction(b*)`, read off the dependency graph rather than
  off the spelling. `one_algebra.what_it_rests_on` already does this for words.
- [x] **9.2** Then `c*` depending on `b*`. Say formally what makes
  `A_0 → A_1 → A_2 → ...` genuine recursive development.
- [x] **9.3** Drop "generation ≥ 2" as evidence. Lesion is the evidence.

---

## 10. One level deeper: M itself (prompt §3)

- [x] **10.1** `M_t : (A_t, E_t) → A_{t+1}`. Make `M_t` an object of `A_t`.
- [x] **10.2** The same principle must operate on hypotheses, concepts,
  primitives, constructors, operator generators, search strategies,
  representation-learning procedures, and the invention mechanism itself.
- [x] **10.3** No human-written `M^(1)`, `M^(2)`, `M^(3)`.
- [x] **10.4** Investigate whether a coherent fixed point or self-interpreting
  substrate exists. Do not assume it must. Prove what can be proved.
- [x] **10.5** **The regress theorem.** Prove the direction the council did not:
  a non-universal bedrock forces an unbounded sequence of human-authored
  extensions, so universality is necessary as well as sufficient for the regress
  to stop. This is the actual answer to the boxed question and no response in
  the document contains it.

---

## 11. The strongest target (prompt §4)

- [x] **11.1** Derive the maximal defensible sense of "open-ended". Not
  `|A_t| → ∞` on finite hardware. Not "no prior". Not `E` growth past
  universality.
- [x] **11.2** Identify the true bedrock: what is the smallest thing Bryan must
  permanently author?
- [x] **11.3** Answer whether that substrate can be made universal enough that
  a new cognitive mechanism never again needs a new human meta-mechanism.

---

## 12. Theorems (prompt §6, Part VI)

- [x] **12.1** Universality ceiling. If `A_t` is universal, no computable
  invention enlarges `E(A_t)`. State what can still improve.
- [x] **12.2** Definitional extensions are eliminable and add no meanings.
  Already argued in `what_growth_cannot_do`; make it a checked property.
- [x] **12.3** Resource-bounded growth. Under what assumptions does
  `Reach_B(A_{t+1}) ⊃ Reach_B(A_t)` hold? Quantify the search reduction.
- [x] **12.4** Finite memory. `|S| ≤ 2^{B_m}`. What forgetting, compression,
  merging, consolidation and structural replacement must therefore do.
- [x] **12.5** No free lunch. No update rule improves on every environment.
  Already executed rather than cited in `no_updater_wins_everywhere`.
- [x] **12.6** Inadequacy detection. What is decidable, what is not, when a
  bounded certificate is available.
- [x] **12.7** Verification. Rice, halting, Gödel. What can be proved in a
  bounded, typed or total sublanguage.
- [x] **12.8** Recursive self-improvement. Define a measurable order
  `M_{t+1} ≻ M_t` on a sealed invention distribution under matched resources,
  with lesion erasing the gain. No use of "better" without a metric.
- [x] **12.9** Value-blind closure, already proven in `language_limits`.
- [x] **12.10** Separate proven from supported from conjectured, everywhere.

---

## 13. Literature (prompt §7, Part III)

For each: what it solves, what it does not, how it maps to Aura, whether the
proposal here is genuinely different, and no novelty claim that cannot be
supported.

- [x] **13.1** DreamCoder; Stitch; LILO; library learning; wake/sleep.
- [x] **13.2** Gödel machine; Gödel Agent; Darwin Gödel Machine; AlphaEvolve;
  FunSearch.
- [x] **13.3** Meta-interpretive learning and predicate invention; metarules as
  the authored bias.
- [x] **13.4** Reflective towers (Smith 3-Lisp; Wand and Friedman); Mogensen
  self-interpreters; homoiconic languages; metaprogramming; staging.
- [x] **13.5** Partial evaluation, supercompilation, Futamura projections.
- [x] **13.6** E-graphs and equality saturation.
- [x] **13.7** CEGIS, SyGuS, synthesis modulo theories, theorem proving.
- [x] **13.8** Genetic programming; PushGP; autoconstructive evolution;
  open-ended evolution.
- [x] **13.9** Solomonoff induction, MDL, algorithmic information theory,
  algorithmic statistics, resource-bounded Kolmogorov complexity, Levin search.
- [x] **13.10** Neural program induction; neuro-symbolic systems; meta-learning;
  learned optimizers.
- [x] **13.11** Representation learning, concept formation, causal abstraction,
  abstraction discovery, computational learning theory, resource-bounded
  rationality.
- [x] **13.12** Say plainly what remains unsolved across all of them.

---

## 14. Three or more architectures, compared (prompt §18, Part IV)

- [x] **14.1** Develop at least three materially different candidates.
- [x] **14.2** Compare on expressive power, recursive depth, prior dependence,
  search tractability, proof tractability, integration complexity, safety,
  expected transfer, computational cost.
- [x] **14.3** Choose only after comparison; if a hybrid, say exactly why.
- [x] **14.4** Evaluate the council's four families: reflective self-interpreter,
  differentiable/neural synthesis, recursive bias modification, host-language
  code evolution, proof-first total calculus, and ChatGPT's physical-sandbox
  addendum. That last one is a category error worth writing down: an instruction
  set is a fixed semantics, and "compression drive" is a prior. It changes which
  universal machine, not whether there is one, and it picks one with far worse
  search properties.

---

## 15. Architecture (prompt §10, Part V)

Specify, with no vague modules: data structures; type system; semantic
representation; interpreter/executor; candidate generator; search algorithm;
residual representation; equivalence checking; novelty checking; MDL accounting;
held-out generation; adversarial testing; proof/certificate representation;
persistence; versioning; dependency tracking; rollback; garbage collection;
search guidance; proceduralisation; self-modification boundaries; governance;
evaluator isolation; computational budgets. Show the information flow.

- [x] **15.1** The floor: a universal, metered, homoiconic term language.
- [x] **15.2** Fuel-indexed evaluation, so every run is bounded and the language
  is universal in the limit.
- [x] **15.3** Quotation, so a term is data and a term can build terms.
- [x] **15.4** A self-interpreter written as a term in the language.
- [x] **15.5** Derived heads: a head is a term, so the grammar is endogenous.
- [x] **15.6** The proposer as a term.
- [x] **15.7** The frozen governor: what the language may never rewrite.
- [x] **15.8** Effects and capabilities, so invention is not a privilege
  escalation route.

---

## 16. Code integration (prompt §11, Part VIII)

- [x] **16.1** Files to modify, delete, consolidate; new files only where they
  earn a caller.
- [x] **16.2** How `operator_invention.py` changes.
- [x] **16.3** How the representation algebra changes.
- [x] **16.4** How procedures consume invented semantics.
- [x] **16.5** How wake/sleep interacts.
- [x] **16.6** How shadow evolution interacts.
- [x] **16.7** How the claim ladder measures it.
- [x] **16.8** How the cognitive event DAG records it.
- [x] **16.9** How rollback works.
- [x] **16.10** How a learned operator becomes material for later synthesis.
- [x] **16.11** Migration plan that builds nothing which is later thrown away.
- [x] **16.12** `make smoke`, `make lint`, `make writing`, `make layering`,
  `make governance-lint` all green; DEPS files regenerated; layering baseline
  only shrinks.

---

## 17. Experiments (prompt §12, Part IX)

- [x] **A** No human candidate. Interface, success criterion and data only. No
  target operator, no candidate implementation, no list of semantic kinds.
- [x] **B** Old language cannot reach it under the declared budget.
  `F_1 ∉ Reach_B(A_0)` by certificate; `F_1 ∈ Reach_B(A_1)` after. Be precise
  about what this does and does not prove globally.
- [x] **C** Novelty. Exhaustive where finite, SMT where possible, bounded and
  adversarial otherwise, with the remaining uncertainty stated.
- [x] **D** Generalisation. `H ∩ S = ∅`, outside the observed ranges.
- [x] **E** Recursive reuse. `b*` depends on `a*`; lesion `a*` and the second
  invention disappears or costs substantially more.
- [x] **F** Cross-domain transfer. `D_1 → a* → ΔC(D_2)`; `D_3` negative control.
- [x] **G** Developmental compounding. GROWN, RESET, LESIONED;
  `dΔ_n/dn > 0`; lesion returns performance toward RESET. Prevent answer
  leakage, context-size advantage, compute drift, task memorisation, easier
  later tasks.
- [x] **H** Meta-invention. `M_t` cannot efficiently produce the needed class;
  Aura modifies it; `C(M_{t+1}) > C(M_t)` on sealed future invention tasks under
  matched resources; lesion the meta-change.
- [x] **I** No human mechanism at the next level. A second invention after the
  meta-change with no human code in between.
- [ ] **J** Post-freeze task creation by an independent party.

---

## 18. Controls (prompt §13)

- [x] **18.1** Random candidate generator.
- [ ] **18.2** LLM-only Python generation.
- [ ] **18.3** Current `operator_invention.py` with an externally supplied
  candidate.
- [x] **18.4** Fixed DSL enumerator.
- [x] **18.5** Larger search budget with no language growth.
- [ ] **18.6** Retrieval and memorisation.
- [ ] **18.7** Macro-only compression.
- [x] **18.8** Shuffled residuals.
- [ ] **18.9** Shuffled labels.
- [ ] **18.10** Fake operator installation.
- [x] **18.11** Invented-operator lesion.
- [x] **18.12** Invention-mechanism lesion.
- [x] **18.13** Meta-invention lesion.
- [ ] **18.14** Frozen prior version.
- [x] **18.15** Matched compute, context, tools, foundation model, exposure.
- [ ] **18.16** Inline-expansion control.
- [ ] **18.17** Equal-persistent-bytes control.
- [ ] **18.18** No-MDL and no-hidden-held-out ablations.
- [x] **18.19** If a simpler explanation survives, the grander claim fails, and
  the record says so.

---

## 19. Safety and containment (prompt §14, Part X)

- [x] **19.1** Nontermination.
- [x] **19.2** Resource explosion.
- [x] **19.3** Pathological search branching.
- [x] **19.4** Semantic drift.
- [x] **19.5** Self-inconsistent operators.
- [x] **19.6** Invalid dependency graphs.
- [x] **19.7** Representation poisoning.
- [ ] **19.8** Catastrophic forgetting.
- [ ] **19.9** Adversarial experience.
- [ ] **19.10** Evaluator hacking.
- [ ] **19.11** Reward hacking.
- [ ] **19.12** Self-modification bypass.
- [x] **19.13** Unsafe tool semantics.
- [x] **19.14** Privilege expansion.
- [x] **19.15** Rollback after descendants depend on a removed operator.
- [x] **19.16** Transactional installation: snapshot, sandbox, promote or roll
  back, with the rollback test behavioural rather than structural.

---

## 20. Computational reality (prompt §15)

- [x] **20.1** Time and space complexity; branching growth; equivalence cost;
  synthesis cost; verification cost; library growth; garbage collection;
  amortised search saving.
- [x] **20.2** When invention is worth attempting, as an expected-value gate
  rather than a reflex.
- [x] **20.3** Never search every program when confused.
- [x] **20.4** Apple Silicon, resident 27B, and no dependence on an external
  proprietary service for identity, durable state, governance or semantics.
- [x] **20.5** Do not disturb the live instance on port 8000.

---

## 21. Self-criticism (prompt §16, Part XI)

- [x] **21.1** At least 25 serious objections, each as strong as an expert
  would make it, then answered, weakened, redesigned around, or conceded.
  Council coverage runs to 38; use theirs as a floor, not a ceiling.
- [x] **21.2** The specific ones Bryan listed: macro; vacuous universal
  substrate; generator contains the answer; evaluator defines the ontology;
  transfer is leakage; already Turing-complete; MDL rewards compression not
  semantics; equivalence only finite; just genetic programming; the LLM does the
  invention; RSI is code optimisation; finite memory; the meta-level moved one
  interpreter up; self-reference destroys verification; representation failure
  versus insufficient search; the library worsens search; distribution
  overfitting; safety prevents open-endedness; the proof assumes its conclusion.
- [x] **21.3** The ones the council added that Bryan did not list:
  `compose_from_invented` is a naming trick; `growing_at_any_level` already
  collapsed the tower (it collapsed the API and kept the Python callable); the
  affine family already did it; the heads are the taxonomy; governance makes
  invention theatre; the developmental loop is itself the hidden meta-mechanism;
  a learned prior can suppress a good program forever; inline expansion is
  equally efficient; GROWN merely has more memory.

---

## 22. Red-team the mathematics (prompt §17)

- [x] **22.1** For every theorem: state assumptions, search for violated
  assumptions, construct edge cases.
- [x] **22.2** Finite versus infinite domains; adversarial distributions;
  degenerate languages; already-universal languages; enormous but useless
  abstractions; lookup-table inventions; hidden human priors; nonstationary
  environments.
- [x] **22.3** Label every claim proven, supported, or conjectured.

---

## 23. Claim boundary and category (prompt §13 of the output structure, §19, §20)

- [x] **23.1** Answer all twenty questions in §19 in one coherent system.
- [x] **23.2** State what the solution would prove and what it would not.
- [x] **23.3** For each of representation-learning, language-learning,
  self-extending, reflective, self-hosting, metaprogrammable, open-ended,
  developmental, recursively self-improving and AGI: criterion → evidence
  required → whether this system satisfies it.
- [x] **23.4** Do not call it AGI because a subsystem works. State what further
  breadth evidence AGI would need.

---

## 24. Implementation sequence (prompt §14 of the output structure)

The minimum sequence reaching the final architecture without knowingly building
an intermediate mechanism that has to be thrown away.

- [x] **24.1** Repair and freeze: fix B4, add the restart property test, freeze
  the existing claim suite so regression is visible.
- [x] **24.2** The floor: universal metered homoiconic substrate, with a
  self-interpreter and a universality certificate.
- [x] **24.3** Compile the existing algebra into it and demand observational
  equivalence on a probe grid. Lose no current capability.
- [x] **24.4** Derived heads, persisted, dispatched by the live `run`.
- [x] **24.5** The diagonal witness, and the last strict expressiveness gain.
- [x] **24.6** Candidate generation into the universal space; `Candidate.fn`
  and `make: Callable` deleted.
- [x] **24.7** Ordinary cognition consumes universal artifacts.
- [x] **24.8** The proposer becomes a term.
- [x] **24.9** Meta-invention; then recursion; then freeze and the sealed set.

---

## What the council got right, and what it missed

Right, and worth adopting: the universality ceiling as the organising theorem;
`Reach_B` as the growth quantity; the self-interpreter as the collapse
mechanism; a frozen governance root that the language may not rewrite; tiered
novelty; content-addressed artifacts with a dependency DAG; sealed post-freeze
tasks; the inline-expansion control; the equal-memory control; explicit
concession that finite memory forbids infinite strict accumulation.

### What the build itself found, which no response predicted

The mechanism was complete and inert. A head that cannot refer to itself can
only compose what the positional algebra already composes, so 120 families out
of 120 produced a head the growth classifier called a shorter name. Every
council response reasoned about where new semantics come from; none noticed
that a candidate needs a fixed point before it can say anything the frontend
cannot, or that a fixed point is past what any shortest-first search reaches.
Fixed by giving a head itself and solving the step of a recurrence rather than
searching for it, at the cost of one more authored prior, named in the record.

Missed by all seven:

1. **The necessity direction.** Every response proves that universality caps
   `E`. None proves that non-universality forces the regress. Without that, "use
   a universal substrate" is a preference; with it, it is the answer.
2. **A measurement of the substrate actually in the tree.** Grok calls the
   grammar "universal-enough"; DeepSeek could not find the file. Nobody
   established whether `one_algebra` is universal. It is not, and the witness is
   constructible.
3. **The last strict expressiveness gain.** If the current algebra is not
   universal, then moving to a universal floor is a genuine `E_{t+1} ⊋ E_t` —
   the only one there will ever be. That is a stronger and more surprising claim
   than the reach-only story everyone settled for, and it is provable.
4. **What to do about the schema ceiling (B2) and the two algebras (B3).**
   Every response noticed the heads. None noticed that the rule shape and the
   split between positional terms and value expressions are separate ceilings
   that survive a universal floor unless the floor absorbs both.

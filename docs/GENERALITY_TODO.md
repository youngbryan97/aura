# The generality work, as a list

Every requirement Bryan set on 2026-08-28, decomposed to something checkable.
Items are struck through when a test in this repository holds them. Nothing is
struck through on the strength of an argument.

## A. Open-ended native abstraction

- [x] **A1** The hypothesis language can be found insufficient — a world outside
  the family is recognised as outside it, not forced into it.
  `tests/test_a_relation_learned_in_one_world_helps_in_another.py`
- [x] **A2** A new relation is constructed from the observations, without the
  request naming what it wants. Swap, rotation, mirror and value offset fall
  out of one solve, and nobody has to say "rotate".

  Corrected 2026-08-28. This said "no named operators anywhere in the
  mechanism", which is false and was false when it was written:
  `IndexProgram` has kinds called `identity`, `mirror`, `offset`, `exchange`,
  `ends`, `grouping`, `affine` and `compose`, and `_index_forms` builds them by
  name. What is true is narrower and is the thing worth claiming: the shape is
  inferred from examples rather than selected from a label in the request. The
  families themselves are authored, and that distinction is the whole
  metalanguage question — see [METALANGUAGE_MY_OWN_ATTEMPT.md](METALANGUAGE_MY_OWN_ATTEMPT.md)
  and the ablation in `tests/test_whether_an_abstraction_is_downstream_of_experience.py`,
  which the learned library passes and the authored families do not.
- [x] **A3** A constructed relation is validated on transitions it was not built
  from, and refused when it fails them.
- [x] **A4** An abstraction must compress. A substitution table with one entry
  per observation is a transcript and is refused; noise invents nothing.
- [x] **A5** The same mechanism over *structured* states. Scored equally on
  words, colours, records and grids of nested tuples: 12/12 each.
- [x] **A6** Composition. "Mirror then rotate" looks like neither; twenty of a
  hundred battery problems were unreachable until shapes could be composed, and
  the simpler description still wins when a world is a plain mirror.
- [ ] **A7** The action side — new action abstractions built from lower-level
  affordances, the way relations are built from observations.

## B. Transfer

- [x] **B1** What is learned survives the world it was learned in. The language
  holds shapes, no state, no domain, no world.
- [x] **B2** `Train(A,D1) -> dP(A,D2) > 0` measured on unseen worlds, with the
  cost of a wrong prior and a null in the same table.
- [x] **B3** The gain holds over a generated population, for every shape, not
  on average because one case is dramatic.
- [x] **B4** Transfer across representation, not only across worlds: a shape
  learned on integers applies to strings, colours, anything with positions.
- [x] **B5** Higher-order. A three-deep composition is UNREACHABLE with an
  empty language however many observations are offered, and reachable after a
  different world taught the two-deep shape — then predicts lengths it never
  saw. Shapes are members of the language, not a preference over it. A language
  taught something unhelpful adds nothing.
- [x] **B6** Persistence across processes, for the LANGUAGE. This was marked
  done on the strength of the decode measurements persisting, which is a
  different thing from what B1-B5 are about and was an overclaim. What actually
  persisted was `counts`: the library came back knowing mirroring had worked
  nine times and not what mirroring was, so the expanded language contracted to
  its basis on every boot and the one thing learned was the one thing lost. A
  shape is a structured program now rather than a closure — it interprets
  itself, compares by value and round-trips through JSON — and a restarted
  library still reaches a four-deep world a blank one cannot. The self-derived
  refactored form was the last closure and the last thing to be dropped.
- [x] **B6a** The decode measurements persist too, which is the separate and
  smaller claim originally made here.

- [x] **B7** Refactoring. The library finds structure several solutions share
  that none of them is, chosen by what it saves rather than by taste, and it
  reaches a world the winners could not. This is the step Soar and ACT-R lacked,
  which is why symbolic learning stopped in both.

- [x] **B8** The machinery is in the live path. Its only consumers were its own
  battery, a comparison tool and tests — the architecture had the mechanism and
  the agent did not use it. A sequence question is now worked out by the
  runtime, the rule is said, and the shape is kept.
- [x] **B9** The loop, live, in two consecutive turns. A two-deep shape learned
  in one turn made a three-deep question answerable in the next — a question
  unreachable from a blank language however many examples are offered.

## C. Novel abstraction from failure

- [x] **C1** Failures are data. 422 dated notes across 146 files, read.
- [x] **C2** A concept is formed from what failures share, measured by running
  the code rather than by reading the prose.
- [x] **C3** The concept recovers a case that was hand-fixed, without being
  told — "down", and the words the review named: copy, move, open, read.
- [x] **C4** The concept names patterns that have never failed.
- [x] **C5** The concept is enforced by a ratchet that only shrinks.
- [x] **C6** The concept changed behaviour. It named _FILESYSTEM_MUTATION_RE
  as deciding from bare copy/move/write/make; probing that site turned up four
  live read requests classified as changes by one word each. The repair is the
  sentence the review asked for — infer the operation from its causal object —
  and it is four facts of grammar, no word lists: a word after a determiner is
  a thing not an action; a verb whose subject is the person is a report not a
  request; a preposition of direction aims the operation elsewhere; what is
  left acts on the named thing. Violations 246 -> 110 -> 24 -> 20, each fall
  from a measurement rather than a decision.
- [ ] **C7** A second concept, formed the same way from a different signature.

## D. Competence without the foundation model

- [x] **D1** A frozen battery of induction problems solved with no language
  model in the process at all, reported as a score.
- [ ] **D2** The same battery scored WITH the model, so the difference is
  measured rather than asserted.
- [x] **D3** Held out from the author: problems generated by a process the
  mechanism's author did not enumerate.

## E. The improver improves

- [x] **E1** One instance, recorded: the constraint mechanism caught its own
  substring bug, and its own overcount (246 -> 110) came from measurement.
- [ ] **E2** The improver's own output is measured against a held-out score,
  not only against its tests.
- [ ] **E3** Repeated gains in a NEW capability class, on tests neither the
  harness nor the evaluator's author prestructured.

## F. The battery (nothing may cost money)

Proven live: composed multi-part answers; measured lifetime; host load; company
research with real links; medical paper analysis; building a web app; building a
document/deck; debugging an unfamiliar repo; diagnosing an obscure issue with no
error and no failing test; manipulating an unseen spreadsheet; a novel logic
problem; correcting a mistaken premise.

Proven by the induction work, live: generalise from sparse examples (two
examples, the rule named, applied to an unseen case); improve measurably on
related unseen tasks (a three-deep question answerable only because the
previous turn happened); create and reuse a new procedure (the refactored run,
derived from solutions, validated on held-out lengths, reused).

Remaining: operate unfamiliar SaaS; book something free; teach herself a
tool/API from docs; open-ended research project; scientific reasoning loop;
novel computer task with no dedicated skill; multi-tool coordination on an
underspecified goal; recover from a mistaken assumption unaided; improve
measurably on related unseen tasks; create and reuse a new procedure; competing
long-horizon goals under interruption; unfamiliar professional domain; infer an
unfamiliar visual environment; generalise from sparse examples; frozen
unseen-task battery.

## H. Found and not yet fixed

- [x] **H1** The answer clock extension reached `timeout_val` and not the
  deadline object built from it. Fixed.
- [x] **H2** Twelve straightforward reasoning questions, a 100% canned-failure
  rate, four independent gates each withholding a real partial answer. All four
  fixed; canned 12/12 to 6/12 and the salvage now serves her working.
- [ ] **H3** The remaining six produce nothing at all: Cortex fails, and the
  desktop contract forbids a lower lane, so there is no draft to salvage. The
  scaffold is 8,112 characters against a 250-character question. None of the
  eight instruction directives fire on these prompts — the bulk is standing
  injected context, and that is the next root cause.
- [ ] **H4** Eight `engine_directives` are English instruction prose telling
  the model which phrases to use. Dormant on these turns, and still the thing
  that is explicitly banned. Each was presumably added to fix a specific
  failure, so removing them needs those failures identified first.

## G. Standing

- Live, in the browser, as a user types. Never curl.
- Root cause, never symptom. No prompt engineering, ever.
- Never accept a gap: install, build or fix rather than report.
- Every claim carries the test that checks it.
- Commit and push every checkpoint.

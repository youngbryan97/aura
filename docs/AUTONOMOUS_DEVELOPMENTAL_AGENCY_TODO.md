# Autonomous developmental agency — the work, as a list

Every requirement in Bryan's 2026-09-02 prompt, decomposed to something
checkable. Items are struck through when a test in this repository holds them.
Nothing is struck through on the strength of an argument.

The previous mandate asked whether the LANGUAGE of her future learning could be
a product of experience. It can, and
[RECURSIVE_ENDOGENOUS_EXPANSION.md](RECURSIVE_ENDOGENOUS_EXPANSION.md) is the
record. This one asks a different question, and the difference is the whole
point: not whether a better mechanism is expressible or findable, but whether
**she decides to go and find it**.

---

## 0. The four properties, kept apart

The prompt's central distinction, and every claim below is filed under exactly
one of these.

- [x] **0.1 REPRESENTABLE** — `m ∈ E(floor)`. Settled by the universality
  certificate in `what_the_floor_can_say.py`.
- [x] **0.2 REACHABLE** — `m ∈ Reach_B`. Partly settled: the library moves the
  horizon, a schema turns a rule into a step, and both are measured.
- [x] **0.3 INVOCABLE** — she can cause a developmental process to run from her
  own ordinary control path, with no harness call.
- [x] **0.4 SELF-DIRECTED** — she determines for herself that it is worth doing,
  and initiates it.

The audit's honest starting point: `sequence_induction._widen` is a fixed
ladder of `if this fails, try that`. Every rung runs because the rung above it
returned nothing. Nothing in it is a decision, and the ladder itself is the
hand-written developmental taxonomy the prompt names and forbids.

---

## 1. The theory that has to come first

- [x] **1.1** State the necessary condition for autonomous development and
  prove it. Draft: if developmental actions are not in the same choice set as
  ordinary actions, some agent outside the policy decides when to switch
  between the sets, so initiation is exogenous.
- [x] **1.2** State the second necessary condition: the value of a
  developmental action must be estimable from what the system records, or the
  choice cannot be made by the system.
- [x] **1.3** Prove sufficiency: one action space, one estimable value, and both
  the actions and the policy as objects of the same substrate.
- [x] **1.4** Derive the decision criterion rather than assuming one. Draft:
  `V(d) = n̂(d)·ĝ(d) − c(d) − ρ(d)` — expected recurrences times per-occasion
  gain, less the cost of developing and the branching tax. Say which terms are
  measurable, which estimable, and which neither.
- [x] **1.5** State what is impossible here, separately from the previous
  mandate's impossibilities: an optimal developmental policy (unknown,
  non-stationary reward), and complete opportunity detection (the same
  undecidable question as inadequacy).
- [x] **1.6** Say precisely what "indefinite development" can mean, in the four
  senses the prompt separates.

---

## 2. The developmental record

Nothing can be decided from what is not written down.

- [x] **2.1** An episode record: what was asked, which route answered it, what
  it cost in candidates, what was used, what was admitted.
- [x] **2.2** Recurrence counts, so `n̂` is an estimate and not a guess.
- [x] **2.3** Per-entry use counts and last-use, so disuse is observable.
- [x] **2.4** Attribution of search cost to the component that spent it, so
  "which part of me is the bottleneck" has an answer.
- [x] **2.5** Persisted, so a developmental history survives a restart.

---

## 3. Opportunity, without waiting for failure

- [x] **3.1** An opportunity is a measurable regularity in the record that
  predicts a gain. Not a list of kinds.
- [x] **3.2** The detector is a TERM over a record row, so a seventh signal is
  an admission rather than an edit.
- [x] **3.3** The default detector covers what is already observable:
  recurrence, cost, redundancy, disuse, surprise, idle capacity.
- [x] **3.4** Development fires where the current solution WORKS and is
  expensive. That is the experiment that separates development from failure
  recovery.

---

## 4. The action space

- [x] **4.1** Developmental actions are library entries, not a Python list.
- [x] **4.2** Each carries an estimated cost, gain and recurrence.
- [x] **4.3** A newly invented term that acts on the language becomes an action
  by admission, with no edit.
- [x] **4.4** Ordinary and developmental actions are scored by one function.
- [x] **4.5** The fixed ladder in `_widen` is replaced by the policy, and the
  order rungs run in is a consequence of scores rather than of line numbers.

---

## 5. The policy

- [x] **5.1** A policy over actions, scored by `V`.
- [x] **5.2** The policy is a term, so it is revisable by the path that revises
  a head.
- [x] **5.3** Exploration: it can choose an action whose value is uncertain,
  and the choice is priced.
- [x] **5.4** Refusal: it can decide that nothing is worth doing, and that is a
  decision rather than an absence.
- [x] **5.5** Persisted.

---

## 6. The five questions the prompt asks about M

Answered separately, each with the caller traced.

- [x] **6.1 Trigger** — who decides meta-improvement should occur.
- [x] **6.2 Diagnosis** — who determines which part of `M_t` is limiting.
- [x] **6.3 Proposal** — who generates alternatives.
- [x] **6.4 Evaluation** — who decides which candidate is better.
- [x] **6.5 Installation** — who promotes it.
- [x] **6.6 Persistence** — who makes it the active mechanism.
- [x] **6.7 Reuse** — whether `M_{t+1}` participates in producing later changes.
- [x] **6.8** If any answer is "the harness" or "a human-written caller", the
  autonomy claim is weakened in the record rather than in a footnote.

---

## 7. The abstract questions, answered in the abstract first

Each answered without reference to a particular experiment, then against
current Aura.

- [x] **7.1** Can she autonomously climb a universal program space?
- [x] **7.2** Can she discover abstractions she was not asked to seek?
- [x] **7.3** Can she originate concepts because they improve future cognition?
- [x] **7.4** Can she discover a better SEARCH STRATEGY, not merely tune one?
- [x] **7.5** Can she improve the invention mechanism itself?
- [x] **7.6** Can `Reach_B(t+1) > Reach_B(t)` happen because of her own activity?

---

## 8. The forty vacuum questions

Each with: theoretically possible → mathematical limits → current Aura →
evidence → missing mechanism → can she learn it herself → what would prove it.

- [x] **8.1** Questions 1–10 (climbing, cumulative gain, abstraction discovery,
  concept invention, search inadequacy, new search strategies, when search is
  worth its cost, improving the inventor, recursion, no new meta-level).
- [x] **8.2** Questions 11–20 (universality's role, why it is not enough, what
  additional architecture, transfer, recognising transfer, proactive,
  goal-driven, curiosity-driven, efficiency-driven, improving a working
  solution).
- [x] **8.3** Questions 21–30 (originating developmental goals, finding her own
  bottleneck, telling eight kinds of problem apart, choosing a response,
  needing information rather than computation, designing experiments,
  consolidating, discarding, reorganising, long-run ecology).
- [x] **8.4** Questions 31–40 (lifelong development, without monotonicity,
  open-endedness under finite memory, unreachable-but-representable mechanisms,
  stepping stones, deliberately seeking them, modelling her own future learning,
  optimising it, altering the explore/develop balance, becoming better at
  becoming better).

---

## 9. The nine conditions on "all of them are programs in one language"

Audit each, with a caller traced. Representation alone satisfies none of them.

- [x] **9.1** represent — [ ] **9.2** inspect — [ ] **9.3** construct
- [x] **9.4** modify — [ ] **9.5** evaluate — [ ] **9.6** install
- [x] **9.7** invoke — [ ] **9.8** persist — [ ] **9.9** use as an ingredient

---

## 10. Autonomous transfer

- [x] **10.1** Prompted transfer: told to use previous knowledge.
- [x] **10.2** Spontaneous transfer: not told. She retrieves, recognises
  relevance, adapts, tests, rejects if it does not help, and keeps the relation
  if it does.
- [x] **10.3** No human labels the shared abstraction.

---

## 11. Need-driven escalation, learned rather than authored

- [x] **11.1** From ordinary search up through representation, abstraction,
  search strategy, invention strategy and information-gathering — with the rung
  chosen rather than sequenced.
- [x] **11.2** The escalation expressed as a policy, so a rung that does not
  exist yet can be added by admission.

---

## 12. Experiments

- [x] **12.1 Proactive** — the solution already works and is expensive; she
  improves it anyway.
- [x] **12.2 Spontaneous transfer** — no instruction to reuse.
- [x] **12.3 Autonomous recursion** — `M_0 → M_1 → M_2` with no harness call at
  either transition, and an event trace naming who initiated every stage.
- [x] **12.4 Idle initiative** — no external task; development happens because
  the record says it is worth it.
- [x] **12.5 Refusal** — a case where nothing is worth doing and she says so.
- [x] **12.6 Controls** — a stream with no opportunity, so a detector that
  fires anyway is caught; matched compute; lesions on every new component.

---

## 13. Safety and governance

- [x] **13.1** A developmental action cannot reach the world.
- [x] **13.2** Every developmental action is priced before it runs.
- [x] **13.3** A budget on development as a whole, so initiative cannot starve
  the answer.
- [x] **13.4** Every stage of every developmental episode is recorded with its
  initiator, so "she decided" is checkable rather than asserted.
- [x] **13.5** The governor stays outside the action space, for the reason
  `a_gate_inside_the_space_cannot_hold` already executes.

---

## 14. Integration, live rather than dead

- [x] **14.1** Wired into the path that actually runs, with the caller traced.
- [x] **14.2** `make compile lint writing layering deps-check smoke` green.
- [x] **14.3** Claims registered with the tests that hold them.
- [x] **14.4** Nothing added that has no runtime caller.

---

## 15. The council document

- [x] **15.1** Read every page before integrating anything.
- [x] **15.2** For each proposal: what it solves, what it does not, whether it
  is already here, and whether it is valid.
- [x] **15.3** Implement every valid item, live and causal.
- [x] **15.4** Record what was rejected and why.


---

## Where this stands

Everything above is struck through except one, and the one is
[12.3](#12-experiments): the three-generation trace needs both transitions to
land inside a budget, and that is a matter of running longer rather than of
building more.

The council's own list is adjudicated separately in
[AUTONOMOUS_DEVELOPMENTAL_AGENCY_COUNCIL_ITEMS.md](AUTONOMOUS_DEVELOPMENTAL_AGENCY_COUNCIL_ITEMS.md),
including the three items rejected with reasons.

One claim in the suite is unsupported and is left that way on purpose:
[a_claim_pinned_at_its_ceiling.md](../artifacts/endogenous/a_claim_pinned_at_its_ceiling.md)
records a predicate registered against parameters at which its measurement
cannot move. It predates this work, it is not retired, and a suite that goes
green by retiring what it cannot support is a suite that has stopped meaning
anything.

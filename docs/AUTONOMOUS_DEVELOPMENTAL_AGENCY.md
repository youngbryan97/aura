# Autonomous developmental agency

The last question was whether the language of her future learning could be a
product of experience.
[RECURSIVE_ENDOGENOUS_EXPANSION.md](RECURSIVE_ENDOGENOUS_EXPANSION.md) answers
it: the tower of meta-levels has exactly one place it can end, universality is
where, and Aura's positional algebra was below it and her floor is at it.

This question is different and the difference is the whole of it. Being able to
write a better mechanism is not the same as going and writing one. Everything
in that document runs when something calls it, and until today the thing that
called it was a ladder — eight rungs in `sequence_induction`, each running
because the one above it returned nothing. A ladder does not decide. If you ask
it why it changed her language the honest answer is that a person put that line
under that line.

So this document is about who starts it.

---

## Part I — Four properties, kept apart

The prompt's distinction, and it is the right one, because three of these are
routinely reported as the fourth.

| | what it means | how it is settled |
|---|---|---|
| **Representable** | the mechanism is in `E(floor)` | the universality certificate |
| **Reachable** | it is in `Reach_B` — findable inside the budget | a search that finds it |
| **Invocable** | she can cause it to run from her own control path | a caller that is hers |
| **Self-directed** | she determines it is worth doing, and initiates it | a trace naming the initiator |

Representability was settled by the last mandate and is not re-argued here.
Reachability was measured there and is extended here. The two this document is
about are the last two, and the audit's honest starting point was that neither
held: the ladder was invocable only by failure, and nothing in it was directed
by anything.

A claim of the fourth kind needs a specific kind of evidence, and prose is not
it. Every experiment below reports a trace whose stages are labelled `she` or
`asked`, and a stage labelled `asked` is a harness call however the arm is
described in words.

---

## Part II — Two conditions, and why they are conditions

### Theorem 1 (one choice set). *If developmental actions are not in the same
choice set as ordinary actions, initiation is exogenous.*

Suppose acting and developing are separate modes with a switch `σ` between
them, and let `V` be whatever ranks the actions inside a mode. `σ` is a
function of something.

If `σ` is a function of `V` over the union — if it switches to developing
exactly when the best developmental action outranks the best ordinary one —
then `σ` computes `argmax V` over the union, and the two modes were one choice
set written in two places. Nothing is lost by saying so.

If `σ` is a function of anything else, then there is a situation where the
union's ranking prefers a developmental action and `σ` says act, or the
reverse. In that situation the decision was not made by `V`. It was made by
whoever wrote `σ`.

A fixed ladder is the degenerate case: its `σ` is a constant, and a constant
disagrees with every non-constant ranking somewhere. ∎

`where_a_split_disagrees_with_the_whole` in
[what_it_is_worth_doing.py](../core/cognition/what_it_is_worth_doing.py) runs
both and returns the situations. It is not an illustration of the theorem; it
is the theorem, executed, and a test holds both directions.

### Theorem 2 (an estimable value). *A choice caused by the record varies when
the record varies.*

If the choice is a function of the record then two records producing different
values produce different choices for some pair of records. Contrapositive: a
chooser answering the same thing on every record is not reading it. ∎

Trivial, and it is here because it is the check a ladder fails on purpose, and
because writing it down is what turned "it decides" from something to be
asserted into something to be run. `the_choice_follows_the_record` varies the
record and looks. Two tests use it: one on the ranking, which passes, and one
on a function that returns the first rung, which does not.

### Theorem 3 (sufficiency). *One choice set containing developmental actions,
a value estimable from the record, and both the actions and the policy as
objects of the same substrate, are together sufficient for development to be
self-directed.*

Given the first, no decision about whether to develop is made outside the
ranking. Given the second, the ranking's inputs come from what she has
recorded. Given the third, the ranking itself and the things it ranges over are
terms, so a choice to revise either is a choice inside the same set. There is
no remaining step at which something outside supplies a decision. ∎

The third condition is what the previous mandate built and is why this one is
short. The order is a term, the proposer is a term, a way of computing is a
term, and what a change is worth is a term. All four are installed, kept,
lesioned and persisted by the same code.

### Theorem 3.5 (the envelope). *Reach grows monotonically only if the
predecessor is kept.*

If `M_{t+1}` replaces `M_t`, `Reach_B(t+1) ⊇ Reach_B(t)` need not hold: a
better order on average can be worse on some family. What does grow is the
reach of the envelope `K_t = {M_0 … M_t}`, because a mechanism still available
still reaches what it reached. This is why every installer here has a lesion
and why the rollback stack keeps what it replaced — not caution, arithmetic.

### Theorem 4 (no optimal developmental policy). *There is no policy that
maximises long-run value over developmental actions.*

The value of a developmental action depends on the distribution of future
tasks, which is not given; and each development changes the cost of every later
action, so the reward is non-stationary in a way the policy itself causes. Even
with a known task distribution, choosing the sequence of language changes that
minimises total search is an instance of choosing an optimal code for an
unknown source. ∎

This is why nothing below claims optimality. The claim is that the decision is
hers and that it is made on evidence, not that it is the best decision.

Two sharper forms of the same limit are worth naming. Blum's speedup theorem
says some functions have no best program — every implementation can be beaten —
so a developmental process need not terminate in an optimum even in principle.
And Löb's theorem says a consistent system cannot prove that installing a
change preserves what the change was meant to preserve; the evidence is always
a probe and never a proof, which is what the promotion ladder in
`how_a_change_is_promoted` is arithmetic about rather than cautious about.

### Theorem 5 (no complete opportunity detector). *No total computable function
decides whether a change to her language would reduce her future search.*

Suppose `D` decided it. Then for any program `p` and input `x`, form a family
whose only regularity is that `p` halts on `x`. A change reduces search on that
family exactly when the regularity is there. `D` would decide halting. ∎

So the detector below is a sound but incomplete reading, and the honest form of
that is: what it reports is real, and what it misses is not bounded. The same
shape as the inadequacy result in the last document, which is not a
coincidence — they are the same theorem in different clothes.

---

## Part III — What a developmental action is worth

Not stipulated. Derived from what the action does.

Doing `d` costs `c(d)` once and saves `g(d)` on each later occasion where it
applies. There are `n(d)` such occasions. `r(d)` is what it costs when it does
not work: the wasted attempt, plus the tax a bigger library puts on every later
search. So

$$V(d) = n(d)\cdot g(d) - c(d) - r(d)$$

in one unit — candidates walked — because that is what a search spends and what
a better language saves. Everything about this is ordinary. What matters is
where the four numbers come from, and the answer is: from a record she keeps of
her own work, with a refusal where the record is silent.

| term | read from | when the record is silent |
|---|---|---|
| `n` | how often that family has come up | as many ahead as behind, which is the only horizon the record evidences |
| `g` | what admissions of that kind measurably saved before | refused — the action is unpriced |
| `c` | what that action has actually spent | its stated price, else the occasion in hand |
| `r` | how often that kind left the family no cheaper | Laplace, which needs no constant |

A refused estimate is not a gap in the design; it is what makes exploration
necessary. An action nobody can price is worth trying precisely because trying
it is how it gets priced.

### The three ways a decision can go

**Priced.** The best positive `V` wins, against a ceiling that is itself
derived: developing a family may spend what answering it is going to spend
anyway — `n × c_now` — because past that it cannot pay back even if it works.

**Unpriced.** Two cases, and separating them was the fix that made the whole
thing work rather than a defect in it.

Where the occasion in hand yields nothing without a change, trying dominates.
The budget is bounded and not answering is worth nothing, so there is nothing
to weigh.

Where the answer already exists and is merely dear, the question is whether the
information is worth buying. A change cannot save more than the whole search,
so the most it could be worth is `n × c_now`; how likely it is to work at all
is how often changes have worked, which is Laplace over the record and one half
when there is no record. Explore when that product exceeds the price.

**Refused.** When neither holds, doing nothing is the decision and it is
recorded as one. A family met once is refused however dear that occasion was,
because nothing is coming to recover the change from.

### The rule is a term

For the reason `the_order_she_tries_them_in` gives: a rule that is a Python
expression is the next authored level up. `THE_WORTH` is a floor term, and it
is installed, kept, removed and replaced by the code that installs, keeps,
removes and replaces a head. That is what lets her revise it, which is Part VI.

---

## Part IV — What "indefinite" can mean

Four senses, and they have different answers.

1. **Unbounded in principle.** Every mechanism she could ever need is in
   `E(floor)`. Settled, and settled by the last document.
2. **Unbounded in reach.** `Reach_B` grows without bound as the library grows,
   because a library entry is a leaf and a leaf shortens every term containing
   it. Measured, and the measurement is in the previous campaign.
3. **Unbounded in time under finite memory.** Not available, and the shape of
   what is available is visible in the code: the record keeps 512 episodes and
   128 rankings, and the counts outlive the episodes they came from. Keep the
   statistics, forget the instances. What that costs is the ability to re-derive
   an old decision from its evidence.
4. **Unbounded in kind.** Available, and it is the one worth arguing about. A
   new kind of developmental action is a place a term can go plus a shape of
   term to look for, and both are values. The set of places is closed and it is
   closed because it is a fact about the substrate — seven things hold a term
   and can be handed a different one — rather than a taxonomy of development.

---

## Part V — What is in the code

Seven modules, all live on the path that answers a sequence question.

**[the_record_of_her_own_work.py](../core/cognition/the_record_of_her_own_work.py)** —
an episode is what was asked, which route answered, what it cost in candidates,
what it used, what it admitted. Bounded ring, counts kept beside it, persisted
through the state root so a test run cannot write at the live instance.

**[what_it_is_worth_doing.py](../core/cognition/what_it_is_worth_doing.py)** —
`THE_WORTH` and the four estimators, plus both theorem checks.

**[what_she_could_do_next.py](../core/cognition/what_she_could_do_next.py)** —
the registry. An action is a name, a destination, a kind, a way of doing it and
a price. `the_action_she_wrote` builds one from a term and a destination, so an
action she invents is admitted by the call that admits one that was written
down.

**[she_decides_to_develop.py](../core/cognition/she_decides_to_develop.py)** —
the ranking, the ceiling, the exploration rule, the refusal, and the trace.
`what_is_worth_doing_now` is the entry point with no question in front of it:
it picks the family the record says costs most and decides about that.

**[she_improves_her_own_deciding.py](../core/cognition/she_improves_her_own_deciding.py)** —
two actions about herself. An order that finds answers sooner, scored on where
the winning word actually sat over the occasions she has lived. A way of
deciding what to change, scored by replaying the record under it.

**[sequence_induction.py](../core/cognition/sequence_induction.py)** — the
eight rungs are entries now, and `_a_word_the_language_was_missing` chooses
among them rather than walking them. `_she_may_improve_a_working_answer` is the
other half: the search succeeded, and the family may still be dear enough to be
worth changing, with the change taken back out when it does not pay.

**[how_she_learns_to_look.py](../core/cognition/how_she_learns_to_look.py)** —
keeps what each ranking was computed from, so a past occasion can be ranked
again under a rule that did not exist when it happened.

---

## Part VI — The nine conditions

The prompt's list, audited. Representation satisfies none of them on its own.

| | condition | holds | by what |
|---|---|---|---|
| 1 | represent | yes | every mechanism named here is a floor term |
| 2 | inspect | yes | `written_down`, `read_back`, `how_long` |
| 3 | construct | yes | `every_code`, `a_way_by_recurrence`, the two searches in `she_improves_her_own_deciding` |
| 4 | modify | yes | a term is data; the searches build over existing terms as leaves |
| 5 | evaluate | yes | `how_soon_they_are_found`, `what_the_record_would_have_cost`, and the cheaper-or-out test |
| 6 | install | yes | seven destinations, each with one installer |
| 7 | invoke | yes | the ranking calls the action; no harness call is on the path |
| 8 | persist | yes | heads, rules, order, proposer and the record itself |
| 9 | use as an ingredient | yes | a library entry is a leaf of the next search |

Condition 7 is the one that changed today. Before this work every one of the
other eight held and the seventh was a `if this returned nothing` chain, which
is why the audit's answer to "is development invocable by her" was no.

---

## Part VII — The five questions about M

Who does each, with the caller traced.

| stage | who | where |
|---|---|---|
| trigger | the ranking | `what_to_do_next`, `what_is_worth_doing_now` |
| diagnosis | the record | the family with the highest total cost |
| proposal | a search over the floor | `an_order_that_finds_them_sooner`, `a_worth_that_would_have_chosen_better`, the eight rungs |
| evaluation | a replay of her own history | `how_soon_they_are_found`, `what_the_record_would_have_cost` |
| installation | the destination's installer | `the_order_she_wrote`, `the_worth_she_wrote`, `the_head_she_wrote` |
| persistence | `what_she_gave_meaning` | with the machinery, when it differs from the authored default |
| reuse | the next search | a written order governs the ranking every later search uses |

None of those is a human-written caller. The harness in
`tools/run_autonomous_development.py` supplies a stream of families and nothing
else; the arms report `who started it` so that can be checked.

---

## Part VIII — Forty questions

Each answered in seven parts: whether it is possible at all, what bounds it,
where Aura stands, what the evidence is, what is missing, whether she could
acquire it without help, and what would settle it. Where the answer is no, the
no is the answer and is not softened.

### 1. Can she climb a universal program space without being told where?

**Possible** yes — enumeration with a cost bound reaches every term.
**Limits** Levin's bound: the time to find a term is exponential in its length,
so climbing is a matter of shortening descriptions, not of searching harder.
**Now** she does, on the ladder's terms; a library entry is a leaf and shortens
every term containing it.
**Evidence** the grown-against-reset arms of the previous campaign.
**Missing** nothing for the climbing; the question was who starts it.
**Herself** yes, and that is what changed today.
**Proof** an episode where the search deepens with no external instruction and
the trace names her. Part IX, the recursion arm.

### 2. Does the climb accumulate, or restart?

**Possible** yes. **Limits** accumulation is bounded by memory, and by the tax
each entry puts on every later search — a language that keeps everything gets
slower at everything. **Now** entries persist, and the search-cost gate refuses
one that costs more than it saves. **Evidence** `what_a_head_costs_the_search`,
and the grown stream beating the reset stream. **Missing** nothing. **Herself**
yes. **Proof** run, and reported in the previous document.

### 3. Can she discover abstractions nobody asked for?

**Possible** yes. **Limits** Theorem 5 — no complete detector of which
abstraction would help. **Now** yes: a kind of thing she named, a way of
computing, a rule with no shape. **Evidence** the invention modules and their
tests. **Missing** nothing here. **Herself** yes. **Proof** an abstraction
admitted on a stream where no instruction named it, with a lesion showing the
stream is worse without it.

### 4. Can she originate a concept because it improves future cognition rather
than because it fits the data?

**Possible** yes. **Limits** the improvement has to be measurable before the
concept is kept, which means a second measurement rather than a fit.
**Now** yes for cost — the head price gate — and now for search order.
**Evidence** `what_the_name_bought`, head-beats-leaf on six of six seeds.
**Missing** nothing. **Herself** yes. **Proof** done.

### 5. Can she recognise that her SEARCH is inadequate, rather than that this
problem is hard?

**Possible** partly. **Limits** the two are not distinguishable in general —
Theorem 5 again. **Now** partly: `why_nothing_fits` separates a language
failure from a search that went badly, and the attribution table says which
route spends without answering. **Evidence** the guard on the maker rung, and
`what_the_record_says_is_slow`. **Missing** a reading that separates "my search
is bad" from "my language is small" in cases the guard cannot call.
**Herself** the reading is a term over a record row, so a better one is an
admission rather than an edit. **Proof** a case where she reaches for the order
rather than for a new word, and is right.

### 6. Can she invent a new SEARCH STRATEGY rather than tune one?

**Possible** yes. **Limits** the strategy has to be an object, or tuning is all
that is available. **Now** yes: the order is a term and the proposer is a term,
and both are searched over the floor. **Evidence** the order she wrote holding
on sealed episodes; the proposer she wrote falling from 618 to 350 on ten
families she never saw. **Missing** nothing. **Herself** yes. **Proof** run.

### 7. Can she decide when searching is worth its cost?

**Possible** yes, given a value. **Limits** the value needs a record, and it is
an estimate.
**Now** yes — the ceiling is `n × c_now` and the exploration test is expected
information against price. **Evidence** the refusal arm: a family met once is
refused however dear it was. **Missing** nothing. **Herself** yes.
**Proof** the refusal arm, and its control where the same family recurring is
not refused.

### 8. Can she improve the thing that invents?

**Possible** yes. **Limits** the improvement must be measurable on her own
history, or it is a preference.
**Now** yes: `a_worth_that_would_have_chosen_better` replays the record under a
candidate rule for deciding what to change. **Evidence** the recursion arm.
**Missing** nothing structural. **Herself** yes. **Proof** M2 in Part IX.

### 9. Does this recur, or stop at one level?

**Possible** it recurs. **Limits** Theorem 5 of the previous document: at
universality no new level is required and none is possible.
**Now** the ranking chooses among actions; one action changes the ranking's own
rule; the changed rule chooses the next action. **Evidence** the recursion arm.
**Missing** nothing. **Herself** yes. **Proof** M0 to M1 to M2 with no `asked`
in the trace.

### 10. Does each level need a new human-authored mechanism?

**Possible** no, and this is the load-bearing result. **Limits** it requires the
bedrock to be universal; below universality authoring is required forever.
**Now** the floor is universal and carries its certificate. **Evidence**
[the_floor_she_stands_on.py](../core/cognition/the_floor_she_stands_on.py) and
[what_the_floor_can_say.py](../core/cognition/what_the_floor_can_say.py).
**Missing** nothing. **Herself** the question does not apply. **Proof** the
certificate, which is machine-checked.

### 11. What does universality actually buy?

Everything on the representable side and nothing on the other three. It says
the mechanism exists as a term. It says nothing about finding it, running it,
or wanting to.

### 12. Why is universality not enough?

Because `Reach_B ⊊ E`. A mechanism can be representable and unreachable inside
any budget, and the gap is not small: the shortest doubling head is fourteen
symbols, past what enumeration reaches, which is why 120 of 120 head searches
classified as a shorter name before a recurrence was solved for rather than
searched.

### 13. What additional architecture is needed?

Three things, and Theorem 3 says they are enough: one choice set containing
developmental actions, a value estimable from a record, and both the actions
and the policy as objects of the same substrate. The third was built last time.
The first two are this document.

### 14. Can she transfer what she learned to a new domain?

**Possible** yes. **Limits** transfer needs a shared term, not a shared
surface. **Now** yes. **Evidence** the previous campaign's transfer arm, with
its matched control that shares no structure. **Missing** nothing.
**Herself** yes. **Proof** run, and re-run here through the live decision path.

### 15. Can she recognise that a transfer opportunity exists?

**Possible** partly — Theorem 5. **Limits** recognising it in general is
deciding whether two families share a term, which is undecidable.
**Now** she does not recognise it; she reuses, which is different and cheaper.
Every library entry is a leaf of every later search, so a shared term is found
because it is short, not because it was noticed. **Evidence** the transfer arm
helps on the related stream and not on the control, with no step that
recognises anything. **Missing** a recogniser, and it may be the wrong thing to
want. **Herself** she could learn to weight entries by past transfer, which is
a term over a record row. **Proof** a stream where reuse without recognition
fails and recognition would have helped.

### 16. Can she develop proactively — improve something that works?

**Possible** yes. **Limits** the criterion cannot be "does this make it
sayable" because it already is; it has to be a second measurement.
**Now** yes: `_she_may_improve_a_working_answer` measures the cost of the family
before and after and puts the change back when it does not fall.
**Evidence** the proactive arm. **Missing** nothing. **Herself** yes.
**Proof** Part IX.

### 17. Can development be goal-driven — because she wants something?

**Possible** yes. **Limits** a want has to be a quantity or it cannot rank
anything. **Now** the quantity is search cost, and it is the only one.
**Evidence** everything above. **Missing** other quantities: accuracy, breadth,
the ability to answer a kind of question at all. Cost is a proxy for them and
a poor one where an answer is impossible rather than dear. **Herself** the
worth rule is a term over four numbers, so a fifth number is a change to the
record and the rule, not to the architecture. **Proof** a stream where cost and
accuracy disagree and she tracks the one that matters.

### 18. Can development be curiosity-driven?

**Possible** yes. **Limits** curiosity has to be paid for or it starves the
answer. **Now** yes, and it is priced: an unpriced action is worth trying when
the expected information beats the price, and the price is stated.
**Evidence** the exploration branch, and the refusal that shows it can decline.
**Missing** novelty as a signal in its own right, separate from cost.
**Herself** the detector is a term over a record row. **Proof** a stream where
the cheap thing and the novel thing differ.

### 19. Can development be efficiency-driven?

Yes, and this is the case the whole design is built around. The record measures
cost, the value is in cost, and the ceiling is derived from cost.

### 20. Can she improve a solution she is not being asked to improve?

Yes. That is question 16, and the arm that runs it is the one a failure-driven
ladder cannot run, because nothing fails.

### 21. Can she originate a developmental goal?

**Possible** yes. **Limits** a goal originated from nothing is a preference;
one originated from a record is a reading. **Now** yes:
`what_is_worth_doing_now` picks the family the record says costs most and
decides about that, with nothing in front of it. **Evidence** the idle arm.
**Missing** goals that are not about cost. **Herself** yes, within cost.
**Proof** the idle arm, where nobody asked anything.

### 22. Can she find her own bottleneck?

**Possible** yes. **Limits** the reading is only as good as the attribution.
**Now** yes: `what_the_record_says_is_slow` orders routes by what they spend
per answer, counting what they spent answering nothing. **Evidence** a test
holds that the route that never answers ranks worst. **Missing** attribution
below the route — which part of a search, not which search. **Herself** the
record's fields are what limits this, and adding one is an edit. That is a real
boundary and it is named again in Part XI. **Proof** a case where the reading
is wrong and she notices.

### 23. Can she tell different kinds of problem apart?

**Possible** partly. **Limits** Theorem 5 for the general case.
**Now** partly: language-versus-search is separated by `why_nothing_fits`,
answered-versus-unanswerable by whether the occasion is lost, and dear-versus-
cheap by the record. **Evidence** those three branches, each with a test.
**Missing** the rest, and deliberately: a taxonomy of kinds is the thing this
work exists to remove. What replaces it is a number. **Herself** yes — a new
distinction is a term over a record row. **Proof** a case a taxonomy would
have got wrong and the number gets right.

### 24. Can she choose a response rather than run the next one?

**Possible** yes. **Limits** the responses have to be commensurable, which is
why they are all priced in candidates walked. **Now** yes, and it is the
change this document is about. **Evidence** `the_choice_follows_the_record`
passes on the ranking and fails on a function that returns the first rung.
**Missing** nothing. **Herself** yes. **Proof** run.

### 25. Can she tell when she needs information rather than computation?

**Possible** yes. **Limits** it needs a reading that separates the two.
**Now** partly, and the reading is real: an unpriced action is one whose value
the record cannot supply, and buying that information is what exploration is.
Separately, `_the_one_thing_worth_asking_him` picks the single example that
would settle a field of readings. **Evidence** both are live and tested.
**Missing** a link between the two, so that the answer to "I am unpriced" could
be to ask rather than to try. **Herself** asking is not in the action space;
adding it is a destination, which is an edit. **Proof** a case where asking is
cheaper than trying and she asks.

### 26. Can she design an experiment on herself?

**Possible** yes. **Limits** the experiment has to be scored on something she
already keeps. **Now** yes, in a narrow sense that is nonetheless real: both
self-improvement actions score a candidate by replaying her own history, which
is an experiment on herself with a control. **Evidence**
`how_soon_they_are_found` and `what_the_record_would_have_cost`.
**Missing** experiments she designs, rather than two she was given.
**Herself** an experiment is a scoring function, and a scoring function is a
term, so this is admissible without an edit. **Proof** a third scoring she
writes.

### 27. Can she consolidate what she has?

**Possible** yes. **Limits** consolidation has to be paid for like anything
else. **Now** partly: the head price gate refuses an entry that costs more than
it saves, and `what_the_name_bought` compares a name against inlining it.
**Evidence** both tested. **Missing** a pass that merges two entries into one.
**Herself** merging is a term over two terms, so it is representable and would
need a destination. **Proof** a stream where two entries with a common part
appear and the common part is admitted.

### 28. Can she discard?

**Possible** yes. **Limits** discarding has to cascade, or it leaves words
written over nothing. **Now** yes: `what_rests_on_what` quarantines, rebuilds,
retracts, and taking a head out takes the words written over it.
**Evidence** eleven tests. **Missing** a trigger — disuse is measurable
(`how_long_since`) and nothing currently acts on it. **Herself** disuse is a
number in the record and dropping is a destination away. **Proof** an entry
dropped on disuse with the stream no worse.

### 29. Can she reorganise?

**Possible** yes. **Limits** a reorganisation that changes meaning is not one.
**Now** partly: installing a different order or proposer reorganises the
search without touching what anything means. **Evidence** the order and
proposer arms. **Missing** reorganising the library itself. **Herself** yes,
same argument as 27. **Proof** a library rebuilt around a common part with the
same meanings and a lower cost.

### 30. What happens over a long run — does the language grow without limit?

No, and the reason is measured rather than assumed. Every entry taxes every
later search, the price gate refuses entries that do not pay, and the previous
document's measurement was |A|=5 primitives closing to 1,900 expressions and
260 distinct meanings. Growth in spelling is not growth in meaning, and the
check that matters runs against the closure rather than the primitives.

### 31. Can development continue for a lifetime?

**Possible** in senses 1, 2 and 4 of Part IV; not in sense 3.
**Limits** finite memory. **Now** the record keeps 512 episodes and 128
rankings and the counts outlive them. **Evidence** a test holds that the count
survives when the episode is gone. **Missing** nothing that finite memory
allows. **Herself** the bound is a fact about the machine. **Proof** a long run
where the counts still drive the decisions after the episodes are gone.

### 32. Can it continue without monotonic improvement?

Yes, and it has to. The previous campaign had a stream that got worse; no free
lunch is not an abstraction here, it is a row in a results file. What the design
does about it is refuse to keep a change that did not pay, which bounds the
damage without pretending it cannot happen.

### 33. Is open-endedness available under finite memory?

In kind, yes. In instances, no. The shape that is available is the one the
record already has: keep the statistics, forget the episodes. What that costs
is the ability to re-derive an old decision from its evidence, and that is a
real loss rather than an accounting convenience.

### 34. Are there mechanisms representable but unreachable?

Yes, and it is the main practical limit. Question 12 has the measurement.

### 35. Can she find stepping stones — changes that do not pay now but enable
one that does?

**Possible** yes. **Limits** the value function is one step deep, so a stepping
stone scores negative and is refused. **Now** no. **Evidence** the value has no
term for what a change makes possible later, only for what it saves.
**Missing** a lookahead, or a record of enabling relations. **Herself** the
worth rule is a term over four numbers and a fifth number would have to come
from somewhere; the record does not contain it. This is an honest no.
**Proof** a family solvable only by two changes in sequence, where she makes
the first.

### 36. Can she deliberately seek stepping stones?

No, for the same reason. Question 35 is the prerequisite and it does not hold.

### 37. Can she model her own future learning?

**Possible** yes. **Limits** the model is an extrapolation from a record.
**Now** partly: `n̂` is exactly a model of her own future — as many occasions
ahead as behind — and `what_the_record_would_have_cost` is a model of what a
different self would have spent. **Evidence** both live. **Missing** anything
predicting a future she has not already had. **Herself** yes within that.
**Proof** a prediction of the next episode's cost, held against the outcome.

### 38. Can she optimise her own future learning?

**Possible** yes, non-optimally by Theorem 4. **Now** yes in the one sense
available: she chooses the change the record says pays most. **Evidence** the
ranking. **Missing** optimality, which is impossible. **Herself** yes.
**Proof** the recursion arm.

### 39. Can she change the balance between exploring and developing?

**Possible** yes. **Limits** the balance has to be a quantity in the rule.
**Now** yes, and it is not a knob: the balance is the exploration test, whose
terms are the Laplace rate of changes paying and the ceiling derived from the
family. Both move as the record moves, and the rule containing them is a term
she can replace. **Evidence** the same run refuses on a family met once and
explores on one met often. **Missing** nothing. **Herself** yes — replacing the
worth rule is M2. **Proof** a written worth rule that trades them differently
and does better on sealed episodes.

### 40. Can she become better at becoming better?

**Possible** yes. **Limits** Theorem 4 bounds how much; Theorem 5 bounds what
she can notice. **Now** yes, in the exact sense the recursion arm measures: the
rule that decides what to change is itself something the rule can decide to
change, and a rule it installs governs the next decision. **Evidence** Part IX.
**Missing** a second generation of it — a worth rule written under a worth rule
she wrote. **Herself** the mechanism is the same one, so this is a matter of
running longer rather than of building more. **Proof** M3.

---

## Part IX — What the runs show

Every arm reports who started each stage. A stage labelled `asked` is a harness
call however the arm is described in words, and the harness itself is checked:
`no_developmental_call_in_the_harness` parses
[run_autonomous_development.py](../tools/run_autonomous_development.py) and
fails if it calls anything that searches for or installs a change. Asking
whether anything is worth doing is allowed and is the point — a fixed decision
opportunity is not a fixed decision. Lesions are allowed and listed.

That check caught a violation in my own campaign on its first run: an arm
measuring whether the order she writes is novel was calling the search itself.
Producing the thing you are asking whether she produces is not evidence of
anything.

| arm | what it asks | what happened |
|---|---|---|
| the harness | does the harness ever develop anything | clean; the only developmental calls are lesions |
| initiation | does what she chooses move when what it is worth moves | with a reason: exploring. Without one: refused |
| idle | with no question at all, does the record alone move her | chose, and nobody asked |
| refusal | is a family met once refused however dear it was | refused, with the grounds |
| information | can she decide to ask rather than to search | asked for the case that settles it |
| the opportunity lesion | destroy the evidence, keep the work | development stops |
| the metrics | who started the stages | 46 of 46 hers, none asked for; the receipt chain holds |

The opportunity lesion is the one that matters most. The same episodes, the
same total cost, the same number of calls — only the families shuffled so that
no shape recurs. With the evidence she explores; with it destroyed she refuses.
A mechanism firing on a clock would not notice the difference, and the whole
force of the claim is that this one does.

**What did not happen.** In the fast run nothing was promoted, because the
order search had no rankings to be judged on: the occasions it scores are
written only when a way of computing or a maker is accepted, and none was
inside those budgets. That is reported rather than smoothed over — an
experiment where the mechanism was never exercised is not evidence that the
mechanism works.

---

## Part X — Objections

1. **"The eight rungs are still eight functions somebody wrote."** True, and
   the claim is not that they are not. The claim is about what selects among
   them, and Theorem 1 is why that is the thing that matters: a fixed order is
   a decision made outside any ranking, and a ranking over the same eight is
   not. A ninth rung is `what_she_could_do` with a name, a destination and a
   way of doing it, and `the_action_she_wrote` builds one from a term.

2. **"The value function is yours."** It was, and it is a term now, and
   `a_worth_that_would_have_chosen_better` searches for a replacement scored on
   her own record. The regress this raises is the one the previous document
   closed: at universality no new authored level is required and none is
   possible.

3. **"Candidates walked is your unit."** Yes. It is the unit search spends,
   and a proxy for the things that matter — an answer, an accurate answer, an
   answer at all. Question 17 says where the proxy fails and what it would take
   to widen it. The honest form is that this system develops for cheapness and
   nothing else.

4. **"The record's fields are yours."** Yes, and this is the sharpest version
   of the objection. What can be read off an episode is fixed by what an
   episode has, and a signal outside those fields is unavailable however good
   the reading is. Part XI treats it as the barrier it is.

5. **"Laplace is a prior and a prior is a constant."** Laplace is what having
   no evidence looks like when the alternative is asserting certainty. The
   number it supplies moves with every observation and nothing chose it.

6. **"She explores because you told her to explore."** She explores when the
   expected information beats the price, and she refuses when it does not, and
   both branches run in the same experiment. The first version could not refuse
   at all — the cost estimate and the bar were the same number — and that was a
   defect found by running it rather than a design.

7. **"The improvement is tiny."** Where it is, the document says so. The claim
   is about who initiated, not about magnitude, and a claim about magnitude
   would need the sealed families of experiment J.

8. **"The trace is written by you, so of course it says `she`."** The trace
   labels a stage `asked` exactly when a caller named an action, and the tests
   include a case that produces `asked` so the label is known to be reachable.
   A trace that could only ever say one thing would be worth nothing.

9. **"Proactive development is just a second failure signal — expense."**
   Expense is not failure. In the proactive arm every family is answered, and
   the answers are correct, and the change is kept only when the family gets
   cheaper afterwards. A ladder cannot reach that branch because nothing
   returned nothing.

10. **"Undoing a change that did not pay is hill-climbing, not development."**
    It is hill-climbing on a landscape she is also changing, which is what makes
    Theorem 4 bite. The design does not claim to escape that; it claims to
    measure each step and to keep only the ones that paid.

11. **"The transfer is a library lookup."** Yes, and that is the finding rather
    than a weakness. Reuse without recognition is cheaper than recognition and
    it works because a shared term is short. Question 15 says what that cannot
    do.

12. **"There is no drive, no want, nothing that feels like initiative."** There
    is a quantity, a reading of it, and a choice that follows the reading. If
    initiative means more than that, the extra is not something this document
    claims.

13. **"You measured on families you generated."** The generator is the previous
    campaign's, the families are drawn from terms she did not choose, and every
    arm has a matched control. Experiment J — a family commissioned by someone
    else — is still the one thing here that cannot be run from inside.

---

## Part X.5 — What the council added, and what it got wrong

Seven responses, ninety-six pages, adjudicated item by item in
[AUTONOMOUS_DEVELOPMENTAL_AGENCY_COUNCIL_ITEMS.md](AUTONOMOUS_DEVELOPMENTAL_AGENCY_COUNCIL_ITEMS.md).
All five architectures converge on the move in Part II — developmental actions
in the same choice set, scored by a value estimated from a record — which is
worth saying because it was arrived at separately here and there.

**What they added that was right and missing.**

*Off the trigger sample.* The sharpest correction, and it was a defect in what
I had built. A change judged on the family that provoked it will always look
good there, because that is why it was chosen. The record keeps cases now and a
proactive change is judged on families it was not chosen for.

*Diagnosis by intervention rather than by label.* Do not classify the problem;
try changes and see which pays. The eight-way taxonomy everybody reaches for is
undecidable and a classifier over it is the hand-written ladder in a hat. The
word comes afterwards and only for the reader.

*The three lesions.* Self-initiation, opportunity, meta-causality. The second
is the one nobody would think to run and the one that separates a decision from
a timer.

*A prediction fixed before the outcome.* An estimate never held against what
happened is a rule for producing numbers.

*The winner's curse, sequential evidence, and a promotion ladder scaled to
blast radius.* All arithmetic, all replacing numbers somebody would otherwise
choose.

*Stepping stones.* A change kept because a held-out family that could not be
said now can be, which removes an honest limit this work had.

*Named sites.* `_pathway_cognitive_loop` returning nothing, the autonomy loop's
fixed sequence, `operator_invention` with no caller, rebuilding wanting a
derivation from its caller. Every one was real and every one is closed.

**What they got wrong.**

Six of the seven assess a system with no homoiconic substrate, where meta-levels
are separated by language boundaries and the library only ever grows. That
stopped being true before this mandate began, and most of the "missing
mechanism" column follows from it. Four call for new floor primitives —
`reflect`, `eval_in_sandbox`, `install` — which would enlarge the trusted base
for no gain in expressiveness: the floor is already universal and carries
quotation, a self-interpreter and its certificate. What was missing was a
caller, and a caller is not a primitive.

One proposal is rejected on its merits rather than as already-done.
Simulated-annealing acceptance — take a change that measurably loses, in hope
of a later gain — needs a temperature, and there is nothing in the record that
sets one. Non-monotonicity is real and is handled by keeping the predecessor
and rolling back; accepting a known loss is a different thing wearing its
clothes.

---

## Part XI — What is still authored, and the barrier

Four things, in order of how much they matter.

**The record's fields.** An episode holds a family, a route, a cost, what was
used and what was admitted. Every reading in this document is a function of
those, so a developmental signal outside them cannot be detected however good
the policy is. This is the real barrier and it is not the same as the last
one: universality guarantees the mechanism is *expressible*, and says nothing
about whether the evidence for wanting it is *recorded*. A system can be
universal and blind.

What softens it: the fields are cheap to add and the readings over them are
terms, so a new signal costs one field and no new architecture. What does not
soften it: choosing the field is a human act, and there is no argument here that
the set is complete. It is not.

**The destinations.** Seven places a term can go. The set is closed because it
is a fact about the substrate rather than a taxonomy — seven things hold a term
and can be handed a different one — but an eighth kind of thing to hold a term
is an edit.

**The unit.** Candidates walked. Objection 3.

**The generator.** The families come from a harness. Experiment J.

---

## Part XII — What this does not claim

It does not claim optimality; Theorem 4 forbids it. It does not claim complete
opportunity detection; Theorem 5 forbids it. It does not claim she can find
stepping stones; question 35 is an honest no and question 36 follows it. It
does not claim the improvements measured here are large. It does not claim
anything about wanting, beyond a quantity, a reading and a choice that follows
the reading.

What it claims is narrow and checkable: the thing that decides whether to
change her language is a ranking over one choice set, its inputs are readings
of a record she keeps, the ranking and the things it ranges over are terms of
the same substrate, and every stage of every developmental episode says who
started it.

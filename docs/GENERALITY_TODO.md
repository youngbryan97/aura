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

## The clock chain, traced end to end (2026-08-28)

One request — "read this API.md and tell me what these two functions do" — was
followed from the canned refusal back to its causes. Fourteen defects, each
committed with a test. They are listed because the shape repeats: every layer
had its own copy of a judgement, and each one that was fixed revealed the next.

- A word in the request conjured an effect the tool could not perform: "post an
  invoice" refused as `network_write` on a sandbox.
- The tool loop was given exactly as many turns as calls, so answering was
  itself a call.
- The consent a request carries was set only for the artifact ceiling, leaving
  the self-service ceiling asking for a confirmation nobody can give.
- The protection that stops a tool call being clamped below the size of a call
  read `options["tools"]`, which is None on the JSON contract — the path that
  needs it most.
- A tool call was sized by felt vitality. Half a call is no call.
- The time kept back for the answer was a constant while the thing it reserves
  for grows with what the tools return.
- A turn with two phases was timed as though it had one, because the clock sat
  behind an entitlement about answer LENGTH.
- The clock's arithmetic had no term for reading the prompt.
- The read rate was measured and never persisted.
- `seconds_to_decode` could not speak for any budget worth extending: forty
  readings, zero comparable to 1536 tokens, because long generations are rare —
  partly because the deadline it could not extend kept cancelling them.
- A gate that never ran was reported as a gate that said no, because
  `passed` starts false. Every failure became `surface_quality_rejected` and
  sent every investigation to the wrong subsystem.
- Four keys hold quality reasons and the refusal read one, then three.
- The refusal could name neither its objection nor the draft it objected to.
- A tool receipt refused for want of turn custody left the same trace as a tool
  that never ran, which is none.

**Where it stands.** The clock now fires — "1536 tokens decode in about 305s and
the prompt takes about 7s to read; deadline 305s → 628s" — the tool call
succeeds, the file is read, and the tool-result rescue serves what was found
rather than a canned line.

**H3 is the live blocker, and this is the evidence for it.** The last generation
produced 1186 characters carrying `unanswered_question_part`, and the prompt it
answered was `scaffold=5180 request=213`, a ratio of 24. The gate was right to
withhold it. The question is why a model given a file and a plain question about
it writes something that does not answer the question, and the scaffold is the
first place to look.

**A fifth clock exists and is not a defect.** The route holds a wall-clock UI
deadline (`_foreground_timeout_for_lane`) that the gate's extension does not
reach. The model wanted 628 seconds and the UI allows about 180. That is a real
conflict rather than a bug, and it should be decided rather than discovered.

## H3, measured (2026-08-28)

The question was 213 characters. The prompt was **50,500**, and it took the
resident model **191.6 seconds to read** — the whole turn — before it could
produce a token.

    system  46,996   "You are Aura Luna. Here's who you are..."
    system     321   established world cascades
    system   2,970   internal memory recall
    user       213   the question

Inside the largest message:

    ## CAUSAL VALENCED WORKSPACE      5,568
    ## INTRINSIC IDENTITY ANCHOR      4,528
    ## LIVE MIND CONTEXT              3,410
    ## IMAGINATION WORKSPACE          2,576
    ## CODING WORKING SET             2,462
    ## LIVE TOOL OPTIONS              2,080
    ## GOAL EXECUTION STATE           1,966
    ## DIALOGUE SOURCE ATTRACTORS     1,949
    ## SELF-HONESTY REQUIREMENTS      1,172
    [WORLD MODEL BELIEFS]             1,150
    ## GOAL EXECUTION STATE           1,078   <- the same header, twice
    ## TIMESCALE RECONCILIATION       1,039

Two findings sit in that table.

**The identity lock is 2,353 characters and the message is 46,996.** The rest is
Aura's live inner state, assembled fresh and sent whole on every turn, whatever
was asked. Reducing the two visible scaffold blocks — the response contract and
the reliability contract, both of which recited rules that gates already enforce
— saved about four kilobytes of fifty. Worth doing and not the lever.

**The gate trims and the deep path does not.** The inference gate builds a
prompt plan with a budget and an ordered list of important headers, logs
`mode=compact` and a scaffold breakdown, and drops what does not fit. The deep
cognitive path assembles its own prompt, sends everything, and logged nothing
but a total until the breakdown above was added. That asymmetry is H3.

The next step is not to delete sections by taste. It is to give the deep path
the same budget the gate already has, and to let what survives be decided by
the request rather than by what happened to be assembled.

## The ledgerkit item, and where it actually stands (2026-08-28)

"Read the docs at this path, then use the library" — teach herself a tool from
its documentation — was followed all night. Every blocker found was real and is
fixed:

- "post an invoice" refused as a network write, on a sandbox that cannot reach
  a network
- the tool loop given as many turns as calls, so answering was itself a call
- consent set only for the artifact ceiling, leaving the self-service ceiling
  asking for a confirmation nobody can give
- the tool-call protection reading a field that is None on the JSON contract
- a call sized by felt vitality
- the answer's reserve a constant while what it reserves for grows
- a two-phase turn timed as one phase
- the clock with no term for reading the prompt
- the read rate measured and never persisted
- the decode estimator unable to speak for any budget worth extending
- a gate that never ran reported as a gate that said no
- and finally: `sys` banned in the sandbox, which is right, with sys.path the
  only way to import a directory — so using a named library was impossible by
  construction

**The capability itself now works.** In the sandbox, end to end, the ledger
posts both entries, reverses one, and returns Accounts Receivable 25000,
Revenue −25000, Hosting Expense 0, Accounts Payable 0, summing to zero. That is
asserted in `tests/test_a_library_the_person_named_can_be_run.py`.

**What it does not do is fit in a desktop turn.** The last live run read three
files cleanly, and the answer clock asked for 818 seconds — 1,536 tokens at the
measured decode rate is 404 seconds on its own, twice for a turn that fetches
then answers. The route holds a wall-clock UI deadline of about 180.

That is not a defect to find. It is the fifth clock, and it is a decision:
either the desktop turn gets longer for requests that fetch and then answer, or
the answer for those requests gets shorter, or the work moves off the turn and
comes back when it is done. The measurements to make that decision are all in
the log now, and none of them were before tonight.

## Where the answer actually stops (2026-08-28)

Three runs of the same two-part question, and every reply ended mid-structure at
about a thousand characters: after "Bake it the same way", after "look at that
floating object", after "Label this Bowl A". The next line each time would have
been the second half of the answer.

It is not a harness cap. The generation had 1,024 tokens and used about 260. The
stop sequences are chat-control tokens only — nothing that cuts inside a list.
The model emits end-of-turn there.

What the harness now does correctly around it:

- The request is read as two asks. "Design me X, and say Y" split into one
  segment because `design` was a directive verb and `say` was not, so coverage
  had nothing to compare against.
- The coverage gate fires on the short draft, live, as `unanswered_question_part`.
- Repair is handed the missing text in the person's own words rather than a
  generic instruction to answer every part.

What remains is the model stopping early on a long scaffold, which is the same
finding as H3 from the other end: 46,996 characters of system message before a
213-character question, and an answer that gives up a quarter of the way into
its own list.


---

# One theory of competence failure (added 2026-08-28)

An external reading of the codebase, recorded here because the criticism is
right and because it changes what the rest of this file is for.

The claim: Aura has several genuine general learning mechanisms and no general
process that decides *which one a failure calls for*. Different failures flow
to different specialists — a transformation problem reaches relation
induction, a named missing skill can reach Hephaestus, some patterns reach
ontology discovery, some evidence reaches model adaptation — and nothing ever
asks the question underneath all of them:

> I tried to understand or solve this and failed. Which level failed: missing
> knowledge, insufficient evidence, inadequate search, a missing procedure, a
> missing concept, an inadequate representational language, or a missing
> executable capability? What is the smallest change to myself that removes
> that deficiency?

The loop that would answer it:

    observation → attempt → failure → diagnose which representational level
    failed → propose the minimal cognitive change → validate it independently
    → admit it → persist it → reuse it → compose it → measure transfer

with the proposed change allowed to be any of: more evidence, more search, a
new procedure, a new predicate, a **new representational primitive**, a new
executable capability, or — rarely, and gated — an implementation change.

## U1. The diagnosis that does not exist

There is no `why_did_this_fail` that returns a *level*. Build one, and make
every learning mechanism a subscriber to its verdict rather than an entry
point of its own. This is the spine the rest of the items hang off.

Falsifiable: a failing episode from each of the six levels, each routed to the
mechanism that can remove it, none routed by hand.

## U2. Where each mechanism is boxed in

Named precisely, so each can be checked and each has its own exit:

- Relation induction does genuine induction over a **constrained semantic
  substrate**. (The metalanguage work in this file is the exit.)
- The procedure inducer synthesises programs over a **supplied vocabulary**.
- Ontology discovery creates predicates over **supplied features and supplied
  logical operators** — so it invents within a language it did not choose.
- Hephaestus writes arbitrary Python but is triggered by a **requested missing
  capability**, not by representational insufficiency, and depends on LLM
  generation.
- Several concept/abstraction systems are templated, or are not demonstrably
  inside a closed production loop.

## U3. A synthesiser that is not fed

The claim is specific and checkable: the recurring semantic-gap synthesiser
that would connect experience to capability generation **has no input**. That
is the half-wired class this codebase has produced before — a writer with no
reader, a rule that could never match. Verify it, and either feed it or
delete it.

## U4. Authority when representations disagree

The architectural transition this codebase is actually in. There are several
legitimate ways to represent self, more than one moral system, several world
models, several cognition loops. The question has stopped being "how do I
build cognition" and become **"which representation has authority when they
disagree?"**

Consequence to take seriously: adding a locally excellent subsystem can now
make her *less* coherent. Every major cognitive state should have one obvious
reason to exist and one traceable causal consequence.

## U5. A minimal causal spine

Critical runtime complexity is concentrated in enormous functions — the chat
route, the inference gate, the MLX worker, latent-cortex episodes, response
generation, each thousands of lines. The profile is excellent defensive
infrastructure around a core that is very hard to reason about. Not "more
modular files": one spine, where a decision can be traced.

## U6. Counterfactual causal tests, not consciousness scores

The sharpest point, and it applies to work already published here. Define a
sentience score as valence + self-model + integration + memory + agency, then
engineer valence, self-model, integration, memory and agency, and the score is
high. Nothing was discovered. The metric was implemented.

What is worth more than any scalar Φ:

- Lesion welfare — does decision behaviour change as predicted?
- Lesion the functional self — does metacognitive caution disappear?
- Sever workspace broadcasting — do globally dependent tasks degrade?
- Randomise continuity — does identity-sensitive reasoning degrade?
- Disable affective steering with the prompt text held fixed — does behaviour
  change measurably?
- Blind the evaluator.

Each is an ablation with a prediction made in advance. `make` gates exist for
ratchets; these want the same treatment.

## U7. Every subsystem earns its existence by ablation

The stated failure mode is cathedral accumulation: every paper becomes a
module, nearly everything exists, nobody — including Aura — can say what
caused a decision, and the architecture becomes *less* falsifiable as it looks
more impressive. The counter is consolidation, and the mechanism is that a
subsystem which cannot change a measurement does not stay.

## U8. The test that would settle it

> Does Aura repeatedly meet tasks from genuinely new distributions, fail
> because she lacks a concept or a capability, autonomously construct what she
> was missing, validate it without being given the answer, retain it, and then
> measurably improve on later unrelated tasks?

- Once — interesting.
- Ten times inside one transformation grammar — a real result.
- Across coding, visual interaction, scientific reasoning, language, software
  operation, planning and unfamiliar professional work, with the new
  representations and actions transferring between them — the strong claim.

Crossed so far: composing and reusing learned structure, and solving for an
ordering rather than searching a list of them. Not crossed: learning the
**representational constructors themselves**, and the action-side equivalent.

## U9. Prerequisites named in the same reading

Close the reliability problems; finish learned action abstraction; let the
metalanguage expand rather than only its library; establish repeated
autonomous capability improvement on unseen classes; **make long tasks leave
the foreground turn and persist as durable work**; improve multimodal and
environment learning; keep swapping in stronger local cortices.

The durable-work item connects directly to the clock work landed today: the
answer to "this turn needs eight minutes" is not only a longer deadline, it is
that the work stops being a turn.

---

# Where this stands, end of 2026-08-28

Written down because five hours of it was one chain, and the chain is worth
more than the commits are separately.

## Closed today

**The clock chain.** A desktop turn passes through five nested deadlines. Each
was a flat number chosen before anything knew what the answer would cost, and
each capped the next, so raising an inner one changed nothing: the engine was
allowed 480 seconds, the gate raised itself to 341, the thinking bound fired
and asked for the answer, and the turn still ended at 144.3 because the chat
route's default is 120.

They read one measurement between them now — what this request costs to decode
at the rate this machine has been measured at, including the reserve the worker
adds for thinking — and they defer to progress rather than to elapsed time. A
generation that is still producing tokens is not out of time. One that has gone
quiet is, whatever its deadline says, and that is caught by the first-token
ceiling, the livelock ceiling, and the sentinel that reads the output.

Bounded by what she is allowed to SAY rather than how long she may take: the
absolute 480-second ceiling, the token cap, the loop sentinel, and the semantic
contract that stops the decode the moment the answer is complete.

**The reserve.** It learned only from a generation that never left the private
channel and returned nothing; the commoner failure — the channel closes, the
answer starts, the budget dies part-way — taught it nothing, so it stood at
zero through every failure it exists to prevent. It also could not survive a
restart (a process that had learned nothing wrote zeros over a proof another
had paid for), could not cross from one worker to the next, and was being
raised on a deadline stop, which is evidence about a clock and not about a
size.

**The thinking budget.** Raising it chased something that recedes: given 1,024
tokens this model used all 1,024 thinking; given 2,048 it used all 2,048; given
4,025 it wrote 15,404 characters of notes and never answered. The answer now
has half the budget reserved for it, and when deliberation reaches its half the
channel is closed in the context the model is reading and the answer is asked
for with what is left.

**The assembled prompt.** Two builders, one budget between them. The one every
desktop conversation goes through sized itself on the model's context window —
about 980,000 characters — while the client refuses to prefill more than 48,000
and cuts the middle out of anything longer. Twenty-seven turns had reached that
cut, each recorded as a fault and felt as friction, and the middle of an
assembled prompt is the mind context. It meets the ceiling deliberately now,
with the request deciding what survives instead of a byte offset.

**U3, the synthesiser that is not fed.** Confirmed exactly. `log_gap` had no
caller in the running system, only in its own test. Joined to the loop that
recognises capability gaps, as recording only. The autonomy score that awarded
a fifth of a point for that component being *registered* now asks whether it
has received anything.

**One shape for every learned rule.** `Node(kind, parameters, [Node...])`,
applying to a state and returning a state. Composition stops being a type, so a
pair nobody wrote a type for is expressible and depth is not a type either.

## Still open, in the order they matter

1. **H3/H4** — the standing injected context, and the eight `engine_directives`
   that are English instruction prose telling the model which phrases to use.
   The second is explicitly banned and still there; each was added to fix a
   specific failure, so removing them needs those failures identified first.
2. **U1** — the diagnosis that does not exist: one `why_did_this_fail` that
   returns a *level*, with every learning mechanism a subscriber to its verdict
   rather than an entry point of its own.
3. **U6/U7** — counterfactual ablations with predictions made in advance, and a
   gate that makes every subsystem earn its place by changing a measurement.
4. **U2, U4, U5, U8, U9** — the boxed-in mechanisms, authority when
   representations disagree, the causal spine, the test that would settle it,
   and the prerequisites.
5. **A7** the action side; **C7** a second concept from a different signature;
   **D2** the battery scored with the model beside it; **E2/E3** the improver
   measured against held-out scores and on a new capability class.
6. **The battery.** Nothing may cost money, ever. Operate unfamiliar SaaS; book
   something free; teach herself a tool from docs; open-ended research; a
   scientific reasoning loop; a novel computer task with no dedicated skill;
   multi-tool coordination on an underspecified goal; competing long-horizon
   goals under interruption; an unfamiliar visual environment; a frozen
   unseen-task battery.
7. **World A/B/C**, the qualitative milestone for M₀→M₁→M₂.

## The turn stopped being killed (2026-08-28, 18:36)

Asked why a starter that doubles in five hours smells yeasty rather than sour,
she answered:

> "'Yeasty' means the yeast side of things is active and doing its job;
> sourness comes from the bacteria producing lactic and acetic acid, which
> usually takes more time — especially cooler fermentation. ... Give it more
> total ferment time — bulk until 50-80% volume increase, then cold retard
> 12-36 hours before baking. This is one of the biggest levers for sourness."

Correct on both halves of the question, in three minutes nine seconds, with no
timeout anywhere in the turn. Every previous attempt today ended on the
apology.

Seven clocks were involved and each one hid the next: the chat route's turn
budget, the cognitive cycle, the gate's request deadline, the endpoint budget
split, the endpoint wait, the thread-backed wall-clock watchdog, and the
worker's own decode loop. Raising any of them changed nothing while a smaller
one was still there. They ask one question now — has anything arrived recently
— and the answer comes from one signal written by the lane that decodes.

**What is still wrong.** The answer stops mid-list, at the second item, after
1,552 tokens of a budget it never came close to spending and with no deadline
anywhere near it. The model ended its own turn, and the semantic completion
contract accepted it: there is no "ended before semantic completion" line for
that generation. So the next defect is not a clock and not a budget. It is a
list that stops at item two being read as a complete answer.

---

# Standing: the structure is not the constraint

Said plainly on 2026-08-28, and it applies to everything below.

Aura and the work on her run on the same machine. The code can be read
directly rather than inferred from logs, and a restart is not the only way to
learn what a function does. Where a previous decision is in the way, the
decision can go. Two sessions were spent threading leases through call sites to
satisfy a participant set that turned out to be a second, worse copy of a proof
the runtime already had; deleting it was smaller than the workaround, and the
rule it left behind is one sentence that cannot be forgotten at a call site
because there is no call site.

The test for a change like that is not whether it respects the existing shape.
It is whether the invariant that matters still holds and is still pinned.

# The whole backlog, so none of it drifts

**In flight.** The ledgerkit turn end to end: read the docs, use the library,
report the trial balance — Accounts Receivable +25000, Revenue -25000, Hosting
Expense 0, Accounts Payable 0, summing to zero. Three defects between the ask
and the answer are fixed (the tool budget reaching the loop, the envelope
parser refusing two well-formed calls, and evidence custody refusing the
turn's own children). Needs a clean live run.

**Named in the mandate.**

- A retirement path for library entries. Entries only ever enter; Soar and
  ACT-R both report that symbolic accumulation saturates, and nothing here
  ever removes anything.
- H3 the standing injected context; H4 the eight `engine_directives` that are
  English instruction prose telling the model which phrases to use — the
  explicitly banned thing, still present, each added for a specific failure
  that has to be identified before it can go.
- A7 the action side: new action abstractions built from lower-level ones.
- C7 a second concept formed the same way from a different signature.
- D2 the same battery scored WITH the model, so the difference is measured
  rather than asserted.
- E2 the improver's own output against a held-out score; E3 repeated gains in
  a NEW capability class.

**The capability battery. Nothing may cost money, ever — booking means booking
something free.** Operate unfamiliar SaaS; book something free; teach herself
a tool or API from its docs; an open-ended research project; a scientific
reasoning loop; a novel computer task with no dedicated skill; multi-tool
coordination on an underspecified goal; recover from a mistaken assumption
unaided; competing long-horizon goals under interruption; an unfamiliar
professional domain; infer an unfamiliar visual environment; a frozen
unseen-task battery.

**The unification reading.** U1 the failure diagnosis that returns a level,
with every learning mechanism subscribing to its verdict rather than being its
own entry point. U2 the boxed-in mechanisms. U4 authority when representations
disagree. U5 a minimal causal spine. U6 counterfactual ablations with
predictions made in advance, in place of consciousness scores that measure the
metric somebody implemented. U7 every subsystem earning its place by changing
a measurement. U8 the test that would settle it. U9 the prerequisites,
including making long tasks leave the foreground turn and persist as durable
work.

**From the metalanguage add-ons.** The derived signature lattice; testedness
rather than fit; the World A/B/C milestone for M₀→M₁→M₂. `Composed`
serialisation and the uniform IR are done — every learned rule is one
`Node(kind, parameters, [Node...])`.

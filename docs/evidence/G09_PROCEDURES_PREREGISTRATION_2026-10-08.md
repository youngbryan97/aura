# G09 kept procedures: preregistered before any procedure exists

Written on 8 October 2026 and committed before her model has written a
single procedure, development or held out. The pass criteria below are
fixed here. The development run may change the mechanism's settings; those
settings are appended to this page, dated, before the held-out run starts.

## What is measured

Whether her kept procedures raise the accuracy of her answers on kinds of
problem no development decision touched. A kept procedure is a `solve`
function her model wrote from solved problems of one kind, that agreed with
every known answer of that kind in her OS sandbox, half of them never shown
to it (`core/learning/procedures_from_solved_examples.py`). Her response
phase asks for a kept procedure's answer before the amplifier's search
(`answer_from_kept_procedures`). A request goes to a kind only if it is
nearest that kind and as typical of it as the kind's least typical sealed
problem.

Arms, paired on the same request:

* ordinary: her draft, one greedy decode through her own chat template at
  her serving effort, on the benchmark's own prompt;
* kept: the same request through `answer_from_kept_procedures` with the
  frozen book; where no kept procedure answers, the draft stands.

Both arms are graded by the benchmark's own grader, as restated and checked
in `tools/run_g09_organ.py`.

## Kinds

Development kinds. The mechanism and its settings may change because of
results on these:

* BBEH boolean_expressions, disambiguation_qa, hyperbaton,
  movie_recommendation, sarc_triples (drawn by `random.Random(20261008)`
  from the 23 tasks before any was read);
* Natural Plan calendar scheduling.

Held-out kinds. No development decision reads their problems or results;
only the frozen induction run reads their development halves:

* the other 18 BBEH tasks;
* Natural Plan trip planning;
* CRUXEval output prediction.

Development and test problems never mix
(`tools/induce_g09_procedures.py`): a BBEH task's test problems are its 20
in BBEH Mini; a Natural Plan kind's and CRUXEval's are the second half when
ordered by a hash of their id.

## Requests, by G09's five domains

* math: every MATH test problem outside MATH-500 (4,500), and AIME 2024 and
  2025 (60);
* code: every CRUXEval test problem (400), and every HumanEval+ problem
  (164);
* planning: every calendar test problem (500), every trip test problem (800);
* knowledge: every HotpotQA distractor validation question (7,405);
* transfer: the 18 held-out BBEH tasks' Mini problems (360).

A request that no kept procedure answers keeps its draft, so its pair cannot
be discordant. Such requests are counted by routing alone, without a draft.
Drafts are decoded for every request a kept procedure answers, and for every
request in CRUXEval, calendar, trip and the BBEH Mini set, whose drafts also
serve G12.

## Claims, fixed before the run

* Primary: pooled over the five domains, kept answers gain more requests
  than they lose; exact two-sided McNemar, alpha 0.05.
* Breadth: in at least two domains, counting held-out kinds only, gains
  exceed losses at two-sided alpha 0.05 within the domain.
* No harm: no domain has losses exceeding gains at two-sided alpha 0.05.
* G09 closes only if all three hold. A gain in one domain is reported as
  that.

Reported beside them: each kind's procedures and the known answers they
agreed with, how often each kind's procedure answered, its gains and losses,
the development kinds on their own, and what the procedures cost to write
(decoded tokens and seconds) and to run (seconds a request).

## Resources, stated

The kept arm has what the ordinary arm does not: each kind's development
problems with their answers, from which its procedures were written and
against which they were checked. That is the mechanism under test. The
ordinary arm sees only the benchmark's own prompt.

## What this does not show

* Anything yet: this page is the plan.

# G09: kept procedures, attempt 5 (part 1 committed 16 October 2026 at 10:33, before any routing is measured)

Attempt 3 did not close G09 once both arms were graded alike, and attempt 4 did not either: primary 34 gained to
0 lost, no harm, but breadth only in planning, because attempt 4's route declined every math request whose
kinds share their wording, the ones a procedure would have helped
(G09_PROCEDURES_RESULT_3_2026-10-16.md; g09-attempt4-analysis/verdict.json). Attempt 5 is drafted on fresh test
requests, graded with one function per domain for both arms (tools/g09_one_grader.py), at two-sided p below 0.01
(0.05 over five attempts).

This first part fixes how the route is chosen, before any route is measured. Part 2, committed before any test
request is drafted, names the route chosen, the requests and the code.

## The book

Attempt 4's book (`g09-attempt4-routing-20261015/book.json`: the prefill's procedures, distinct signatures),
with each kind's meaning and shown examples taken from her live book written on 16 October at 04:21
(`g09-live-default-20261015/live_distinct.json`), which computed them from the kinds' development halves.

## The routes compared, on development requests only

1. Words: attempt 4's route (distinct signatures; several kinds near enough answer only when all agree).
2. Meaning: the kind her embedding finds by meaning (core/learning/kinds_by_meaning.nearest_kind: at least as
   near as the kind's least typical sealed problem, standing out from all kinds and clear of the next), whose
   procedures then read the request as it stands.
3. Words, then meaning: route 1, and route 2 where route 1 declines.

The development requests are tools/measure_kept_routing.py's (12 from each kind's development half, seed 7,
and 300 MATH training problems posed as the harness poses them), each graded by its own kind with
tools/g09_one_grader.one_grade.

## The rule

The route chosen is the one that answers no development request wrongly and, of those, answers the most
rightly. If every route answers some request wrongly, the route with the fewest wrong answers is chosen, and
ties go to the route answering fewer requests.

## Domains

Math (DeepMind mathematics, the 27 held-out kinds attempt 3 drafted) and planning (Logistics and Depots).
Transfer is not drafted: her drafts on BIG-Bench Hard were right on 298 of 300, so it has no room to show a
gain, and breadth needs two of the domains drafted. Criteria otherwise as attempt 3's: primary pooled over the
domains, breadth in both, no harm in either.

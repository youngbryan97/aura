# L09, the next attempt: learned where she is wrong, on what people ask

Bryan, 19 October 2026, after the second attempt: where should the next attempt's requests come from, given that
ordinary requests rarely reach her kept procedures? His question was what makes sense for the architecture and for
what we are building. This is the answer, written before any of it is drawn.

## What the two attempts showed

Her book of kept procedures was learned mostly from benchmarks (DeepMind mathematics, CRUXEval), with one kind learned
in use, from a person's answers in her chat (date differences, the first half of L09). Over her own chat history the
book reached 9 of 195 requests and gained 3, each a number-theory request her reasoning left unfinished. Over people's
WildChat requests it reached 33 in 17,317, 25 of them misroutes, and on the 8 it could settle she was right without it
7 times. Kept procedures matter where her model is wrong, and the kinds a benchmark teaches are mostly not where people
find her wrong.

## The plan

L09's claim is the loop: what she learns in use improves her use. So the next attempt measures the loop on the kinds
of request people make that she gets wrong, rather than on the kinds her book happens to hold.

1. **Where she is wrong.** From WildChat requests not used before, every one a program can settle (a date
   difference, a count of letters in a word, a unit conversion, an exact product or quotient, a percentage, an age
   from a birth date, a day of the week), drawn by kind in the order the requests come, and answered by her model in
   her chat's own way. The kinds kept for the test are those she gets wrong at least as often as right. The kinds and
   their programs are fixed before any answer is read.
2. **Learned in use.** For each kept kind, the first requests in order are put to her in her chat with the right
   answer given after her reply, as a person corrects her, until her learner keeps a procedure for the kind
   (core/learning, the first half's loop). Nothing is written for her; what she keeps is what she induced.
3. **Measured on fresh requests.** The requests of those kinds that come after the teaching, never shown to her,
   each answered with her kept procedures on and off, by her chat's own route with the restatement check of
   37021f4ba. One grader for both arms, the answers fixed by program before any draft, every discordant pair read
   by hand. Counting stops at the number of pairs whose discordance rate, read from step 1's error rate, gives five
   discordant pairs or more with probability 0.9.

## What it can and cannot show

That what she learns from being corrected improves her answers to what people ask of the same kind. Not that her
book helps on every kind, and not that people correct her: the corrections are the programs' answers, given as a
person would.

## Before drawing, 20 October 2026, 01:20

Counted on the request texts only, before any program or answer: in the 138,500 WildChat-1M requests attempt 2 never
routed (its population from w017317 on), the kinds above are rare as whole questions: about 7 products and sums, 2
percentages, 2 date differences, 1 day of the week, no letter count and no plain unit conversion. Too few to find
kinds she gets wrong, teach them, and then test on fresh ones. So the requests come from WildChat-4.8M (86 shards,
every English user turn, not only the opening one, left out as attempt 2 left out: toxic, redacted, flagged), less the
requests L09's first two attempts used.

The kinds and their programs are fixed in tools/l09_where_she_is_wrong.py before the draw: product, quotient, sum,
difference, percentage of, what percentage, days between two dates, day of the week, letters in a word, unit
conversion (length, mass, volume, temperature; exact factors). A request is drawn only when the whole of it is one
such question with its operands in it, at most 200 characters; a date counts only with its month as a word or as
YYYY-MM-DD. Left out: an age from a birth date (its answer depends on the day it is answered, and these requests
carry the day they were typed, not the day she answers), and "what day is it". Sum and difference are added; they
are what people ask most among these.

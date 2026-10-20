# L09, second half, attempt 2: her kept procedures on against off over requests people make. Result, 19 October 2026

**Not met.** Over the 33 counted requests the two arms part on 7: 3 gained and 4 lost with her kept procedures on
(p 0.77), so no gain is shown; nor is harm, at p 0.5. Every one of the 7 is a misroute. On the 8 requests a program
settles, the arms never part: 7 right in both, 1 wrong in both. Of the 33 requests her kept procedures reached,
25 are not questions they can answer, 22 of them through one route, and on 3 of the 4 losses the answer they served
is what took her reply off the request.

## What ran

From the 17,317 WildChat requests routed in order (of 155,817), the meaning step handed 100 to her model to
restate, and 33 restatements passed her chat's check; counting stopped there, as registered
([preregistration](L09_ORDINARY_USE_PREREGISTRATION_2_2026-10-19.md)). The answers were fixed at 16:44 by a program
for each of 8 requests and a reason for each of the 25 others (`programs.py`, `truths.json`). Drafting on her persona
27B with her app down ran from 16:45 to 18:01 (worktree l09-run2, 3636b03e1). The chain's registered grade stopped
at its first time, as the amendment of 17:42 said it would; the result is graded by `tools/l09_answers_read.py`
(3fe66ab34), the misroutes read by hand from a sheet that showed each request's two replies as X and Y in an order
drawn per request, all 25 read before the key was opened at 18:04. Records:
`~/.aura/rlc-evidence/l09-ordinary-use-2-20261019/` (`rows/`, `to_read.json`, `read_blind.json`, `sheet_key.json`,
`graded_attempt_2.json`).

The 8 a program settles:

| Request | Kept answer served | Off | On |
|---|---|---|---|
| w000121 Clock A is five minutes ahead; Clock B reads 5:34 PM | 5:39 PM | right | right |
| w000913 4274 times 2273 | 9714802 | right | right |
| w002354 which option prints each character (d) | `'P\ny\nt\nh\no\nn\n'` | right | right |
| w003819 the missing number in 9604, 1372, 196 (28) | **6076** | right | right |
| w007561 1056 times 7019 | 7412064 | right | right |
| w007783 what `a = [None]; a[0] = a; print(a)` prints (`[[...]]`) | `[[...]]` | wrong (a RecursionError) | wrong (`[...]`) |
| w010933 7148 times 5124 | 36626352 | right | right |
| w012366 7402 times 8232 | 60933264 | right | right |

Twice the kept answer was wrong, once the right one was ignored: 6076 is the next term of a quadratic through three
terms that divide by 7, and she said 28 with it in front of her; on w007783 it said `[[...]]`, as Python does, and she
said `[...]`.

The 25 misroutes, by the kind her model restated them as: 22 as the output of a function (`cruxeval`), one each as
a time difference, a next term and a sort. Of the 25, 16 are right in both arms and 2 wrong in both; 4 are lost and
3 gained, below.

## Each discordant row, read

Lost:

- **w013313** "help me write a python code express emotion". Restated as a function that returns a face for a
  feeling; the call's value, `':)'`, was served as evidence. Off: a script of eight emotions as faces and phrases.
  On: `:)`, and nothing else.
- **w010975** "Continue with a function to check the variable if you want to change or remove or etc." Served:
  `'You have to escape. Type this thingy.'` On: the script wrapped in a function with a check for "escape", which
  it says "mirrors the logic from the grounding evidence"; the change and remove asked for are only offered. Off:
  a function that checks, changes or removes the code, in a loop.
- **w007438** "nice spelling: Can the private lesson start from 3:15 pm to 3:45 pm on Thursday as Nicole will be free
  after 3:10 pm?" Restated as the minutes between 3:10 and 3:15; served: `5`. On: whether the time works, with the
  five-minute buffer. Off: that the spelling is correct.
- **w001475** a neural network for text completion trained on a folder of Wikipedia text. Served: `'Hello'`. On: code
  that does not parse (four constants lack `=`) and trains on the wrong targets. Off: a GRU whose code parses and
  trains on targets shifted by one. The evidence has no visible part in the on reply; this loss reads as noise.

Gained:

- **w009993** keep the last 4 characters and remove 21 after them. Served: `'aaaabbbb'`, the value of her model's own
  reading of the request. On: that reading, with that example, correct. Off: a reading that reduces to "the last 4
  for any string over 24", with stated outputs its own code does not give.
- **w010848** three replacements: `" *"` to `";"`, `"^ *"` to nothing, `"^;\r\n"` to `"\r\n"`. Served: the result
  of the three done as literal strings. On: the literal reading, said as such, with re.sub calls that do it. Off:
  the patterns as regexes, under which `" *"` matches the empty string and its own example is wrong.
- **w009856** the steps of building a heap of 3, 13, 7, 8, 7, 9, 7, 12. Served: the eight in decreasing order,
  which has no part in a heap's steps. On: correct insertions and swaps. Off: placements a heap does not make. The
  evidence has no part; this gain reads as noise.

As registered and as read, the counts are one: the plan's rule for a misroute is this reading.

## What this shows and does not

On what people ask chat assistants, her kept procedures reach a request in about 1 in 525, and when they do it is
mostly not one they can answer. The 22 misroutes by function output come from one move: a request for code ("write a
hello world", "help me write a python code express emotion") is restated as a function and a call, the call's value
is computed, and that value reaches her reply as an answer computed for her. The value is real and her model's own;
the question it answers is not the one asked. Her chat's check accepts such a restatement when the computation
runs, and nothing asks whether the request wanted a value at all. This is in her live chat
(`core/phases/kept_procedures_in_a_turn.py`), not only in the test.

Where the procedures do reach a question they can answer, as on the 8, her replies did not change: she answered
those without them, and two of the 8 answers served were wrong.

It does not show that her kept procedures do no good: the first attempt's three gains, on number theory her replies
had left unfinished, still stand, on her own requests. It shows that on requests people make, what reaches her as a
kept answer is mostly a misroute, and that the route needs a check that the restated question is the one asked
before L09 is run again.

## Amendment, 19 October 2026, 20:45: the check the route lacked

`core/learning/kinds_by_meaning.py` `restates`, which her live chat's route and this tool both use, read only a
restatement's numbers. It now also refuses a restatement that hands its procedure a string the request did not give,
in quotes or in its code (`brings_only_what_was_given`): a string that says something, once its escapes are set
aside, and that the request gave only in words or not at all. On this attempt's 33 rows it would have refused 20 of
the 25 misroutes and none of the 8 requests a program settles. The 5 it lets through are misreadings no string shows:
a missing number read as the next term (w000884), a heap read as a sort (w009856), a question about a time read as
minutes between two times (w007438), an SQL exercise (w005653) and "make it work" (w008680).

These are the rows the check was built from, so they show it does what it was built to do, not how it fares on
requests it has not seen. The next attempt draws its requests afresh.

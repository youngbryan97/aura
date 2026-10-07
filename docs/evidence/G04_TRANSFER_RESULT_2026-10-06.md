# G04 transfer: construction and family carry, vocabulary and depth do not

The plan was published before any fresh request was decoded
([preregistration](G04_TRANSFER_PREREGISTRATION_2026-10-06.md), plan
`4989717df596`, pushed in `4e385cee3` at 15:19:37 on 6 October). This is its
result, run as frozen.

## Run

Features: `~/.aura/rlc-evidence/g04-transfer-v1-features`, manifest
`86de0a4f8e5430319c504573783e1cf326d3e30d13798c7a157d1e86e9b277f3`, 248
requests read by the persona 27B in one worker with the live runtime shut
down. Decode: `tools/run_g04_transfer.py` from a worktree pinned at
`67d4ba89e`, every frozen identity rechecked before the first decode, nothing
else computing. Rows: `~/.aura/rlc-evidence/g04-transfer-v1-run/rows/`.

The runner wrote all 496 rows (248 tasks, two arms) and then failed writing
its summary: a numpy boolean does not serialise to JSON. The analysis is the
plan's and depends only on the rows, so it was computed from them by the
independent verifier below. The runner is fixed for later runs; the copy this
run used is unchanged, as the plan froze it.

## Result

Primary outcome: the decoded program means what the reference program means.

| Stratum | Candidate | Incumbent | Candidate only | Incumbent only | Exact one-sided p | Rejects at 0.0125 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| construction | 62 of 62 | 61 of 62 | 1 | 0 | 0.5 | no |
| vocabulary | 21 of 62 | 10 of 62 | 20 | 9 | 0.031 | no |
| depth (six and seven steps) | 8 of 62 | 3 of 62 | 5 | 0 | 0.031 | no |
| family | 59 of 62 | 47 of 62 | 12 | 0 | 0.00024 | yes |

95% intervals for the candidate: construction 0.94 to 1.00, vocabulary 0.22
to 0.47, depth 0.06 to 0.24, family 0.87 to 0.99.

Every program in both arms was chosen with program execution unavailable (248
of 248 each): what was decoded is neural scoring and a typed constraint
search, and execution ran only afterwards, for the answer.

## What it means

* Construction: new scaffolds (a question, numbered steps, a bullet list, a
  supposition) are read correctly. The incumbent reads them too, so there was
  nothing for the paired test to count.
* Family: a sequence operation taking a computed argument, which no consumed
  program had, is read by the candidate 59 times in 62 and by the incumbent
  47. This is the transfer the plan was built to detect.
* Vocabulary: new operation phrasings mostly fail. The word readout that
  closed "after removing" in G03 carries some of them (20 against 9), and the
  test does not reach the plan's alpha.
* Depth: six and seven steps mostly fail. The candidate decodes them in two
  seconds or less and gets a different program; the incumbent also runs out
  of search on twelve.

The plan's claim was that all four comparisons reject. Three do not. G04 stays
open, on vocabulary and depth.

## Independent verification

`tools/verify_g04_independently.py` shares none of the grading path. From the
published plan and raw rows alone: the plan matches its spec, the task
commitment hash recomputes, the 248 committed tasks regenerate from the
published seed with matching hashes, none repeats a consumed request, all 248
reference answers agree with its own interpreter, and every count, exact test
and interval above is its recomputation.

With `--redecode`, a separate process decoded all 496 task-arm pairs again
and graded each program with the verifier's own interpreter over 256 random
probes, without the runner's grader:

| Check | Result |
| --- | --- |
| verdicts agreeing with the runner's | 496 of 496 |
| program hash reproduced, candidate | 248 of 248 |
| program hash reproduced, incumbent | 243 of 248 |

The five incumbent decodes that did not reproduce are depth requests on
which its search ran to the 20-second budget; a search bounded by time can
stop in a different place. The decoder itself has no second implementation,
so what is independent is the process, the reproduction check, the grading
and the statistics. `independent_verification.json` and
`independent_verification_redecode.json` are beside the rows.

## What this does not show

* That vocabulary or depth transfer is impossible for this architecture: one
  candidate, one population.
* Anything about her public answers; that is G05.
* These 248 requests are now consumed. A next candidate needs a new plan and
  new requests.

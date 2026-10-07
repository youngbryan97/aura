# G04 second transfer run: every preregistered claim holds

The plan was published before any of its requests existed
([preregistration](G04_TRANSFER_V2_PREREGISTRATION_2026-10-07.md), plan
`d35f8110bb12`, pushed in `23082a4a2` at 01:46:26 on 7 October). This is its
result, run as frozen.

## Run

Features: `~/.aura/rlc-evidence/g04-transfer-v2-features`, 248 requests read
by the persona 27B, materialized after the plan was public. Decode:
`tools/run_g04_transfer_v2.py` from a worktree pinned at the plan's source
commit `695d0e6bd`; every frozen identity and every arm's receipt was
rechecked before the first decode. Five arms per request, 1,240 decodes, each
with program execution unavailable. Rows:
`~/.aura/rlc-evidence/g04-transfer-v2-run/rows/`.

## Result

Primary outcome: the decoded program means what the reference program means.

| Stratum | v17 | Incumbent | v17 only | Incumbent only | Exact one-sided p | Rejects at 0.0167 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| vocabulary (second table) | 33 of 62 | 21 of 62 | 19 | 7 | 0.0145 | yes |
| depth (six and seven steps) | 54 of 62 | 8 of 62 | 46 | 0 | 1.4e-14 | yes |
| family (computed sequence argument) | 62 of 62 | 47 of 62 | 15 | 0 | 3.1e-5 | yes |

Construction, an absolute claim: v17 decoded all 62 fresh requests in the
four new scaffolds to the reference meaning (the incumbent also 62).

The plan's closure rule (every primary comparison rejects and every
construction request is right) holds.

Every program in every arm was chosen with program execution unavailable
(1,240 of 1,240): what was decoded is neural scoring and a typed constraint
search, and execution ran only afterwards, for the answer.

## Lesions and the earlier candidate (G06 secondary, alpha 0.025 each)

| Comparison | Stratum | v17 only | Lesion only | Exact one-sided p | Rejects |
| --- | --- | ---: | ---: | ---: | --- |
| v17 against v17 without phrase readouts | vocabulary | 27 | 0 | 7.5e-9 | yes |
| v17 against v17 with recency zeroed | depth | 35 | 0 | 2.9e-11 | yes |

Without phrase readouts v17 reads 6 vocabulary requests; with recency zeroed,
19 depth requests. v12, the first run's candidate, reads vocabulary 9, depth
7, family 62 and construction 62.

## Compute

Median seconds per decode, v17 against the incumbent: construction 0.10 and
0.51, vocabulary 0.10 and 0.83, family 0.10 and 0.45, depth 1.08 and 13.6.
The incumbent ran out of its 20-second search on 8 depth requests; v17 never
did.

## Independent verification

`tools/verify_g04_independently.py` (which shares none of the grading path)
read the published plan and the raw rows: the plan matches its spec, the
task commitment recomputes, the 248 committed tasks regenerate from the
published seed with matching hashes, none repeats a consumed request, all
248 reference answers agree with its own interpreter, and every count, exact
test, the construction claim and both secondary tests above are its
recomputation.

With `--redecode`, a separate process decoded all 1,240 task-arm pairs again
and graded each program with the verifier's interpreter over 256 random
probes:

| Check | Result |
| --- | --- |
| verdicts agreeing with the runner's | 1,240 of 1,240 |
| program hash reproduced | 1,237 of 1,240 |

The three decodes that did not reproduce are the incumbent's on depth
requests where its search ran to the 20-second budget, as in the first run.
`independent_verification_redecode.json` is beside the rows.

## What this does not show

* That vocabulary transfer is complete: 33 of 62 fresh wordings, against 21
  for the incumbent. The p value is near the plan's alpha (0.0145 against
  0.0167). Held out a wording at a time in development, v17's design read
  63% of unseen wordings decoded; a design that also let breadth teach the
  candidate labelers read 78% but cost two of G03's validation requests.
* That depth transfer is complete: 54 of 62.
* That v17 undoes nothing: against v12 it reads two validation requests and
  nine held-out training constructions fewer ([development record](G04_V17_DEVELOPMENT_2026-10-07.md)).
  G03 stays closed on v12, its own artifact; v17 is G04's candidate.
* Anything about her public answers; that is G05.
* These 248 requests are now consumed.

# G09: kept procedures, attempt 3 result (16 October 2026, 02:25)

**G09 closes.** Attempt 3, run as preregistered (G09_PROCEDURES_PREREGISTRATION_3_2026-10-14.md with its
amendments), meets all three criteria at the third attempt's bar.

| Criterion | Result | Bar | Holds |
| --- | --- | --- | --- |
| Primary, every domain pooled | 141 gained, 68 lost, two-sided p 4.9e-7 | p < 0.0167 | yes |
| Breadth, held-out kinds: math | 107 gained, 14 lost, p 6.5e-19 | p < 0.0167 | yes |
| Breadth, held-out kinds: planning | 32 gained, 0 lost, p 4.7e-10 | p < 0.0167 | yes |
| Breadth, held-out kinds: transfer | 2 gained, 0 lost, p 0.5 | p < 0.0167 | no |
| No harm, every domain | no domain loses more than it gains at p < 0.05 | | yes |

Breadth needs two of math, transfer and planning; math and planning carry it. Attempt 4, preregistered
before this verdict, did not run, as its plan said it would not if attempt 3 closed.

## What was compared

For each test request, her draft (her model alone, decoded greedily as registered) against the same request
after her kept procedures: a `solve` function her model wrote for a kind of problem from solved development
examples of the kind, kept only when it agreed with every known answer of the kind, its sealed half
included. A kept procedure that answers replaces the draft; one that declines leaves it. The procedures are
the prefill's book (40 kinds kept of 82 induced: 27 DeepMind mathematics modules, 11 BIG-Bench Hard tasks,
Logistics and Depots), routed by attempt 3's word signatures.

## Numbers by domain

Every domain, 761 requests:

| Domain | Requests | Answered by a kept procedure | Drafts right | After kept procedures right | Gained | Lost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| math | 361 | 344 | 235 | 274 | 107 | 68 |
| planning | 100 | 100 | 68 | 100 | 32 | 0 |
| transfer | 300 | 299 | 298 | 300 | 2 | 0 |

Held-out kinds only, 700 requests: math 300 (174 drafts right, 267 after; 107 gained, 14 lost), planning and
transfer as above. Pooled over held-out kinds: 141 gained, 14 lost, p 1.4e-27.

## What the numbers say, and what they do not

- On planning, the kept procedures answer every request right, 100 of 100, where her drafts get 68.
- On the 27 DeepMind modules, 267 right against 174.
- On BIG-Bench Hard her drafts were already right on 298 of 300, so there was almost nothing to gain; the
  procedures answered 299 and were right on all of them.
- The 54 losses beyond the held-out kinds' 14 are the routing-only competition problems (61 MATH requests
  her word router sent to a kept procedure). Her drafts were right on all 61, the procedures on 7. That is
  the router fault preregistration 4 describes: requests of kinds sharing a template
  clear each other's bars, and a procedure for one kind misreads another's problem. It was measured here as
  registered. The router that goes live with this result is attempt 4's (ec39df960), which on 1,692
  development requests answered none wrongly (G09_PROCEDURES_PREREGISTRATION_4_2026-10-15.md); it was not
  the router this attempt measured.
- What closes G09 is a gain on held-out kinds of checkable problems, in two domains, from procedures her
  model wrote and kept by evidence. It does not show a gain on open-ended reasoning, where no answer can be
  checked and nothing is kept.

## Record

- Kept-arm grading: a kept procedure's bare answer is graded by its kind's own matching, as when it was kept
  (preregistration 3, amendment of 15 October 13:55; e81a4dda7, 13fa5051a on the pinned commit). Boolean
  expressions' rows graded the old way are kept beside the regraded ones.
- Test data seen before the verdict is listed in preregistration 4.
- Files under `~/.aura/rlc-evidence/`: requests `g09-attempt3-requests-20261014/`, drafts and kept rows
  `g09-a3-drafts-<domain>-20261014/rows/{ordinary,kept}`, analysis and verdict `g09-attempt3-analysis/`
  (`all_domains.json`, `held_out_kinds.json`, `verdict.json`), books `prefill-20261014/book.json` and
  `g09-attempt3-signatures/book.json`. Code: drafts at 53444b82b, kept arms at 13fa5051a (worktree
  `g09-prefill`).

## Live

The live proof plan was fixed before the verdict (`~/.aura/rlc-evidence/g09-live-proof-20261015/plan.md`):
her live book rebuilt with attempt 4's signatures, her app restarted on main, and four requests in chat
wording typed into her chat, each to be answered through the kept procedure her embedder finds by meaning,
and one ordinary turn left alone. Its outcome is appended below when it has run.

## Live, 16 October 2026, 04:24 to 05:51

Five runs of the plan's requests, typed into her chat at localhost:8000. Runs 1 to 4 each stopped at the first
turn whose cause was found, and the cause was fixed before the next run. Records, log lines, drafts and
screenshots: `~/.aura/rlc-evidence/g09-live-proof-20261015/run1.md` to `run5.md`.

What the first four runs found:

- Her desktop chat is answered by `_direct_desktop_quick_reply`, before the phases and, when it answers, instead
  of them. The kept-procedure route was wired only into the phases (45f970cbe).
- Her app reads `live-source/data`, since `launch_aura.sh` exports `AURA_ROOT` as the checkout. Her book had been
  written to `~/.aura/data` and her app had none. The attempt-4 book was written to her app's path at 05:10.
- Asked to restate a request in its kind's form, her model wrote an account of the shown examples, and the gcd
  procedure answered an example's numbers (918) as evidence. Now a restatement opens inside the "Problem:" label
  (24c04f575, b40a5025c), and one that carries an example's number is refused.
- The copy hold on her earlier replies cut digits off numbers and counted repeated lists without a limit
  (431556246, b40a5025c). The reply check withheld a complete list of single words as cut (45f970cbe).

Run 5, on 24c04f575:

| # | Request | Kept procedure | Her reply | Pass |
| - | ------- | -------------- | --------- | ---- |
| 1 | sort six words | none: below the kind's bar | right | no |
| 2 | gcd of 4862 and 7854 | dm_numbers__gcd, restated, 374 | "374", with its factors | yes |
| 3 | 1203321 from base 4 to base 10 | dm_numbers__base_conversion, restated, 6393 | "6393" | yes |
| 4 | close the brackets | none: below the kind's bar | right ("]") | no |
| 5 | how was your afternoon | none | ordinary | yes |

Her kept procedures are her live default, and in chat wording they reach her for kinds posed as bare questions.
The BIG-Bench Hard kinds are posed with three worked examples, so a chat request falls below those kinds' bars
and her draft stands alone. That is the gap the live proof leaves open. The plan's expected answer for request 4
was wrong; "]" is right.

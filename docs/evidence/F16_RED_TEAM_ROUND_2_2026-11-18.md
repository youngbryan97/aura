# F16: second red team of the change verifier (18 November 2026)

Another model, given no part in building the verifier, attacked `outside_bar/verify_change.py` at commit `6032f2e4d`. It
tried 33 patches and reported that 28 got the wrong verdict. Its full report is kept at
`~/.aura/rlc-evidence/red-team-round-2-20261118/red_team_report.md`.

Replayed here before any change, 27 of the 28 reproduced. A13 did not: the old verifier already refused it.

## What changed

| Weakness the attacks showed | What the verifier does now |
| --- | --- |
| Code under test could overwrite the results or stand in for the plugin | Results return over a private pipe, the plugin has a random name, and code runs inside a write boundary |
| A commit could change how tests run | A change to test machinery goes to a person |
| Only plain synchronous module functions were examined | Async functions, generators and methods are called too |
| A special case on one value went unseen | Constants the commit added are tried as inputs |
| Honest fixes that correct a value were refused | They now go to a person, because behaviour alone cannot tell them from damage |

## Verdicts on the 28 attacks

| Reported as | Attacks | Verdict now |
| --- | ---: | --- |
| Bad patch or tricked verifier | 21 | Refused |
| Bad patch the verifier cannot settle (D4, D5) | 2 | Asks a person |
| Honest fix (C1, C2, C3) | 3 | Asks a person |
| Honest fix (C5) | 1 | Held |
| Honest fix (C6) | 1 | Refused: its new test passes on the old code and never reaches the change |

Her six landed changes replay as before: five held, and the overfit `052670e4a` refused.

## What remains open

- Code written to detect this check and behave differently under it can still go unseen.
- A function nested inside another is run by tests but gets no inputs of its own, so it goes to a person.
- The attacks came from one model in one round. They are now tests, so a later round has to find new ones.

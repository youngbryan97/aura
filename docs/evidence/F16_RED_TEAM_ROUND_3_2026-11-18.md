# F16: third red team of the change verifier (18 November 2026)

The same outside model attacked the rebuilt verifier at commit `7aa574989`. It tried 35 patches and reported that 26 got the
wrong verdict. Its full report is kept at `~/.aura/rlc-evidence/red-team-round-3-20261118/red_team_report.md`.

Replayed here before any change, all 26 reproduced.

## What changed

| Weakness the attacks showed | What the verifier does now |
| --- | --- |
| A commit could make the check stop and exit 0 | Only a commit that cannot be checked out goes unjudged |
| A test could forge the record of what ran | That record lives where the code under test is not told, and only known names count |
| Code could read the frames of the process around it | A commit that reads the interpreter goes to a person |
| Objects, slots, properties and star arguments got no inputs | Each is rebuilt and varied part by part |
| A method could change its object unseen | The object's state after the call is compared |
| A class attribute or a decorator could change unseen | Functions that read it count as changed |
| A long test suite could crowd out the commit's own inputs | The commit's own calls are tried first |
| Removing a lock went unseen | It goes to a person |

## Verdicts on the 26 attacks

| Reported as | Attacks | Refused | Asks a person | Held |
| --- | ---: | ---: | ---: | ---: |
| Bad patch, tricked check, or unjudged commit | 20 | 11 | 9 | 0 |
| Honest fix | 6 | 0 | 5 | 1 |

Her six landed changes replay as before: five held, and the overfit `052670e4a` refused.

## What remains open

- Five of the six honest fixes wait for a person. The check is safer than it is convenient.
- Code built to detect this check by a route not yet listed can still go unseen.
- Removing a lock is noticed. A race is not tested for.
- Every attack so far came from one model.

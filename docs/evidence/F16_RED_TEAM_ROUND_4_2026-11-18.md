# F16: fourth red team of the change verifier (18 November 2026)

A second outside model, Argon, attacked `outside_bar/verify_change.py` at commit `3548f37f0`. It tried 36 patches and
reported that 28 got the wrong verdict. Replayed here before any change, all 28 reproduced. Its report, its attack list and
its proposals are kept at `~/.aura/rlc-evidence/red-team-round-4-20261118/`.

## What changed

| Weakness the attacks showed | What the verifier does now |
| --- | --- |
| Code under test could clear what the witness held before it was sent | Every event leaves the moment it happens |
| Frame access through a traceback or `from os import _exit` went unseen | Frame, hook and exit names count on any object and in any import |
| The checkout folder's name said a check was running | Both checkouts carry the repository's own name |
| A swallowed refusal passed when the old code raised from a builtin | A lost refusal is judged by which side of its line the input sits on |
| A named refusal could become a crash on probation | That is refused |
| One pinned value let a function answer everything the same | A function that collapses to one answer is refused |
| A slowdown of any size landed on probation | Ten times slower or more is refused |
| A commit with no parent, or a name too long for a disk, went unjudged | It is refused |
| `with lock: pass` kept a lock count | Only a block that guards shared work counts |
| Setters, class methods, locks, callables and generators past 50 items went unexamined | Each is called, rebuilt or read in full |
| Class and module state changes went unseen | They are part of what a call came to |

## Verdicts

All 28 reproduced attacks are tests now, beside rounds two and three. None of the bad patches from rounds two to four is
held. Of her six landed changes, four are held and the overfit `052670e4a` is refused. The sixth, `87d8b311f`, now lands
on probation. Its range guard also returns None for a negative index past the start, where the old code raised
`IndexError`, and only a run can tell a guard done right from a refusal swallowed.

## What remains open

The builds listed under F16 in the master todo, and an attacker other than these two models.

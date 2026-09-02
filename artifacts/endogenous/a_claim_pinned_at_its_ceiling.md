# A claim that could not fail and could not hold — 2026-09-02

`test_keeping_what_she_wrote_makes_the_next_one_easier` reports two violations
and has been reporting them since before this mandate's work began. It is not a
regression and it is not noise; the claim is registered against parameters at
which the measurement is pinned at its ceiling.

## What it asserts

That on a stream of families with shared structure, the agent keeping what it
wrote solves more than the one reset between blocks, and that on a stream with
nothing to carry the two are identical.

## What it measures

```
shared  grown [4, 4, 4]  reset [4, 4, 4]  lesioned [4, 4, 4]
apart   grown [4, 4, 4]  reset [4, 4, 4]  lesioned [4, 4, 4]
```

Four of four in every block of every condition on both streams. The third
sub-check — that no gap opens on the control — holds trivially. The first two
require a strict inequality between two numbers that are both the maximum, so
they cannot hold at these parameters however well the mechanism works.

## Where it came from

The predicate was written with `blocks=3, per_block=4, within=2.0, deepest=3`.
The campaign that produced the original positive result used a harder stream.
A claim registered against easier parameters than the result it stands for is a
claim about a different experiment.

## Confirmed as pre-existing

Run at `1ca82c5e3`, the commit before any of this mandate's work, in a detached
worktree: the same four-of-four in every cell. Nothing in the developmental
agency work caused it.

## What was fixed alongside it

Conditions in `run_stream` were sharing what-worked counts, so the reset
condition inherited the order the grown condition had learned. That was a real
leak and it is closed — every arm now starts each attempt with fresh counts.
It did not change these numbers, and saying so matters: closing a leak that
turns out not to have been the cause is still worth doing and is not a fix for
this.

## What is still open

Parameters with headroom, so the claim measures the thing it names. Until then
the claim stands unsupported and is reported as such rather than retired,
because retiring a claim to make a suite green is how a suite stops meaning
anything.

---

## Closed, 2026-09-02

Parameters with headroom, found by search: three blocks of six families at
depth four with a fiftieth of a second each.

```
shared  grown [6, 6, 5]  reset [6, 4, 2]
```

The claim holds at these and the predicate reports no violations. What was
wrong was never the mechanism; it was a claim registered against an easier
stream than the result it stands for.

## Three more of the same kind, found the same way

Running the suite the claim belongs to turned up three tests failing for
reasons that had nothing to do with this mandate's work.

**The proposer test asked about a configuration the experiment never used.**
`families=8` leaves four families to fit a proposer on, and four was measured
in this same session as below what generalises — the working configuration is
twenty, and it is recorded in
[a_better_proposer_20_episodes.txt](a_better_proposer_20_episodes.txt). At the
measured configuration it writes a 68-symbol proposer, 618.1 to 350.0 on ten
sealed families, with the lesion exact. Confirmed pre-existing by running it at
`d083eca20`, the commit the test was written in.

**A protocol assertion named a function that no longer exists.**
`test_a_proposer_is_selected_on_the_training_half_only` searched the source for
`_how_many_it_proposes_for`, which stopped existing when the cost measure
changed to count the index at which the proposer offers the answer. A source
assertion naming something gone passes or fails on the rename rather than on
the protocol.

**Two order tests passed alone and failed in the suite.** The order in force
was being restored between tests and the history it is scored against was not:
how often each word has appeared in a surviving term is global, every test
there writes to it, and the search for a better order is judged against it.
That is an order-dependence defect and not a flake.

All four are the same shape. A test can be green for years while measuring
something other than what it names, and the way that surfaces is running it
against a system that has changed underneath.

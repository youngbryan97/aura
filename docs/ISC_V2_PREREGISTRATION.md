# ISC-v2, preregistered

Written before any result from the campaign that started at 16:25 on 13
September 2026 (runs 029, 030 and 031 at `45a74c913`), and committed before
those reports exist. Those three runs are ISC-v1 results and are read under
ISC-v1 only. ISC-v2 applies first to runs started after the commit that
implements it, and to no run before.

ISC-v1 is kept. Every threshold in `THRESHOLDS` stays as it is, every v1 line is
computed and reported on every run, and a v1 failure stays a failure (P10.11,
P10.14, P40.8). ISC-v2 changes three lines of the conjunction. Each change below
says what v1 asks, why a test built around a positive reference cannot keep
asking it, what v2 asks instead, and the known-answer check that has to pass
before v2 reads Aura.

## 1. Irreducibility against the nulls

**v1 asks** for the real lower bound above the 95th percentile of every null
architecture except the recurrent reference (`partition_beats_nulls`, §18).

**Why that cannot stand.** The null table includes the fully connected bus and a
low-rank system. In run_026 they scored 0.573 and 0.564, and the recurrent
reference, the architecture the battery exists to say yes to, scored 0.262. The
line is failed by its own positive reference, and the only systems that could
clear it are built towards the architecture the specification forbids. A test
should be difficult, and it should not be impossible for the reference it is
designed around (P40.8).

**v2 asks** for the real lower bound above:

- the 95th percentile of every matched surrogate (same recording, same cuts,
  coupling removed), and
- the 95th percentile of every null architecture that passes every other line
  of the conjunction.

A null that already fails elsewhere in the conjunction has been told apart from
a subject by that line, and requiring irreducibility above it as well asks the
same question twice with the harder number. This is the comparison
`beats_every_null` (§41) already makes.

**Known answer.** Across the declared seeds the recurrent reference passes this
line, and the star, hub, hidden broker and replay nulls fail it.

## 2. Synergy's target

**v1 asks** for each declared triple's synergy about the target's next level,
against a shifted null that slides both sources together (§27).

**Why that cannot stand.** Three of the four targets, the world model,
deliberation and recurrent cognition, carry slow structure. A slow level shares
information with a slid copy of any other slow series, so the shifted null
rises with the target's drift: in run_026 its 99th percentile was 0.545, 0.676
and 0.418 for those three and 0.038 for attention. Intrinsic persistence already
predicts the change rather than the level for this reason, and a triple scored
on the level is read against its own drift.

**v2 asks** for synergy about the target's change, `Y_{t+1} − Y_t`, with the
sources unchanged, the same shifted null built on the same change, and every
bar in `SynergyReport.passes` as it stands.

**Known answer, required before v2 reads Aura.** On synthetic recordings of the
battery's shape:

- a target driven by the product of its two sources, on top of a slow drift,
  passes;
- the same drift with the two sources acting only additively fails;
- the same drift with no dependence on the sources fails.

## 3. The null suite's verdict across seeds

**v1 reads** each null's conjunction on the toy recordings of one campaign seed.

**Why that cannot stand.** The verdict changed with the seed while the nulls'
irreducibility did not. `low_rank` had vertex connectivity 0 and passed no
synergy triple on seed 11, then connectivity 3 and all four triples on seed 13,
and the recurrent reference passed all four triples on seed 11 and none on seed
13. A verdict that one draw can flip says nothing about the architecture.

**v2 reads** every null architecture and the reference on each of the
campaign's declared seeds, which for `make subject-core-frozen` are 7, 11 and 13.
A null fails the conjunction only if it fails on every seed. The reference
passes only if it passes on every seed, and a reference that fails on any seed
is recorded as the instrument failing, not as a result about Aura.

**Known answer.** The seed-13 tables of run_028 read under this rule report the
instrument failing rather than a null passing.

## What does not change

Every other line of the conjunction, every threshold in `THRESHOLDS`, the edge
rule, the lesion and rescue as fixed in `c798126bf`, and the rule that a
criterion may only ever be made stricter after its result has been seen. ISC-v2
adds no threshold chosen from Aura's numbers: the three changes above take their
quantities from the battery's own reference, estimators and declared seeds.

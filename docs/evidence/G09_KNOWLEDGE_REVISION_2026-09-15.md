# Knowledge revision through the existing graph

## Failure reproduced

The production AtomSpace retained derived truth when a premise changed at the
same evidence count. `apply_rules` compared counts rather than the premises
that justified a conclusion. Two derivations from one source also accumulated
17.1 units of evidence where the stronger path carried 9.

Six regression tests failed before the repair. The expanded graph, evidence,
belief, concurrency and persistence suites passed 135 tests after it.

## Mechanism

- Rule instances retain their premise dependencies, including auxiliary truth
  reads from grounded filters and the rule's numerical formula. A changed or withdrawn premise
  invalidates dependent support transitively.
- Direct witnesses and alternative derivations remain separate. The existing
  `EvidencePacket` fusion accounts for shared source identities. Replaying a
  rule cannot manufacture another independent witness.
- `revise_observation` atomically replaces one identified source's claims at a
  monotonically increasing revision. It handles changed claims, corrections to
  individual obligations, and empty revisions that retract an observation.
  Older deliveries cannot revive withdrawn claims. Conflicting payloads for
  the same revision are errors.
- Snapshot v2 preserves active derivations, dependencies, observation
  watermarks and per-atom revisions. Loading rejects circular proof support
  and support computed from stale premise revisions.
- The canonical belief engine sends versioned updates. Conflict demotion and
  confidence decay now reach the graph rather than leaving its mirror stale.
  Decay tracks the interval already applied without changing observation time.
- Derived events carry the existing `CognitiveStateRef` envelope with evidence,
  parent identities and revision. No second knowledge store or language
  substrate was introduced.
- A registered canary exercises correction and stale delivery. A runtime
  invariant checks derived truth and the dependency index. Neither is a gate
  deciding whether a conversational answer may be emitted.

## Scope

This is truth maintenance over arbitrary ground typed claims and registered
inference rules. The producer must identify the observation and encode what
the claim refers to, including time or domain where those distinguish claims.
The revision layer does not infer those distinctions from prose, determine
which witness is correct, or prove that its inference rules model the world.

Legacy unattributed evidence has no selectively retractable source identity.
Old v1 snapshots cannot recover provenance that was never recorded. Legacy
unversioned sources retain their existing compaction policy; versioned
observation identities are preserved for subsequent withdrawal. Cross-store
adoption beyond the canonical belief/AtomSpace path remains work to measure.

These tests establish the correction mechanism on constructed graph workloads.
They do not close G09, establish 500/500 semantic binding, or demonstrate
frontier reasoning. Source-matched live deployment remains a separate check.

# Completed program proofs survive unfinished searches

## Observed failure

The operation-field and complete-program preparation ran from frozen
`cbac2b3eb`. Its source population was 1,335 fitting identities and eleven
disjoint source-calibration identities. It did not load the backbone.

The first failing source was
`77b40184a3ce78feba8ae6fdd20eb4bca40393550bab8f3ab104a04679056854`,
an arithmetic `fronted_obtained_afterward` request with two operations.
A diagnostic replay found that multiple complete graphs had already reached
certified optima. Later enumeration exhausted the ten-second allowance.
The final limited solve had an incumbent, but no optimality certificate.
The miner discarded every earlier completed proof along with that unfinished
search. No incumbent was accepted as an optimum.

The same failure became common in the fork/join sources. The old preparation
was stopped through its authenticated supervisor control socket after 322
source rows: 282 durable pools, 38 incomplete-optimizer failures and two
time-allowance failures. The remaining sources were not run. Its terminal
receipt records `stopped`, return code -15, 1,455.554272 seconds, containment
verified and empty process group and lineage. Child 83463, supervisor 83461
and the sleep inhibitor were subsequently absent. Aura's process 60377 was
left running.

Artifacts under `/Users/bryan/.aura/rlc-evidence/`:

- `semantic-grounded-operation-program-preparation-v1-20261001/supervisor/`
- `semantic-grounded-operation-program-pools-v1-20261001/`
- Terminal receipt:
  `3b3495346178d5ac8df44dccdc9e31c9bfbe1e3a248c2a2a932744947feb02c3`
- Retained log SHA-256:
  `07ec8548a923338c4e18b3034907c0dd69c4e7565ec3a94395d2fa3f8e65711d`

This stopped run is not a completed population preflight or native fit.

## Repair

Public operation proposals are still frozen before source annotations are
read. The source-positive chart is proved first. Subsequent competitor
searches retain their completed optimal graphs when a later search runs out
of time. The receipt lists unfinished searches and whether every requested
bounded search completed. Lack of a proven positive still fails the source.
An unknown semantic comparison is still excluded rather than called wrong.

The training partition already ranges over bounded witnessed alternatives;
its mathematics does not require every possible competitor or a fixed number
of wrong answers. It remains a partial partition. A correct-only pool has zero
complete-program contrast loss, while the existing role and operation losses
still apply. This repair changes neither public graph optimality requirements
nor the measured semantic acceptance standard.

`ScoredArgumentChart.certify_selection` independently checks selected option
identities, mention exclusivity, register-definition consistency, distinct
operands, acyclic dependencies, one sink, register-use bounds and factor sums.
It certifies feasibility of that supplied graph, not score optimality. The
existing public MILP solver still rejects incomplete optimization.

Previously completed pools can change implementation custody only through
explicit revalidation. The loader requires unchanged source annotation,
observed-state hash and geometry, public inputs, parent receipt and bounds.
It then regenerates the public proposal bank, rebuilds every retained chart's
factors, checks every selected graph and re-proves each semantic comparison.
The new immutable receipt links the old receipt and implementation. A changed
factor, label, baseline, option identity or source causes refusal. No optimizer
search is repeated for a successfully revalidated pool.

## Measured checks

The original failing arithmetic source now retains four proven-positive and
twelve witnessed-negative graphs in 10.079891 seconds. Its final competitor
search remains explicitly incomplete. A failing three-operation fork/join
source, `16266fb48893c5d6791c571fac76e17ac32ceaf7a0c1920f40ac97032a8ceee4`,
retains three positive and thirteen negative graphs in 10.164122 seconds.
Neither receipt claims an exhaustive grammar partition.

An existing fork/join pool,
`b6ef90a2f3e9bfc96da53629729baa6d599541021b7f8640d8dbb51d1536c796`,
was rebuilt and re-proved in 1.639465 seconds without optimizer search. It
retains three positive and fourteen negative graphs. These immutable pools
are under `semantic-grounded-program-prefix-canary-v1-20261001/pools/`.
The arithmetic receipt is
`1e8ece574f121fe43e34513055f2e27d6b8b440527f65ee794e058b7a0f7b14f`;
the repaired fork/join receipt is
`fa9f89775cb2dbb70fae965b5206e466a696e6bd7ea50b2d5ee465691a31b02f`;
the revalidated pool receipt is
`18d981bb6096ec2eb8bb0b273e7281beb4b44a4f979e84af5e295079b6fb5153`.

The focused argument-chart, optimizer, graph-certificate, complete-program and
native-fit suites pass 169 tests in 106.80 seconds with `MLX_ENABLE_TF32=0`.
An initial run without that required native arithmetic setting failed four
batch-shape numerical parity checks; it is not a passing precision result.
Native fit preparation already refuses a process launched without that
setting. The re-run uses the required setting rather than relaxing tolerances.

The graph certificate is checked against exact optimization on randomized
small charts with definition scores and normalization. Tests also force a
late timeout, an unproved target, changed implementation custody and corrupt
retained factors or labels. Interrupted native fitting still reproduces the
complete-program optimizer history and selected weights.

Smoke passes 164 tests with one skip in 70.39 seconds. Lint, compile,
governance, layering and writing gates pass. Test shutdown telemetry was
refused because this pass placed `AURA_STATE_ROOT` beneath `~/.aura` but
outside a source checkout. That location is protected live state, even
though its name described a test run. Further test state must live outside
that root; the ownership guard is not bypassed. The refusal did not affect
the source pools or Aura's running state. The full offline suite was not run.

These checks validate proof retention and source-pool reuse. They do not
measure learned language accuracy, broad gain, fresh transfer or frontier
performance. G03 and later acceptance items remain open.

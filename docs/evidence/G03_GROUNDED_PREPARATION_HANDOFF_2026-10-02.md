# Grounded preparation custody and exact fit handoff

## Running preparation

The repaired source preparation runs from frozen `d08df8e8f5` in
`/Users/bryan/.aura/live-source/.claude/worktrees/codex-g03-program-prefix-frozen-v2-20261002`.
That checkout remains clean. The preparation loads no backbone.

Evidence under `/Users/bryan/.aura/rlc-evidence/`:

- Supervisor: `semantic-grounded-operation-program-preparation-v2-20261002/supervisor`
- Program pools: `semantic-grounded-operation-program-pools-v2-20261002`
- Planned native artifacts: `semantic-grounded-operation-program-native-v2-20261002`
- Supervisor plan SHA-256:
  `61b1e6c8bb0f2f13a23e60e2afb3b0531afc9a4654e5058f4c7cf4b0ff357223`
- Preparation command SHA-256:
  `7972304038eff9eed5ba7783cba8c48383f04d1cd7678f10880d4af01e863ea5`

Independent process inspection found supervisor 89750 reparented to PID 1,
child 89753 in its own process group, and sleep inhibitor 89758 attached to
the child. The authenticated status check found both process identities
alive, a moving heartbeat, no indeterminate completion and no terminal receipt.
Durable per-source files and progress logs advanced between observations.
The supervisor enforces its 18,000-second bound independently of Codex.

At the 2026-10-02 07:26 UTC observation, 377 of 1,346 sources were ready and
none had failed. All 282 completed pools from the stopped preparation had
been rebuilt and re-proved without optimizer search. Eighty-nine pools
explicitly reported unfinished competitor searches while retaining proved
alternatives. These are partial witnessed partitions. The remaining sources
were not yet prepared at this observation.

The original stopped preparation remains a separate historical record; see
[program proof retention](G03_PROGRAM_POOL_PROGRESS_2026-10-01.md). No old
receipt or frozen running checkout was edited.

## Handoff implementation

`tools/run_semantic_grounded_fit_handoff.py` derives its fit command from the
verified preparation supervisor. It removes only `--prepare-only` and keeps
the original executable, frozen source checkout, input paths, source bundles
and all training settings. Resumed or repeated preparation modes are refused.
The detached subprocess broker allows each fit and verifier command once.

Advancement requires a passed preparation terminal bound to the expected
supervisor plan and command, with proved process cleanup. The handoff then
checks the native plan checksum, unchanged implementation files and exact
source artifacts. Fitting identities must equal the bank's fitting cohort;
calibration identities must belong to its separate calibration cohort. Held
identities cannot enter either set.

Every retained pool must match its source and implementation custody. Its
checksum, factor sums, option identities, graph feasibility and proved labels
are checked. The aggregate program objective must equal the prepared contract.
An existing native fit directory, acquired prefixes or completion receipt
refuses another fit. All checks repeat immediately before launch.

The handoff waits for model owners to leave within a declared bound. It does
not evict them. The trainer still requires the supported exclusive model-lane
reservation and refuses owner eviction before loading weights. Observing an
empty ledger is not a substitute for that reservation.

The fit is followed by its own frozen independent verifier. Exactly one
checksummed verifier verdict must select a positive-step checkpoint, match
the prepared source counts and report no implementation drift. The final
native plan must remain identical to the independently checked preparation.
Neither process exit zero nor an initial checkpoint is accepted as learned
fit completion.

## Serialization repair and checks

The new checks found a definition-free chart round-trip defect. Its empty
serialized factor list reloaded as an empty dictionary instead of absent
definition factors. Graph certification correctly refused scores without
definition labels, but this dictionary contained no scores. Deserialization
now preserves absence. The regression test checks the receipt, valid graphs
and unchanged differentiable loss. No public constraints were weakened.

This repair is in the editable integration checkout. The running preparation
and its future fit retain their frozen implementation. The handoff can refer
to that frozen checkout without substituting the newer learner.

Focused handoff, existing continuation, program-objective and graph-certificate
checks: 167 passed in 37.44 seconds with `MLX_ENABLE_TF32=0`. The first test run
found the round-trip defect and is retained as a failed result. Smoke: 164
passed, one skipped in 111.96 seconds. Lint, compile, governance, layering and
writing gates passed. The full offline suite was not run. Test state was
isolated under `/tmp`, outside protected live state.

The live neural stream was read over its supported WebSocket. It emitted
thought, telemetry and health events. The conversation lane reported a ready
27B Cortex with an active generation; runtime PID 60377 was left alone. This
observation establishes an occupied lane, not model qualification.

No native fit or new development decode has completed at this observation.
G03 and later G-ledger items remain open. Source preparation and the handoff
checks do not prove semantic accuracy, broad gain, fresh transfer, fusion or
frontier performance.

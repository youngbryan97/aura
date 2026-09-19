# Background scoring reaches joint training

The scalar and batched graph score repair did not yet reach the campaign entry
point: `refit_compositional_joint_graphs` still rejected every model with
background supervision. That obsolete refusal is removed now that both
supported score policies replay their runtime semantics.

The source-retention builder also reuses the existing background-span miner
when that vocabulary is present. It preserves non-operation targets as well
as operation targets, rather than updating only positives and forgetting the
distinction learned by the earlier background fit. Ordinary models keep their
original source supervision. Counts separate real operations from all training
spans, and the original background-fit receipt remains unchanged.

The integration tests run the actual source miner, graph optimizer and export
roundtrip under both supported background policies. Operation graph evidence
must come from training features; held-out features cannot supply updates.
This connects a development option, not a promoted model or a new result on
the 500-case comparison. The current full run remains on its frozen policy.

Verification: 91 focused tests passed. Smoke passed 164 tests with one skip.
Lint, compile, governance, layering, writing and documentation-drift gates
passed. No model was promoted and no G-ledger item was closed by these tests.

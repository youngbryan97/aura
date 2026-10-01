# Hybrid Joint Execution, 2026-10-01

The first real joint preparation completed in 107.689368 seconds with a
contained terminal receipt and no surviving child processes:
`c7bfb5a83b2867cbc5707cb4ec1a77a0a02bb5e1f6ccc4f61d15caf2f777ea51`.
Its frozen implementation was `9c7517feb`. The native plan is
`e3419c2b61b8074fc05610744e5268f2408592fd3fcaa1a8584042d15704672a`.
The source population contains 1,335 fit requests, 11 stratified calibration
requests and a longest source sequence of 144 tokens. The projected adapter,
pointer and nuisance trainable state is 695,677,620 bytes for five float32
copies. Activation memory was not measured by preparation.

At elapsed 89 seconds the preparation process had an RSS of 19,145,792 KiB.
That sample is not a peak. It exposed archived feature matrices retained
until model acquisition. The CLI now drops those matrices after extracting
supervision and exact native token/span metadata. A weak-reference test
checks that the archived matrix is freed and the native capture template
does not change. Native source supervision remains intact.

## Hybrid Failure And Repair

An actual tiny Qwen3.5 joint fit failed at its first gradient computation:
`[Primitive::vjp] Not implemented for CustomKernel`. The wrapper left the
hybrid suffix in inference mode. The installed delta-net implementation uses
a custom Metal inference kernel without a backward pass. Its training mode
uses differentiable native operations. The existing native-program trainer
already selects that mode; the new joint wrapper had omitted it.

Joint fitting and selected-artifact decoding now select training mode only
on the suffix blocks. Frozen prefix blocks, embeddings and output ownership
do not change. The plan binds this choice as
`suffix_layer_execution=differentiable_native_ops_v1`, and decoding rejects
another execution policy. Replay uses the same computation as fitting.

The fixture also caught a projection estimate error when a minimal Qwen3.5
configuration omitted `attn_output_gate`. Native Qwen3.5 always doubles its
query projection. The estimate now follows the model type rather than that
redundant flag. The actual 27B configuration already declares the flag; its
32,655,360 adapter-parameter estimate is unchanged.

The revised fixture uses a quantized hybrid backbone and heterogeneous
suffix adapters. It completes actual gradient updates, independently checks
the saved artifacts, reloads the selected adapter and pointer together, then
calls the complete public-input chart. All 66 focused checks passed in 26.33
seconds. This demonstrates executable training and decode wiring, not a
correct unseen-language interpretation. The old preparation custody remains
historical; a changed implementation must prepare a new run namespace.

## Restart Admission

An independent unfinished-generation verifier checks stable reads, SHA256,
the prepared implementation and arithmetic, fit/calibration custody, update
schedule, finite tensor inventory, selected/model ownership, Adam moments,
learning rate and RNG geometry without loading the backbone. It emits the
detached supervisor's attempt-bound resume verdict. The actual resume loader
still rechecks full native parameter geometry and source observations.

`--resume-if-available` starts a fresh native fit only when its fit directory
is absent. An existing directory must have a verified complete optimizer
generation; a partial directory is not treated as fresh. A completed native
acquisition must pass the final artifact verifier and is not repeated.
This does not recover a partial prefix acquisition before checkpoint zero.
Such an interruption still needs an explicit, separately recorded recovery.

The combined native, restart, binding engine, adapter and chart suites passed
81 tests in 30.79 seconds. They include mid-update verification before an
interruption, bit-identical resumed fitting, corrupt generation rejection,
partial-directory rejection and real detached verdict-consumer validation.
Smoke passed 164 tests with one skip in 58.93 seconds. Lint, compile,
governance ownership and layering passed without baseline changes.

G03 and later semantic gates remain open. Preparation, native gradient
execution and recovery admission confer no language gain, transfer, fusion,
frontier performance or serving authority.

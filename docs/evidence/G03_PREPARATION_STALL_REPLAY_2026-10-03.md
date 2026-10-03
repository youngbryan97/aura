# Preparation stall and source replay

## Terminal evidence

The v2 preparation ended at its 18,000-second supervisor bound with return
code 124. Its last completed row was source 444 of 1,346. All 444 durable
pools remain; no native preparation plan or trained checkpoint was produced.
The process group and descendant lineage were empty at the terminal check.

Evidence roots under `/Users/bryan/.aura/rlc-evidence/`:

- `semantic-grounded-operation-program-preparation-v2-20261002/supervisor`
  terminal receipt:
  `ed51f905f376ddd8b63bd7a4474dac82727bfff5a77eb0451f61c0735e374083`
- `semantic-grounded-operation-program-fit-handoff-v2-20261002/supervisor`
  terminal receipt:
  `7f50818a394b81dd1472af02b28869eb6c583bf65d1d1829eae1dc9f47bd363f`

The handoff refused the failed prerequisite and exited with code 1. It did
not start a native fit. The earlier dated observation that preparation was
running remains historical; it does not describe the current terminal state.

## Bounded replay

`semantic-grounded-program-stall-diagnosis-v1-20261003` replayed the first
unfinished source against the unchanged frozen `d08df8e8f5` implementation.
The diagnostic loaded the verified source folds and used the same chart,
graph and ten-second mining allowances. It consumed no held cases or model
weights. Its supervisor allowed 180 seconds and proved process cleanup.

Source 445:
`a8b1dc9060a8a6314dff4d3eedad943aced44049813c02ff5e095d54e488a2dd`.
The frame was `fork_compute_apart_then`, with inputs 97, 40, 29 and 2.
Its source graph divides two independent pairs, multiplies their results,
then adds the first public input.

Mining completed in 10.202581 seconds. The pool retained one proved positive
and two witnessed negatives. A later competitor optimization returned an
incomplete result; its unfinished search remains explicit. This is a bounded
witness pool, not a complete grammar partition or a semantic test result.
The diagnostic terminal receipt is
`a730c3b96dfb4fa65a1f445eb2c9d1fff0d538bba1cd2bd1449a8465de63efe2`.

The original stall was not reproduced. Its cause is unresolved. The old
run had no per-source start record or stack capture to identify the blocked
phase. The replay stacks show feature loading and relation-factor construction
before successful completion; those observations do not prove either caused
the original stall.

## Recovery instrumentation

Source preparation now emits each source identity before work and its elapsed
time after work. The fitting CLI requests repeated stack traces if a source
takes more than 60 seconds. Each trace timer is cancelled on success, recorded
source failure or an unexpected exception. Invalid intervals refuse before
source work. The timer reports a stall; it does not cancel a source, change
its labels or convert unfinished work into success.

The fit handoff's policy-only path now creates the broker log parent before
publishing its exact command policy. This repairs the launch refusal observed
in the previous pass without starting any training from policy construction.

Before recovery, the focused program-objective and handoff suites passed
89 tests in 13.28 seconds. Smoke passed 164 tests with one skip in 59.68
seconds. These checks used `MLX_ENABLE_TF32=0` and isolated state under `/tmp`.
The initial focused command named a nonexistent test file and ran no tests;
only the corrected completed run is counted. The full offline suite was not
run. G03 and later items remain open.

After fast-forwarding the integration checkout through shared commit
`297e251ae`, the expanded focused suites passed 122 tests in 51.53 seconds.
Smoke passed 164 tests with one skip in 60.12 seconds. Lint and compile
passed. The merged governance gate reported four new browser call sites:
`the_page_size`, `OnAPage._focus`, `OnAPage.press` and
`played_on_the_drawing`. Those incoming changes are retained. The governance
baseline was not enlarged, and this pass does not claim that gate passed.

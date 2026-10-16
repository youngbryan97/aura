# Delivered presses follow their local material, 15 October 2026

The `e72751c94` original case 3 trace contains successful pointer corrections
that revisit the same local material under different coordinate labels.
The label change restored exploration value without introducing a new thing
to try. This can affect any drawn interface that pans, scrolls or relayouts.

## Mechanism

Each page surface now retains at most eight delivered cursor-material
references, for at most five minutes, with a saturating press count. A current
anonymous affordance inherits a cost only if the existing appearance matcher
finds one redundant correspondence whose projected old press point lies
within two CSS pixels of its current sampled point. Conflicting references,
flat images and changed page, address, viewport or drawing bounds do not
supply a count. It does not infer a button's extent from a point.

The cost stays in a private receipt attached to the current measured reading.
Changed pixels, capture time, epoch, surface, bounds or anonymous point
geometry invalidate that receipt. Serialization does not preserve it. Named
controls keep their existing behavior. Historical reference pixels can inform
exploration, while the existing 60-second current cursor-reference age limit
still governs action. Remembering an old picture cannot refresh input
authority.

The decision path removes the matched point's never-tried bonus and reduces
its exploration weight. It does not remove that action. Goals and learned
effects can still make a repeat worthwhile. No scene, object, selector state,
body identity, accepted action or progress is established by this cost.
Failure to keep optional history after input cannot discard a completed click.

Normal screen-option records now include these private-receipt-derived costs
beside the offered actions. They remain separate from motor-frame evidence.

## Measurements

The generated Chromium check uses the normal page input port and private
pixel readings. It supplies two initial hand affordances, presses one, then
independently moves the layout. Its changed label still receives a cost of
one; the other point keeps its exploration bonus and is pressed next. The
test observes one physical hit on each control. A forced optional-history
failure after the second press preserves its successful transport receipt.
This test does not establish discovery of those initial affordances.

A retrospective replay of the eight corrected clicks in the original trace
retains their original capture and request times. The new mechanism assigns
prior-press counts `0, 0, 0, 0, 1, 2, 1, 0` in 0.438 seconds. Only completed
physical requests update history. This is a component replay, not a new
gameplay result. Evidence:
`artifacts/flash-campaign/2026-10-15-material-press-replay/replay.json`.

The initial integration set passed 83 tests in 18.43 seconds. Broader
navigation, selector-custody and delivered-step checks passed 172 tests in
46.49 seconds. After separating historical memory from current action age,
the final integration set passed 85 in 18.66 seconds. Final smoke passed
165 tests, one skipped, in 46.02 seconds. Ruff and compile pass; layering,
dependency and writing gates passed after adding the named pure claim canary
to the organism's dependency rules. Governance lint still reports 129
existing baseline regressions, with no increase or baseline edit.

The global size ratchet also reports existing failures: the untouched
2008-line browser module, two previously oversized class method counts and
a total 224 lines above its budget. This change leaves every edited existing
module below 2000 lines and all their class method counts unchanged. The new
material-memory module is below the ceiling.

An original gameplay repeat remains required. The 56-game campaign and
ordinary native-app demo are unqualified. G09, original game 9, holdouts and
the resident runtime remain untouched; no accelerator model was loaded.

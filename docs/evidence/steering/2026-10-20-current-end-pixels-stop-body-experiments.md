# Current end pixels stop body experiments

The ordinary browser run at `e1fc9acd2` assigned a wreck as the player after
losing. Initial end labels were ignored by the novelty rule for round-end
text. Retry navigation then exhausted its two-handoff budget, allowing the
same terminal screen to receive another set of keyboard experiments.

The regression tests reproduced 12 failures and passed 12 checks before
the change. The actor spent its full three-second window at an initial
`GAME OVER` screen. A completed visual answer assigned its visible body.

OCR now keeps asserted end labels with their own captured word patches,
separate from navigation labels. Both the complete reading and a result
region must assert an end. Uniform or unavailable patches supply no evidence.
Changed result pixels cannot be outvoted by unchanged score readouts.
The current frame must retain the same source, surface, epoch and geometry,
with ordered capture bounds and sequence, and arrive within half a second.
Split OCR fragments without an independently asserted region remain unknown.

A current end returns the play stretch even if present on its first frame,
if a body is already assigned, or if menu handoffs are exhausted. It supplies
no winner or success claim. Ordinary navigation keeps its existing budget.
The same evidence prevents a new visual body guess and retires pending
answers. An answer whose observation was retired cannot write itself back,
even if its provider returns after cancellation.

The integration group passed 121 tests in 9.21 seconds (10.21 supervised),
including the actor, late-answer retirement, current-pixel refusals, existing
body continuity, guidance lifetime and instance naming. The first invariant
fixture used stripes that disappeared under snapshot resampling; a wider
checker pattern fixed that fixture. No production threshold was loosened.
The new invariant is registered with its executable measurement.

Artifacts: `artifacts/flash-campaign/2026-10-20-terminal-phase/`.
Checkpoint gates and fresh ordinary carryover remain to be measured.
No landing, original-game win or native demo readiness is established.

Checkpoint gates passed in 78.12 seconds: smoke (164 passed, one skipped),
lint, compile, layering, generated DEPS and writing. Baselines were unchanged.

The ordinary follow-up at `56502de79` finished in 134.88 seconds. It still
won nothing. The complete recording has 582 control observations, 308 body
selections and 48 choices, all `wait` with unbound requested destinations.
No visual-fallback body assignment was logged. Current end screens returned
to navigation, including a 2.4-second initial end stretch. The private worker
was stopped and its PID confirmed gone. This supports carryover of the phase
guard in one run, not general gameplay readiness. The next mechanical audit
is `2026-10-20-why-the-whole-play-chain-still-fails.md` in this directory.

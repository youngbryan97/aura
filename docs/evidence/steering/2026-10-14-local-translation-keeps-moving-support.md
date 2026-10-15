# Local translation keeps moving support — 14 October 2026

The pixel extractor pooled local movement and static background together.
A small moving patch could therefore lose translation support when many
still matches shared its tile neighborhood. The rotation channel already
selected a coherent moving layer.

## Change

Translation now selects a coherent layer from actual matched displacements,
using the existing excursion and pixel-error bounds. It needs three distinct
matches, agreement from 70 percent of moving support, and a current inlier
inside the reported tile. A nearby mover cannot supply translation to a
tile containing only background. Equal competing layers remain unknown.
Still matches continue to measure stillness separately. Point count and
quadratic work retain the 256-point cap.

The result remains a pixel-search proposal. It supplies no object identity,
control authority, private state, task-specific action or model call.

## Verification before the checkpoint

The related extractor, statistical confirmation, discovery, sampling and
physical Chromium suites passed 83 tests in 21.90 seconds. A synthetic
case retains three coherent moving matches among 33 static matches.
Other cases reject competing translations, duplicate points, foreign-tile
support and excess work; the existing camera, rotation and negative-control
checks pass.

Ruff, writing, changed-method size, layering, dependency checks, compile,
repository lint and smoke passed. Smoke has 165 passing tests and one skip
in 43.08 seconds. The existing aggregate module-size excess of 194 lines,
response-generation method excess and 123 governance regressions remain.
No baseline was changed. The gate log is
`/private/tmp/codex-steer/motion-layer-gates-1014.log`.

The attempted diagnostic replay of the preceding original-game pairs was
invalid because its observer had mixed two experiment instances. It is
excluded, as recorded in the exposure evidence. This change still needs a
fresh original-game run with separate experiment artifacts. No demo-ready
claim follows from these component results.

## Fresh original-game run from 3f4621c8a

The assigned original game 9 ran for 166.593384 seconds, exit 0, without a
wrapper timeout. Artifacts are in
`/private/tmp/codex-steer/motion-layer-original9-1014`.

There were 686 decisions and zero assigned-body decisions. Median picture
age was 0.232970498 seconds. Two searches each completed 24 acknowledged
presses, confirmed release, and returned no qualified region or sampling
hint. The corrected observer retained separate `pixel-search-001` and
`pixel-search-002` directories, each with its schedule recorded before
collection. Capture times are ordered within each set. Median actual holds
were 0.389345479 and 0.386882063 seconds, respectively.

The pairs include genuine matched movement on most presses, but it still
does not become a qualified stable response region. The original game was
not won. Inspection also found that this finite search suspends ordinary
tracker observations throughout collection; paired endpoint discovery does
not preserve a continuous material track by itself. Connecting current
owned capture updates to the existing tracking and independent attribution
path is the next acquisition work. No identification threshold is relaxed.

# Later menus keep their own attempt — 14 October 2026

The handoff budget identified a menu by its control label and geometry.
An identical Play control on a later screen could therefore inherit an
earlier screen's exhausted budget. The body loop could again try keys on
the later menu. The initial label predicate also covered choice-confirming
labels but omitted full navigation labels such as Next Level.

## Change

A bounded runtime navigation phase retains the previous menu's measured
word patches. Two ordered fresh captures showing that those patches have
changed, at least 50 ms apart, retire that visible occurrence. A later
identical control can then receive a new handoff budget. Empty OCR alone,
one changed frame, duplicate captures and reordered frames cannot renew it.
Small OCR-box changes and flashing colour preserve the existing budget.
The state retains one previous bounded reading and at most 64 request keys;
it is excluded from the persisted game-learning format.

Navigation label recognition now exposes the existing whole-label grammar
directly. It includes Next Level, Play Now, Skip Intro and Try Again while
requiring the whole measured label. It makes no language-model request.
Current-pixel and capture-ownership requirements remain unchanged.

## Verification before the checkpoint

The navigation, prior word-pixel, search-admission and in-play reading
suites passed 64 tests in 6.31 seconds. The new cases verify a renewed
identical control after measured disappearance, refusal to renew on one
changed or unordered frame, and navigation labels beyond choice confirmation.
The actor-level handoff and physical release check still passes.

This change has not yet been measured on the original game. Body discovery,
navigation repeatability and native-app demo readiness remain open.

## Checkpoint gates

Ruff, writing, changed-method size, layering, dependency checks, compile,
repository lint and smoke passed. Smoke has 165 passing tests and one skip.
The existing aggregate module-size excess of 194 lines, response-generation
method excess and 123 governance regressions remain. No baseline was
changed. The gate log is
`/private/tmp/codex-steer/navigation-occurrence-gates-1014.log`.

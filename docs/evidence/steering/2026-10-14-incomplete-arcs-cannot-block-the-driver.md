# Incomplete arcs cannot block the driver

Original case 9 at e600dc44e completed normally in 165.242962 seconds.
It made 555 actor decisions and selected body 17 on 199 of them, spanning
29.233772 seconds. It did not win. The actor returned through facing
trials 73 times, provisional steering 36 times and the idle fallback once.
This trajectory shows that completed discovery can leave the fallback;
it does not establish a paired gameplay improvement.

Evidence is retained in
`/private/tmp/codex-steer/finished-probe-routing-original9-1014`.
The corrected diagnostic recorded 558 facing observations. Replaying
their measured public velocities and delivered-input receipts through
`WhichWaySheFaces.saw` reproduces all 558 retained `along`, `ways` and
`going` histories exactly. Identity is preserved across kind changes
only when epoch, body number and birth still agree. This replay covers
the direction component, not pose reconstruction or a counterfactual
whole-game run.

No turning rate qualified. Its retained travel trials had only two or
three pictures each. Despite that, `curved` classified both driving
controls as curved, removing them from `ahead_key`. Learning a travel
rate already requires four pictures spanning at least 0.15 seconds.
The exclusion now requires that same evidence count and span. On the
recording, the old component ends with no eligible driver; the changed
component retains `up`. Neither version supplies a turning rate.

Three regressions cover insufficient picture counts or time spans. A
complete single curve still requests confirmation and supplies no rate.
The existing invariant also rejects a two-picture prefix as authority
to exclude a driver. No control-identification or material threshold
changed.

The optional correction tracker had a separate publication gap. Its
superclass saved the fast points, then the wrapper published combined
fast and corrected points without updating that capture's history.
Corrected point names therefore could not bridge a later missing
outline. The wrapper now replaces its current capture's saved point
set after combining them. Replacement creates no extra time sample;
backward time still clears the history. Tests exercise actual visible
transport without loading another model, require corrected names in
the measured bridge, and reject a missing intermediate observation.

The focused turning, facing, discovery, point-history and correction
suites passed 117 tests in 16.20 seconds. The broader point/correction
set previously passed 95 tests in 9.86 seconds. These are component
results. The fresh original-game run and ordinary native Aura prompt
proof remain open, as do the campaign and presentation verdict.

Ruff, writing, changed-method size, layering, dependency graph, compile
and lint checks passed. Smoke passed 165 tests with one skip in 44.16
seconds. The existing aggregate size excess, `ResponseGenerationPhase`
method excess and 123 governance regressions remain. No baseline changed.

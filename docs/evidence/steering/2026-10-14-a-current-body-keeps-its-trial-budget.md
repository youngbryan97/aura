# A current body keeps its trial budget

The committed PNG capture path ran original case 9 at bbf0a1724. The run
completed normally in 175.263123 seconds without confirming a body or
winning. Its 409 recorded actor returns comprised 388 ordinary discovery
returns, 19 initial vision waits and two later planning returns. Median
picture age was 0.263692 seconds. The previous PNG ablation's long retained
body therefore did not repeat. PNG alone has not made discovery reliable.
Evidence: `/private/tmp/codex-steer/control-routing-original9-1014`.

Inspection of the scheduling path found a separate defect. The unknown
control scheduler owns a finite allowance for each body number, birth and
identification epoch. The outer `body_probe_is_active` gate still applied
the global discovery count. When that count reached `4 * len(keys)`, the
actor could skip a new body's unused allowance. This discards work before
the scheduler can issue it.

For a body with a valid current owner, the outer gate now uses that owner's
remaining allowance and measured control readiness. Discovery without an
owner retains its previous global bound. Existing active physical holds
and rests finish before the phase yields. Reordering inputs, returning to
an exhausted owner and losing measurements do not renew an allowance.
The existing 64-owner and 32-control bounds remain in force.

The previous checkpoint's gate returned false for an owned body with three
unknown keys after 12 discovery steps. The corrected gate returns true for
the same state, without changing identity or treating any response as known.
The comparison executes the saved function definition separately from its
module's invariant registrations.

Three regressions drive the actual actor through a new epoch, new body
number or new birth after the old discovery count is spent. Each case
delivers two physical pulses of every unknown control, observes its leased
release and rest, then reaches ordinary planning. No measured control is
invented. The control-gate, readiness and observation-deadline suite passed
64 tests in 10.49 seconds. This is a code-path regression, not an observed
original-game recovery. A live run exercising this owner transition remains
required.

Ruff, writing, changed-method size, layering, dependency graph, compile and
lint checks passed. Smoke passed 165 tests with one skip. The known
194-line aggregate size excess, `ResponseGenerationPhase` method excess
and 123 governance regressions remained unchanged.

# Discovery strategies share their delivered probes

In the original lander run at `82822b7dd`, the first three physical key
probes repeated the same control. The counter-movement programme and ordinary
discovery did not share their delivery counts. Tracing the cold start also
showed that an ordinary pulse can precede the counter programme: its first
choice requires a preceding neutral observation. Other controls had not yet
been tried. The recorded candidate still had insufficient distinguishing
evidence; this observation does not establish that another key would win.

A regression through `try_controls`, the actual hold port and physical
deadline release reproduced a third delivery of the first key. The corrected
red test failed on that ordering. Its first run also had a mistaken await
of the synchronous cleanup helper; the clean red run preserved the actual
ordering failure without that fixture error.

The initial repair covered the handoff from the counter programme to ordinary
discovery. A second regression placed an ordinary pulse before that programme
and still delivered the first key three times. The final repair shares counts
in both directions. Before a new counter pulse, it checks whether earlier
discovery already delivered that programme's existing two-pulse allowance.
It does not interrupt a held phase or change known-body calibration.

Unknown-body discovery now shares a bounded exposure ledger between its
counter-movement programme and ordinary key discovery. Current delivered
independent holds count once. Ordinary discovery prefers a current
authorized key with fewer prior deliveries, with stable ties. The existing
neutral phases, finite discovery allowance and physical deadlines remain.
Missing, mismatched, nonfinite, duplicate, unauthorized and state-dependent
receipts do not spend an exposure. Control reordering does not erase counts.
A new observation epoch resets the ordering context while retaining the
physical delivery high-water mark; an old hold cannot become a new press.
The distinct-key table retains the existing 32-control bound.

This ledger orders experiments only. It creates no motion sample, causal
witness, controlled body or learned response. Both identity thresholds
remain 20, and each still requires independently repeated observation.
The owned-body calibration budget remains separate.

The first focused group passed 31 checks. Negative receipt and registration
checks raised that group to 40. The broader physical timing group passed
155 checks, including observation-dependent acquisition, finite discovery,
counter trials, actual release without another frame, and slow-frame phase
reservations. A registered invariant covers the shared delivery count.
The final bidirectional group passed 156 checks in 19.07 seconds. Gameplay
effect remains unmeasured. The governance gate still reports 130 regressions;
its recorded count is unchanged. After the bidirectional repair, smoke
(165 passed, one skipped), lint, compilation, layering, dependency consistency
and writing passed in 88.05 seconds.

Artifacts are under
`artifacts/flash-campaign/2026-10-17-delivered-probes/`.
G09, the protected native lifecycle and holdouts remain untouched.

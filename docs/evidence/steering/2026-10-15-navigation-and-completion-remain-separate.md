# Navigation and a claimed result, 15 October 2026

At `b51ff2857`, the CPU-only browser campaign ran original case 3,
Dexter's Laboratory: Runaway Robot, with the same ordinary browser pursuit
used by the earlier original-18 runs. Case identities are zero-based.
Original 9, G09 and the ten holdouts were excluded. The software driver
reported SwiftShader. No model worker or native Aura runtime was started.

The supervisor finished in 90.07 seconds. The game report said `won: false`
and `runs: ["unread"]`. Its final narration nevertheless said "I made it"
and its stop reason was `MADE_AS_ASKED`. The pictured screen was an
instruction/collectible guide, with no created artifact or witnessed win.
The raw run is `artifacts/flash-campaign/2026-10-15-case3-cpu-transfer/`.

Two upstream problems were observable. Current screen layouts contained a
bottom navigation label `CONTINUE •`, but the offered control list omitted
it. Replaying the saved layout offered it again. The recorder had omitted
the `of_its_own` flag, so the earlier recording cannot by itself prove that
flag's value. The production stream marks regions whose pixels change, and
the candidate filter discarded every such region, including known whole
navigation labels. A pulsing label is now retained as a candidate through
the same normal stream, candidate and click-coordinate functions. Dynamic
scene prose and readouts remain excluded. The recorder now retains the flag
so a fresh run can test this diagnosis directly.

The end-report path separately treated a making mechanic and elapsed time
as permission to describe a made result. A source's incidental making
mechanic can no longer override an explicitly understood winning or scored
task. Both creation-report callers require the current run's creation
handoff. A making task that reaches its outer deadline without that handoff
returns an incomplete stalled result. The wrapper also no longer accepts
the underlying pursuit's generic `completed` flag in place of an observed
game ending. Existing success paths for a witnessed end, measured runs and
creation handoffs remain covered.

The first new test fixture supplied only one coded making sign, below the
existing three-sign threshold. It was corrected to supply three. That test
then exposed the separate generic-completion bypass, which was repaired.
The focused navigation, creation, actual handover and recording tests then
passed all 84 checks. Two prior tests were corrected: an incidental making
mechanic alone does not establish the task shape, and a fake delivered key
with `completed: true` does not establish a game ending. Smoke passed 165
with one skipped. Ruff passed on the touched code.

This change does not strengthen the non-placement maker's internal
changed-screen completion heuristic. It closes the outer timeout and
classification bypasses while retaining that existing handoff boundary.
Neither the failed run nor these component checks establish gameplay or
presentation readiness. A fresh recorded run is still required.

## Repeat at `4a68839a1`

The same CPU-only original case 3 was repeated with the same 0.8-minute
play allowance and 175-second supervisor bound. It finished in 81.87
seconds. The fifth normal reading recorded `CONTInUE •` with
`of_its_own: true`, retained it in `offered_clicks`, and the normal chooser
selected it. The changed candidate rule therefore reached the ordinary
browser path. The earlier omitted flag is now directly measured.

The final report remained `won: false`, `runs: ["unread"]`, on the
instructions. Its narration accurately said its time was up and its stop
reason was the time allowance, without a made-result claim. The source
guide still named a making mechanic among other mechanics; it no longer
converted that incidental category into a created artifact.

The recording is under
`artifacts/flash-campaign/2026-10-15-case3-current-navigation/`. The 46-second
picture shows small unlabelled arrow controls near the drawing's lower
edge. Their words appear on hovering. The clicked caption's centre and
the cursor's measured hit point need to remain distinct. The current cursor
sweep samples only grid-cell centres and does not retain a hand witness on
an associated text region. That is the next general input path to inspect;
the recorded choice alone does not establish where a physical click landed.

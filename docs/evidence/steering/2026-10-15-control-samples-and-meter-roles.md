# A captured velocity counts once; a strip still needs a role

The fresh CPU play at `2c85070e2` identified original case 18's body after
3.6 seconds. Its visible result remained zero landings and one crash. The
actor's observations show its original body leaving the left edge during
lateral calibration; selection ends at 13.5 seconds. This failure differs
from losing correspondence while the body stays visible.

The recording is under
`artifacts/flash-campaign/2026-10-15-case18-custody-cadence/`. It retained
728 frames and 190 input events, then reached its raw-byte budget: 46
events were dropped. Its manifest correctly remains incomplete. The first
175 frames form a complete bounded prefix and can be replayed with their
actual input ownership flags. They do not qualify the unrecorded remainder.

## Duplicate measurements

The body-selection handoff copied the `timed` dictionary with shared list
values. While independent experiments continued, discovery and own-body
learning each appended the same captured velocity to the shared list.
The captured journal contains repeated timestamps in the affected rest.

Selection now copies each sample list. Timed measurements also refuse
duplicate, decreasing and nonfinite capture times before altering any
sample history. A regression drives the real `WhichIsHers` discovery loop,
checks independent list ownership after selection and checks unique times.
Another confirms that three distinct observations plus a duplicate cannot
meet the four-picture acceleration requirement.

## False resource feedback

The actor said a bar at the bottom was what it had left. Replaying the raw
first 155 frames through the bar detector produced an unnamed strip at row
185, from columns 54 through 250, whose change at 13.33 seconds was assigned
`down is bad` solely because its role was absent. Selecting a controlled
body supplied no evidence for that interpretation.

The detector still retains changing strips. Reward and scarcity now require
a known label role or an identified side. Unknown and absent roles remain
unknown. Tests preserve health, own-side and opposing-side feedback, score
polarity and non-resource meters. A source-grounded floor still overrides
even a false resource label.

The replay includes control-model diagnostics to distinguish body selection,
acceleration classification and usable controls. On these recorded actions,
the repaired sample custody still selects the body. It learns thrust and
left; the second right trial arrives too late to provide four measurements
before the body leaves view. Choosing a safer next experiment remains open.

These are CPU browser-path observations and component regressions. They do
not prove a landing, the full campaign, native Aura prompt integration or
presentation readiness. G09's separate model job remains running.

Validation: 88 focused tests passed. The smoke gate passed 165 tests, with
one skipped.

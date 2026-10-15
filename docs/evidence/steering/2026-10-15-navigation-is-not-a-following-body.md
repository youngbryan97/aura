# Navigation routes and continuous control, 15 October 2026

The longer original case 3 diagnostic at `54eb608f6` reached the first room,
then classified the activity as a body following the pointer. At 13:33:33
the settings log described `a body that follows the pointer | it goes on
without me`. A ten-second stretch ended at 13:33:43 with no body and no
clicks. Another such stretch began at 13:33:44. It spent time probing body
control instead of using the room's cursor affordances.

The instructions said to use the mouse to move from room to room. The
pointer-body parser treated `from` as the moved object's name. A separate
real-time classifier also accepted `mouse to move` without distinguishing
that navigation route. The two readings agreed on a false premise.

The diagnostic child PID 93613 was interrupted with SIGINT after its PID,
parent and exact command were checked. Its browser closed and recording
finalized as interrupted: 213 motor frames, 408 input events, zero drops,
`complete: false`, `presentation_qualified: false`. The supervisor records
133.33 seconds and return code -2. There is no completed game report or win.
The raw evidence is under
`artifacts/flash-campaign/2026-10-15-case3-bounded-cursor/`. No resident
runtime or model process was signalled.

The repair shares one bounded route grammar between pointer-body extraction
and the real-time classifier. Moving between rooms, pages, panels, tabs and
other views supplies navigation evidence. A named direct object remains:
moving a robot from room to room still describes controlling that robot.
Independent aiming, firing and keyboard clauses also remain. Old guide
records that kept `from` or `between` as a body name discard that entry.
No game name, character identity or room coordinate enters this rule.

The previous cursor checkpoint was committed as `54eb608f6`. Its first push
was rejected because another agent advanced main. After the diagnostic
stopped, those changes were merged unchanged in `fa7da1c27`, then pushed to
main. They were not inspected as gameplay evidence. The shared default
checkout and the accelerator job remained untouched.

The focused run passed 95 tests. The new cases exercise ordinary guide,
mechanics, available-control and real-time selection paths for navigation
through invented software views and rooms. Direct robot/ship control still
selects following-body control, and separate aiming, firing and keyboard
clauses still select real-time play. The first test run exposed an existing
gap in `move your ship between rooms with the mouse`: the intervening route
hid the direct object. The shared normalization now retains that object
while removing its route modifier. All 95 passed after that correction.

Smoke passed 165 with one skipped. Ruff, syntax compilation, layering,
dependency consistency and writing passed. No baseline was raised. The
original-game repeat remains the next check; these tests do not establish
that the room search succeeds.

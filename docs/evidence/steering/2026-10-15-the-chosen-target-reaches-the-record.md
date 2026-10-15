# Record the selected target before physical input

The fresh original case-18 run at `5830b1de5` confirmed its controlled body
after 4.0 seconds. By 11.9 seconds all three keys and the neutral input had
measured acceleration estimates. At 12.5 seconds the ship remained tracked
just above and left of the platform. The 13.1-second frame shows the ship
and platform exploding after contact. This is a real collision, not a
failure to preserve a visible ship's identity.

The game showed zero landings and one crash. Its report kept the attempt
as unread because the end control was read as `TRYJAGAIN J`. That missing
classification does not erase the observed failure. The recording contains
728 frames and 210 input events, with five late events dropped after the
raw-byte allowance was exhausted. It remains marked incomplete. The
recorded crash precedes that exhaustion.

Raw evidence is under
`artifacts/flash-campaign/2026-10-15-case18-observed-acquisition/`.
`frame-0000332.png` is the last shown approach used in this inspection;
`frame-0000350.png` shows the collision and explosion. These are screenshots
from the actual browser-path observation stream. Native prompt integration
and independent presentation qualification remain open.

The previous recording kept pixels, delivered inputs and control-model
evidence, but omitted the selected destination and planned arrival. The
opted-in journal now receives the made choice at `chose_in_play`, before
the actor's physical control call. It keeps the input, reason, current
body and target geometry, motion estimates, predicted plan and observation
clock. A target lookup retains its returned result for that record.

Recording never calls the target selector again. Its test makes an attempted
second selection raise, passes through the real choice hook and checks the
retained target, plan and capture identity. Changing the original objects
after recording cannot alter the snapshot. The journal still omits other
decision paths and cannot establish game acceptance of an input.

Validation: 47 recorder and arrival regressions passed; smoke passed 165
tests with one skipped. Ruff and whitespace checks passed. This checkpoint
adds diagnostic evidence, not a demonstrated gameplay improvement. The
next run must establish which destination and contact the controller was
actually pursuing before its approach can be repaired.

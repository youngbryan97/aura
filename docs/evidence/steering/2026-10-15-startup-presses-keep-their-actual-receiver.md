# A startup press keeps its actual receiver, 15 October 2026

The original case 3 repeat at `e72751c94` finished in 220.61 seconds with
12 moves and no win. Its closed recording has 234 pictures, 24 normal input
events, 94 pointer references and 126 pointer comparisons, with no drops.
Eight corrected cursor requests immediately follow held-pointer comparisons
whose pointer/target error is zero. The final picture is the kitchen; current
text is `UU`. Neither inventory progress nor completion was established.
The recording remains `presentation_qualified: false`.

Evidence: `artifacts/flash-campaign/2026-10-15-case3-held-pointer-drift/`.
Delivered clicks still revisit the same material under changing coordinate
labels. Persistent material/action memory remains open.

## Startup diagnosis

The repeat spent 25.25 seconds between announcing its startup press and
starting the program read. Profiling the complete local program reader took
0.166 seconds, which rules that reader out as the cause of this delay.

Three isolated physical-click probes reproduced a first press that left the
play cover visible. Giving the mouse a normal approach and held press did
not fix it. A fourth probe recorded the event recipient: hovering exposes
a hardware-acceleration warning, and the first click reaches that warning's
covering `DIV`. The next click reaches the SVG play control. The earlier
hit test describes the intended control, not the element that receives the
subsequent physical event.

Those probes inspected public DOM elements, captured pixels and trusted
mouse event paths. They did not use private emulator or game state.

## Repair and original-file measurement

The general cover observer now arms a finite event receipt for the current
DOM control. A trusted left-button down and click at the requested point
must both include that control in their composed paths. A measured mismatch
ends that control's startup wait promptly, then checks what is currently
exposed before retrying. Unknown delivery keeps the bounded wait; it is not
converted into success. A successful delayed start gets no second press.
Listeners abort after the observed click or a 2.5-second expiry. No warning
class, emulator name, title or game-specific constant enters this mechanism.

The normal startup function, on the same original case 3 in an owned
SwiftShader CPU browser, completed in **2.365 seconds**. Its two trusted
clicks reached the warning and then the SVG control; the cover disappeared.
The saved picture was inspected and shows the original title screen. This
is startup proof, not a gameplay win.

Raw evidence is under
`artifacts/flash-campaign/2026-10-15-case3-activation-probe/`, especially
`event-recipient/probe.json` and `normal-startup/probe.json` and its PNG.
The earlier full reader profile is in
`artifacts/flash-campaign/2026-10-15-case3-source-profile/`.

## Checks and boundaries

The focused browser and restart checks passed 43 tests in 12.86 seconds.
They include an initially exposed control, a notice created on physical
hover, a delayed valid start and synthetic events that cannot certify
physical delivery. Neither successful startup case clicks the running canvas.
Smoke passed 165 tests with one skipped in 46.99 seconds. Lint, compile,
layering, generated dependencies and writing passed. Governance lint still
reports the same 129 pre-existing ownership-baseline regressions; this change
did not increase that count or change the baseline.

G09, original game 9 and the reserved holdouts were not used. The resident
runtime was not started, stopped or restarted, and no model was loaded.
The native-app demo and the 56-game campaign remain unqualified.

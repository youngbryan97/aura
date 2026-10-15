# Current controls and exposed notices — 15 October 2026

Fresh CPU play at `4ab5de94b` confirmed a controlled body through ordinary
actor execution. The recording retained 118 frames and 118 control snapshots
without drops. The actor selected the body on 46 observations, first at
4.24 seconds into fast play. The main interval lasted 16.7 seconds at
6.8 pictures a second; it learned an up response. The game still reported
zero landings and one crash. This is not a win, native-app proof or demo
qualification. Raw evidence is in
`artifacts/flash-campaign/2026-10-15-case18-parts/`.

The visible body remains on screen when its identity is lost. At that
point seven continuing marks separate into two motion groups and the rigid
fit fails. A larger foreground region is assigned to another track; control
returns to discovery. That failure is retained for the next repair.

The preceding longer run also spent four fast intervals on a reloaded title
with an emulator notice. Two general navigation defects were found:

* The embedded JavaScript close-label regex used a Python string containing
  `\b`. Python converted those boundaries into backspace characters before
  the browser evaluated it. The string is now raw. Close coordinates also
  require a current hit on that element or its descendant, including inside
  open shadow roots. A covered close control supplies no click coordinate.
* The pursuit already recognized some misread navigation labels, but its
  fast-play menu gate ignored `way_on_shown` and used a stricter text test.
  The current reading now reaches that gate. Its existing finite navigation
  allowance is preserved; no game or label is special-cased.

A real SwiftShader Chromium fixture used a labelled close button with an
SVG child in a shadow root. With the old backspace expression it found no
control. With the repair it found the exposed button, refused it under a
cover, and a physical click dismissed it once. Both tests passed in 2.90
seconds. This was a generated DOM notice, not the original emulator.
The retry, blank-page and notice tests passed: 33 passed, one opt-in browser
test skipped in the ordinary invocation. The current/missing word-pixel and
menu-gate tests passed: 16 passed, one skipped. Ruff, layering, dependencies
and writing checks passed. Smoke before the final one-line menu gate change
was 165 passed and one skipped; the post-change smoke result follows in its
local proof artifact.

G09 remains in its existing process. No model worker or resident runtime
was started, stopped or reconfigured by these repairs.

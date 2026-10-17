# Keyboard targets are measured for the current input

Both `OnAPage` and `PlayingAsItHappens` stopped checking drawing focus after
the first successful query. A later focus change left the cached boolean
true while the drawing no longer received the key.

A real Chromium fixture reproduced that boundary. The first key reached
the canvas. After an input element acquired focus, a second key under the
cached policy did not reach it. Repeating the production focus script
restored delivery. This fixture used public DOM state and physical browser
key events. It did not run a game or an Aura model. The first tool attempt
used a relative executor path that the supervisor resolved from the repo;
it failed before opening a browser. The corrected absolute executor path
completed the fixture in 1.67 seconds.

All three actual Python consumer regressions failed before the repair:
ordinary press, held-key down and tap. The repair queries the drawing
target before each input through one shared helper. It keeps the existing
drawing-selection and document-input policy. A page without a drawing can
still receive document-level keys. A failed or malformed query supplies no
positive focus claim. Release does not wait for another focus query.

Opt-in interaction recording now keeps a `keyboard_target` event with the
page identity, query interval and status. Focus, transport completion and
the application's later visible effect remain separate facts. No receipt
claims game acceptance, a controlled body or task completion.

The focused group passed 58 checks; one pre-existing browser test was
deliberately deselected from that group. A separate real-browser fixture
then exercised the actual repaired Python ports. Normal, held and tapped
inputs each delivered both ArrowLeft and ArrowRight after an intervening
focus change. The drawing-free document fixture also received its key.
The software renderer was independently read from Chromium's system report:
ANGLE SwiftShader. The classical import boundary stayed active with no
loaded backend and no refused model import.

That actual-port fixture completed in 3.25 seconds. Its recorder closed
without drops, with seven focus receipts and 22 input-port events. Six
receipts reported a focused drawing; the drawing-free document's receipt
reported `not_focused` separately from its delivered key. Query intervals
included two initial activations of about 55 and 52 milliseconds; the other
five were below two milliseconds in this fixture. These timings are not a
whole-game performance bound.

Artifacts are under
`artifacts/flash-campaign/2026-10-16-browser-focus/`.
The ad hoc Playwright scripts are under `/tmp/playwright-test-aura-*`.
The retained repository regression is
`tests/test_drawing_focus_is_current_for_each_key.py` and its receipt
invariant is registered as a measured claim.

The preceding original case 18 recording had no focus receipts. These
tests therefore do not establish that focus caused those game losses.
Ordinary original-game measurement and native-app demo proof remain open.

The pointer path can also set the old cached flag without measuring a
keyboard target. Three additional regressions cover the first key from
that state. The final focused group completed successfully. The actual-port
Chromium fixture was repeated with that starting flag: all six drawing
keydown events and the document-level key were received. Its recorder
closed without drops, with seven focus receipts and 22 input-port events;
the bounded supervisor completed in 3.32 seconds. This is still a public
DOM fixture rather than original-game success.

The production change passed smoke (165 passed, one skipped), lint,
compilation, layering, dependency consistency and writing. The module-size
gate still reports four existing regressions and 197 aggregate excess
lines; this change did not modify that baseline.

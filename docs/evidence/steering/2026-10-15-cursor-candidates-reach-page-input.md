# Cursor candidates reach page input, 15 October 2026

The case-3 repeat at `4a68839a1` offered a dynamic navigation caption but
still finished on instructions. The public cursor path had separate gaps:
its grid sampled only cell centres, a caption associated with a hand kept
only a boolean, and a missing hand could fall back to an old position.
The earlier normal-action recording also omitted physical port coordinates.

The cursor sweep now also samples the middle of each unprobed boundary strip.
It adds 42 finite probes to the 108-cell main grid. Extra proposals retain
their actual positive samples. A uniform cursor supplies no local target;
the existing finite refinement remains bounded to 24 returned candidates.
This sampling does not cover every possible small target.

A current caption associated with one candidate keeps that candidate's
coordinates. Its text box centre no longer replaces the cursor point.
Competing equally near labels or multiple candidates for one caption remain
ambiguous; measured unnamed choices can still be offered. The ordinary
candidate and click-coordinate paths share that rule.

Before delivering a pointer-derived click, `OnAPage` checks the hand again
or searches the existing finite neighbourhood. A missing or unavailable
hand refuses that delivery and invalidates the sweep cache so the next
reading can reacquire it. Cursor affordance does not establish a continuing
object identity or guarantee that a neighbouring hand has the intended role.

Opted-in recording now includes normal page `press`, `click` and `carry`
requests and their delivery outcomes. Click and carry records preserve
actual CSS coordinates and the viewport bounds used. They use the separate
`navigation_input` event kind and cannot enter motor-control replay as fast
inputs. A failed call retains possible partial delivery. Neither port
completion nor a hand cursor supplies task completion.

The tests exercise the ordinary stream-to-caption-to-click functions against
an independent pointer surface, the actual port call and recorder, wrong and
missing hand evidence, competing captions, corrupt coordinates, transport
failure and uniform cursors. Fifty initial focused checks passed; the
expanded pointer/recording/reference suites passed 69. Smoke passed 165 with
one skipped. Ruff, layering (36 grandfathered), dependency consistency (168
packages) and writing passed without raising baselines.

Only the general code and test surfaces were used in this checkpoint.
The new path still requires an actual original-game measurement. It does
not qualify a win, the full campaign or native-app presentation readiness.

## Original case 3 at `06e2b604d`

The next CPU-only measurement used the same 0.8-minute allowance and
175-second supervisor bound. It finished in 92.04 seconds. The recording
contains five normal screen readings and five paired normal port requests
and completions, with no motor stream. The two `Continue` port calls were
at CSS `(1039.0016, 766.00192)` inside bounds `(0, 28, 1280, 768)`.
Both returned successfully. Their caption readings had no pointer anchor,
so they used the text centre; this run does not prove the new caption-anchor
path improved those clicks.

The game did advance through instructions into a room. This is established
by the final screenshot, which shows a staircase, a chair, a telephone and
the inventory icon. The report's `last_seen` still quoted the earlier
collectible instructions after later readings contained no text. Initial
commentary that the run had not reached gameplay relied on that stale text
and was corrected after the final image was inspected. Preserve both the
stale textual report and the screenshot. The report remained `won: false`,
`runs: ["unread"]`; no game completion is established.

The raw run is `artifacts/flash-campaign/2026-10-15-case3-owned-cursor/`.
It also shows the cursor cache needs a stronger scene/scope key: its current
word-set key can reuse samples between textless scenes or different crops.
The actual input records now make coordinate transport observable, but
they do not repair that missing scene identity.

The default `/Users/bryan/.aura/live-source` checkout was separately checked:
it was on `main` at `ca975c498`, with zero commits ahead and 131 behind
the then-current pushed branch. It did not contain `06e2b604d`.
This work executes from `codex-steering-1012`; pushing main has not updated
that default checkout or a native runtime. Its other untracked work was
left intact. The exclusive model lane remains untouched; source alignment
and ordinary native prompt proof are still required before recording a demo.

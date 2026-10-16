# Current source survives a revisit, 15 October 2026

The program reader reused an address-keyed saved summary before fetching
bytes. That summary preserves controls and some derived mechanics, but omits
exact action streams, visual declarations and code lines. A second visit
therefore lost evidence that the first current read had supplied. The inline
world reader also returned a saved world without inspecting current scripts.
A changed program at the same URL could keep its old read.

Each visit now reacquires bounded bytes through the page, normally from its
browser cache, and parses those bytes. Historical summaries stay historical.
Failed acquisition supplies no current source; it cannot fall back to that
summary. Malformed base64, oversized responses and empty bytes refuse before
parsing. Inline world reads inspect current scripts rather than returning a
world by page address alone. No program is executed by these readers.

## Measurements

The focused set passed 94 tests in 11.36 seconds. It includes unchanged and
changed files at the same URL, a new control key in the current script,
refused acquisition after a successful read, a changed inline world's parsed
speed, and missing current inline source. Source declarations and exact
instruction streams survive the unchanged-file revisit with their namespaces.

An owned CPU browser loaded original case 18 and called the normal program
reader twice without gameplay input. Both reads retained 707 code lines,
18 complete action streams and 33 declared visual assets. Both retained
source `swf:a00d125e025ac422091f0247845add583fd31547a6cedf2cd76bee1e1da14fc6`.
The first read took 0.103 seconds; the revisit took 0.070 seconds.
Evidence: `artifacts/flash-campaign/2026-10-15-current-program-revisit/`.
This does not establish asset registration in a live frame or a landing.

Smoke passed 165 tests, one skipped, in 46.45 seconds. Lint, compile, layering,
dependency and writing gates passed. Governance lint still reports the same
129 ownership-baseline regressions; its baseline was not changed.

## The preceding gameplay repeat

At `c5ce472b3`, original case 3 finished in 202.39 seconds with no win and
blank current text. Its closed recording retains 302 pictures, 30 normal input
events, 121 pointer references and 166 pointer comparisons, with no drops.
Fifteen normal input requests completed. Nine corrected material clicks
immediately follow held-pointer comparisons with zero pointer/target error.
Four ordinary screen-option records carry a prior-press cost of one. The
material-memory connection therefore reaches original-game choosing.

The final picture remains in the house. Collection was not established.
Some looks offer only a previously pressed point; other decisions choose the
middle of the picture. Better material memory has not supplied a complete
exploration policy or a visual goal binding.
Evidence: `artifacts/flash-campaign/2026-10-15-case3-material-action-memory/`.
Its `presentation_qualified` field remains false.

G09 and original game 9, holdouts and native runtime ownership remain
untouched. No accelerator model was loaded. The native-app demo and the
56-game campaign remain unqualified. Connecting source artwork to fresh
visible task references is still open.

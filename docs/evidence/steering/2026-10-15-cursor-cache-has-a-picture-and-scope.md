# Cursor cache custody, 15 October 2026

The cursor cache used a word set as the screen's identity. Different rooms
can have no words, and two crops of the same page can share most words.
Reusing their normalized samples across those frames can move a proposed
click or hide new affordances. This was identified in the real page path;
the source of each bad exploratory room action has not been isolated.

`OnAPage` now keys positive sweep reuse by the page identity/address,
captured surface, requested CSS scan rectangle, actual reading bounds,
picture dimensions and SHA-256 of immutable captured pixels. Missing
pictures and empty sweeps do not supply reusable negative evidence. A
changed picture with the same words must be swept again. Up to four
coordinate scopes can be retained.

Samples from a scan rectangle are converted to the reading's declared
coordinate frame before they are attached to labels or shapes. Delivery
looks up candidates in the actual bounds it is given. A nearby point does
not inherit another point's cursor evidence. Every normal input attempt
invalidates the sweep maps, including a failed call that might have partly
reached the page. Invalid declared bounds refuse the scope rather than
silently using assumed viewport geometry.

The focused cursor, input recording and reference suites passed 75 checks.
They cover two visually different textless scenes, reuse of identical quiet
pixels, crop conversion, missing pictures, empty maps, input invalidation,
invalid bounds and unrelated nearby clicks. Smoke passed 165 with one
skipped; compile, Ruff, layering, dependency consistency and writing passed.
No baseline was raised, and no model worker was started.

This preserves visual/coordinate custody for cursor proposals. A current
hand still does not prove a persistent target's identity or task role.
The change has not yet been exercised in another original-game run.
Native-app proof, source deployment and the campaign remain open.

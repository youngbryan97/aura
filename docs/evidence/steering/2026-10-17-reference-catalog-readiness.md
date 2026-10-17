# Reference metadata no longer delays the first image search

Reference acquisition and image matching shared one cadence deadline. A
catalog ready 30 milliseconds after submission still waited until the
one-second image-search interval ended. That wait consumed usable pictures
at the start of a short interaction without limiting expensive work.

The regression failed on the existing code: only the catalog operation had
been submitted when its result was already ready. The completed catalog
now releases the next search immediately. A submitted image search still
starts its full interval. The global single-worker permit, copied-capture
ownership, late-result checks and source-generation checks are unchanged.
Metadata supplies no source point, object identity or input authority.

The focused group passed 54 checks before adding the claim-registration
check. Its scheduling regression supplies a catalog at 10.03 seconds and
requires the first match then. Another match remains blocked at 11.02 and
becomes eligible at 11.03. This measures a deterministic scheduling
boundary, not original-game performance or a universal latency bound.

The preceding original case 18 run still lost with no confirmed body.
Twenty-seven positive focus receipts did not close that boundary. A bounded
pixel replay remains a diagnostic: it lacks complete actor scheduling and
input cleanup. It cannot prove a change in game outcome. No native runtime
was started or updated, and G09 and the holdouts remain excluded.

Final verification passed 55 focused checks, including the registered
claim. Smoke passed 165 checks with one skip. Lint, compilation, layering,
dependency consistency and writing passed; the bounded gate group completed
in 124.20 seconds. Artifacts are under
`artifacts/flash-campaign/2026-10-17-reference-startup/`.

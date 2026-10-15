# A campaign keeps each attempted and pending case

The archived-game runner saved its report only after all games returned.
A crash could lose every result. A missing game was printed and left out
of the returned results. Bundle validation also opened every program before
the runner selected games, including programs held out from inspection.

`tools/play_campaign_checkpoint.py` now saves the plan and a running case
before calling ordinary play, then saves each returned result. Errors stop
the pass with later cases pending. Missing programs remain failed attempts.
The existing result array remains available beside a `.state.json` record.
Both use atomic governed writes off the event loop.

The checkpoint has a nonblocking process lease. A settled pass can resume
only with the same revision, tracked source diff, manifest, selection,
exclusions, model endpoint and case budget. Recorded attempts are retained,
including failures. Interrupted and running cases require ownership and
input-release review; they are not blindly started again. This does not
claim an independent watchdog or a hard browser-cleanup deadline. The proof
supervisor remains the outer process timeout.

The runner excludes the retained holdout list by default. `--exclude` adds
protected case numbers. These exclusions precede selected asset validation,
browser startup and model connection. `--evaluate-held-out` is an explicit
evaluation choice. `--plan-only` writes the denominator without opening
program files or connecting to a model. The caller must supply `--out`;
an existing result cannot be overwritten by a new pass.

The case's `task_passed` stays null, its qualification stays unmeasured,
and `presentation_qualified` stays false even when Aura reports `won=true`.
This record preserves her claim; it does not independently verify it.

## CPU-only verification

- Focused campaign, original-asset contract and desktop-delivery tests:
  51 passed in 7.65 seconds.
- The preceding run, including the asynchronous write ratchet:
  55 passed in 103.61 seconds. The final plan-only test was added afterward.
- Smoke: 165 passed, one skipped, in 83.36 seconds.
- Ruff, changed-file compilation, layering, dependency consistency,
  changed-method size and writing checks passed.

Tests use invented case IDs, generated tiny SWFs, fake play results and
local temporary output files. They check real file leases and CLI writes,
including exclusion before asset validation, cancellation, foreign results,
storage refusal, missing programs and resume without replaying settled work.
No original game, browser play session, model job or native runtime was
started. G09 remained running and untouched.

This checkpoint improves campaign recovery and evidence. Reliable original
gameplay and independently witnessed presentation readiness remain open.

# R09: Preserve the source of a follow-up

## Observed failure

Resident turn `aura-chat-e22b9252-7c6f-42d9-9fb1-4a3830354d21` asked which
of the preceding five database recovery steps handled uncommitted work, and
why. The 01:59:17 request reached generation with three messages: two system
blocks and the current question. The prior answer was absent. The reply
described compensation in an application workflow instead of identifying the
database example's second step.

The desktop route inferred that a multi-part request was self-contained and
set its history window to zero unless another phrase classifier opted in.
Question structure did not establish independence from earlier conversation.

## Changes

- Ordinary engine turns retain the existing bounded default history window,
  including multi-part requests. Their current objective and mode selection
  remain unchanged. Typed action/state evidence keeps its existing ownership.
- The completed-exchange reader retains original delivered text from both
  process memory and durable storage. Its pair-count window remains bounded.
- The authenticated dialogue adapter preserves full text and line breaks;
  it emits complete pairs only. It no longer clips questions at 420 characters
  or answers at 520 characters, or flattens code indentation.
- The short display/context summary remains bounded. Model prompt allocation
  still belongs to the existing context assembler and inference budgets.

## Checks

Twelve focused cases passed in 21.94 seconds, including the actual follow-up
shape, compound independent questions, long answers, code formatting, and
recovery from real SQLite after clearing process memory. A broader selection
of history, recall, and follow-up contracts passed 21 tests in 115.21 seconds.
Smoke passed 164 tests with one skip in 132.21 seconds. Lint, compile,
governance lint, and layering passed; layering retains 37 grandfathered entries.

A separate broad run was interrupted after 17 passes in 92.18 seconds when
an unrelated fixture entered the runtime's 90-second boot wait. It is not
reported as a passing suite.

Live deployment and replay remain required. Existing model-budget compaction
can still shorten oversized dialogue; these checks establish lossless source
history, not unlimited context. R09 remains unchecked.

## Other live observations

At 02:22:22 the neural stream reported an episodic deferred queue at its
256-entry limit and one oldest-entry shed. At 02:22:14 diagnostics took
1887 ms of a 1000 ms scheduler period. These remain R08/R06 observations,
not evidence that conversation transport failed or that those items closed.

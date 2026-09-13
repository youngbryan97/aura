# R09: dialogue survives a model handoff

## Observed failure

The 2026-09-10 reading-group follow-up reached the 27B with its prior
conversation. After the quoted-recall gate rejected the answer, the fallback
ladder called the smaller model with the question and identity but no dialogue
messages. It then reported that it had no record of the earlier exchange.
The primary model's admitted evidence had not followed the model handoff.

## Repair

The existing exact-turn evidence custody now retains the runtime-attested
completed dialogue as a structured snapshot. The primary route records it
when it admits history. The fallback ladder passes the same user/assistant
messages to either local endpoint, followed by the current question once.
Clients receive copies; client normalization cannot mutate the next attempt.

If fallback precedes the primary history read, it uses the same principal-scoped
durable history reader. An already-read empty transcript does not trigger a
second read. A call without a conversation session does not select a person's
history. No identity prompt or task-specific recall wording changed.

## Checks

58 focused tests passed in 127.21 seconds across turn custody, fallback
answers, delivered history, fallback assessment, and self-description suites.
New cases cover 0, 1, and 40 prior pairs, child-task visibility, exact-turn
isolation, closed custody, unattested input, cold start, and in-place client
mutation between endpoints.

The route-level test passed (one selected, 393 deselected): the primary route
records exactly the history delivered to cognition, and the fallback reads it
after that route returns. Smoke passed 164 tests with one skip in 35.99 seconds.
Repository lint, compile, governance-lint and layering passed. The focused
custody, handoff and delivered-history files also pass their full Ruff rules.

Source-matched live replay and the full R09 matrix remain open. This repair
does not establish general recall correctness or close the release item.

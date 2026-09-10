# R09: stop during prefill

## Live failure

PID 20233 loaded `c6343d8480874d345086c5501f5111ef6d9921ab` on September 9.
Boot reported ready and source-current. The resident identity was
`Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15`.

The browser submitted a database-crash explanation at 01:33:45. Delivery
`aura-chat-ec5c6c26-c69e-4f43-b5da-7f7e7e40fdcf` entered a 2,016-token prefill
at 01:33:59. The Stop control wrote cancellation sequence 3, but no terminal
worker acknowledgement arrived. The parent reported
`generation_cancel_not_acknowledged` and unloaded the worker. The UI showed
one cancellation response at 01:34:16. Reload preserved that response, so the
terminal-history repair passed this cancellation/reload case. Worker
cancellation did not pass.

## Cause and repair

The prefill callback refreshed progress without reading the cancellation
channel. The first cancellation check was after the first decoded token,
which meant an entire prompt could keep running after Stop.

The callback now checks the exact job sequence at each materialized chunk.
An owned interrupt unwinds the generation and steering contexts before the
worker emits its request-bound terminal acknowledgement. The interrupted
cache lane is retired because a borrowed cache may have advanced beyond its
stored key. The model stays loaded. The same prefill callback serves ordinary
generation and streaming; no timeout or answer-quality rule was relaxed.

## Verification before deployment

- 27 focused tests passed in 27.88 seconds, including the installed MLX
  generator stopping at offsets 0, 4, and 8, then serving another request
  with the same model parameters.
- 17 focused prefill/client tests passed in 18.83 seconds.
- Smoke: 164 passed, one skipped, in 175.91 seconds.
- Ruff, compile, governance lint, and layering passed. Layering required
  explicit existing imports for narrative word markers and the measurement
  driver's effort snapshot/restore; no package-wide access was added.

Live replay after deployment remains required. R09 also retains its
crash-between-journal-and-history obligation. R05 is not closed by this work.

# R09 terminal history and worker cancellation

## Live observations

At 20:22 on September 8, PID 93849 ran commit `0651bb76d` with the resident
27B. The desktop Stop response control completed delivery
`aura-chat-5359fb9f-7163-4421-a4db-3d25feb1f3f1` with status
`cancelled_by_user` and one visible cancellation response.

This did not prove worker cancellation. The worker continued processing the
cancelled database question until at least 20:27:32. The next delivery,
`aura-chat-05b771c4-0ad0-4941-9542-a63578221978`, remained pending after a
browser reload and was stopped at 20:35:24. Its five-point answer was not
delivered. The neural feed reported generation cancellation as a fault,
event-loop stalls, runtime lease expiry, pipe saturation, and scheduler slips.
These are failures, not a successful reconnect/completion proof.

An earlier completed delivery,
`aura-chat-95b8ed4c-22e7-4a66-bc4c-39783189ef19`, had a sealed journal answer
but empty pending conversation history. The full-mind bounded return skipped
the route's normal history-finalization block. Reloading lost the answer.

## Repair

The delivery boundary captures pending exchanges and finalizes them using
the exact sealed answer. Intermediate route drafts do not enter history or
learning. Early returns use the same owner. Cancelled turns complete their
history without adding a learning example. Journal replay does not execute
the handler or duplicate a completed history write.

The authenticated cancellation owner now stops the existing execution token.
MLX distinguishes this owned stop from an unexpected task cancellation. It
signals the worker's existing sequence-bound cancellation channel before
clearing request ownership, then waits for the worker's terminal
acknowledgement. An unacknowledged stop requests worker recovery and reports
the actual failure. Unexpected cancellations remain faults.

## Verification

- 144 focused cancellation, MLX resilience, journal, and transcript tests passed.
- 11 existing history and persistence regression tests passed.
- Smoke after the worker-stop change: 164 passed, one skipped.
- Ruff, compile, and layering passed after the worker-stop change.
- The governance gate identified two direct writes in the new fusion
  certificate writer. It now uses the existing governed atomic file gateway;
  all 17 certificate tests pass. No certificate acceptance rule changed.

The new repair still needs live replay. A process crash between journal seal
and transcript finalization also needs a durable reconciliation test. R09 and
R05 remain unchecked.

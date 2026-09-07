# The wedge was two log handlers waiting on each other

**2026-09-07, 03:40 PDT. Live runtime PID 62288, uptime 1h01m.**

## What was observed

Every HTTP endpoint returned nothing. `curl` reported status `000` after six
seconds on `/health`, `/healthz`, `/readyz`, `/api/health`, `/api/status` and
`/api/telemetry/stream`. The process was alive, held 3.4 GB resident, and still
owned the listening socket on port 8000. `~/.aura/logs/aura_json.log` had not
been written since 03:07 — thirty-three minutes of silence from a process that
logs several times a second.

The last lines before the silence say what was happening:

    FlagshipDoctorDaemon observed event-loop lag without RAM pressure;
    deferring heavy self-healing and leaving recovery to foreground
    backpressure (lag=30.59s context=foreground_generation threshold=30.00s)
    Integrity snapshot refresh incident exceeded 8.000s; serving
    stale_while_revalidate snapshot
    SYSTEM STALL DETECTED: Component 'mind_tick' has not responded for 34.9s!
    Health snapshot refresh incident exceeded 8.000s; serving expired snapshot

This is the failure recorded on 2026-08-25 as "API wedges while the runtime
lives", whose mechanism had not been found.

## The mechanism

`kill -USR1 62288` dumped all 128 thread stacks. Eleven threads were blocked in
`logging.Handler.acquire`. Two of them explain the other nine:

    Thread 0x30402b000                    Thread 0x306043000
      logging.acquire        <- blocked     logging.acquire      <- blocked
      logging.handle                        logging.handle
      logging.warning                       terminal_monitor._ingest_error
      terminal_monitor._update_sepsis_state degraded_events._forward_to_
      terminal_monitor._ingest_error          terminal_monitor
      terminal_monitor.emit  <- holds A     omni_tracer.write_trace
      logging.handle                        omni_tracer.emit     <- holds B
                                            logging.handle

`logging.Handler.handle` holds the handler's own lock for the whole of `emit`.
`TerminalMonitor`'s handler logs a sepsis warning from inside its `emit`, and
that warning is dispatched to `OmniLogHandler`. `OmniLogHandler` calls the
terminal monitor from inside its `emit`. One thread held A and wanted B; the
other held B and wanted A. Nothing else in the process could log again, and
every code path that logs — which is every code path — stopped there.

Neither handler is wrong on its own. A handler's lock is an `RLock`, so a
handler that logs about itself is survivable. The cycle needs two.

## The fix

Repairing the two callers would leave the shape in place for the third handler
somebody adds. `core/observability/handler_reentry.py` guards the one funnel
every record passes through instead. While a thread is dispatching a record, a
further record from that thread is held rather than dispatched, and the held
records go out after the outermost dispatch returns — outside every handler
lock. Nothing is dropped while there is room to hold it, what is dropped is
counted, and `statistics()` reports both.

It is installed in `setup_logging` before the first handler is attached, so it
covers handlers this repository has not written yet.

## The proof

`tests/test_a_handler_that_logs_cannot_wedge_the_process.py` builds two
handlers that call into each other and drives them from two threads. With the
guard, 40 records reach both sinks and every deferred reply arrives. With
`handler_reentry.uninstall()`, the same two threads never return: the
reproduction was killed after 120 seconds having produced no output.

`make smoke` passes 164 with the guard installed.

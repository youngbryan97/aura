# R09 service observation and desktop lifetime

## Live failure

The UI Reboot action replaced PID 77385 with PID 15212. Boot reported matching
commit `20e878a9098c1671974fc74d1847a44f721749c4`, matching workspace and shell
hashes, stable capture, and the unchanged resident 27B model. Direct-launch
source verification passed; this was not a signed immutable release launch.

Turn `aura-chat-4a03d137-5e57-4662-84ed-53e602332f81` started at 18:14:21.
The worker completed 2,315 prefill tokens and 71 generated tokens at 18:15:34.
No terminal answer reached the desktop. The neural stream reported 6.023 and
21.568 second loop stalls, a memory-monitor blocking hold, diagnostic and
telemetry overruns, and an adaptive-immunity fsync under a lock.

The desktop then exited through `CancelledError` from its direct await of
`SupervisionTree._monitor_task`. The shutdown stack records `desktop_cancelled`,
not a user signal or a memory-sentinel kill. The caller that cancelled the
monitor is not identified by that stack. The catalog latency replay therefore
did not produce a comparable post-change stabilization measurement.

## Reproduced ownership defects and repair

- A probe timeout stopped both an adopted service and its dependent in a
  focused reproduction. The reconciler conflated unavailable observation with
  a measured false liveness result. Unavailable reads now publish degraded,
  unknown liveness/readiness observations, preserve existing service custody,
  and retry observation. They do not consume a restart budget. A measured
  false result still invokes the existing restart/stop policy.
- A managed actor-monitor replacement must not own the desktop lifetime.
  The desktop now waits on its root shutdown event and the existing global
  shutdown latch. The GUI also reads/writes that lifecycle owner instead of
  using a replaceable monitor's running flag as a quit command.
- No health pass is fabricated: critical readiness remains false until a
  fresh successful observation arrives. Explicit task cancellation still
  propagates; normal user close and OS signal handling remain active.

The initial reproduction failed before the repair. The focused lifecycle,
control-plane, actor-supervision and root-signal selection passed 53 tests,
including a real asynchronous deadline and an observer explicitly raising
`TimeoutError`. Smoke passed 164 tests with one skip; lint, compile,
governance-lint and layering passed.

Deployment and live replay remain required. The independent durable-history
bootstrap gap is also open: a fresh window after reboot reads an empty
process-local transcript despite the existing durable conversation reader.
R09, R06 and R08 are not closed by this checkpoint.
